//! `omni model` — inspect the model seam.
//!
//! Two subcommands, and neither of them talks to a model:
//!
//! - `omni model peregrine [--url U] [--key K]` builds a client and prints what
//!   it would send, then tries `/health`. It is a **smoke test**, and it reports
//!   a server that is down as down rather than as an empty reply.
//! - `omni model dialects` prints the peregrine dialect differences this build
//!   is built against.
//!
//! Printing the request before sending it is the point of the first one. A
//! harness whose first contact with a model is a completion is a harness whose
//! first bug report is "it said something odd" — and the request is where the
//! bug will be.

use std::process::ExitCode;

use omni_model::{ChatRequest, Message, Model as _};
use omni_model_peregrine::{Config, Peregrine, WireRequest, build_request, wire};

use crate::cli::Usage;

pub fn dispatch(args: &[String]) -> Result<ExitCode, Usage> {
    let Some(sub) = args.first() else {
        return Err(Usage::new("model needs a subcommand: peregrine | dialects"));
    };
    let rest = &args[1..];
    match sub.as_str() {
        "peregrine" => peregrine(rest),
        "dialects" => dialects(),
        other => Err(Usage::new(format!("unknown `omni model` subcommand {other:?}"))),
    }
}

fn parse_flags(args: &[String]) -> Result<Flags, Usage> {
    let mut f = Flags::default();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--url" => {
                f.url = Some(
                    args.get(i + 1).ok_or_else(|| Usage::new("--url needs a value"))?.clone(),
                );
                i += 2;
            }
            "--key" => {
                f.key = Some(
                    args.get(i + 1).ok_or_else(|| Usage::new("--key needs a value"))?.clone(),
                );
                i += 2;
            }
            "--model" => {
                f.model = Some(
                    args.get(i + 1).ok_or_else(|| Usage::new("--model needs a value"))?.clone(),
                );
                i += 2;
            }
            "--max-tokens" => {
                let v = args.get(i + 1).ok_or_else(|| Usage::new("--max-tokens needs a value"))?;
                f.max_tokens = Some(
                    v.parse::<usize>()
                        .map_err(|_| Usage::new(format!("--max-tokens {v:?} is not a number")))?,
                );
                i += 2;
            }
            "--complete" => {
                f.complete = true;
                i += 1;
            }
            other => return Err(Usage::new(format!("unknown option {other:?}"))),
        }
    }
    Ok(f)
}

#[derive(Default)]
struct Flags {
    url: Option<String>,
    key: Option<String>,
    model: Option<String>,
    complete: bool,
    max_tokens: Option<usize>,
}

fn peregrine(args: &[String]) -> Result<ExitCode, Usage> {
    let flags = parse_flags(args)?;
    let config = Config {
        base_url: flags.url.unwrap_or_else(|| Config::default().base_url),
        api_key: flags.key,
        ..Config::default()
    };

    let history = vec![Message::user("Reply with exactly: ok")];
    // 16 is enough for a non-thinking model and far too few for one that emits
    // reasoning tokens first — which surfaces as `finish_reason: length` with an
    // empty body. That is a useful thing for a smoke test to show, so the flag
    // exists rather than the number being quietly generous.
    let req = ChatRequest::new(&history).with_max_tokens(flags.max_tokens.unwrap_or(16));
    let req = ChatRequest { model: flags.model.as_deref(), ..req };
    let wire_request: WireRequest = build_request(&config, &req)
        .map_err(|e| Usage::new(format!("building the request: {e}")))?;

    println!("would POST {}", wire_request.url);
    for (k, v) in &wire_request.headers {
        // The key is a credential: show that a header exists, never its value.
        let shown = if k == "authorization" { "<redacted>" } else { v.as_str() };
        println!("  {k}: {shown}");
    }
    println!("body:");
    println!("{}", pretty_json(&wire_request.body));
    println!();

    // Then actually try. Reported separately from the request, because "the
    // request looks right" and "the server answered" are different findings and
    // conflating them is how a dead server gets debugged as a malformed request.
    let mut client = Peregrine::new(config.clone()).map_err(|e| Usage::new(e.to_string()))?;
    if let Some(name) = &flags.model {
        client = client.named(name).map_err(|e| Usage::new(e.to_string()))?;
    }

    // `/health` is not part of the chat dialect and not every OpenAI-compatible
    // server has it (ollama returns 404 there). A 404 therefore means "this
    // server has no health endpoint", not "this server is down" — and the two
    // want opposite conclusions, so they are reported separately.
    match probe_health(&client) {
        Ok(body) => {
            println!("health: {}", truncate(&body, 400));
        }
        Err(e) if e.to_string().contains("HTTP 404") => {
            println!("health: no endpoint at /health (normal for non-peregrine servers)");
        }
        Err(e) => {
            // A server that is down is reported as down. Rule 10.
            println!("server: NOT reachable ({e})");
            println!("the request above is correct; nothing answered it.");
            return Ok(ExitCode::FAILURE);
        }
    }

    if !flags.complete {
        println!();
        println!("not sending. Pass --complete to make a real call.");
        return Ok(ExitCode::SUCCESS);
    }

    // The real round trip. This is the only part of omni that exercises the
    // transport, the status classification and the reply parser together, and it
    // is what a gate for the model seam actually means.
    println!();
    println!("sending...");
    match client.chat(&req) {
        Ok(reply) => {
            println!("finish_reason: {}", reply.finish_reason.as_str());
            println!("reply: {}", truncate(reply.text().unwrap_or("(no text)"), 400));
            if let Some(u) = reply.usage {
                println!("usage: {} prompt + {} completion = {}", u.prompt_tokens, u.completion_tokens, u.total());
            }
            Ok(ExitCode::SUCCESS)
        }
        Err(e) => {
            println!("call failed: {e}");
            println!("retryable: {}", e.retryable());
            Ok(ExitCode::FAILURE)
        }
    }
}

fn probe_health(client: &Peregrine) -> Result<String, omni_core::Error> {
    // `/health` is unauthenticated on the server side, so this needs no key and
    // cannot fail for a reason that has anything to do with the credentials.
    let url = format!("{}/health", wire::normalize_base(client.base_url()));
    client.health(&url)
}

/// Re-indent a JSON string. A body printed on one 400-character line is a body
/// nobody reads, and this is the part of the smoke test a person is looking at.
fn pretty_json(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| serde_json::to_string_pretty(&v).ok())
        .unwrap_or_else(|| format!("(not valid JSON; {body}"))
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n).collect();
        format!("{head}…")
    }
}

fn dialects() -> Result<ExitCode, Usage> {
    println!("omni speaks peregrine's dialect. The differences from OpenAI's:");
    println!();
    println!("  {:<22} {:<14} {:<14}", "field", "openai", "peregrine");
    println!("  {:<22} {:<14} {:<14}", "temperature", "1.0", "0.0");
    println!("  {:<22} {:<14} {:<14}", "top_p", "1.0", "0.95");
    println!("  {:<22} {:<14} {:<14}", "top_k", "absent", "sent, before top_p");
    println!("  {:<22} {:<14} {:<14}", "tool_choice", "4 values", "only \"none\"");
    println!();
    println!("peregrine accepts and discards:");
    println!("  model, n, stop, seed, logprobs, presence_penalty,");
    println!("  frequency_penalty, response_format, stream_options");
    println!();
    println!("omni sends none of them, because a field the server drops is a");
    println!("field whose value can drift out of sync with what the server does");
    println!("without anything failing.");
    println!();
    println!("Verified by the `peregrine` certificate group:");
    println!("  scripts/verify-certificates.sh --group peregrine");
    Ok(ExitCode::SUCCESS)
}

