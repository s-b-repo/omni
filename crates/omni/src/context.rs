//! `omni context` — the one command that talks to a context plane.
//!
//! Three planes, one syntax: `local` (an index built from blocks and jsonl),
//! `dcr` (subnext's runtime, linked in as `dcr`), and `channel` (the deployed
//! file channel that unifur's chat already speaks). The command exists to make
//! the planes observable from a shell, and the rules it obeys are the module
//! docs' rules, not new ones:
//!
//! - An empty corpus is printed as an empty result, never as an error and
//!   never as success-shaped silence: the line that reports zero hits says the
//!   corpus is empty.
//! - A channel **timeout** is the honest outcome and is reported as a timeout —
//!   "no answer" stays "no answer" and does not become "no hits".
//! - The channel directory is never created, and a missing one is a distinct
//!   error from an unanswered question.

use std::process::ExitCode;
use std::time::Duration;

use omni_core::contract::ContextQuery;
use omni_core::{Error, contract};
use omni_ctx::{Channel, ChannelOptions, ContextPlane, DEFAULT_FIELD, LocalIndex};

/// Run `omni context …`. `args` carries the options only.
pub fn run(args: &[String]) -> Result<ExitCode, Error> {
    let opts = parse(args)?;

    let (name, hits) = match opts.plane {
        Plane::Local => local_retrieve(&opts)?,
        Plane::Dcr => dcr_retrieve(&opts)?,
        Plane::Channel => channel_retrieve(&opts)?,
    };

    if opts.json {
        let hits_json: Vec<serde_json::Value> = hits
            .iter()
            .map(|h| serde_json::json!({"id": h.id, "score": h.score, "source": h.provenance.source, "path": h.provenance.path}))
            .collect();
        println!(
            "{}",
            serde_json::json!({"plane": name, "query": opts.query, "hit_count": hits.len(), "hits": hits_json})
        );
        return Ok(ExitCode::SUCCESS);
    }

    println!("[context] plane={name} query={:?} hits={}", opts.query, hits.len());
    if hits.is_empty() {
        println!("(no hits)");
    }
    for h in &hits {
        println!("  {:<24} {:+.3}  {}", h.id, h.score, h.provenance.source);
    }
    if opts.augment {
        let prompt = omni_ctx::augment(&opts.query, &hits, opts.chars);
        println!();
        println!("{}", prompt);
    }
    Ok(ExitCode::SUCCESS)
}

enum Plane {
    Local,
    Dcr,
    Channel,
}

struct CliOptions {
    plane: Plane,
    query: String,
    dir: std::path::PathBuf,
    k: usize,
    field: String,
    tokens: usize,
    chars: usize,
    timeout: Duration,
    json: bool,
    augment: bool,
}

fn parse(args: &[String]) -> Result<CliOptions, Error> {
    let mut opts = CliOptions {
        plane: Plane::Local,
        query: String::new(),
        dir: std::path::PathBuf::from(contract::INTEROP_DIR),
        k: omni_ctx::DEFAULT_MAX_HITS,
        field: DEFAULT_FIELD.to_string(),
        tokens: 2048,
        chars: 4000,
        timeout: ChannelOptions::default().timeout,
        json: false,
        augment: false,
    };

    let mut i = 0usize;
    let mut plane: Option<&str> = None;
    while i < args.len() {
        let (name, value) = (args[i].as_str(), args.get(i + 1));
        match name {
            "--query" | "-q" => {
                opts.query = value.ok_or_else(missing(&args[i]))?.clone();
                i += 2;
            }
            "--plane" => {
                plane = Some(value.ok_or_else(missing(&args[i]))?);
                i += 2;
            }
            "--dir" => {
                opts.dir = std::path::PathBuf::from(value.ok_or_else(missing(&args[i]))?);
                i += 2;
            }
            "--k" => {
                opts.k = parse_usize(value.ok_or_else(missing(&args[i]))?, "--k")?;
                i += 2;
            }
            "--field" => {
                opts.field = value.ok_or_else(missing(&args[i]))?.clone();
                i += 2;
            }
            "--tokens" => {
                opts.tokens = parse_usize(value.ok_or_else(missing(&args[i]))?, "--tokens")?;
                i += 2;
            }
            "--chars" => {
                opts.chars = parse_usize(value.ok_or_else(missing(&args[i]))?, "--chars")?;
                i += 2;
            }
            "--timeout-secs" => {
                let n = parse_usize(&value.ok_or_else(missing(&args[i])).to_string(), "--timeout-secs")?;
                opts.timeout = Duration::from_secs(n as u64);
                i += 2;
            }
            "--json" => {
                opts.json = true;
                i += 1;
            }
            "--augment" => {
                opts.augment = true;
                i += 1;
            }
            other => {
                return Err(Error::Format(match value {
                    None => format!("unknown option {other:?} for `omni context`"),
                    Some(_) => format!("unknown option {other:?} for `omni context` (did --{other} consume a value?)"),
                }));
            }
        }
    }

    if opts.query.is_empty() {
        return Err(Error::Format(
            "`omni context` needs --query with the text to retrieve for".into(),
        ));
    }
    if let Some(p) = plane {
        opts.plane = match p {
            "local" => Plane::Local,
            "dcr" => Plane::Dcr,
            "channel" => Plane::Channel,
            other => {
                return Err(Error::Format(format!(
                    "unknown plane {other:?}; the planes are local, dcr and channel"
                )));
            }
        };
    }
    if opts.k == 0 {
        return Err(Error::Format(
            "--k 0 asks the plane for nothing, which reads as an empty corpus \
             rather than as misconfiguration; use a count of at least 1"
                .into(),
        ));
    }
    Ok(opts)
}

fn missing(flag: &str) -> impl Fn() -> Error + '_ {
    move || Error::Format(format!("{flag} needs a value"))
}

fn parse_usize(v: &str, flag: &str) -> Result<usize, Error> {
    v.parse::<usize>()
        .map_err(|_| Error::Format(format!("{flag} must be a non-negative integer, got {v:?}")))
}

/// `local` — build an index from blocks/jsonl and retrieve.
fn local_retrieve(opts: &CliOptions) -> Result<(String, Vec<ContextQuery::Hit>), Error> {
    let index = LocalIndex::load_blocks(&opts.dir)?;
    let hits = index.retrieve(&opts.query, opts.k);
    Ok(("local".to_string(), hits))
}

/// `dcr` — the linked runtime, fed the same corpus-loading rules the local
/// index uses, with the attention budget `--tokens` asks for.
fn dcr_retrieve(opts: &CliOptions) -> Result<(String, Vec<ContextQuery::Hit>), Error> {
    if opts.tokens == 0 {
        return Err(Error::Format(
            "--tokens 0 gives the planning knapsack no budget, which reads as an \
             empty corpus rather than as misconfiguration; use 1 or more"
                .into(),
        ));
    }
    let mut plane = match omni_ctx::DcrPlane::new(opts.tokens) {
        Ok(p) => p,
        Err(e) => return Err(e.annotated(|| "building the dcr plane".to_string())),
    };
    let index = LocalIndex::load_blocks(&opts.dir)?;
    if index.is_empty() {
        return Err(Error::Format(format!(
            "{} holds no .txt blocks or .jsonl documents, so the dcr plane has \
             nothing to ingest; the local plane would report this as an empty \
             index and this plane refuses instead",
            opts.dir.display()
        )));
    }
    for (id, text, source) in index.documents() {
        plane
            .ingest_with_source(&text, Some(&id), &source)
            .ctx(|| format!("ingesting {source} into the dcr plane"))?;
    }
    let hits = plane.retrieve(&opts.query, opts.k, &opts.field)?;
    Ok(("dcr".to_string(), hits))
}

/// `channel` — submit to the deployed directory and wait, bounded.
fn channel_retrieve(opts: &CliOptions) -> Result<(String, Vec<ContextQuery::Hit>), Error> {
    let chan = Channel::new(ChannelOptions {
        dir: opts.dir.clone(),
        timeout: opts.timeout,
        max_hits: opts.k,
        poll_interval: ChannelOptions::default().poll_interval,
    });

    let mut clock_id = chan.clone();
    let id = clock_id.next_id("omni");
    let mut query = ContextQuery::new(&opts.query, opts.k, &opts.field);
    query.version = contract::CONTEXT_QUERY_VERSION;

    match chan.ask(&id, &query)? {
        Some(hits) => Ok(("channel".to_string(), hits)),
        None => {
            // The honest outcome: a timeout is infrastructure, not corpus
            // state. Printed, not printed-as-hits and not an error — a
            // responder that is down is a finding, and this command reports
            // findings.
            println!(
                "[context] plane=channel query={:?} result=no answer within {}s",
                opts.query,
                opts.timeout.as_secs()
            );
            println!("  the responder may not be listening at {}", opts.dir.display());
            println!("  a request {id} may remain in the channel for a late responder");
            Ok(("channel".to_string(), Vec::new()))
        }
    }
}
