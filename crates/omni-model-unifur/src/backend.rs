//! The `Model` impl itself. Gated behind `#[cfg(feature = "dblocks")]` so a
//! default build compiles and stays honest about having no backend.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use omni_core::{Context, Error};
use omni_model::prelude::*;

/// Decoding strategy passed to `dblocks lm generate --sampling`.
///
/// Carried as an enum rather than two floats so the determinism contract on
/// [`Model::deterministic_at_zero`] is exact: greedy is deterministic, top-k
/// is not, and no caller has to reason about "but temperature 0.0 with top-k
/// is actually..." (it is not, because top-k 1 is greedy under a different
/// name and that path is `Greedy`, not `TopK{top_k: 1, ..}`).
#[derive(Debug, Clone, Copy)]
pub enum Sampling {
    /// `--sampling greedy`. Byte-for-byte reproducible at the same checkpoint.
    Greedy,
    /// `--sampling topk --top-k K --temperature T`.
    TopK { top_k: u32, temperature: f32 },
}

/// A `Model` backed by `dblocks lm generate`.
#[derive(Debug, Clone)]
pub struct Dblocks {
    binary: PathBuf,
    checkpoint: PathBuf,
    name: String,
    timeout: Duration,
    sampling: Sampling,
}

impl Dblocks {
    /// Build a backend. Verifies the binary and checkpoint exist up front;
    /// a per-call 500 from a missing checkpoint would disguise the absence
    /// as a transport failure (AGENTS.md rule 10).
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when the binary is not a regular file or the
    /// checkpoint path does not exist.
    pub fn new(binary: impl Into<PathBuf>, checkpoint: impl Into<PathBuf>) -> Result<Self, Error> {
        let binary = binary.into();
        let checkpoint = checkpoint.into();
        if !binary.is_file() {
            return Err(Error::Format(format!(
                "dblocks binary not found or not a regular file: {}",
                binary.display()
            )));
        }
        if !checkpoint.is_file() {
            return Err(Error::Format(format!(
                "dblocks checkpoint not found: {}",
                checkpoint.display()
            )));
        }
        let name = derive_name(&checkpoint);
        Ok(Self {
            binary,
            checkpoint,
            name,
            timeout: Duration::from_secs(600),
            sampling: Sampling::Greedy,
        })
    }

    /// Change the per-call timeout. Not currently used (std::process::Command
    /// has no built-in timeout), but held so a future `std::process::Child`
    /// with a watchdog thread does not need a signature change.
    #[must_use]
    pub fn with_timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
        self
    }

    /// Switch to top-k sampling. `deterministic_at_zero()` then returns
    /// `false` — top-k is not deterministic at temperature 0.0 in general,
    /// and the one case that would be (`top_k = 1`) is already available
    /// as [`Sampling::Greedy`].
    #[must_use]
    pub fn with_sampling(mut self, s: Sampling) -> Self {
        self.sampling = s;
        self
    }

    /// Flatten a chat history into the single prompt string `dblocks lm
    /// generate` expects.
    ///
    /// Each turn is rendered as `[role] text`, separated by a blank line.
    /// Tool-call turns and tool-result turns both land here as `[assistant]`
    /// and `[tool]` lines respectively; a tool-call-only assistant turn is
    /// skipped (it has no text), which keeps the concatenation shaped like
    /// a conversation and not like a JSON envelope.
    fn render_prompt(messages: &[Message]) -> String {
        let mut out = String::new();
        for m in messages {
            let body = m.text().unwrap_or("");
            if body.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push('[');
            out.push_str(m.role());
            out.push_str("] ");
            out.push_str(body);
        }
        out
    }

    /// Parse a `dblocks lm generate` stdout blob and recover the continuation.
    ///
    /// The format is:
    /// ```text
    /// backend=... device=... dtype=...
    /// architecture from ...
    /// loaded ...
    /// layers=... hidden=... context=... attention=... positions=...
    /// decoder=... | N tokens in Tms
    /// ---
    /// <prompt echoed verbatim><generated continuation>
    /// ---
    /// ```
    ///
    /// Everything between the first two `---` fence lines is the fenced
    /// block; stripping the prompt as a prefix (if it appears) leaves the
    /// continuation. A missing fence is an `Error::Format`, because a toy
    /// checkpoint that printed only its preamble and crashed before
    /// generating would otherwise look like an empty reply.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when fewer than two `---` lines are present in
    /// `stdout`.
    fn extract(stdout: &str, prompt: &str) -> Result<String, Error> {
        let fences: Vec<usize> = stdout
            .lines()
            .enumerate()
            .filter_map(|(i, line)| (line.trim() == "---").then_some(i))
            .take(2)
            .collect();
        if fences.len() < 2 {
            let head: String = stdout.chars().take(400).collect();
            return Err(Error::Format(format!(
                "dblocks lm generate output had {} fence line(s), expected 2; first 400 chars: {head:?}",
                fences.len()
            )));
        }
        let start = fences[0];
        let end = fences[1];
        let lines: Vec<&str> = stdout.lines().collect();
        let fenced = lines
            .get(start + 1..end)
            .map(|slice| slice.join("\n"))
            .unwrap_or_default();
        let fenced = fenced.trim();
        let prompt_trim = prompt.trim();
        if !prompt_trim.is_empty() && fenced.starts_with(prompt_trim) {
            Ok(fenced[prompt_trim.len()..].trim_start().to_string())
        } else {
            Ok(fenced.to_string())
        }
    }

    /// Build the argv `dblocks lm generate` is invoked with. Split out from
    /// [`chat`] so a test can assert the shape without a real subprocess —
    /// shell-out correctness is a property of the argv, and a test that
    /// could only check it by reading the running binary's /proc/self/cmdline
    /// is proving something about the operating system rather than the code.
    fn build_argv(&self, prompt: &str, max_tokens: usize) -> Vec<String> {
        let mut argv: Vec<String> = vec![
            "lm".into(),
            "generate".into(),
            "--checkpoint".into(),
            self.checkpoint.to_string_lossy().into_owned(),
            "--prompt".into(),
            prompt.to_string(),
            "--max-new".into(),
            max_tokens.to_string(),
            "--cached".into(),
        ];
        match self.sampling {
            Sampling::Greedy => {
                argv.push("--sampling".into());
                argv.push("greedy".into());
            }
            Sampling::TopK { top_k, temperature } => {
                argv.push("--sampling".into());
                argv.push("topk".into());
                argv.push("--top-k".into());
                argv.push(top_k.to_string());
                argv.push("--temperature".into());
                argv.push(format!("{temperature:.4}"));
            }
        }
        argv
    }
}

fn derive_name(checkpoint: &Path) -> String {
    let stem = checkpoint
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    format!("dblocks:{stem}")
}

impl Model for Dblocks {
    fn chat(&mut self, req: &ChatRequest<'_>) -> Result<ChatReply, Error> {
        let prompt = Self::render_prompt(req.messages);
        let argv = self.build_argv(&prompt, req.max_tokens.max(1));

        let output = Command::new(&self.binary)
            .args(&argv)
            .output()
            .ctx(|| format!("spawning dblocks at {}", self.binary.display()))?;

        if !output.status.success() {
            let code_text = output
                .status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "signal".to_string());
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stderr_head: String = stderr.chars().take(800).collect();
            return Err(Error::Format(format!(
                "dblocks exited {code_text}; stderr: {stderr_head}"
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let text = Self::extract(&stdout, &prompt)?;

        // 4-chars-per-token heuristic for the length vs stop signal. `dblocks`
        // does not emit a stop reason over stdout, and asking `lm generate`
        // to print one is a dblocks change, not an omni one. Documented in
        // the crate header.
        let approx_tokens = text.len().saturating_add(3) / 4;
        let budget = req.max_tokens.max(1);
        let finish = if approx_tokens >= budget {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };

        Ok(ChatReply {
            message: Message::assistant(text),
            finish_reason: finish,
            // `None` rather than a fabricated count. A caller that needs
            // counts should drive a real tokenizer; a caller that does not
            // sees an explicit "backend did not report" instead of a zero
            // that reads as "zero tokens".
            usage: None,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn deterministic_at_zero(&self) -> bool {
        matches!(self.sampling, Sampling::Greedy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shape every test's unexpected-variant path takes: return an
    /// `Error::Format` naming what was expected and what arrived. Avoids the
    /// `assert!(false, ...)` idiom, which expands to `panic!` and would be
    /// caught by the strict gate.
    fn unexpected(what: &str, got: &str) -> Error {
        Error::Format(format!("expected {what}, got: {got}"))
    }

    #[test]
    fn render_prompt_joins_with_blank_lines_and_role_tags() -> Result<(), Error> {
        let msgs = [
            Message::system("be helpful"),
            Message::user("hi"),
            Message::assistant("hello"),
        ];
        let rendered = Dblocks::render_prompt(&msgs);
        if rendered != "[system] be helpful\n\n[user] hi\n\n[assistant] hello" {
            return Err(unexpected(
                "role-tagged blank-line-joined render",
                &rendered,
            ));
        }
        Ok(())
    }

    #[test]
    fn render_prompt_drops_text_free_turns() -> Result<(), Error> {
        // An assistant turn with only tool calls has `content: None`.
        // Rendering it as `[assistant] ` would leave a trailing role tag
        // with no text, which a greedy decoder reads as "the model started
        // an answer and gave up".
        let msgs = [
            Message::user("hi"),
            Message::assistant_tool_calls(Vec::new()),
            Message::user("still here"),
        ];
        let rendered = Dblocks::render_prompt(&msgs);
        if rendered != "[user] hi\n\n[user] still here" {
            return Err(unexpected(
                "text-free turn dropped from render",
                &rendered,
            ));
        }
        Ok(())
    }

    #[test]
    fn extract_picks_the_fenced_block_and_strips_the_prompt() -> Result<(), Error> {
        let stdout = "\
backend=cpu device=NdArray(Cpu) dtype=f32
architecture from ./ck.state: layers=4 hidden=32 context=128
loaded ./ck.mpk
layers=4 hidden=32 context=128 attention=4D positions=learned routing_state=0 vocab=259
decoder=greedy+kv-cache | 24 tokens in 497.4ms
---
[user] hi
there is more here
---
";
        let got = Dblocks::extract(stdout, "[user] hi")?;
        if got != "there is more here" {
            return Err(unexpected("`there is more here`", &got));
        }
        Ok(())
    }

    #[test]
    fn extract_returns_fenced_body_verbatim_when_prompt_is_not_a_prefix() -> Result<(), Error> {
        // If the toy model munges whitespace or the tokenizer normalises the
        // prompt bytes, the prompt may not appear at the head of the fenced
        // block. Returning the fenced body verbatim is strictly more honest
        // than erroring: the caller can see what dblocks actually said.
        let stdout = "preamble\n---\n<unexpected head> output\n---\n";
        let got = Dblocks::extract(stdout, "what the caller sent")?;
        if got != "<unexpected head> output" {
            return Err(unexpected("verbatim fenced body", &got));
        }
        Ok(())
    }

    #[test]
    fn extract_rejects_output_with_no_fences() -> Result<(), Error> {
        // A dblocks run that printed its preamble and then crashed would
        // leave an un-fenced stdout. Treating that as an empty reply would
        // make a backend-side crash look like a quiet model.
        let stdout = "backend=cpu device=NdArray(Cpu) dtype=f32\nloaded ./ck.mpk\n";
        match Dblocks::extract(stdout, "whatever") {
            Err(Error::Format(msg)) => {
                if !msg.contains("fence") {
                    return Err(unexpected("`fence` in the error message", &msg));
                }
                Ok(())
            }
            Ok(text) => Err(unexpected(
                "Error::Format for no-fences, got Ok",
                &text,
            )),
            Err(other) => Err(unexpected(
                "Error::Format for no-fences",
                &format!("{other:?}"),
            )),
        }
    }

    #[test]
    fn extract_rejects_output_with_one_fence() -> Result<(), Error> {
        // A single `---` with no closing fence is not a valid block.
        let stdout = "backend=cpu device=NdArray(Cpu) dtype=f32\n---\nhalf\n";
        match Dblocks::extract(stdout, "whatever") {
            Err(Error::Format(msg)) => {
                if !msg.contains("1 fence line(s), expected 2") {
                    return Err(unexpected(
                        "fence-count diagnostic in the error message",
                        &msg,
                    ));
                }
                Ok(())
            }
            Ok(text) => Err(unexpected(
                "Error::Format for one-fence, got Ok",
                &text,
            )),
            Err(other) => Err(unexpected(
                "Error::Format for one-fence",
                &format!("{other:?}"),
            )),
        }
    }

    #[test]
    fn new_rejects_a_missing_binary() -> Result<(), Error> {
        let err = match Dblocks::new("/nonexistent/dblocks", "/also/nonexistent.mpk") {
            Err(e) => format!("{e}"),
            Ok(_) => return Err(unexpected("Err for missing binary", "Ok")),
        };
        if !err.contains("dblocks binary not found") {
            return Err(unexpected("`dblocks binary not found` in error", &err));
        }
        Ok(())
    }

    #[test]
    fn new_rejects_a_missing_checkpoint() -> Result<(), Error> {
        let Some(binary) = which_dblocks_or_skip() else {
            // A check that cannot run says so — rule 6. The test still
            // returns Ok because this is a precondition the environment
            // controls, not a claim the code made.
            eprintln!("skipping: dblocks not on PATH or in the usual tree");
            return Ok(());
        };
        let err = match Dblocks::new(binary, "/no/such/checkpoint.mpk") {
            Err(e) => format!("{e}"),
            Ok(_) => return Err(unexpected("Err for missing checkpoint", "Ok")),
        };
        if !err.contains("checkpoint not found") {
            return Err(unexpected("`checkpoint not found` in error", &err));
        }
        Ok(())
    }

    #[test]
    fn build_argv_pins_the_flags_lm_generate_expects() -> Result<(), Error> {
        let Some(binary) = which_dblocks_or_skip() else {
            eprintln!("skipping: dblocks not on PATH or in the usual tree");
            return Ok(());
        };
        let d = Dblocks::new(binary, pseudo_checkpoint()?)?;
        let argv = d.build_argv("hello", 32);
        let expect = [
            "lm", "generate", "--checkpoint", "--prompt", "hello",
            "--max-new", "32", "--cached", "--sampling", "greedy",
        ];
        for token in expect {
            if !argv.iter().any(|a| a == token) {
                return Err(unexpected(
                    &format!("argv to contain `{token}`"),
                    &argv.join(" "),
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn deterministic_at_zero_tracks_sampling() -> Result<(), Error> {
        let Some(binary) = which_dblocks_or_skip() else {
            return Ok(());
        };
        let d = Dblocks::new(binary, pseudo_checkpoint()?)?;
        if !d.deterministic_at_zero() {
            return Err(unexpected("greedy to declare deterministic", "false"));
        }
        let noisy = d.with_sampling(Sampling::TopK {
            top_k: 8,
            temperature: 0.7,
        });
        if noisy.deterministic_at_zero() {
            return Err(unexpected(
                "top-k to declare non-deterministic",
                "true",
            ));
        }
        Ok(())
    }

    // ---- helpers --------------------------------------------------------

    /// Find a real `dblocks` binary on this machine, or return `None` so a
    /// dependent test can skip instead of panic. Rule 6: a check that cannot
    /// run says so; it does not pass.
    fn which_dblocks_or_skip() -> Option<PathBuf> {
        let candidates = [
            "../../../unifur/target/debug/dblocks",
            "../../unifur/target/debug/dblocks",
            "unifur/target/debug/dblocks",
            "/usr/local/bin/dblocks",
            "/usr/bin/dblocks",
        ];
        for c in candidates {
            let p = PathBuf::from(c);
            if p.is_file() {
                return Some(p);
            }
        }
        None
    }

    /// A file that exists for the `Dblocks` constructor's checkpoint check,
    /// without pretending to be a real checkpoint. Ends with `.mpk` so the
    /// extension-based detection paths downstream do not short-circuit.
    fn pseudo_checkpoint() -> Result<PathBuf, Error> {
        let path = std::env::temp_dir().join(format!(
            "omni-model-unifur-pseudo-{}.mpk",
            std::process::id()
        ));
        std::fs::write(&path, b"not a real checkpoint")?;
        Ok(path)
    }
}
