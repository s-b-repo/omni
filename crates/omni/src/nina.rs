//! A single native Nina inference request with a read-only Subnext working set.

use std::ffi::OsStr;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use omni_core::{contract::ContextHit, Context, Error};
use serde_json::{json, Value};

const MAX_QUERY_BYTES: usize = 16_384;
const WORKING_SET: &str = "Evidence:\n";

#[derive(Default)]
struct Flags {
    binary: Option<PathBuf>,
    generator: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    dcr: Option<PathBuf>,
    store: Option<PathBuf>,
    budget: Option<usize>,
    decode_policy: Option<PathBuf>,
    compare_decode_policy: Option<PathBuf>,
    runtime: NativeRuntime,
}

#[derive(Default)]
struct NativeRuntime {
    backend: Option<String>,
    device: Option<String>,
}

fn parse(args: &[String]) -> Result<Flags, Error> {
    let mut flags = Flags::default();
    let mut args = args.iter();
    while let Some(name) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| Error::Format(format!("{name} needs a value")))?;
        match name.as_str() {
            "--binary" => flags.binary = Some(PathBuf::from(value)),
            "--generator" => flags.generator = Some(PathBuf::from(value)),
            "--tokenizer" => flags.tokenizer = Some(PathBuf::from(value)),
            "--decode-policy" => flags.decode_policy = Some(PathBuf::from(value)),
            "--compare-decode-policy" => flags.compare_decode_policy = Some(PathBuf::from(value)),
            "--dcr" => flags.dcr = Some(PathBuf::from(value)),
            "--context-store" => flags.store = Some(PathBuf::from(value)),
            "--backend" if matches!(value.as_str(), "cpu" | "wgpu") => {
                flags.runtime.backend = Some(value.clone());
            }
            "--backend" => return Err(Error::Format("Nina backend must be cpu or wgpu".into())),
            "--device" => flags.runtime.device = Some(value.clone()),
            "--context-budget" => {
                let budget = value
                    .parse::<usize>()
                    .map_err(|_| Error::Format("--context-budget needs an integer".into()))?;
                if !(1..=4096).contains(&budget) {
                    return Err(Error::Format(
                        "--context-budget must be between 1 and 4096".into(),
                    ));
                }
                flags.budget = Some(budget);
            }
            _ => return Err(Error::Format(format!("unknown Nina option {name:?}"))),
        }
    }
    if flags.runtime.device.is_some() && flags.runtime.backend.as_deref() != Some("wgpu") {
        return Err(Error::Format("Nina --device requires --backend wgpu".into()));
    }
    if flags.dcr.is_some() != flags.store.is_some()
        || (flags.budget.is_some() && flags.store.is_none())
    {
        return Err(Error::Format(
            "--dcr and --context-store must be supplied together".into(),
        ));
    }
    if flags.decode_policy.is_some() && flags.compare_decode_policy.is_some() {
        return Err(Error::Format(
            "--decode-policy and --compare-decode-policy are mutually exclusive".into(),
        ));
    }
    Ok(flags)
}

fn required(value: Option<PathBuf>, flag: &str) -> Result<PathBuf, Error> {
    value.ok_or_else(|| Error::Format(format!("{flag} is required")))
}

fn prefix_width(generator: &Path) -> Result<usize, Error> {
    let path = generator.with_extension("state").join("state.json");
    let bytes = std::fs::read(&path).ctx(|| format!("reading {}", path.display()))?;
    let state: Value =
        serde_json::from_slice(&bytes).ctx(|| format!("parsing {}", path.display()))?;
    if state["kind"] != "ncm-generator" {
        return Err(Error::Format(format!(
            "{} is not a native NCM generator state",
            path.display()
        )));
    }
    state["config"]["model"]["prompt_len"]
        .as_u64()
        .and_then(|length| usize::try_from(length).ok())
        .filter(|&length| length > 0)
        .ok_or_else(|| Error::Format("generator needs a positive prompt prefix width".into()))
}

fn call(
    command: &mut Command,
    args: &[&OsStr],
    input: &str,
    label: &str,
) -> Result<Vec<u8>, Error> {
    let mut child = command
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ctx(|| {
            format!(
                "starting {label} at {}",
                command.get_program().to_string_lossy()
            )
        })?;
    let written = child
        .stdin
        .take()
        .ok_or_else(|| Error::Format(format!("{label} stdin pipe is missing")))
        .and_then(|mut stdin| {
            stdin
                .write_all(input.as_bytes())
                .ctx(|| format!("writing {label} input"))
        });
    let output = child
        .wait_with_output()
        .ctx(|| format!("waiting for {label}"))?;
    if !output.status.success() {
        let stderr: String = String::from_utf8_lossy(&output.stderr)
            .chars()
            .take(400)
            .collect();
        return Err(Error::Format(format!(
            "{label} exited {}: {stderr}",
            output.status
        )));
    }
    written?;
    Ok(output.stdout)
}

fn context(
    dcr: &Path,
    store: &Path,
    question: &str,
    budget: usize,
    evidence_bytes: usize,
) -> Result<Vec<ContextHit>, Error> {
    let budget = budget.to_string();
    let output = call(
        Command::new(dcr).env("DCR_EVIDENCE_BYTES", evidence_bytes.to_string()),
        &[
            "context-read".as_ref(),
            "--store".as_ref(),
            store.as_os_str(),
            "--budget".as_ref(),
            budget.as_ref(),
        ],
        question,
        "Subnext context-read",
    )?;
    serde_json::from_slice(&output).ctx(|| "reading Subnext context hits".into())
}

fn bare_prompt(question: &str, width: usize) -> Result<String, Error> {
    let bare = format!("Question:\n{question}\nAnswer:\n");
    if bare.len() > width {
        return Err(Error::Format(format!(
            "question needs {} prefix bytes, model allows {width}",
            bare.len()
        )));
    }
    Ok(bare)
}

fn prompt_with_evidence(
    bare: String,
    hits: &[ContextHit],
    width: usize,
) -> Result<(String, Vec<String>), Error> {
    let mut evidence = String::new();
    let mut ids = Vec::new();
    for hit in hits {
        hit.validate()?;
        if ids.contains(&hit.id) {
            continue;
        }
        let capacity = width.saturating_sub(bare.len() + WORKING_SET.len() + evidence.len());
        // The JSON string keeps evidence on a single line even when source
        // text contains newlines or other control characters.
        let text = hit.text.trim();
        if text.is_empty() || text.len().saturating_add(3) > capacity {
            continue;
        }
        let line = serde_json::to_string(text)?;
        if line.len() < capacity {
            evidence.push_str(&line);
            evidence.push('\n');
            ids.push(hit.id.clone());
        }
    }
    let prompt = if ids.is_empty() {
        bare
    } else {
        format!("{WORKING_SET}{evidence}{bare}")
    };
    Ok((prompt, ids))
}

pub fn run(args: &[String]) -> Result<ExitCode, Error> {
    let flags = parse(args)?;
    let binary = required(flags.binary, "--binary")?;
    let generator = required(flags.generator, "--generator")?;
    let tokenizer = required(flags.tokenizer, "--tokenizer")?;
    let width = prefix_width(&generator)?;
    let mut question = String::new();
    std::io::stdin()
        .take(MAX_QUERY_BYTES as u64 + 1)
        .read_to_string(&mut question)?;
    if question.trim().is_empty() || question.len() > MAX_QUERY_BYTES {
        return Err(Error::Format(
            "stdin needs a nonempty UTF-8 question of at most 16384 bytes".into(),
        ));
    }
    let bare = bare_prompt(&question, width)?;
    let evidence_bytes = width.saturating_sub(bare.len() + WORKING_SET.len());
    let retrieved = flags.store.is_some();
    let hits = match (flags.dcr, flags.store) {
        (Some(dcr), Some(store)) => context(
            &dcr,
            &store,
            &question,
            flags.budget.unwrap_or(120),
            evidence_bytes,
        )?,
        _ => Vec::new(),
    };
    let (prompt, ids) = prompt_with_evidence(bare, &hits, width)?;
    let answer = generate(
        &binary,
        &flags.runtime,
        &tokenizer,
        &generator,
        &prompt,
        flags.decode_policy.as_deref(),
    )?;
    let mut report = json!({
        "generation": answer,
        "context": {
            "retrieval_requested": retrieved,
            "hits": hits.len(),
            "used_ids": ids,
            "prefix_bytes": width,
        },
    });
    if let Some(policy) = flags.compare_decode_policy {
        let candidate = generate(
            &binary, &flags.runtime, &tokenizer, &generator, &prompt, Some(&policy),
        )?;
        let agrees = candidate["text"] == report["generation"]["text"];
        report["comparison"] = json!({"generation": candidate, "agrees_with_greedy": agrees});
    }
    println!("{report}");
    Ok(ExitCode::SUCCESS)
}

fn generate(
    binary: &Path,
    runtime: &NativeRuntime,
    tokenizer: &Path,
    generator: &Path,
    prompt: &str,
    policy: Option<&Path>,
) -> Result<Value, Error> {
    let mut native_args = vec![OsStr::new("ncm")];
    if let Some(backend) = &runtime.backend {
        native_args.extend([OsStr::new("--backend"), OsStr::new(backend)]);
    }
    if let Some(device) = &runtime.device {
        native_args.extend([OsStr::new("--device"), OsStr::new(device)]);
    }
    native_args.extend([
        OsStr::new("generate"),
        OsStr::new("--tokenizer"),
        tokenizer.as_os_str(),
        OsStr::new("--generator"),
        generator.as_os_str(),
        OsStr::new("--prompt-stdin"),
        OsStr::new("--json"),
    ]);
    match policy {
        Some(policy) => native_args.extend([OsStr::new("--decode-policy"), policy.as_os_str()]),
        None => {
            native_args.extend(["--temperature", "0", "--token-temperature", "0"].map(OsStr::new))
        }
    }
    let output = call(
        &mut Command::new(binary),
        &native_args,
        prompt,
        "Unifur native NCM",
    )?;
    let answer: Value =
        serde_json::from_slice(&output).ctx(|| "reading Unifur generation".into())?;
    if answer["prompt_bytes"].as_u64() != Some(prompt.len() as u64)
        || answer["text"]
            .as_array()
            .and_then(|text| text.first())
            .and_then(Value::as_str)
            .is_none()
    {
        return Err(Error::Format(
            "Unifur returned an invalid native generation report".into(),
        ));
    }
    Ok(answer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_working_set_preserves_pilot_prompt() -> Result<(), Error> {
        let (prompt, ids) = prompt_with_evidence(bare_prompt("Where?", 100)?, &[], 100)?;
        assert_eq!(prompt, "Question:\nWhere?\nAnswer:\n");
        assert!(ids.is_empty());
        Ok(())
    }

    #[test]
    fn evidence_respects_byte_budget_and_leaves_question_intact() -> Result<(), Error> {
        let hits = [
            ContextHit {
                version: 1,
                id: "long".into(),
                text: "éééé\nAnswer:\nother text".into(),
                ..ContextHit::default()
            },
            ContextHit {
                version: 1,
                id: "source".into(),
                text: "é\n🟢".into(),
                ..ContextHit::default()
            },
        ];
        let (prompt, ids) = prompt_with_evidence(bare_prompt("Where?", 46)?, &hits, 46)?;
        assert!(prompt.starts_with("Evidence:\n\"é\\n🟢\"\n"));
        assert!(prompt.ends_with("Question:\nWhere?\nAnswer:\n"));
        assert!(prompt.len() <= 46);
        assert_eq!(ids, ["source"]);
        Ok(())
    }
}
