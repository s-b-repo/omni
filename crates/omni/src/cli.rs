//! Argument parsing and the subcommand table.
//!
//! Hand-rolled on purpose. The binary is a harness front door, and a front door
//! that can panic while parsing `--help` is a front door that fails exactly when
//! someone is trying to find out why it is failing. Every path here returns a
//! `Result`; unknown input is an error that names the thing it did not
//! recognise, never a default that silently does something else.

use std::process::ExitCode;

use crate::model;
use omni_core::{Error, contract};



/// A usage error: bad input, as opposed to a failure doing the thing asked.
///
/// A separate type from [`Error`] because the two want different exit codes and
/// different wording. "You typed it wrong" and "the operation failed" are
/// different problems, and a CLI that reports both as `omni: <error>` sends the
/// reader to the wrong place.
#[derive(Debug)]
pub struct Usage(pub String);

impl Usage {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

/// Run one invocation.
///
/// `args` is the argument vector with the program name already removed.
pub fn dispatch(args: &[String]) -> Result<ExitCode, Error> {
    let Some(cmd) = args.first() else {
        print_usage();
        return Ok(ExitCode::from(2));
    };

    match cmd.as_str() {
        "verify" => cmd_verify(&args[1..]),
        "list" => cmd_list(),
        "contract" => cmd_contract(),
        "model" => model::dispatch(&args[1..]).map_err(|u| Error::Format(u.0)),
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(ExitCode::SUCCESS)
        }
        "version" | "--version" | "-V" => {
            println!("omni {}", env!("CARGO_PKG_VERSION"));
            Ok(ExitCode::SUCCESS)
        }
        other => Err(Error::Format(format!(
            "unknown subcommand {other:?}; try `omni help`"
        ))),
    }
}

fn print_usage() {
    println!(
        "\
omni — an agent harness whose runtime is itself editable

USAGE
    omni <SUBCOMMAND>

SUBCOMMANDS
    verify [--group <NAME>]        run the certificate suite; non-zero on failure
    list                           list the certificate groups
    contract                       print the cross-process contracts this build speaks
    model peregrine [--url U] [--key K]
                                   show the exact request this build would send,
                                   then try /health on the server
    model dialects                 print where peregrine differs from OpenAI
    help                           this text

GATES
    cargo test --workspace
    cargo clippy --workspace --all-targets
    scripts/audit-bad-patterns.sh --strict
    scripts/verify-certificates.sh"
    );
}

fn cmd_verify(args: &[String]) -> Result<ExitCode, Error> {
    let mut group: Option<&str> = None;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--group" => {
                let Some(value) = args.get(i + 1) else {
                    return Err(Error::Format("--group needs a name; see `omni list`".into()));
                };
                group = Some(value.as_str());
                i += 2;
            }
            other => {
                return Err(Error::Format(format!(
                    "unknown option {other:?} for `omni verify`; the only option is --group"
                )))
            }
        }
    }

    let report = match group {
        Some(g) => omni_verify::run_group(g).map_err(Error::Format)?,
        None => omni_verify::run_all(),
    };

    print!("{}", report.render());
    if report.passed() {
        Ok(ExitCode::SUCCESS)
    } else {
        // The reason is on stdout in the table; this line names the exit so a
        // CI log that truncates still says what happened.
        eprintln!("omni: certificate suite FAILED");
        Ok(ExitCode::FAILURE)
    }
}

fn cmd_list() -> Result<ExitCode, Error> {
    println!("certificate groups:");
    for g in omni_verify::GROUPS {
        println!("    {g}");
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_contract() -> Result<ExitCode, Error> {
    // The deployed context contract, plus omni's own. Printed by the binary
    // that enforces them, so a reader comparing against a peer's `contract.json`
    // is comparing against what this build actually speaks rather than against
    // a document that can drift from the code.
    println!("protocol: {}", contract::PROTOCOL);
    println!("channel directory: {}", contract::INTEROP_DIR);
    println!("  (read, never assumed to exist, never created as a side effect)");
    println!("contract document: {}", contract::CONTRACT_FILE);
    println!("request suffix:   {}", contract::REQUEST_SUFFIX);
    println!("response suffix:  {}", contract::RESPONSE_SUFFIX);
    println!();

    let deployed = "deployed (unifur <-> subnext)";
    let omni_own = "omni's own (harness-edit evidence)";
    let rows: [(&str, u32, &str); 5] = [
        ("ContextQuery", contract::CONTEXT_QUERY_VERSION, deployed),
        ("ContextHit", contract::CONTEXT_HIT_VERSION, deployed),
        ("ChannelRequest", contract::CHANNEL_ENVELOPE_VERSION, deployed),
        ("ChannelResponse", contract::CHANNEL_ENVELOPE_VERSION, deployed),
        ("HookPacket", contract::HOOK_PACKET_VERSION, omni_own),
    ];

    let (header_name, header_status) = ("message", "status");
    println!("{:<18} {:>8}  {}", header_name, "version", header_status);
    for (name, version, status) in rows {
        println!("{name:<18} {version:>8}  {status}");
    }

    println!();
    println!("An unknown version, and a missing one, are both refused: a missing");
    println!("field parses as 0 and 0 is simply an unknown version.");
    Ok(ExitCode::SUCCESS)
}