//! The `omni-verify` binary — the certificate suite as a gate.
//!
//! Separate from the `omni` binary so that the gate can be run on its own. Its
//! dependency chain is deliberately flat: `omni-core`, `omni-model`,
//! `omni-model-peregrine` (for the dialect certificates) and `omni-ctx` (so the
//! `contract` group's gated rows name real message types rather than a
//! structural stand-in for them). A harness binary that grew a feature could
//! not be used as a gate without pulling that feature's dependencies into
//! whatever ran it.
//!
//! ```text
//! omni-verify verify [--group <NAME>]
//! omni-verify list
//! ```

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::process::ExitCode;

use omni_core::Error;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match dispatch(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("omni-verify: {e}");
            ExitCode::from(2)
        }
    }
}

fn dispatch(args: &[String]) -> Result<ExitCode, Error> {
    let Some(cmd) = args.first() else {
        print_usage();
        return Ok(ExitCode::from(2));
    };
    match cmd.as_str() {
        "verify" => cmd_verify(&args[1..]),
        "list" => {
            println!("certificate groups:");
            for g in omni_verify::GROUPS {
                println!("    {g}");
            }
            Ok(ExitCode::SUCCESS)
        }
        "preflight" => {
            // The cheap subset, for running before a long job. A smoke alarm,
            // not a substitute for the full suite.
            let report = omni_verify::preflight();
            print!("{}", report.render());
            Ok(if report.passed() { ExitCode::SUCCESS } else { ExitCode::FAILURE })
        }
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(ExitCode::SUCCESS)
        }
        other => Err(Error::Format(format!(
            "unknown subcommand {other:?}; try `omni-verify help`"
        ))),
    }
}

fn print_usage() {
    println!(
        "\
omni-verify — the numerical certificate suite

USAGE
    omni-verify verify [--group <NAME>]   run the suite; non-zero on failure
    omni-verify preflight                 run the cheapest load-bearing checks
    omni-verify list                      list the certificate groups
"
    );
}

/// Run one certificate group by name.
///
/// # Errors
///
/// [`Error::Format`] carrying the message `run_group` produced, which already
/// names the unknown group and lists the ones that exist.
///
/// This is the one `String`-to-`Error` conversion in the binary, and it is
/// written out rather than compressed into `map_err` because AGENTS.md rule 2
/// forbids `map_err`: annotate with [`omni_core::Context`] instead. `Context`
/// cannot be used here — its blanket implementation is over `E: Into<Error>`,
/// and `omni_core::Error` has no `From<String>`, which is what makes adding one
/// a change to a crate this binary does not own. So the conversion is a `match`
/// with both arms written down.
fn run_named_group(group: &str) -> Result<omni_verify::Report, Error> {
    match omni_verify::run_group(group) {
        Ok(report) => Ok(report),
        Err(msg) => Err(Error::Format(msg)),
    }
}

fn cmd_verify(args: &[String]) -> Result<ExitCode, Error> {
    let mut group: Option<&str> = None;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--group" => {
                let Some(value) = args.get(i + 1) else {
                    return Err(Error::Format("--group needs a name; see `omni-verify list`".into()));
                };
                group = Some(value.as_str());
                i += 2;
            }
            other => {
                return Err(Error::Format(format!(
                    "unknown option {other:?} for `omni-verify verify`"
                )))
            }
        }
    }

    let report = match group {
        Some(g) => run_named_group(g)?,
        None => omni_verify::run_all(),
    };

    print!("{}", report.render());
    if report.passed() {
        Ok(ExitCode::SUCCESS)
    } else {
        eprintln!("omni-verify: certificate suite FAILED");
        Ok(ExitCode::FAILURE)
    }
}