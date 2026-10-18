//! The `omni` binary.
//!
//! One entry point for the workspace, shaped like peregrine's: `verify` is the
//! gate, and everything else is a subcommand that reads like the thing it
//! does. Argument parsing is hand-rolled rather than pulled from `clap` so the
//! binary has no dependency whose failure mode is a panic at startup — which
//! for a harness that runs model-written code is the wrong place to own one.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod cli;
mod model;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match cli::dispatch(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("omni: {e}");
            ExitCode::FAILURE
        }
    }
}