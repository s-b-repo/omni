//! The `dblocks` backend for the `Model` trait.
//!
//! **There is no unifur checkpoint this can load.** Stated plainly because
//! AGENTS.md rule 10 requires it: a capability that does not exist is reported
//! as absent, with the reason. What exists is a 4.6 MB set of Burn `.mpk`
//! checkpoints for 2-D geometric scenes (6 points, 5 question kinds), a
//! 22 GB `/srv` registry of *routing classifiers* whose `router_top1` is near
//! 1.0 and whose job is to classify a prompt rather than answer it, and
//! `quality_coder/`, whose own module header says it "does not load any model,
//! run any network calls, or spawn any subprocess".
//!
//! So the feature is off by default, and the default build is the honest one.
//! Loading a geometry model and calling it an agent would be the failure this
//! crate exists to avoid.
//!
//! This file is the crate's declared shape, not its behaviour.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![forbid(unsafe_code)]
