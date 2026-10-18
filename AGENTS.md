# AGENTS.md — the law of this repository

`omni` is an agent harness whose runtime is itself editable. A trained engineer
reads a batch of failed agent trajectories, writes a runtime patch, and the
patch is scored by rerunning a frozen agent on exactly the same tasks. The
harness improves by being edited; everything below exists because a system that
edits itself needs stronger evidence than one that does not.

The crates, in dependency order:

| Crate | What lives there |
|---|---|
| `omni-core` | errors, versioned contracts, ids, canonical json, atomic writes |
| `omni-verify` | **the numerical certificate suite** (`verify`), a gate like the audit |
| `omni-model` | the `Model` trait and the `ToolRegistry` |
| `omni-model-peregrine` | OpenAI-compatible client |
| `omni-model-unifur` | `dblocks` backend — **feature-gated, no code checkpoint yet** |
| `omni-ctx` | the `ContextPlane` over canonical `dcr` |
| `omni-hook` | the sandboxed hook runtime |
| `omni-graph` | typed node graph, frontier, idempotent apply |
| `omni-loop` | the orchestration loop, council, round budget |
| `omni-profile-pentest` | ROE, scope, findings, coverage, combinations |
| `omni-eval` | failure packets, rerun-delta reward, identity pinning |
| `omni` | the `omni` binary |

`omni-core`, `omni-verify`, `omni-model`, `omni-model-peregrine`, `omni-ctx`,
`omni-model-unifur`, `omni-graph`, `omni-loop`, `omni-eval` and `omni-hook`
exist. The remaining one (`omni-profile-pentest`) is a milestone; do not
create a crate whose job is described in a table row that nothing imports yet.
An existing row is not an excuse to exist by itself: a crate whose only callers
are its own tests has to say so in `docs/CONTRACT.md`'s "What is not true yet",
and the state it claims has to be one its tests can prove. (`omni-loop::rollout` hosts
patches today — scripted agent, demonstration-grade; the note in
`docs/CONTRACT.md` tracks exactly what that proves and what it does not.)

The live smoke test for the model seam:

```bash
omni model peregrine --url http://127.0.0.1:8080 --complete
```

It prints the exact request this build would send, then makes the call and
reports the finish reason, the text and the usage. A server that is down is
reported as down, with the reason, and separately from the request — because
"the request looks right" and "the server answered" are different findings.

Start at [`docs/CONTRACT.md`](docs/CONTRACT.md).

## The gates — run all of these before you stop

```bash
cargo test --workspace                            # 0 failures
cargo clippy --workspace --all-targets            # no new warnings
scripts/audit-bad-patterns.sh --strict            # exit 0 required
scripts/verify-certificates.sh                    # every certificate, exit 0 required
scripts/verify-cross-repo.sh --strict             # no drift from the other repos
```

Plus, before you trust either audit script itself:

```bash
scripts/audit-bad-patterns.sh --self-test         # the gate still bites
scripts/verify-cross-repo.sh --self-test          # so does this one
```

The last four have the same shape on purpose: one command each, non-zero on
failure, reason on stderr.

`verify-cross-repo.sh` is the only gate that looks outside this tree. It exists
because the four repos are joined by versioned contracts and a vendored port,
and neither of those rots quietly. It checks that the canonical subnext checkout
is the one carrying the live responder, that `peregrine-ctx` still differs from
`subnext/src` in exactly the two files the prefix rename allows, that `GATED`
covers every message the deployed contract declares, that unifur's and subnext's
message shapes agree field-for-field, and that every citation in
`docs/NEGATIVE-RESULTS.md` still resolves. It has no `--accept-drift` flag, and
that is deliberate: a finding you can switch off is a finding you will switch
off.

`--all-targets` on clippy is not optional. Without it `cargo check` skips test
targets, so a signature change can leave a `#[cfg(test)]` module uncompilable
while the check still reports success.

A gate you cannot run in this environment is not a gate you ran. Say so.

## The rules

### 1. No panicking error handling. Anywhere.

`#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]` at every
crate root, and `clippy.toml` sets all three `allow-*-in-tests = false`.

That last flag is load-bearing. At its default (`true`) the lint exempts
`#[test]` bodies, which is exactly the exemption that lets a panic-based test
suite coexist with a claim that the suite has none. peregrine sets the same
three flags for the same reason.

Tests return `Result` and use `?`. `assert_eq!` is fine — clippy's `panic` lint
matches the macro path, not the expansion, and asserting is what a test is for.
`panic!` and `.expect_err()` are not.

### 2. One error type, and it is a closed enum.

`omni_core::Error`. No `anyhow`, no `Box<dyn Error + Send + Sync>`.

A harness that installs model-written code has failures that must be
classifiable: a hook that blew its instruction budget, a contract whose version
is from the future, and a full disk are three different bugs. A stringly-typed
error erases that distinction exactly where it costs most.

Annotate with `Context`, never `map_err`:

```rust
foo().ctx(|| format!("reading {path}"))?      // the one map_err lives in the trait
```

For an error you constructed rather than propagated, `Error::annotated`.

### 3. Nothing is swallowed.

`let _ = f()?` discards the difference between "we tried and failed" and "we
never tried", and the first of those can leave debris for the next process.

Say it out loud instead:

```rust
omni_core::note_advisory("removing a leftover scratch file", &Error::from(e));
```

Section B of the audit counts `let _ =`, `if let Err(_)`, an empty `Err(_) => {}`
and a bare `.ok();`. Budget 0. `let _ = &borrow` and `let _: T = expr` are
excluded — they drop a value, not a failure.

### 4. No lint suppression.

No `#[allow]`, no `#[expect]`, no `#[cfg_attr(..., allow)]`, no
`clippy::allow_attributes`, no `// audit-allow:` escape hatch. Section C, budget 0.

If a lint is wrong, the code is wrong. If a lint cannot be satisfied without
`unsafe`, the function belongs in a crate that has `#![forbid(unsafe_code)]` and
does not do it.

### 5. Versioned contracts, never shared memory.

Every cross-process message carries a `version`. An unknown version is refused.
A **missing** version is refused identically, because `#[serde(default)]` makes
it `0` and `0` is an unknown version — a peer that drops a field must not be
mistaken for a peer speaking a newer dialect.

One implementation of that rule: `omni_core::check_version`. One enumeration of
what it gates: the `GATED` table in `omni-verify`'s contract group. If you add a
message and do not add it to that table, the certificate is no longer describing
the workspace.

The context message shapes are already deployed at version 1 on both sides of
`/srv/m-sda/unifur/interop/`. **Implement them; do not redefine them.**

### 6. A claim that cannot be reduced to a residual is not a certificate.

`omni-verify` exists to make load-bearing claims falsifiable: the claim in prose,
a residual that is zero exactly when it holds, and a tolerance chosen from the
arithmetic rather than from whatever the code happens to produce.

Two consequences worth internalising:

- **A check that cannot run is not a passing check.** `failed()` returns
  `f64::INFINITY`, never a skip.
- **A certificate needs a control.** A check that passes because everything is
  distinct has certified nothing. Each one here that could degenerate that way
  carries a case that goes the other way (`array_order_is_identity` has two
  empty arrays that must agree; `non_finite_is_refused` requires a finite float
  to still build).

Never widen a tolerance to make a suite green. That is a review conversation.

### 7. Enforcement is pure and dry-runnable.

`fn evaluate(&self, tool: &str, args: &Value) -> Decision` performs no I/O.
That is what lets the graph frontier run *the same gate* on a candidate action
that has not been dispatched yet, and feed the block back into the graph as a
learnable edge. Policy that can only be consulted after the fact is not a gate.

### 8. Matched baselines must actually be matched.

A reward is `patched − baseline` over pinned task identities. Identities are
content-derived (`omni_core::content_id`), never positional — **matching integer
indices is not proof of a paired comparison.** Every knob that can change a run's
output belongs in the identity fingerprint, including environment variables.

An invalid patch is a no-patch and scores zero. An environment failure is a
missing evaluation and is reported separately. The two are never conflated, and
nothing is ever retried until it turns green.

### 9. `unsafe` is confined.

`omni-core` and `omni-verify` carry `#![forbid(unsafe_code)]`. `omni-hook`
carries it too: the interpreter whose host boundary was expected to need
`unsafe` was replaced by a declarative rule-set patch language, so the
exception this rule anticipated does not exist. Every crate in the workspace
forbids `unsafe`. Section N of the audit, budget 0, catches it appearing
anywhere else.

### 10. Say what is not true.

A capability that does not exist is reported as absent, with the reason. A model
backend with no checkpoint says so. A reward that was never computed says so. A
suite that ran zero cases does not report success.

This is borrowed directly from the systems omni is built beside: unifur's
`route_prompt` refuses to return a chance-level pick dressed up as a route,
subnext's responder never fabricates an empty hit list (a peer with no answer
times out, which is the honest outcome), and peregrine's reward loop hands a
certificate failure back to the model that broke it rather than widening the
tolerance.

## The audit gate

`scripts/audit-bad-patterns.sh --strict` ratchets: it fails when a section
*exceeds* its budget, so reviewed debt could stay visible while new code is held
to zero. **omni's budgets all start at 0** — it is greenfield with no debt to
carry. peregrine opens at A 285 / B 109 because those measure its existing debt
across 13 crates; copying those numbers here would be importing a debt that was
never incurred.

Budgets only move down. A budget that needs raising because new code added a hit
is a review conversation, not a bookkeeping update.

Sections: **A** panicking error handling · **B** swallowed errors · **C** lint
suppression · **J** secrets · **M** injection · **N** UB and concurrency · **T**
Cargo.toml hygiene. All gated, all at 0.

`--self-test` plants known-bad code and asserts each section still catches it.
Run it after touching the patterns: a gate that only ever passes is
indistinguishable from one whose regex rotted. It has already earned its keep
here — it caught a `.unwrap(` written as `x.unwrap()` being missed because the
pattern required a non-word character before the dot.

## Where things come from

| From | What was taken |
|---|---|
| **Harness-R1** (Apache-2.0, arXiv 2608.02276) | the idea and the discipline: patches validated by sandbox, scored by real reruns |
| **subnext** / DCR | the context runtime, the `#ESCALATE` contract, the "harness that can signal" thesis |
| **peregrine** | the error-and-`Context` shape, the certificate discipline, the ratchet, the pinned toolchain |
| **unifur** | the certificate-as-law, the version-gate wording, `note_advisory`, the cheap-first answer for a contract you do not own |
| **openhack** | pure dry-runnable enforcement, deterministic council tally, combos-as-set-difference, the controller that degrades byte-identically |

Third-party code keeps its own licence and its own style. `peregrine-token` is
vendored upstream in peregrine and is not "fixed" there for that reason; if a
vendored crate is added here, the same applies.