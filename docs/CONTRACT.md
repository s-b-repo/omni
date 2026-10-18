# CONTRACT.md — the seams, and what crosses them

`omni` joins five systems that were written separately, by different people, for
different purposes. This document states what crosses between them, who owns each
shape, and which version this build speaks.

The rule the whole design rests on, quoted from the system it comes from:

> Context lives outside the model. Capability is trained into weights here;
> the corpus, memory and long documents are owned and served elsewhere. The two
> repos coordinate through a versioned, file-based contract — **never shared
> memory** — so either side can be developed and tested alone.

That is why this is a list of messages and not a list of libraries. Every seam
below could be reimplemented by the other side, on another language, on another
machine, and the contract would be the only thing that had to agree.

## The owners

| Seam | Owner | How omni reaches it |
|---|---|---|
| model inference | **peregrine** | HTTP, `POST /v1/chat/completions`, OpenAI dialect |
| context / working set | **subnext** (`dcr`) | library link + the file channel below |
| trained capability | **unifur** (`dblocks`) | process, feature-gated; no code checkpoint yet |
| agent loop, ROE, graph | **openhack** | ported as `omni-graph` / `omni-loop` / `omni-profile-pentest` |
| harness editing | **Harness-R1** | the patch protocol and the rerun-delta reward, in `omni-hook` / `omni-eval` |

## The version gate

One implementation: `omni_core::check_version(kind, got, supported)`. It refuses
on inequality, and returns `Error::ContractVersion` naming **both** numbers.

A **missing** version is refused identically, and this is not a detail. A field
annotated `#[serde(default)]` parses as `0` when absent, so "the peer dropped a
field" and "the peer speaks a newer dialect" are the same input by the time the
code sees it. If those two paths ever diverge, one of them is silently
interpreting garbage.

The gate is only as good as the list of things it gates, so the list is
enumerable and the certificate checks it:

```rust
// crates/omni-verify/src/contract.rs
const GATED: &[Gated] = &[
    Gated { kind: "ContextQuery",    supported: CONTEXT_QUERY_VERSION,     sample: /* … */, accept: /* … */ },
    Gated { kind: "ContextHit",      supported: CONTEXT_HIT_VERSION,       sample: /* … */, accept: /* … */ },
    Gated { kind: "ContextPut",      supported: CONTEXT_PUT_VERSION,       sample: /* … */, accept: /* … */ },
    Gated { kind: "MemoryAppend",    supported: MEMORY_APPEND_VERSION,     sample: /* … */, accept: /* … */ },
    Gated { kind: "ChannelRequest",  supported: CHANNEL_ENVELOPE_VERSION,  sample: /* … */, accept: /* … */ },
    Gated { kind: "ChannelResponse", supported: CHANNEL_ENVELOPE_VERSION,  sample: /* … */, accept: /* … */ },
    Gated { kind: "HookPacket",      supported: HOOK_PACKET_VERSION,       sample: /* … */, accept: /* … */ },  // omni's own
];
```

Each row carries a **wire document** and the **gate that must accept it**, so a
row cannot name a message nothing parses. `contract/every_declared_message_is_a_gated_row`
holds the table against `omni_core::contract::VERSIONED_MESSAGES` — what
omni-core *declares* — so a message added without a row is a failing
certificate rather than a quietly shorter table.

If you add a versioned message, add it here. A certificate that does not
enumerate what it covers cannot certify that it covers everything.

## The context channel — deployed, do not redefine

**Directory:** `/srv/m-sda/unifur/interop/`
**Protocol:** `unifur-subnext-interop`, version 1
**Contract document:** `contract.json` in that directory, generated from the same
constants the code enforces so it cannot drift

These shapes are **already deployed at version 1 on both sides**, pinned in
`contract.json`, and guarded by a cross-repo drift test on the subnext side.
`omni` implements them; it does not fork them. Forking the struct would create a
second shape that is correct today and silently incompatible after the next
bump.

| Message | Fields | Version |
|---|---|---|
| `ContextPut` | `version`, `id`, `kind`, `text`, `meta` | 1 |
| `ContextQuery` | `version`, `text`, `k`, `field` | 1 |
| `ContextHit` | `version`, `id`, `score`, `text`, `provenance` | 1 |
| `MemoryAppend` | `version`, `payload` | 1 |
| `ChannelRequest` | `version`, `id`, `query` | 1 |
| `ChannelResponse` | `version`, `id`, `hits` | 1 |
| `Provenance` | `source`, `path` | — |

`ContextPut` and `MemoryAppend` are declared by the contract document but not yet
sent by omni: its corpus is built by the indexer, not by a put, and the memory
log belongs to the peer. They are implemented so a message a peer *does* send is
recognised and version-checked rather than reinterpreted by accident.

`omni` adds one message of its own:

| Message | Fields | Version |
|---|---|---|
| `HookPacket` | `version`, `profile`, `summary`, `episodes`, `signals` | 1 |
| `HookEpisode` | `task_id`, `score`, `status`, `turns`, `transcript` | — |

### File-channel rules

- A request is `<id>.request.json`, an answer is `<id>.response.json`.
- The response `id` **must** equal the request `id`.
- Every write is atomic: full contents to a sibling temp, then `rename(2)`.
- Reads are bounded by a caller-chosen timeout. Expiry is a loud error naming the
  id, the directory and the seconds waited — never an infinite hang.
- Ids may contain only `[A-Za-z0-9-_]`, because an id becomes a file name.
  Certificate: `contract/ids_cannot_escape_the_channel`, with
  `contract/ids_match_the_deployed_alphabet` as its control in the other
  direction.

### Two things about `ContextHit` worth knowing before you use it

- `text` is the **rendered ladder line** (`[node_id L2] …`), not the raw
  document.
- `score` is the planner's **utility**, not a similarity.

Recording that here is the difference between a consumer that knows what it is
being handed and one that re-scores garbage.

### The `field` hint is reported, not filtered

`ContextQuery::field` is unifur's taxonomy (`pentest`, `coding`, …). The DCR
store does not own that taxonomy — its documents carry a filesystem `source` —
so the field is recorded in the poll report and not used to filter. An honest
over-retrieval is the better failure.

## The model seam — peregrine

Not a contract in the file sense: an HTTP API, versioned by its own conventions.

`omni` speaks the OpenAI dialect `peregrine-serve` already implements, and the
awkward parts of that dialect are specified as certificates rather than as
comments:

| Field | peregrine's behaviour | Consequence for omni |
|---|---|---|
| `temperature` | defaults to **0.0**, not OpenAI's 1.0 | send 0.0 explicitly |
| `top_k` | applied **before** `top_p` | match the precedence |
| `tool_choice` | only the exact string `"none"` disables | `"required"` is accepted and ignored |
| `model`, `n`, `stop`, `seed`, `logprobs`, penalties, `response_format`, `stream_options` | ignored | omni must not depend on them |
| `X-Peregrine-Priority` | `high` \| `normal`, two-tier queue | a real knob |
| 503 `overloaded_error` | the peer's own load signal | retryable, with backoff |
| 413 | prompt over `--max-prompt-tokens` | a hard budget error, never retried |

### The event stream

`Peregrine::stream` speaks SSE. The dialect was read out of `peregrine-serve`'s
source rather than from the OpenAI spec, and three of its properties are not what
a client built from the spec would expect:

| | OpenAI | peregrine |
|---|---|---|
| a tool call | fragmented across deltas | **one whole call per delta** (`main.rs:1468`) |
| `usage` on a stream | with `stream_options.include_usage` | **never** — `usage` exists only in the non-streaming body (`main.rs:1252`) |
| a failed generation | an error frame, then the stream ends | **no finish chunk and no `[DONE]`** — the pump task returns (`main.rs:1054`) |

The third is the one that matters. A finished generation and a failed one close
the socket the same way; only the `data: [DONE]` sentinel tells them apart, so
end-of-body is not end-of-turn and the client treats it as an error rather than
as an answer. Both halves are certified in the `peregrine` group — nineteen
certificates, one per failure mode.

`omni_model::Model` is *not* implemented for streaming and is not meant to be:
`chat` returns a completed turn, and a stream has no completed turn to return yet.
`Peregrine::stream` is a second, separately named entry point.

One caveat: `peregrine-serve` has no test asserting a stream's raw bytes — it
asserts against `axum`'s `Sse` type — so the framing (`data: <payload>\n\n` per
event, `:\n\n` per keep-alive) is a property of `axum` 0.7.9 as much as of
peregrine. The client implements the event-stream rules rather than only what
`axum` emits, and the certificates pin those rules.

Two of peregrine's own environment knobs change token values and therefore belong
in any reward fingerprint: `COLI_KV_DTYPE`, `COLI_ACT_F32`,
`COLI_GPU_DENSE`, `COLI_GPU_DENSE_LAYERS`, `COLI_DSA`, `COLI_ROUTE_MIN_SHARE`,
plus `COLI_RLM` and `COLI_MLA_ABSORB`. Everything else it documents as
performance-only, and performance-only knobs do not belong in an identity.

**omni links no peregrine crate.** Compatibility is at the protocol layer. A
path dependency on a workspace you do not build with would make the harness
unbuildable anywhere peregrine is not present, to buy a type alias.

### DCR identity

omni uses canonical `dcr` (subnext), not `peregrine-ctx`. The latter is a
six-week-old port missing `interop`, `sessions`, `skills` and `fit`.

Two DCR implementations in two processes is fine and intended — they coordinate
over the version-1 file channel above. peregrine's `COLI_CTX=1` plane is
per-request and stateless; omni's working set is per-session. They must not be
pointed at the same store.

## The harness-edit seam — Harness-R1

omni adopts the idea and the discipline, not the benchmark bindings.

The loop:

```text
frozen target rollout
  -> batch failure packet            (omni_core::HookPacket)
  -> harness engineer                (a Model impl; not yet trained)
  <think>...</think><patch>...</patch>
  -> parse, validate, sandbox the hooks
  -> rerun the same target on the same task identities
  -> reward = patched metric - baseline metric
```

Four properties carried across from Harness-R1, all of which are the point:

1. **Rewards come from reruns, not a judge.** A patch that validates but does
   nothing scores zero, because the score is a difference of measured reruns.
2. **Task identity is pinned by content.** `omni_core::content_id`, never a
   position. Matching integer indices is not proof of a paired comparison.
3. **An invalid patch is a no-patch, and an environment failure is a missing
   evaluation.** Different outcomes, reported differently, never conflated and
   never retried until green.
4. **The failure packet carries deterministic signals, not model-authored labels.**
   If a debugger model graded the failures, the reward would be measuring the
   grader.

The one place omni improves on the source: Harness-R1 pins WebShop's `goal_seed`
and manifest hashes but has no equivalent list of knobs that change token values.
omni folds the resolved knob set into the identity hash, because a "matched
baseline" that differs in one field while sharing an identity is worse than an
unmatched one — it looks paired.

## What is not true yet

Stated plainly because rule 10 requires it:

- **No peregrine server has been exercised.** No `peregrine-serve` is running on
  this machine, so the twenty-seven `peregrine` certificates and the client tests
  prove the dialect and the transport, not peregrine's behaviour. The transport
  was proven end to end against **ollama on :11434**, which is OpenAI-compatible
  but *not* peregrine — so "it works against a compatible server" is a weaker
  claim than "it works against peregrine", and it is the weaker claim that is
  true. The streaming reader is in the same position: its fixtures are
  transcribed from `peregrine-serve`'s source and nothing has been read off a
  live stream.
- **The event stream is read, not consumed.** `Peregrine::stream` exists and is
  certified, but no loop uses it yet and `omni model peregrine --complete` still
  takes the blocking path, so streaming is a tested capability rather than an
  exercised one.
- **No live integration test runs in CI.** `omni model peregrine --complete` is
  the gate; it needs a server and reports its absence rather than skipping.
- **There is no harness engineer model.** `HookPacket` exists and is tested; the
  thing that reads one and emits a patch does not. Every current patch is
  hand-authored or hand-seeded.
- **The hooks are built; nothing hosts them yet.** `omni-hook` exists: a
  harness-engineer patch is a declarative rule set (not model-written code —
  the interpreter design that would have needed `unsafe` was removed), parsed,
  validated against budgets at install, and hosted at the four lifecycle
  positions (`on_init`, `make_pre_hint`, `on_before_action`, `on_post_step`)
  I/O-free and deterministically. `on_before_action` implements
  `omni_model::Gate` pure, so a dry-run verdict is the dispatch verdict. What
  is still not true: no harness-engineer model exists (every current patch is
  hand-authored), and the rollout that feeds rewards is scripted
  (`omni-loop::rollout`) — demonstration-grade, not evidence-grade. What IS
  true: **one reward has been computed through the full loop** — baseline
  rollout, failure packet, hand-authored patch, sandbox, rerun,
  identity-paired delta (`omni-loop::rollout::evaluate_patch`, five tests:
  gate-a-trap prices positive, invalid patch prices zero, abort prices the
  cheaper failure, identity pairing is an exact-zero control).
- **`omni-model-unifur` will not work.** unifur's shipped checkpoints are
  370 KB–1 MB geometry experiments and `quality_coder/` is explicitly "no model
  loaded, no training". The seam will be feature-gated and will report the
  absence with the reason rather than loading a geometry model and calling it an
  agent.
- **The subnext responder is uncommitted** in `/home/cortix/Downloads/subnext`
  (`?? src/interop.rs`). omni links `dcr` at a revision; if that file is not yet
  in the canonical tree, the channel has no server and every `ContextQuery`
  times out.
- **The graph and the loop shape records, nothing records yet.** `omni-graph`
  and `omni-loop` exist with certified unit semantics, but no runtime writes a
  graph file or runs a council — the only callers are their own tests.
- **No reward has been computed.** `omni-eval` exists and its rule-8 semantics
  are certified (pairing by identity, invalid patch = zero, environment
  failure = named missing evaluation, three `eval` certificates), but no
  baseline/patch batches have been produced by real runs, so there is no `Δ`
  to report — and the `omni` binary has no command that would compute one.

## Research-backed improvements

Subnext now ships research-backed modules for context assembly, eviction, unified scheduling, durability, and authorization. See `/home/cortix/Downloads/inc/subnext/docs/research-improvements.md`.
