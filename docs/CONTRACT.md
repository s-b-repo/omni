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
| context / working set | **subnext** (`dcr`) | library link and `context-read` subprocess |
| trained capability | **unifur** (`dblocks`) | native NCM subprocess for Nina |
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

## The native Nina route

`omni model nina --binary P --tokenizer P --generator P` reads one UTF-8
question from stdin. When given `--dcr P --context-store P`, it calls Subnext's
`context-read` over stdin and validates each returned `ContextHit` at version 1.
It passes the available evidence bytes in `DCR_EVIDENCE_BYTES` and packs whole
JSON evidence strings within the generator's recorded `prompt_len`. The full
question stays in the prompt. A missing or malformed store response is an
error. An empty working set preserves the native pilot template.

Omni sends the active prompt through stdin to `dblocks ncm generate --json`
and returns the answer, supplied byte count, evidence IDs, and measured NCM
level and trunk evaluations. Subnext owns durable memory. Nina performs its
trained latent vector updates and geometric scoring within each level.
The request is read-only and the route returns text. The direct route uses
native `.mpk` codec and generator files with their state sidecars.
Use `--backend wgpu --device discrete:0` with a WGPU-capable Unifur build
to run that same request on an accelerator.

The Unifur worker records the generator's codec digest and keeps the
checkpoint and optimizer sidecars with the run. Transfer the matching pair
from one run to Omni. Subnext's storage and retrieval budget is independent
of the accelerator budget for training the model.
[Worker dispatch and checkpoints](https://github.com/s-b-repo/unifur/blob/main/docs/Training-Guide.md#dispatch-a-worker)

`--decode-policy P` passes a native JSON policy to the generator. Use
`--compare-decode-policy P` to replay that policy and the greedy control
against the same checkpoint and evidence. The response adds the candidate
report and its text agreement with the control. The greedy answer remains
the primary response of the comparison. The policy can budget level calls,
drafts, particles and probes, and the runner includes the measured work.

For a compatible target and separately trained DFlash draft, Omni can call
the enabled inference server through its existing OpenAI-compatible client.
The server owns the acceptance algorithm and draft metrics.
[Reference pair](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash)

### DCR identity

Build the `dcr` binary from the Subnext checkout beside Omni. Run
`scripts/verify-cross-repo.sh --strict` with Subnext, Unifur and Peregrine as
sibling checkouts and their cited Git revisions available. The script runs the
live Nina route on CPU and validates Unifur's emitted file-channel manifest.
For an existing file-channel deployment, pass `--deployed-contract PATH`
to check its manifest against the same parser.

Use `omni context-respond CHANNEL DCR STORE [BUDGET]` to answer pending
file-channel requests from a scoped Subnext store. The command validates the
request versions and filenames, caps each response at the requested hit
count, and publishes each response once. Repeated invocations can pick up
new requests without rewriting existing responses.

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

Stated plainly because rule 10 requires it. Items **crossed off** have had
their absence named and the absence removed; they stay listed with what is
true now, because moving them silently out of the list would make a reader
of a cached copy think they were never claimed.

- **No peregrine server has been exercised.** Still true for
  `peregrine-serve`; the twenty-seven `peregrine` certificates and the
  client tests prove the dialect and the transport, not peregrine's
  behaviour. ~~The transport was proven end to end against ollama on
  :11434~~ — it has now **also** been proven against the sibling
  `mesh-shim` (Python stdlib, shells out to `dblocks`): see
  `scripts/verify-live-mesh.sh`, which boots the shim, drives
  `omni model peregrine --complete` at it, and asserts four markers the
  live run must produce. "It works against a compatible server" is still
  the weaker claim; the stronger "it works against peregrine" waits on
  peregrine-serve actually running on this machine.
- ~~**The event stream is read, not consumed.**~~ **Closed.**
  `omni model peregrine --stream` wires `Peregrine::stream` end-to-end:
  manual iteration, delta-at-a-time stdout, server-error and
  finish-reason chunks reported, `collect()` called after exhaustion for
  the matched usage. A streaming request to the current shim returns
  `HTTP 400 stream_unsupported`, which the client classifies into
  `Error::Format` and `retryable: false` — the correct "server declined"
  report, not a bug. A real stream-capable backend hits the same code
  path through the other branch.
- ~~**No live integration test runs in CI.**~~ **Closed.**
  `scripts/verify-live-mesh.sh` is that live gate: one command, non-zero
  on failure, four grep-reducible markers, tears down its own subprocess
  and tempdir. Not yet added to AGENTS.md's standing gate list — a
  review conversation, not a bookkeeping one.
- **There is no harness engineer model.** `HookPacket` exists and is
  tested; the thing that reads one and emits a patch does not. Every
  current patch is hand-authored or hand-seeded.
- **The hooks are built; nothing hosts them yet.** `omni-hook` exists: a
  harness-engineer patch is a declarative rule set (not model-written
  code — the interpreter design that would have needed `unsafe` was
  removed), parsed, validated against budgets at install, and hosted at
  the four lifecycle positions (`on_init`, `make_pre_hint`,
  `on_before_action`, `on_post_step`) I/O-free and deterministically.
  `on_before_action` implements `omni_model::Gate` pure, so a dry-run
  verdict is the dispatch verdict. Still not true: no harness-engineer
  model exists (every current patch is hand-authored), and the rollout
  that feeds rewards is scripted (`omni-loop::rollout`) —
  demonstration-grade, not evidence-grade. True now:
  **rewards are being computed and written to disk** —
  `omni rollout --out <path>` runs the demo baseline, evaluates a patch
  (built-in or `--patch-file`), and writes a versioned
  `RolloutRecord` JSON via `omni_core::write_atomic` (five tests:
  round-trip, missing-version refused, wrong-version refused,
  non-object-top-level refused, refused-patch still records).
- ~~**`omni-model-unifur` will not work.**~~ **Partially closed.**
  Code side: with `--features dblocks`, `Dblocks` implements
  `omni_model::Model` by shelling out to a built `dblocks lm generate`
  binary; 10 unit tests pass (render_prompt, fence parser, argv
  pinning, deterministic-at-zero tracking, missing-binary and
  missing-checkpoint refusals). CLI side: `omni model unifur` mirrors
  `omni model peregrine`'s show-request → probe → `--complete` shape.
  Still not true: unifur ships no agent-capable checkpoint of its own.
  The toy at `nina-data/tiny-model/lm-12f979ad92611014.mpk` works for
  smoke-testing the transport but produces noise — exactly what an
  8-step training run produces. The `nina chat` backend variant
  (routing in the loop) waits on a local registry.
- **The subnext responder is uncommitted** in
  `/home/cortix/Downloads/subnext` (`?? src/interop.rs`). omni links
  `dcr` at a revision; if that file is not yet in the canonical tree,
  the channel has no server and every `ContextQuery` times out.
- ~~**The graph and the loop shape records, nothing records yet.**~~
  **Closed.**
  Loop half: `omni-loop::RolloutRecord` serializes a `PatchVerdict`
  with the baseline `Batch`, `Reward`, `patched` `Batch` and episode
  summaries under `version: 1`; written atomically; read refuses any
  `version` that is not this build's (including missing).
  Graph half: `omni-graph::GraphSnapshot` mirrors the same discipline
  for the typed node graph — version-gated read, mirror counts for
  `node_count`/`edge_count`/`last_round` that are checked against the
  embedded graph body (so a silent `BTreeMap` truncation on replay is
  caught at read time, not hours later as a drift). `omni rollout
  --graph-out <path>` writes an 8-node / 6-edge snapshot (2 assets, 6
  actions, Requires edges) built from the patched episodes.
  What is *still* not true: the graph writer parses transcripts
  post-hoc rather than observing hook calls directly. A richer writer
  is the obvious follow-up.
- ~~**No reward has been computed.**~~ **Closed.**
  `omni rollout` computes a reward on every invocation and writes it
  into the record (`matched`, `delta`, `skipped_for_env`,
  `unpaired_baseline`, `unpaired_patched`). The demo delta is a stable
  `+1.0` because the built-in patch closes the trap on both demo tasks
  (`reward: matched=2 delta=1.0000` on every run).

## Research-backed improvements

Subnext now ships research-backed modules for context assembly, eviction, unified scheduling, durability, and authorization. See `/home/cortix/Downloads/inc/subnext/docs/research-improvements.md`.
