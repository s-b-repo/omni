//! The sandboxed hook runtime — the third stage of the harness-edit loop.
//!
//! The loop (see `docs/CONTRACT.md`): a frozen target rollout is batched into
//! an [`omni_core::HookPacket`], a harness engineer reads it and emits a
//! *patch*, the patch is parsed, validated and hosted here, the same target is
//! rerun on exactly the same task identities, and `omni-eval` scores the
//! difference of measured reruns. This crate owns the middle stage.
//!
//! The design decision that makes "sandbox" mean something: a patch is not
//! code. It is a **declarative rule set** — matchers over data the rollout
//! already produced, effects from a fixed vocabulary. An interpreter for
//! model-written code needs a host boundary and an instruction budget to be
//! safe; a typed rule set needs neither, so this crate carries
//! `#![forbid(unsafe_code)]` like every other crate. AGENTS.md rule 9
//! expected an interpreter whose host boundary would need `unsafe`; the rule
//! set design removed the interpreter, and the exception with it.
//!
//! The three stages, and where each lives:
//!
//! 1. **Parse** — `serde_json` against the versioned schema. A missing version
//!    parses as `0`, and `0` is refused like any unknown version (rule 5).
//! 2. **Validate** — budgets and positional validity, checked at install time
//!    so the runtime can never be surprised mid-rollout: rule count, hint and
//!    reason lengths, state caps, and the rule that each effect and matcher
//!    makes sense at the lifecycle position it was declared for. An invalid
//!    patch is an error, which omni-eval scores as a zero — the "invalid
//!    patch is a no-patch" rule, as a type.
//! 3. **Host** — four lifecycle positions, evaluated in declaration order,
//!    deterministic, and **I/O-free by construction**: a hook sees only the
//!    data handed to it and writes only into a bounded state map and an audit
//!    trail. `on_before_action` is additionally pure (`&self`), so
//!    [`HookRuntime`] implements [`omni_model::Gate`] and the frontier can
//!    dry-run a candidate action through the *same* verdict the dispatcher
//!    will get — the omni-graph property, carried one stage up.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use omni_core::{Error, HookEpisode, HookPacket, canonicalize, check_version, content_id_str};
use omni_model::{Decision, Gate, GateKind};
use serde::{Deserialize, Serialize};

/// The patch schema version. A missing field parses as `0` and is refused like
/// any unknown version — a peer that drops the field must not be mistaken for
/// a peer speaking a newer dialect.
pub const HOOK_PATCH_VERSION: u32 = 1;

// ── budgets ─────────────────────────────────────────────────────────────────
//
// Checked at install, so a patch that would exceed them is refused before any
// rollout spends a token on it. The numbers are not performance tuning: they
// are the shape of "a patch, not a program".

/// Most rules one patch may carry.
pub const MAX_RULES: usize = 64;
/// Longest single injected hint, in characters.
pub const MAX_HINT_CHARS: usize = 2_000;
/// Longest total pre-hint, in characters, across every hint rule that fires.
pub const MAX_PREHINT_TOTAL_CHARS: usize = 4_000;
/// Longest deny/abort reason, in characters.
pub const MAX_REASON_CHARS: usize = 300;
/// Longest annotation note, in characters.
pub const MAX_NOTE_CHARS: usize = 500;
/// Most state entries one session may hold.
pub const MAX_STATE_ENTRIES: usize = 32;
/// Longest state key, in characters.
pub const MAX_STATE_KEY_CHARS: usize = 100;
/// Longest state value, in characters.
pub const MAX_STATE_VALUE_CHARS: usize = 500;
/// Longest patch name, in characters.
pub const MAX_NAME_CHARS: usize = 100;

// ── schema ──────────────────────────────────────────────────────────────────

/// Where a rule fires in the rollout's life.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Lifecycle {
    /// Once, when the rerun starts. Seeding state and annotations only.
    OnInit,
    /// Before the model's first turn. The classic harness edit: prepend what
    /// the failure packet taught.
    MakePreHint,
    /// Before each candidate action. Pure; [`HookRuntime`] implements
    /// [`omni_model::Gate`] with exactly these rules.
    OnBeforeAction,
    /// After each episode. The only position that may abort a run.
    OnPostStep,
}

impl Lifecycle {
    /// Stable label for audit rows.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Lifecycle::OnInit => "on_init",
            Lifecycle::MakePreHint => "make_pre_hint",
            Lifecycle::OnBeforeAction => "on_before_action",
            Lifecycle::OnPostStep => "on_post_step",
        }
    }

    /// Whether `effect` may fire at this position. The vocabulary is
    /// deliberately narrower than "whatever the engineer typed": each position
    /// gets the effects that mean something there, and validation refuses the
    /// rest by name.
    #[must_use]
    pub fn admits(self, effect: &Effect) -> bool {
        match self {
            Lifecycle::OnInit => {
                matches!(effect, Effect::Remember { .. } | Effect::Annotate { .. })
            }
            Lifecycle::MakePreHint => matches!(effect, Effect::InjectHint { .. }),
            Lifecycle::OnBeforeAction => matches!(effect, Effect::Deny { .. }),
            Lifecycle::OnPostStep => {
                matches!(
                    effect,
                    Effect::Remember { .. } | Effect::Annotate { .. } | Effect::Abort { .. }
                )
            }
        }
    }

    /// Whether `when` can be evaluated at this position. `on_before_action`
    /// runs *before* any episode exists, so matchers over episode data are
    /// refused there rather than silently never firing.
    #[must_use]
    pub fn admits_matcher(self, when: &Matcher) -> bool {
        match self {
            Lifecycle::OnBeforeAction => matches!(
                when,
                Matcher::Always | Matcher::TranscriptContains { .. } | Matcher::StatusEquals { .. }
            ),
            _ => true,
        }
    }
}

/// What a rule matches on. Every matcher reads data the rollout already
/// produced — a hook never reaches outside its arguments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Matcher {
    /// Fires unconditionally.
    Always,
    /// At episode positions: the episode's transcript contains this text
    /// (the packet's transcripts are already redacted and truncated
    /// upstream). At `on_before_action`: the tool name contains it.
    TranscriptContains { text: String },
    /// At episode positions: the episode ended with this status. At
    /// `on_before_action`: the tool name equals it.
    StatusEquals { status: String },
    /// The episode reached at least this many turns.
    TurnsAtLeast { n: usize },
    /// The packet carries a deterministic signal of at least this value.
    SignalAtLeast { key: String, at_least: u64 },
}

impl Matcher {
    fn matches_episode(&self, episode: Option<&HookEpisode>, signals: &[(String, u64)]) -> bool {
        match self {
            Matcher::Always => true,
            Matcher::TranscriptContains { text } => episode
                .map(|e| e.transcript.contains(text.as_str()))
                .unwrap_or(false),
            Matcher::StatusEquals { status } => episode.map(|e| e.status == *status).unwrap_or(false),
            Matcher::TurnsAtLeast { n } => episode.map(|e| e.turns >= *n).unwrap_or(false),
            Matcher::SignalAtLeast { key, at_least } => {
                signals.iter().any(|(k, v)| k == key && v >= at_least)
            }
        }
    }
}

/// What a rule does when it fires.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Effect {
    /// Prepend guidance before the model's first turn.
    InjectHint { text: String },
    /// Refuse one candidate action, with a reason. Never empty — a refusal
    /// without a cause cannot be acted on or learned from.
    Deny { reason: String },
    /// Record an observation in the audit trail.
    Annotate { note: String },
    /// Write one bounded entry into the session state.
    Remember { key: String, value: String },
    /// End the run after this episode, with a reason.
    Abort { reason: String },
}

/// One rule: a position, a matcher, an effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HookRule {
    pub at: Lifecycle,
    pub when: Matcher,
    pub effect: Effect,
}

/// A harness-engineer patch: the unit the loop installs, reruns and scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HookPatch {
    #[serde(default)]
    pub version: u32,
    /// A stable name for audit rows and reward records.
    pub name: String,
    /// Rules fire in declaration order. Order is semantics, not presentation:
    /// first matching deny wins, hints concatenate in order.
    pub rules: Vec<HookRule>,
}

impl HookPatch {
    /// Parse patch text and validate it fully, or refuse it by name. This is
    /// the crate's front door: everything [`HookRuntime`] hosts has passed
    /// through here.
    pub fn install(patch_text: &str) -> Result<HookPatch, Error> {
        let patch: HookPatch = serde_json::from_str(patch_text)?;
        patch.validate()?;
        Ok(patch)
    }

    /// Every structural and budget claim, checked once, naming the failure.
    pub fn validate(&self) -> Result<(), Error> {
        check_version("HookPatch", self.version, HOOK_PATCH_VERSION)?;
        if self.name.is_empty() {
            return Err(Error::Format("hook patch has an empty name".to_string()));
        }
        if self.name.chars().count() > MAX_NAME_CHARS {
            return Err(Error::Format(format!(
                "hook patch name exceeds {MAX_NAME_CHARS} characters"
            )));
        }
        if self.rules.len() > MAX_RULES {
            return Err(Error::Hook {
                hook: self.name.clone(),
                reason: format!("{} rules exceeds the budget of {MAX_RULES}", self.rules.len()),
            });
        }
        let mut prehint_total = 0usize;
        for (i, rule) in self.rules.iter().enumerate() {
            if !rule.at.admits(&rule.effect) {
                return Err(Error::Hook {
                    hook: self.name.clone(),
                    reason: format!(
                        "rule {i}: effect {} is not valid at {}",
                        effect_kind(&rule.effect),
                        rule.at.as_str()
                    ),
                });
            }
            if !rule.at.admits_matcher(&rule.when) {
                return Err(Error::Hook {
                    hook: self.name.clone(),
                    reason: format!(
                        "rule {i}: matcher {} is not evaluable at {} — no episode exists there",
                        matcher_kind(&rule.when),
                        rule.at.as_str()
                    ),
                });
            }
            match &rule.effect {
                Effect::InjectHint { text } => {
                    if text.is_empty() {
                        return Err(Error::Hook {
                            hook: self.name.clone(),
                            reason: format!("rule {i}: injected hint is empty"),
                        });
                    }
                    let chars = text.chars().count();
                    if chars > MAX_HINT_CHARS {
                        return Err(Error::Hook {
                            hook: self.name.clone(),
                            reason: format!(
                                "rule {i}: injected hint exceeds {MAX_HINT_CHARS} characters"
                            ),
                        });
                    }
                    prehint_total += chars;
                    if prehint_total > MAX_PREHINT_TOTAL_CHARS {
                        return Err(Error::Hook {
                            hook: self.name.clone(),
                            reason: format!(
                                "rules {i}: combined pre-hint exceeds {MAX_PREHINT_TOTAL_CHARS} \
                                 characters — a hint the model cannot read whole is not guidance"
                            ),
                        });
                    }
                }
                Effect::Deny { reason } | Effect::Abort { reason } => {
                    if reason.trim().is_empty() {
                        return Err(Error::Hook {
                            hook: self.name.clone(),
                            reason: format!(
                                "rule {i}: refusal without a reason cannot be acted on"
                            ),
                        });
                    }
                    if reason.chars().count() > MAX_REASON_CHARS {
                        return Err(Error::Hook {
                            hook: self.name.clone(),
                            reason: format!(
                                "rule {i}: reason exceeds {MAX_REASON_CHARS} characters"
                            ),
                        });
                    }
                }
                Effect::Annotate { note } => {
                    if note.chars().count() > MAX_NOTE_CHARS {
                        return Err(Error::Hook {
                            hook: self.name.clone(),
                            reason: format!(
                                "rule {i}: annotation exceeds {MAX_NOTE_CHARS} characters"
                            ),
                        });
                    }
                }
                Effect::Remember { key, value } => {
                    if key.is_empty() || key.chars().count() > MAX_STATE_KEY_CHARS {
                        return Err(Error::Hook {
                            hook: self.name.clone(),
                            reason: format!(
                                "rule {i}: state key must be 1..={MAX_STATE_KEY_CHARS} characters"
                            ),
                        });
                    }
                    if value.chars().count() > MAX_STATE_VALUE_CHARS {
                        return Err(Error::Hook {
                            hook: self.name.clone(),
                            reason: format!(
                                "rule {i}: state value exceeds {MAX_STATE_VALUE_CHARS} characters"
                            ),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

/// A short, stable name for the effect, used in refusals and audit rows.
fn effect_kind(effect: &Effect) -> &'static str {
    match effect {
        Effect::InjectHint { .. } => "inject_hint",
        Effect::Deny { .. } => "deny",
        Effect::Annotate { .. } => "annotate",
        Effect::Remember { .. } => "remember",
        Effect::Abort { .. } => "abort",
    }
}

/// A short, stable name for the matcher, used in refusals.
fn matcher_kind(matcher: &Matcher) -> &'static str {
    match matcher {
        Matcher::Always => "always",
        Matcher::TranscriptContains { .. } => "transcript_contains",
        Matcher::StatusEquals { .. } => "status_equals",
        Matcher::TurnsAtLeast { .. } => "turns_at_least",
        Matcher::SignalAtLeast { .. } => "signal_at_least",
    }
}

// ── runtime ─────────────────────────────────────────────────────────────────

/// One audit row: what fired, where, whether it took effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEntry {
    pub lifecycle: &'static str,
    /// Index into the patch's rule list, so a row names its rule exactly.
    pub rule: usize,
    pub effect: &'static str,
    /// True when the effect was *not* applied — a budget refusal at runtime,
    /// which must be visible rather than swallowed.
    pub skipped: bool,
    pub detail: String,
}

/// What a post-step hook decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Flow {
    Continue,
    /// End the run; the reason is in the audit trail.
    Abort(String),
}

/// The hosted patch, its bounded session state, and its audit trail.
///
/// Constructed only from a patch that passed [`HookPatch::install`], so every
/// runtime invariant is a validation invariant.
#[derive(Debug, Clone)]
pub struct HookRuntime {
    patch: HookPatch,
    state: BTreeMap<String, String>,
    audit: Vec<AuditEntry>,
}

impl HookRuntime {
    /// Host a validated patch. Session state starts empty; `on_init` seeds it.
    #[must_use]
    pub fn new(patch: HookPatch) -> Self {
        Self {
            patch,
            state: BTreeMap::new(),
            audit: Vec::new(),
        }
    }

    /// The installed patch.
    #[must_use]
    pub fn patch(&self) -> &HookPatch {
        &self.patch
    }

    /// The session state the hooks have built.
    #[must_use]
    pub fn state(&self) -> &BTreeMap<String, String> {
        &self.state
    }

    /// The full audit trail, oldest first.
    #[must_use]
    pub fn audit(&self) -> &[AuditEntry] {
        &self.audit
    }

    /// The patch's content identity: the hash of its canonical form. Two
    /// reruns that hosted the same patch carry the same id, which is what
    /// makes a reward comparison able to prove it.
    pub fn patch_id(&self) -> Result<String, Error> {
        let value = serde_json::to_value(&self.patch)?;
        let canonical = canonicalize(&value)?;
        Ok(content_id_str(&canonical))
    }

    /// Run every `at` rule that matches, in declaration order, auditing each.
    fn fire(
        &mut self,
        at: Lifecycle,
        episode: Option<&HookEpisode>,
        signals: &[(String, u64)],
    ) -> Vec<Effect> {
        let mut fired = Vec::new();
        for (i, rule) in self.patch.rules.iter().enumerate() {
            if rule.at != at || !rule.when.matches_episode(episode, signals) {
                continue;
            }
            let applied = match &rule.effect {
                Effect::Remember { key, value } => {
                    let room =
                        self.state.len() < MAX_STATE_ENTRIES || self.state.contains_key(key);
                    if room {
                        self.state.insert(key.clone(), value.clone());
                    }
                    room
                }
                _ => true,
            };
            self.audit.push(AuditEntry {
                lifecycle: at.as_str(),
                rule: i,
                effect: effect_kind(&rule.effect),
                skipped: !applied,
                detail: if applied {
                    String::new()
                } else {
                    format!("state budget of {MAX_STATE_ENTRIES} entries is full")
                },
            });
            fired.push(rule.effect.clone());
        }
        fired
    }

    fn annotate(&mut self, at: Lifecycle, note: String) {
        self.audit.push(AuditEntry {
            lifecycle: at.as_str(),
            rule: usize::MAX,
            effect: "annotation",
            skipped: false,
            detail: note,
        });
    }

    /// Lifecycle 1: the rerun starts. Signals come from the failure packet the
    /// engineer conditioned on, so a rule may seed state from them.
    pub fn on_init(&mut self, packet: &HookPacket) {
        for effect in self.fire(Lifecycle::OnInit, None, &packet.signals) {
            if let Effect::Annotate { note } = effect {
                self.annotate(Lifecycle::OnInit, note);
            }
        }
    }

    /// Lifecycle 2: before the model's first turn. Matching hints concatenate
    /// in declaration order, joined with a blank line. `None` when no rule
    /// fired — a patch that hints at nothing adds nothing.
    #[must_use]
    pub fn make_pre_hint(&mut self, packet: &HookPacket) -> Option<String> {
        let fired = self.fire(Lifecycle::MakePreHint, None, &packet.signals);
        let hints: Vec<String> = fired
            .into_iter()
            .filter_map(|effect| match effect {
                Effect::InjectHint { text } => Some(text),
                _ => None,
            })
            .collect();
        if hints.is_empty() {
            None
        } else {
            Some(hints.join("\n\n"))
        }
    }

    /// Lifecycle 4: after one episode. The only position that can end a run.
    pub fn on_post_step(&mut self, episode: &HookEpisode) -> Flow {
        let fired = self.fire(Lifecycle::OnPostStep, Some(episode), &[]);
        for effect in &fired {
            if let Effect::Annotate { note } = effect {
                self.annotate(Lifecycle::OnPostStep, note.clone());
            }
        }
        for effect in fired {
            if let Effect::Abort { reason } = effect {
                return Flow::Abort(reason);
            }
        }
        Flow::Continue
    }
}

/// The pure gate: [`Lifecycle::OnBeforeAction`] rules, evaluated with
/// `&self`, no state writes, no audit appends — dry-run and dispatch get the
/// same verdict from the same inputs, which is the omni-graph property.
impl Gate for HookRuntime {
    // The verdict is a function of the tool name and the patch alone. A rule
    // that means one exact call composes with `args_match_id` at the call
    // site; inside the gate the argument bytes are not read, which is part of
    // what makes the verdict cheap enough to run speculatively.
    fn evaluate(&self, tool: &str, _args: &serde_json::Value) -> Decision {
        for rule in &self.patch.rules {
            if rule.at != Lifecycle::OnBeforeAction {
                continue;
            }
            if let Effect::Deny { reason } = &rule.effect {
                if action_matches(&rule.when, tool) {
                    return Decision::block(GateKind::Custom, reason.clone());
                }
            }
        }
        Decision::allow()
    }

    fn name(&self) -> &str {
        &self.patch.name
    }
}

/// The action-side matcher: `Always`, a tool-name containment, or a tool-name
/// equality. Exact-call targeting composes with these through
/// [`args_match_id`]: the rule matches the tool family, and the caller can
/// additionally compare the call's content id — the same identity the graph
/// nodes use — before dispatching.
fn action_matches(when: &Matcher, tool: &str) -> bool {
    match when {
        Matcher::Always => true,
        Matcher::TranscriptContains { text } => tool.contains(text.as_str()),
        Matcher::StatusEquals { status } => tool == status,
        _ => false,
    }
}

/// Whether the arguments' canonical content id equals `id`. The exact-call
/// matcher an `on_before_action` rule uses when it means one call and not a
/// tool family.
#[must_use]
pub fn args_match_id(args: &serde_json::Value, id: &str) -> bool {
    canonicalize(args)
        .map(|canonical| content_id_str(&canonical) == id)
        .unwrap_or(false)
}
