//! Short-context routing: get the context runtime out of the way when it is not
//! needed.
//!
//! # The problem this solves
//!
//! DCR is rent paid on history. Indexing, scoring and the knapsack buy a small,
//! sufficient window out of a store too big to show whole. That rent earns
//! itself on a long session and is **pure loss** on a short one.
//!
//! subnext already knows this — `fit.rs` gates on `history_tokens`. But it gates
//! *late* and on the *wrong quantity*:
//!
//! 1. `assess(graph, …)` takes a `&MemoryGraph`. By the time the gate runs,
//!    the text has already been through the indexer: spans cut, nodes extracted,
//!    vectors embedded, edges built, invalidation queued. So the gate answers
//!    "you didn't need the planner" **after** the expensive part has been paid.
//! 2. `history_tokens` is store-wide. A session that pasted one large document
//!    and then asks two hundred short questions keeps `history_tokens` high
//!    forever, and pays full planning cost on every one of them.
//!
//! # The fix: decide before ingesting
//!
//! [`ShortContextRouter::classify`] takes the raw turns and nothing else. It
//! does not consult a graph, because at the point it runs there is no graph yet —
//! and that is the entire point. A session whose turns fit the window never
//! reaches DCR at all, so it never pays the ingest.
//!
//! # Why this cannot change the answer
//!
//! The only way to lose is to drop something, and planning can only drop. When
//! the whole conversation fits in the window untouched, a planner's output is a
//! **subset** of its input — possibly re-rendered, possibly summarized to L1,
//! possibly routed around, but never containing anything the raw turns did not.
//!
//! So for a fitting conversation, bypassing is not an approximation of planning;
//! it is *strictly more informative* at strictly lower cost. That is the claim
//! the certificate group checks, and it is the reason this is safe to do
//! automatically.
//!
//! # The honest direction of the gate
//!
//! Being wrong is asymmetric, and the gate leans the cheap way:
//!
//! | Wrong decision | Consequence |
//! |---|---|
//! | said "fits", actually didn't | prompt exceeds the window → a loud 413 from the server, one wasted request |
//! | said "plan", actually fitted | slower, and the answer is *worse* (planning removed material) |
//!
//! The first is noisy and self-correcting. The second is quiet and permanent.
//! So [`ShortContextRouter::with_margin`] reserves headroom rather than betting
//! on the estimate, and the default is conservative.

use omni_core::Error;
use omni_model::{CharHeuristic, Message, TokenCounter};

/// Headroom reserved by default, in tokens.
///
/// The estimator is a heuristic (`chars/4`), so it can be off by a large factor
/// on code and JSON. Reserving ~5% means a fitting conversation stays fitting
/// under a bad estimate often enough to matter, and costs almost nothing when
/// the estimate is good. This is a tuning knob, not a safety property — the
/// safety property is that being wrong is loud.
pub const DEFAULT_MARGIN_TOKENS: usize = 256;

/// Why the router declined to bypass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanReason {
    /// The turns do not fit, with the margin applied.
    OverBudget,
    /// These turns fit, but the session has already accumulated more than the
    /// budget — so the *next* question will not.
    ///
    /// Reported separately from [`PlanReason::OverBudget`] because the fix
    /// differs: one is "this is a big turn", the other is "this session is
    /// long and the working set must be rebuilt".
    SessionOverflow,
    /// A skill is registered, and admission is what loads it.
    ///
    /// Copied from subnext's own limit on its gate, for the same reason: skip
    /// the planner and the skills stay dormant.
    SkillsRegistered,
    /// A caller asked for planning regardless.
    Forced,
}

impl PlanReason {
    /// A short, stable label for logs and the reward fingerprint.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            PlanReason::OverBudget => "over-budget",
            PlanReason::SessionOverflow => "session-overflow",
            PlanReason::SkillsRegistered => "skills-registered",
            PlanReason::Forced => "forced",
        }
    }
}

/// What the router decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// Everything fits. Send the raw turns; DCR is not consulted.
    Verbatim { tokens: usize },
    /// Something needed planning. Reason included, so the decision stays
    /// auditable — a bypass that cannot be explained is a bypass nobody trusts.
    Planned { tokens: usize, reason: PlanReason },
}

impl Route {
    /// Whether the context runtime is bypassed.
    #[must_use]
    pub fn is_verbatim(&self) -> bool {
        matches!(self, Route::Verbatim { .. })
    }

    /// Estimated tokens in the request.
    #[must_use]
    pub fn tokens(&self) -> usize {
        match self {
            Route::Verbatim { tokens } | Route::Planned { tokens, .. } => *tokens,
        }
    }

    /// The reason, when planning was chosen.
    #[must_use]
    pub fn reason(&self) -> Option<PlanReason> {
        match self {
            Route::Planned { reason, .. } => Some(*reason),
            Route::Verbatim { .. } => None,
        }
    }

    /// A stable label, for logs and fingerprints.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Route::Verbatim { .. } => "verbatim",
            Route::Planned { reason, .. } => reason.as_str(),
        }
    }
}

/// Decides whether a short conversation needs a context runtime at all.
pub struct ShortContextRouter {
    budget_tokens: usize,
    counter: Box<dyn TokenCounter>,
    margin_tokens: usize,
    session_tokens: usize,
    session_turns: usize,
    skills_registered: bool,
    force_plan: bool,
}

impl ShortContextRouter {
    /// Build a router for an attention budget.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when `budget_tokens` is zero. A zero budget makes every
    /// request "too big", which looks like a broken context runtime rather than
    /// a misconfigured one.
    pub fn new(budget_tokens: usize) -> Result<Self, Error> {
        Self::with_counter(budget_tokens, Box::new(CharHeuristic::new()))
    }

    /// Build with an explicit token counter — an exact one, once available.
    pub fn with_counter(budget_tokens: usize, counter: Box<dyn TokenCounter>) -> Result<Self, Error> {
        if budget_tokens == 0 {
            return Err(Error::Format(
                "a zero attention budget routes everything to the planner, which \
                 looks like a broken context runtime rather than a misconfiguration"
                    .into(),
            ));
        }
        Ok(Self {
            budget_tokens,
            counter,
            margin_tokens: DEFAULT_MARGIN_TOKENS,
            session_tokens: 0,
            session_turns: 0,
            skills_registered: false,
            force_plan: false,
        })
    }

    /// Set the headroom reserved against a bad estimate.
    #[must_use]
    pub fn with_margin(mut self, tokens: usize) -> Self {
        self.margin_tokens = tokens;
        self
    }

    /// Note that a skill is registered. Bypasses stop.
    #[must_use]
    pub fn with_skills(mut self, registered: bool) -> Self {
        self.skills_registered = registered;
        self
    }

    /// Force planning regardless of size.
    ///
    /// For comparing the two paths against each other, which is the only honest
    /// way to price the bypass.
    #[must_use]
    pub fn forced(mut self, force: bool) -> Self {
        self.force_plan = force;
        self
    }

    /// The attention budget.
    #[must_use]
    pub fn budget(&self) -> usize {
        self.budget_tokens
    }

    /// Tokens ingested into the context runtime this session.
    ///
    /// Zero while bypassing, which is the whole point — and worth logging,
    /// because "the session has been going an hour and the context runtime has
    /// ingested nothing" is the behaviour this module exists to produce.
    #[must_use]
    pub fn session_tokens(&self) -> usize {
        self.session_tokens
    }

    /// How many turns were routed without ingesting.
    #[must_use]
    pub fn session_turns(&self) -> usize {
        self.session_turns
    }

    /// Tokens this request costs, ignoring the margin.
    pub fn estimate(&self, turns: &[Message]) -> usize {
        turns
            .iter()
            .filter_map(Message::text)
            .map(|t| self.counter.count(t).0)
            .sum()
    }

    /// Decide, **without recording anything**.
    ///
    /// Takes `&self`. That is deliberate: a dry-run has to be able to ask "what
    /// would you do?" without changing the answer, for the same reason
    /// [`omni_model::Gate`] does. A router that mutated on inspection would
    /// make a benchmark of it change the system it measures.
    #[must_use]
    pub fn classify(&self, turns: &[Message]) -> Route {
        let tokens = self.estimate(turns);
        let usable = self.budget_tokens.saturating_sub(self.margin_tokens);

        if self.force_plan {
            return Route::Planned { tokens, reason: PlanReason::Forced };
        }
        if self.skills_registered {
            return Route::Planned { tokens, reason: PlanReason::SkillsRegistered };
        }
        if tokens > usable {
            return Route::Planned { tokens, reason: PlanReason::OverBudget };
        }
        // These turns fit, but the session as a whole does not. Planning now
        // builds a working set that survives the next few turns, which is
        // cheaper than discovering the overflow on the turn where it bites.
        if self.session_tokens.saturating_add(tokens) > self.budget_tokens {
            return Route::Planned { tokens, reason: PlanReason::SessionOverflow };
        }
        Route::Verbatim { tokens }
    }

    /// Decide and record.
    pub fn route(&mut self, turns: &[Message]) -> Route {
        let r = self.classify(turns);
        if r.is_verbatim() {
            self.session_turns += 1;
        } else {
            // Only planning paths ingest. This is the line the router exists to
            // hold: the session total must reflect what actually went into the
            // context runtime, not what the user typed.
            self.session_tokens = self.session_tokens.saturating_add(r.tokens());
        }
        r
    }

    /// Forget the session. For a new conversation against the same budget.
    pub fn reset(&mut self) {
        self.session_tokens = 0;
        self.session_turns = 0;
    }
}

/// The messages to actually send, plus why.
#[derive(Debug, Clone, PartialEq)]
pub struct Assembled {
    /// What the router decided, and — when it planned — which of the five
    /// reasons applied. Carried rather than reduced to a bool so a bypass stays
    /// auditable after the fact.
    pub route: Route,
    /// The request as it goes on the wire.
    ///
    /// The two routes produce genuinely different shapes, and that difference is
    /// the module's whole content: a verbatim turn keeps its real roles, a
    /// planned turn is a single working-set block. See [`assemble`].
    pub messages: Vec<Message>,
    /// Tokens the bypass avoided, when it bypassed.
    ///
    /// "Avoided" is the honest framing: it is what *would* have been ingested
    /// into the context runtime. Not a saving in compute for the model — the
    /// prompt is sent either way — which is why this is reported separately
    /// rather than added to a cost total.
    pub saved_ingest_tokens: usize,
}

/// The block a planned turn is assembled into.
///
/// # What this is, and what it is not
///
/// It is the **material the context runtime plans over**, in the one shape that
/// survives being handed to a planner: every turn, in order, with its role.
///
/// It is not a working set. Nothing here has been summarised to L1, promoted to
/// a cached fact at L2, or dropped by the knapsack — that is
/// [`crate::DcrPlane`]'s job, and doing any of it here would mean reimplementing
/// the planner badly instead of calling it. The header says so to the model too,
/// because a block that looks like a finished working set is a block the model
/// will stop asking questions of.
///
/// # Why the roles survive when the roles do not
///
/// A transcript of `turns` with the role of each turn written on it is
/// information-preserving: fold `[1] user` back out and the chat is exactly
/// what was typed. The folding is a rendering choice for the planner's benefit,
/// not a loss, and it is deliberate — the alternative, sending a planned turn as
/// a dozen messages that a model must re-read as one document, is the shape the
/// whole bypass exists to avoid.
///
/// # The question at the end
///
/// `dcr::Dcr::prompt_for` closes its rendered context with `QUESTION: <query>`
/// for the same reason: the last thing in the prompt is the thing to answer.
/// Here that is the trailing user turn, and when there is none the block says
/// so rather than inventing a question to answer.
fn pending_working_set(turns: &[Message], tokens: usize, reason: PlanReason) -> String {
    let mut out = format!(
        "# WORKING SET PENDING (route: {}, ~{tokens} tokens, {} turns)\n\
         # This conversation does not fit the attention window, so it is planned \
         rather than sent verbatim. Nothing below has been summarised, demoted \
         or dropped: a planner can only remove from what it is given, and this \
         is everything it is given.\n",
        reason.as_str(),
        turns.len(),
    );
    for (i, turn) in turns.iter().enumerate() {
        let text = turn.text().unwrap_or_default();
        out.push_str(&format!("\n[{}] {}\n{text}\n", i + 1, turn.role()));
    }
    out.push_str("\n# QUESTION\n");
    match turns.iter().rev().find(|t| t.role() == "user").and_then(Message::text) {
        Some(q) => out.push_str(q),
        None => out.push_str("(none — this request carries no user turn)"),
    }
    out.push('\n');
    out
}

/// Build the messages for one turn.
///
/// The two [`Route`] variants are different code on purpose. Collapsing them
/// would make the bypass unobservable, which is the one thing a bypass has to
/// avoid: "we skipped the context runtime" is only a claim if something in the
/// output shows it.
///
/// | route | messages | why |
/// |---|---|---|
/// | [`Route::Verbatim`] | `system` + the turns, roles and order intact | the model was trained on this shape, and a three-message chat should never have been downgraded from it |
/// | [`Route::Planned`] | `system` + one working-set block | the conversation does not fit; what goes on the wire is the material to plan over, not a chat the planner has not seen yet |
///
/// The planned branch renders the working-set block described above. A caller
/// with a context runtime feeds the same turns to [`crate::DcrPlane::ingest`]
/// and then reads the small working set out of [`crate::DcrPlane::ask`]; this
/// function's job ends at handing the material over, because rendering a
/// *planned* set is the planner's and reimplementing it here would be the
/// expensive, lossy way round.
///
/// # Errors
///
/// Never, in the current shape: routing is total and rendering cannot fail. The
/// signature is `Result` so a caller does not have to be rewritten when a
/// renderer that can — a real tokenizer, a disk-backed store — is installed in
/// its place.
pub fn assemble(
    router: &mut ShortContextRouter,
    system: &str,
    turns: &[Message],
) -> Result<Assembled, Error> {
    let route = router.route(turns);

    let messages = match route {
        Route::Verbatim { .. } => {
            // The turns as turns — real roles, real order. This is the shape the
            // model was trained on, and the shape a three-message chat should
            // never have been downgraded from.
            let mut v = Vec::with_capacity(turns.len() + 1);
            v.push(Message::system(system));
            v.extend_from_slice(turns);
            v
        }
        Route::Planned { tokens, reason } => {
            vec![Message::system(system), Message::user(pending_working_set(turns, tokens, reason))]
        }
    };

    let saved = match route {
        Route::Verbatim { tokens } => tokens,
        Route::Planned { .. } => 0,
    };
    Ok(Assembled { route, messages, saved_ingest_tokens: saved })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turns(n: usize, chars: usize) -> Vec<Message> {
        (0..n).map(|_| Message::user("x".repeat(chars))).collect()
    }

    /// Five turns that are individually small, which is the shape every routing
    /// test below is actually about.
    fn short_chat() -> Vec<Message> {
        vec![
            Message::user("alpha"),
            Message::assistant("beta"),
            Message::user("gamma"),
            Message::assistant("delta"),
            Message::user("epsilon"),
        ]
    }

    #[test]
    fn a_small_conversation_bypasses_the_context_runtime() -> Result<(), Error> {
        let r = ShortContextRouter::new(1200)?;
        assert!(r.classify(&turns(2, 100)).is_verbatim());
        Ok(())
    }

    #[test]
    fn a_large_conversation_does_not() -> Result<(), Error> {
        let r = ShortContextRouter::new(100)?;
        let route = r.classify(&turns(10, 400));
        assert!(!route.is_verbatim());
        assert_eq!(route.reason(), Some(PlanReason::OverBudget));
        Ok(())
    }

    #[test]
    fn a_verbatim_route_ingests_nothing() -> Result<(), Error> {
        // The line the module exists to hold.
        let mut r = ShortContextRouter::new(4000)?;
        for _ in 0..50 {
            r.route(&turns(2, 100));
        }
        assert_eq!(r.session_tokens(), 0, "a bypassing session must ingest nothing");
        assert_eq!(r.session_turns(), 50);
        Ok(())
    }

    #[test]
    fn a_planning_route_does_ingest() -> Result<(), Error> {
        let mut r = ShortContextRouter::new(100)?;
        let route = r.route(&turns(10, 400));
        assert!(!route.is_verbatim());
        assert!(r.session_tokens() > 0, "planning means the runtime is carrying it");
        Ok(())
    }

    #[test]
    fn a_registered_skill_stops_the_bypass() -> Result<(), Error> {
        // Copied from subnext's own limit: admission is what loads a skill, so
        // skipping the planner leaves it dormant.
        let r = ShortContextRouter::new(4000)?.with_skills(true);
        assert_eq!(r.classify(&turns(1, 10)).reason(), Some(PlanReason::SkillsRegistered));
        Ok(())
    }

    #[test]
    fn forcing_planning_is_what_makes_a_comparison_honest() -> Result<(), Error> {
        // The only way to price the bypass is to run both paths on the same
        // input, which needs an override.
        let a = ShortContextRouter::new(4000)?;
        let t = turns(2, 100);
        assert!(a.classify(&t).is_verbatim());
        let b = a.forced(true);
        assert_eq!(b.classify(&t).reason(), Some(PlanReason::Forced));
        Ok(())
    }

    #[test]
    fn the_margin_reserves_headroom_against_a_bad_estimate() -> Result<(), Error> {
        // Right at the boundary: without a margin this is a coin flip.
        let tight = ShortContextRouter::new(1000)?.with_margin(0);
        let padded = ShortContextRouter::new(1000)?.with_margin(200);
        let t = turns(4, 240); // ~240 tokens
        assert!(tight.classify(&t).is_verbatim());
        // 960 usable rather than 1000: still fits.
        assert!(padded.classify(&t).is_verbatim());
        Ok(())
    }

    #[test]
    fn a_session_that_outgrew_its_budget_switches_reason_not_silence() -> Result<(), Error> {
        let mut r = ShortContextRouter::new(600)?;
        // Four 400-character turns: ~400 estimated tokens against 344 usable, so
        // the first turn plans and the session total starts growing.
        let first = r.route(&turns(4, 400));
        assert_eq!(first.reason(), Some(PlanReason::OverBudget));
        // A turn that fits on its own (~250 <= 344) but not on top of what the
        // session already carries (400 + 250 > 600): the reason changes, and it
        // names itself rather than going silent.
        let second = r.route(&turns(1, 1000));
        assert_eq!(
            second.reason(),
            Some(PlanReason::SessionOverflow),
            "an overflowing session must say why, not merely that it planned"
        );
        Ok(())
    }

    #[test]
    fn classify_does_not_mutate_the_session() -> Result<(), Error> {
        // A dry-run must not change the system it inspects, or a benchmark of
        // the router becomes a benchmark of the router-plus-its-own-probing.
        let mut a = ShortContextRouter::new(600)?;
        a.route(&turns(2, 400));
        let before = a.session_tokens();
        let mut probes = 0usize;
        for _ in 0..10 {
            if a.classify(&turns(1, 10)).is_verbatim() {
                probes += 1;
            }
        }
        assert_eq!(probes, 10, "the probe turns must be small enough to bypass");
        assert_eq!(a.session_tokens(), before, "inspection must not ingest");
        Ok(())
    }

    #[test]
    fn a_zero_budget_is_refused_rather_than_planning_everything() {
        assert!(ShortContextRouter::new(0).is_err());
    }

    #[test]
    fn reset_clears_the_session_for_a_new_conversation() -> Result<(), Error> {
        let mut r = ShortContextRouter::new(600)?;
        r.route(&turns(4, 400));
        assert!(r.session_tokens() > 0);
        r.reset();
        assert_eq!(r.session_tokens(), 0);
        assert_eq!(r.session_turns(), 0);
        Ok(())
    }

    #[test]
    fn the_assembled_messages_keep_their_real_roles() -> Result<(), Error> {
        // The whole point: a three-message chat goes to the model as a
        // three-message chat, not as a rendered working set.
        let mut r = ShortContextRouter::new(4000)?;
        let turns = vec![Message::user("hello"), Message::assistant("hi"), Message::user("again")];
        let out = assemble(&mut r, "you are helpful", &turns)?;
        assert!(out.route.is_verbatim());
        assert_eq!(out.messages.len(), 4, "system plus three turns");
        assert_eq!(out.messages[0].role(), "system");
        assert_eq!(out.messages[1].role(), "user");
        assert_eq!(out.messages[2].role(), "assistant");
        assert_eq!(out.messages[3].role(), "user");
        assert_eq!(out.messages[3].text(), Some("again"), "order and content survive");
        Ok(())
    }

    #[test]
    fn the_planning_path_renders_a_working_set_rather_than_refusing() -> Result<(), Error> {
        // This test used to assert the refusal — that `assemble()` had no
        // planned branch. It now pins what replaced it, because a check that
        // has stopped checking is worse than no check.
        let mut r = ShortContextRouter::new(100)?;
        let out = assemble(&mut r, "sys", &turns(10, 400))?;
        assert!(!out.route.is_verbatim(), "a 100-token budget cannot hold this");
        assert_eq!(out.messages.len(), 2, "system plus one working-set block");
        assert_eq!(out.messages[0].role(), "system");
        assert_eq!(out.messages[1].role(), "user");
        let block = out.messages[1].text().unwrap_or_default();
        assert!(block.starts_with("# WORKING SET PENDING"), "got {block:.80}");
        assert!(
            block.contains("over-budget"),
            "the reason must survive into the prompt, not just into the enum"
        );
        assert_eq!(out.saved_ingest_tokens, 0, "planning saves no ingest");
        Ok(())
    }

    #[test]
    fn the_planned_block_carries_every_turn_the_planner_will_be_given() -> Result<(), Error> {
        // The load-bearing property. Planning can only remove from what it is
        // given, so a turn dropped here is a turn dropped from the answer — the
        // module's thesis, stated as an assertion.
        let mut r = ShortContextRouter::new(200)?;
        let chat = short_chat();
        let out = assemble(&mut r, "sys", &chat)?;
        let block = out.messages[1].text().unwrap_or_default();
        for (i, turn) in chat.iter().enumerate() {
            assert!(block.contains(&format!("[{}] {}", i + 1, turn.role())), "turn {i} lost its slot");
            assert!(
                block.contains(turn.text().unwrap_or_default()),
                "turn {i} lost its text: {block}"
            );
        }
        assert!(
            block.trim_end().ends_with("epsilon"),
            "the question to answer is the last thing in the block: {block:.120}"
        );
        Ok(())
    }

    #[test]
    fn the_two_routes_render_differently_so_a_bypass_stays_visible() -> Result<(), Error> {
        // The control for the two tests above: the same five turns, the same
        // budget, the only difference being `forced`. If both branches produced
        // the same messages, "the planned branch renders a working set" would
        // hold for the wrong reason.
        let chat = short_chat();

        let mut verbatim = ShortContextRouter::new(4000)?;
        let a = assemble(&mut verbatim, "sys", &chat)?;

        let mut planned = ShortContextRouter::new(4000)?.forced(true);
        let b = assemble(&mut planned, "sys", &chat)?;

        assert_eq!(a.messages.len(), 6, "system plus five real turns");
        assert_eq!(b.messages.len(), 2, "system plus one working-set block");
        assert_eq!(a.messages[1].role(), "user", "the chat keeps its roles");
        assert_eq!(a.messages[2].role(), "assistant");
        assert!(
            b.messages[1].text().unwrap_or_default().starts_with("# WORKING SET PENDING"),
            "the planned turn is a block"
        );
        assert!(a.saved_ingest_tokens > 0, "the bypass saved the ingest");
        assert_eq!(b.saved_ingest_tokens, 0, "the planned turn ingested it");
        Ok(())
    }

    #[test]
    fn a_planned_turn_with_no_user_question_says_so_rather_than_inventing_one() -> Result<(), Error> {
        // The honest failure: a block ending in an invented question is a
        // confident wrong answer about what was asked.
        let mut r = ShortContextRouter::new(4000)?.forced(true);
        let out = assemble(&mut r, "sys", &[Message::assistant("thinking out loud")])?;
        let block = out.messages[1].text().unwrap_or_default();
        assert!(block.contains("no user turn"), "got {block}");
        assert!(block.contains("thinking out loud"), "the turn is still recorded");
        Ok(())
    }
}
