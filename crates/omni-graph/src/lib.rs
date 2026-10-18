//! A typed node graph with an explicit frontier and idempotent apply.
//!
//! Ported from **openhack**'s `AttackGraph` (see AGENTS.md's "where things come
//! from" table), which states the two properties this crate carries across:
//!
//! > All mutations are idempotent by node id / edge triple so the same
//! > `GraphUpdate` applied twice is a no-op — needed for controller retries
//! > and resume/replay.
//!
//! The load-bearing one is AGENTS.md rule 7: the frontier runs **the same
//! gate** the dispatcher will run, on a candidate action that has not been
//! dispatched yet, and the refusal lands back on the candidate's own node —
//! kind, reason and all — as learnable state the next round starts from. A
//! gate that can only be consulted after the fact is not a gate, so
//! [`omni_model::Gate::evaluate`] takes `&self` and performs no I/O precisely
//! so the frontier and the dispatcher cannot disagree.
//!
//! Ids are content-derived. Two proposals for the same tool with the same
//! arguments are one node, however often each side retries and re-emits them —
//! matching integer indices is not proof of anything here either.
//!
//! This is an in-memory graph. Durable state belongs to the stores that own
//! it; what omni-graph owns is the *replay semantics* — apply twice, one
//! result — which is the part a resume/replay needs to agree on.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use omni_core::{Error, canonicalize, content_id_str};
use omni_model::{Decision, Gate, GateKind};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Actions ─────────────────────────────────────────────────────────────────

/// One candidate dispatch: a tool and its arguments.
///
/// Its identity is the canonical hash of `{tool, args}` — built once, at
/// construction, so no call site can mint an ad-hoc id. The canonical step is
/// what makes key order not matter: two structurally identical calls built by
/// two different serializers are the same action, and two calls that differ in
/// arguments are different actions, whatever their order of arrival.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Action {
    pub tool: String,
    pub args: Value,
    /// `action:` + the content id of the canonicalized arguments.
    pub id: String,
}

impl Action {
    /// Build an action and derive its id.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when `args` contains a non-finite number — the same
    /// refusal canonicalization makes everywhere else in the workspace, and the
    /// same reason: a corrupted value must not silently hash as an absent one.
    pub fn new(tool: &str, args: Value) -> Result<Self, Error> {
        let body = serde_json::json!({ "tool": tool, "args": args });
        let canonical = canonicalize(&body)?;
        Ok(Self {
            tool: tool.to_string(),
            args,
            id: format!("action:{}", content_id_str(&canonical)),
        })
    }
}

/// Where an action stands. A node only ever moves *forward*: application never
/// re-queues a blocked or completed action, because undoing an outcome the
/// harness already saw is how a replay ends up dispatching twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionStatus {
    /// Proposed, not yet dispatched.
    Queued,
    /// The frontier ran the gate and the gate refused.
    Blocked,
    /// The harness dispatched it and recorded the outcome.
    Done,
}

impl ActionStatus {
    /// A short label for logs and reports.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Blocked => "blocked",
            Self::Done => "done",
        }
    }
}

// ── Nodes ───────────────────────────────────────────────────────────────────

/// A discovered target the actions hang off.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssetNode {
    pub id: String,
    /// Human-readable, for reports only — never parsed back.
    pub value: String,
    pub last_seen_round: u64,
}

/// A recorded result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FindingNode {
    pub id: String,
    /// Whether it is verified, not whether it is good. A finding re-recorded
    /// with a stronger verdict is updated, not duplicated.
    pub verified: bool,
}

/// An action on the graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionNode {
    pub id: String,
    pub status: ActionStatus,
    /// The action itself: id, tool, args. [`ActionNode::id`] is identical to
    /// [`Action::id`], so a graph lookup and an action lookup agree without a
    /// conversion.
    pub action: Action,
    /// Front of the queue first. Lower sorts earlier.
    pub priority: i64,
    pub spawned_round: u64,
    /// The frontier's refusal, present exactly when status is
    /// [`ActionStatus::Blocked`]. Kept whole as the `[`omni_model::Decision`]`
    /// the gate returned, so the recorded reason is the gate's own words, not
    /// a paraphrase of them.
    pub refusal: Option<Decision>,
}

impl ActionNode {
    /// Build a queued node for `action`. The id is the action's own id, so two
    /// proposals of the same call are the same node by construction.
    #[must_use]
    pub fn new(action: Action, priority: i64, spawned_round: u64) -> Self {
        Self {
            id: action.id.clone(),
            status: ActionStatus::Queued,
            action,
            priority,
            spawned_round,
            refusal: None,
        }
    }
}

/// The three things a graph holds. A fourth kind is a deliberate extension of
/// the vocabulary, not an accidental `(k, v)` somewhere.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Node {
    Asset(AssetNode),
    Finding(FindingNode),
    Action(ActionNode),
}

impl Node {
    /// The node's id, whatever kind it is.
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Asset(n) => &n.id,
            Self::Finding(n) => &n.id,
            Self::Action(n) => &n.id,
        }
    }

    /// The action view, when the caller already knows the kind.
    #[must_use]
    pub fn as_action(&self) -> Option<&ActionNode> {
        match self {
            Self::Action(a) => Some(a),
            _ => None,
        }
    }
}

// ── Edges ───────────────────────────────────────────────────────────────────

/// What one edge means. Distinct kinds so "links to" is never the only word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// The action needs the asset it targets.
    Requires,
    /// The action expects to produce this finding.
    Produces,
    /// The finding was observed on that asset.
    Observes,
}

// ── Updates and the graph ───────────────────────────────────────────────────

/// One round of proposed state, in the openhack shape: nodes and edges, no
/// in-place semantics, and no "remove". Removal is not an operation this graph
/// supports — findings do not get unsaid, and an unwanted action is simply
/// never dispatched, with its queued node staying queued.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphUpdate {
    pub add_nodes: Vec<Node>,
    pub add_edges: Vec<(String, EdgeKind, String)>,
}

/// The typed node graph.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    nodes: BTreeMap<String, Node>,
    /// Deduplicated `(from, kind, to)` triples, ordered — two runs over the
    /// same updates yield LOG-equal graph state, which is the replay property.
    edges: BTreeSet<(String, EdgeKind, String)>,
    last_round: u64,
}

impl Graph {
    /// Build an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge an update. Returns whether anything changed.
    ///
    /// Merge rules, per openhack:
    ///
    /// - a missing node is inserted;
    /// - an existing [`ActionNode`] progressed past [`ActionStatus::Queued`]
    ///   keeps its outcome; a new emission can re-prioritise a queued node
    ///   but must never un-block or un-dispatch a live one;
    /// - an existing asset or finding is merged in place;
    /// - edges deduplicate by their `(from, kind, to)` triple, so the same
    ///   edge stated on every retry is stated once.
    pub fn apply(&mut self, update: &GraphUpdate, round: u64) -> bool {
        let mut changed = false;

        for node in &update.add_nodes {
            match self.nodes.remove(node.id()) {
                None => {
                    self.nodes.insert(node.id().to_string(), node.clone());
                    changed = true;
                }
                Some(existing) => {
                    let before = existing.clone();
                    let merged = merge(existing, node.clone(), round);
                    changed |= merged != before;
                    self.nodes.insert(merged.id().to_string(), merged);
                }
            }
        }

        for (from, kind, to) in &update.add_edges {
            changed |= self.edges.insert((from.clone(), *kind, to.clone()));
        }

        self.last_round = self.last_round.max(round);
        changed
    }

    /// The top-`k` queued actions, without enforcement.
    ///
    /// Order is `(priority, id)` — a total order that does not depend on
    /// arrival, so the same queue emitted in a different order is the same
    /// queue.
    #[must_use]
    pub fn frontier(&self, k: usize) -> Vec<ActionNode> {
        let mut queued: Vec<&ActionNode> = self
            .nodes
            .values()
            .filter_map(Node::as_action)
            .filter(|a| a.status == ActionStatus::Queued)
            .collect();
        queued.sort_by(|a, b| a.priority.cmp(&b.priority).then_with(|| a.id.cmp(&b.id)));
        queued.into_iter().take(k).cloned().collect()
    }

    /// The frontier with **the real gate** run across it, before anything is
    /// dispatched.
    ///
    /// Every queued action is evaluated — not just the first `k`, because a
    /// refusal from an action outside the top-`k` still shapes the next round.
    /// Allowed candidates come back in frontier order, capped at `k`; the
    /// refusals are written back onto their nodes as [`ActionStatus::Blocked`]
    /// with the gate's own [`Decision`] intact, which is what makes the block
    /// readable by every later round instead of lost in a return value.
    pub fn frontier_gated(
        &mut self,
        k: usize,
        gate: &dyn Gate,
    ) -> (Vec<ActionNode>, Vec<ActionNode>) {
        let candidates = self.frontier(usize::MAX);
        let (mut kept, blocked) = split_by_gate(candidates, gate);

        for b in &blocked {
            if let Some(Node::Action(a)) = self.nodes.get_mut(&b.id) {
                a.status = ActionStatus::Blocked;
                a.refusal = b.refusal.clone();
            }
        }

        kept.truncate(k);
        (kept, blocked)
    }

    /// One action's view, if a node with that id exists.
    #[must_use]
    pub fn action_of(&self, id: &str) -> Option<&ActionNode> {
        self.nodes.get(id).and_then(Node::as_action)
    }

    /// Node count — the simplest replay drift-check.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Edge count.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// The highest round [`Graph::apply`] has been told about.
    #[must_use]
    pub fn last_round(&self) -> u64 {
        self.last_round
    }

    /// Every edge, in from-kind-to order.
    #[must_use]
    pub fn edges(&self) -> Vec<(String, EdgeKind, String)> {
        self.edges.iter().cloned().collect()
    }

    /// Every node, in id order.
    #[must_use]
    pub fn nodes(&self) -> Vec<&Node> {
        self.nodes.values().collect()
    }
}

/// Merge two nodes that share an id.
///
/// A kind disagreement is not silently resolved: the existing node wins
/// wholesale, because replacing an asset with an action under one id is exactly
/// the quiet state corruption the id discipline exists to prevent — and doing
/// it loudly is the caller's bug to find.
fn merge(existing: Node, incoming: Node, round: u64) -> Node {
    match (&existing, &incoming) {
        (Node::Action(a), Node::Action(b)) => {
            let mut merged = a.clone();
            if merged.status == ActionStatus::Queued {
                merged.priority = b.priority;
                merged.spawned_round = merged.spawned_round.max(b.spawned_round);
            }
            Node::Action(merged)
        }
        (Node::Asset(a), Node::Asset(b)) => {
            let mut merged = a.clone();
            merged.last_seen_round = merged.last_seen_round.max(b.last_seen_round).max(round);
            Node::Asset(merged)
        }
        (Node::Finding(a), Node::Finding(b)) => {
            let mut merged = a.clone();
            merged.verified = merged.verified || b.verified;
            Node::Finding(merged)
        }
        _ => existing,
    }
}

/// Run the gate across `candidates`, in their frontier order, recording every
/// verdict.
///
/// Split out from [`Graph::frontier_gated`] so the property that makes rule 7
/// real — *one* gate, run on undispatched candidates, refusals recorded whole —
/// is a function rather than a loop body.
fn split_by_gate(
    candidates: Vec<ActionNode>,
    gate: &dyn Gate,
) -> (Vec<ActionNode>, Vec<ActionNode>) {
    let mut kept = Vec::new();
    let mut blocked = Vec::new();
    for mut candidate in candidates {
        let decision = gate.evaluate(&candidate.action.tool, &candidate.action.args);
        if decision.is_allowed() {
            candidate.refusal = None;
            kept.push(candidate);
        } else {
            candidate.status = ActionStatus::Blocked;
            candidate.refusal = Some(decision);
            blocked.push(candidate);
        }
    }
    (kept, blocked)
}

// ── A real policy the harness can install ───────────────────────────────────

/// Allow only the named tools.
///
/// This is the smallest honest stand-in for openhack's `Scope`, and it is a
/// **real** gate — pure, dry-runnable, and named so audit rows can say what
/// refused a call. A profile replaces it; nothing bypasses it.
#[derive(Debug, Clone, Default)]
pub struct ScopeAllowlist {
    allow: BTreeSet<String>,
}

impl ScopeAllowlist {
    /// A gate allowing exactly the given tools, in any order, duplicates
    /// collapsed.
    #[must_use]
    pub fn new<I, S>(tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self { allow: tools.into_iter().map(Into::into).collect() }
    }

    /// Whether `tool` would be allowed, without building a `Decision`.
    #[must_use]
    pub fn allows(&self, tool: &str) -> bool {
        self.allow.contains(tool)
    }
}

impl Gate for ScopeAllowlist {
    fn evaluate(&self, tool: &str, _args: &Value) -> Decision {
        if self.allow.contains(tool) {
            Decision::allow()
        } else {
            Decision::block(
                GateKind::Scope,
                format!("tool {tool:?} is not in the scope allowlist"),
            )
        }
    }

    fn name(&self) -> &str {
        "scope-allowlist"
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use serde_json::json;

    /// A gate that counts what it was asked about, without influencing the
    /// verdict. The count is an interior-mutable cache — the sanctioned shape
    /// in the trait's own docs — and an atomic because the trait is `Sync`.
    struct RecordingGate {
        allow: BTreeSet<String>,
        asked: AtomicUsize,
    }

    impl Gate for RecordingGate {
        fn evaluate(&self, tool: &str, _args: &Value) -> Decision {
            self.asked.fetch_add(1, Ordering::Relaxed);
            if self.allow.contains(tool) {
                Decision::allow()
            } else {
                Decision::block(GateKind::Scope, format!("recording-gate refuses {tool:?}"))
            }
        }

        fn name(&self) -> &str {
            "recording"
        }
    }

    fn graph_with_action(tool: &str, args: Value, priority: i64) -> Result<(Graph, String), Error> {
        let action = Action::new(tool, args)?;
        let id = action.id.clone();
        let mut g = Graph::new();
        g.apply(
            &GraphUpdate {
                add_nodes: vec![Node::Action(ActionNode::new(action, priority, 1))],
                add_edges: Vec::new(),
            },
            1,
        );
        Ok((g, id))
    }

    fn update_with_action(tool: &str, args: Value, priority: i64) -> Result<(GraphUpdate, String), Error> {
        let action = Action::new(tool, args)?;
        let id = action.id.clone();
        Ok((
            GraphUpdate {
                add_nodes: vec![Node::Action(ActionNode::new(action, priority, 1))],
                add_edges: Vec::new(),
            },
            id,
        ))
    }

    #[test]
    fn the_frontier_runs_the_same_gate_the_dispatcher_will() -> Result<(), Error> {
        // Same cal, same gate, two evaluations: nothing may differ — which is
        // the premise everything else in this module gets to rely on.
        let (mut g, _id) = graph_with_action("curl", json!({"url": "example.com"}), 1)?;

        let gate = RecordingGate {
            allow: BTreeSet::from(["curl".to_string()]),
            asked: AtomicUsize::new(0),
        };
        let direct = gate.evaluate("curl", &json!({"url": "example.com"}));

        let (kept, blocked) = g.frontier_gated(4, &gate);
        assert!(blocked.is_empty(), "the allowed action is not a refusal");
        assert_eq!(kept.len(), 1);
        // The dispatcher evaluates the same call again; the answer must be
        // identical or the "same gate" claim is empty.
        let dispatch_time = gate.evaluate("curl", &json!({"url": "example.com"}));
        assert_eq!(direct, dispatch_time);
        assert_eq!(
            gate.asked.load(Ordering::Relaxed),
            3,
            "frontier once, dispatcher-equivalent twice: all through one gate"
        );
        Ok(())
    }

    #[test]
    fn every_queued_candidate_is_evaluated_even_beyond_the_cap() -> Result<(), Error> {
        let mut g = Graph::new();
        let mut update = GraphUpdate::default();
        let mut ids = Vec::new();
        for tool in ["curl", "nmap", "dig"] {
            let (u, id) = update_with_action(tool, json!({}), 1)?;
            update.add_nodes.extend(u.add_nodes);
            ids.push(id);
        }
        g.apply(&update, 1);

        let gate = RecordingGate { allow: BTreeSet::new(), asked: AtomicUsize::new(0) };
        let (kept, blocked) = g.frontier_gated(1, &gate);
        assert!(kept.is_empty(), "nothing is allowed here");
        assert_eq!(blocked.len(), 3, "all three were evaluated despite k = 1");
        assert_eq!(
            gate.asked.load(Ordering::Relaxed),
            3,
            "a refusal below the cap still shapes the next round"
        );
        for node in g.nodes() {
            if let Node::Action(a) = node {
                assert_eq!(a.status, ActionStatus::Blocked, "{}", a.action.tool);
                assert!(a.refusal.is_some(), "{} must carry its refusal", a.action.tool);
            }
        }
        Ok(())
    }

    #[test]
    fn a_blocked_candidate_carries_the_gates_own_reason() -> Result<(), Error> {
        let (mut g, id) = graph_with_action("write", json!({"path": "/etc"}), 1)?;
        g.frontier_gated(4, &ScopeAllowlist::new(["curl"]));
        let node = g
            .action_of(&id)
            .ok_or_else(|| Error::Format("the action must still be on the graph".into()))?;
        assert_eq!(node.status, ActionStatus::Blocked);
        let refusal = node
            .refusal
            .as_ref()
            .ok_or_else(|| Error::Format("a blocked node carries the gate's own reason".into()))?;
        assert_eq!(refusal.kind, GateKind::Scope);
        assert!(
            refusal.reason.contains("scope allowlist"),
            "the recorded reason is the gate's words: {}",
            refusal.reason
        );
        Ok(())
    }

    #[test]
    fn the_same_args_in_a_different_key_order_are_the_same_action() -> Result<(), Error> {
        let a = Action::new("curl", json!({"url": "http://h", "retries": 2}))?;
        let b = Action::new("curl", json!({"retries": 2, "url": "http://h"}))?;
        assert_eq!(a.id, b.id, "canonical hashing makes key order irrelevant");

        let c = Action::new("curl", json!({"url": "http://h", "retries": 3}))?;
        assert_ne!(a.id, c.id, "changed args are a different action");

        let d = Action::new("dig", json!({"url": "http://h", "retries": 2}))?;
        assert_ne!(a.id, d.id, "a different tool is a different action");
        Ok(())
    }

    #[test]
    fn an_out_of_range_number_does_not_become_an_action() {
        // serde_json refuses `1e999` at the parse boundary: no id path ever
        // sees it. The canonical step has its own refusal of a *constructed*
        // non-finite number, and that one is certified workspace-wide
        // (`canonical/non_finite_is_refused`); what this test pins is the door
        // a wire value would have had to pass first.
        let r = serde_json::from_str::<Value>(r#"{"n": 1e999}"#);
        assert!(r.is_err(), "an out-of-range literal is refused at parse");
    }

    #[test]
    fn apply_is_idempotent_by_node_id_and_edge_triple() -> Result<(), Error> {
        let mut g = Graph::new();
        let (action_update, action_id) = update_with_action("curl", json!({"url": "example.com"}), 2)?;
        let mut update = action_update;
        update.add_nodes.push(Node::Asset(AssetNode {
            id: "asset.example.com".into(),
            value: "example.com".into(),
            last_seen_round: 1,
        }));
        update.add_edges.push((
            action_id.clone(),
            EdgeKind::Requires,
            "asset.example.com".to_string(),
        ));

        g.apply(&update, 1);
        let first = g.clone();
        g.apply(&update, 1);
        assert_eq!(g, first, "applying the same update twice changed nothing");
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1, "the edge dedupes by its (from, kind, to) triple");
        assert_eq!(g.last_round(), 1);
        Ok(())
    }

    #[test]
    fn a_re_emission_cannot_un_block_a_refused_action() -> Result<(), Error> {
        let (mut g, id) = graph_with_action("write", json!({}), 1)?;
        g.frontier_gated(4, &ScopeAllowlist::new(["curl"]));

        // The controller retries and re-proposes the same call, queued again.
        let action = Action::new("write", json!({}))?;
        g.apply(
            &GraphUpdate {
                add_nodes: vec![Node::Action(ActionNode::new(action, 1, 2))],
                add_edges: Vec::new(),
            },
            2,
        );
        let node = g
            .action_of(&id)
            .ok_or_else(|| Error::Format("the node must still exist".into()))?;
        assert_eq!(
            node.status, ActionStatus::Blocked,
            "a re-proposal must not un-block an outcome the harness already saw"
        );
        assert_eq!(node.spawned_round, 1, "the original round survives the replay");
        assert_eq!(g.node_count(), 1, "the re-proposal is the same node, not a new one");
        assert_eq!(g.last_round(), 2);
        Ok(())
    }

    #[test]
    fn the_frontier_order_is_priority_then_id_not_arrival() -> Result<(), Error> {
        let mut g = Graph::new();
        let mut update = GraphUpdate::default();
        let mut built: Vec<(i64, String, String)> = Vec::new();
        for (tool, p) in [("curl", 2), ("nmap", 1), ("dig", 2)] {
            let (u, _) = update_with_action(tool, json!({}), p)?;
            built.push((p, tool.to_string(), u.add_nodes[0].id().to_string()));
            update.add_nodes.extend(u.add_nodes);
        }
        g.apply(&update, 1);

        // The expected order is computed here by the same rule the frontier
        // documents — not hardcoded, because the tie-breaks are content ids
        // whose relative order is a property, not a constant.
        built.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.cmp(&b.2)));
        let got = g.frontier(3);
        for (want, have) in built.iter().zip(got.iter()) {
            assert_eq!(&have.action.tool, &want.1, "frontier order follows (priority, id)");
            assert_eq!(&have.id, &want.2);
        }
        // A second call yields the same queue: the order is a fact about the
        // graph, not about emission order.
        let got_again = g.frontier(3);
        assert_eq!(
            got.iter().map(|a| a.id.clone()).collect::<Vec<_>>(),
            got_again.iter().map(|a| a.id.clone()).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn a_kind_mismatch_merges_onto_the_existing_node_rather_than_replacing_it(
    ) -> Result<(), Error> {
        let mut g = Graph::new();
        let asset = Node::Asset(AssetNode { id: "n1".into(), value: "v1".into(), last_seen_round: 1 });
        g.apply(&GraphUpdate { add_nodes: vec![asset], add_edges: Vec::new() }, 1);
        g.apply(
            &GraphUpdate {
                add_nodes: vec![Node::Finding(FindingNode { id: "n1".into(), verified: true })],
                add_edges: Vec::new(),
            },
            2,
        );
        let n = g.nodes().into_iter().find(|n| n.id() == "n1").ok_or_else(|| {
            Error::Format("the node must survive the kind disagreement".into())
        })?;
        assert!(
            matches!(n, Node::Asset(_)),
            "an id cannot quietly change kind; the existing node wins"
        );
        Ok(())
    }

    #[test]
    fn the_allowlist_gate_refuses_with_a_reason_and_only_when_needed() {
        let gate = ScopeAllowlist::new(["curl", "  curl  "]);
        assert!(gate.evaluate("curl", &json!({})).is_allowed());
        let d = gate.evaluate("rm", &json!({}));
        assert!(!d.is_allowed());
        assert!(!d.reason.is_empty(), "a refusal with no reason cannot be acted on");
        assert_eq!(d.kind, GateKind::Scope);
        assert_eq!(gate.name(), "scope-allowlist");
        assert!(gate.allows("curl"));
        assert!(!gate.allows("rm"));
    }

    #[test]
    fn a_blocked_node_reports_its_status_and_count_of_edges_unchanged() -> Result<(), Error> {
        // The refusal is recorded on the node itself, so a graph that grew no
        // edges while refusing is evidence the refuse path did not invent
        // state anywhere else.
        let (mut g, _) = graph_with_action("write", json!({}), 1)?;
        let edges_before = g.edge_count();
        let (_kept, blocked) = g.frontier_gated(4, &ScopeAllowlist::new(["nothing"]));
        assert_eq!(blocked.len(), 1);
        assert_eq!(g.edge_count(), edges_before, "a refusal adds no edges, only node state");
        Ok(())
    }
}
