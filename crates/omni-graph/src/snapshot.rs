//! On-disk graph snapshots: the smallest honest shape that turns the typed
//! node graph into a file a later run can read back.
//!
//! What this closes: `docs/CONTRACT.md` "What is not true yet" has called
//! out "the graph and the loop shape records, nothing records yet" since
//! the crate opened. The loop half closed in `omni-loop::record`; this is
//! the graph half.
//!
//! Discipline — same as [`omni_loop::record::RolloutRecord`]:
//!
//! * **Versioned.** The snapshot carries `version = 1`. The reader refuses
//!   any other value — including a missing field, which `#[serde(default)]`
//!   would silently turn into `0`. AGENTS.md rule 5: a missing version is
//!   refused identically to an unknown one.
//! * **Atomic.** Writes go through `omni_core::write_atomic`, so a reader
//!   polling the directory never sees a truncated prefix.
//! * **Round-trippable.** `Graph` already derives `Serialize`/`Deserialize`,
//!   and the tests round-trip a built graph through a temp file.
//! * **Rule 10.** `node_count`, `edge_count`, and `last_round` are mirrored
//!   onto the snapshot as top-level fields — redundant with what a reader
//!   can compute from `graph`, but that is the point: a snapshot whose
//!   mirror disagrees with its own graph is a snapshot that was truncated
//!   or tampered with, and the mismatch is caught by [`GraphSnapshot::read`].

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![forbid(unsafe_code)]

use std::path::Path;

use omni_core::{Context, Error, write_atomic};
use serde::{Deserialize, Serialize};

use crate::Graph;

/// Version of the graph snapshot format. Bumped on any breaking shape change.
pub const GRAPH_SNAPSHOT_VERSION: u32 = 1;

/// The on-disk shape of a graph at one moment.
///
/// The two integer mirrors are the same ones [`Graph::node_count`] /
/// [`Graph::edge_count`] expose. They live on the snapshot so a reader can
/// check the file is internally consistent before trusting the graph body:
/// a BTreeMap that lost rows in transit would pass `serde_json::from_value`
/// cleanly but disagree with its own snapshot's claimed counts, and the
/// mismatch is caught here rather than hours later in a replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub version: u32,
    pub recorded_at_unix: u64,
    pub node_count: usize,
    pub edge_count: usize,
    pub last_round: u64,
    pub graph: Graph,
}

impl GraphSnapshot {
    /// Capture a snapshot of `graph` at the current wall-clock time.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when `SystemTime::now()` precedes `UNIX_EPOCH`,
    /// which does not happen on any system that completed its own boot but
    /// is checked rather than skipped because a zero timestamp would
    /// misrepresent the snapshot's age.
    pub fn capture(graph: Graph) -> Result<Self, Error> {
        let recorded_at_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                Error::Format(format!(
                    "system clock is before UNIX_EPOCH ({e}); a zero timestamp \
                     would misrepresent the snapshot's age so it is refused"
                ))
            })?
            .as_secs();
        let node_count = graph.node_count();
        let edge_count = graph.edge_count();
        let last_round = graph.last_round();
        Ok(Self {
            version: GRAPH_SNAPSHOT_VERSION,
            recorded_at_unix,
            node_count,
            edge_count,
            last_round,
            graph,
        })
    }

    /// Serialize and write atomically.
    ///
    /// # Errors
    ///
    /// Whatever [`write_atomic`] returns, plus a [`Error::Json`] if
    /// serialization fails (which it will not for a `Graph` built from
    /// well-formed `Node`s and `(from, kind, to)` triples).
    pub fn write(&self, path: &Path) -> Result<(), Error> {
        let json = serde_json::to_string_pretty(self)?;
        write_atomic(path, &json)
    }

    /// Read, validate the version by hand, and then deserialize. The mirror
    /// counts are checked against the graph body before the snapshot is
    /// handed back — a tampered or truncated file where `node_count` no
    /// longer matches `graph.node_count()` is refused as [`Error::Format`]
    /// rather than quietly loaded.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if `path` cannot be read. [`Error::Format`] for a
    /// non-object top level, a missing or wrong `version`, a non-integer
    /// `version`, or a mirror-count mismatch. [`Error::Json`] when the body
    /// parses as JSON but not as a snapshot.
    pub fn read(path: &Path) -> Result<Self, Error> {
        let bytes = std::fs::read_to_string(path)
            .ctx(|| format!("reading {}", path.display()))?;
        let value: serde_json::Value = serde_json::from_str(&bytes)?;
        if !value.is_object() {
            return Err(Error::Format(format!(
                "{} is not a JSON object",
                path.display()
            )));
        }
        let version_raw = value.get("version").ok_or_else(|| {
            Error::Format(format!(
                "{} has no `version` field. AGENTS.md rule 5: a missing \
                 version is refused identically to an unknown one.",
                path.display()
            ))
        })?;
        let version = version_raw.as_u64().ok_or_else(|| {
            Error::Format(format!(
                "{} `version` is not an integer: {version_raw}",
                path.display()
            ))
        })?;
        if u32::try_from(version).ok() != Some(GRAPH_SNAPSHOT_VERSION) {
            return Err(Error::Format(format!(
                "{} has version {version}, this build supports version {}",
                path.display(),
                GRAPH_SNAPSHOT_VERSION
            )));
        }
        let snap: GraphSnapshot = serde_json::from_value(value)?;
        if snap.node_count != snap.graph.node_count() {
            return Err(Error::Format(format!(
                "{} declares node_count={} but its graph holds {}; the \
                 mirror disagreeing with the body is the exact signal a \
                 truncation would produce",
                path.display(),
                snap.node_count,
                snap.graph.node_count()
            )));
        }
        if snap.edge_count != snap.graph.edge_count() {
            return Err(Error::Format(format!(
                "{} declares edge_count={} but its graph holds {}",
                path.display(),
                snap.edge_count,
                snap.graph.edge_count()
            )));
        }
        if snap.last_round != snap.graph.last_round() {
            return Err(Error::Format(format!(
                "{} declares last_round={} but its graph holds {}",
                path.display(),
                snap.last_round,
                snap.graph.last_round()
            )));
        }
        Ok(snap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Action, ActionNode, EdgeKind, GraphUpdate, Node};

    fn action(tool: &str, step: u64) -> Result<Action, Error> {
        Action::new(tool, serde_json::json!({ "step": step }))
    }

    fn build_sample() -> Result<Graph, Error> {
        let mut g = Graph::new();
        let a1 = action("read_state", 0)?;
        let a2 = action("purge_cache", 1)?;
        let update = GraphUpdate {
            add_nodes: vec![
                Node::Action(ActionNode::new(a1.clone(), 0, 1)),
                Node::Action(ActionNode::new(a2.clone(), 1, 1)),
            ],
            // `Requires` just needs to be a real EdgeKind for the fixture;
            // the test is about round-trip, not semantics.
            add_edges: vec![(a2.id.clone(), EdgeKind::Requires, a1.id.clone())],
        };
        let changed = g.apply(&update, 1);
        if !changed {
            return Err(Error::Format("fixture apply reported no change".to_string()));
        }
        Ok(g)
    }

    #[test]
    fn a_snapshot_round_trips_through_disk() -> Result<(), Error> {
        let g = build_sample()?;
        let snap = GraphSnapshot::capture(g)?;
        let dir = tempdir()?;
        let path = dir.join("snap.json");
        snap.write(&path)?;
        let back = GraphSnapshot::read(&path)?;
        if back != snap {
            return Err(Error::Format(format!(
                "round trip changed the snapshot. wrote {snap:?} read back {back:?}"
            )));
        }
        Ok(())
    }

    #[test]
    fn a_missing_version_is_refused() -> Result<(), Error> {
        let dir = tempdir()?;
        let path = dir.join("bad.json");
        let body = serde_json::json!({
            "recorded_at_unix": 1,
            "node_count": 0,
            "edge_count": 0,
            "last_round": 0,
            "graph": { "nodes": {}, "edges": [], "last_round": 0 },
        });
        write_atomic(&path, &serde_json::to_string(&body)?)?;
        match GraphSnapshot::read(&path) {
            Err(Error::Format(msg)) => {
                if !msg.contains("no `version`") {
                    return Err(Error::Format(format!(
                        "expected `no version` diagnosis, got: {msg}"
                    )));
                }
                Ok(())
            }
            other => Err(Error::Format(format!(
                "expected Err(Format) for missing version, got {other:?}"
            ))),
        }
    }

    #[test]
    fn a_wrong_version_is_refused() -> Result<(), Error> {
        let dir = tempdir()?;
        let path = dir.join("v42.json");
        let body = serde_json::json!({
            "version": 42,
            "recorded_at_unix": 1,
            "node_count": 0,
            "edge_count": 0,
            "last_round": 0,
            "graph": { "nodes": {}, "edges": [], "last_round": 0 },
        });
        write_atomic(&path, &serde_json::to_string(&body)?)?;
        match GraphSnapshot::read(&path) {
            Err(Error::Format(msg)) => {
                if !msg.contains("version 42") || !msg.contains("supports version 1") {
                    return Err(Error::Format(format!(
                        "expected a version-mismatch diagnosis naming both numbers, got: {msg}"
                    )));
                }
                Ok(())
            }
            other => Err(Error::Format(format!(
                "expected Err(Format) for v42, got {other:?}"
            ))),
        }
    }

    #[test]
    fn a_tampered_count_is_refused() -> Result<(), Error> {
        // A hand-crafted snapshot whose `node_count` disagrees with the
        // embedded graph. A truncation that silently dropped a node from
        // the `nodes` map would look exactly like this on disk.
        let dir = tempdir()?;
        let path = dir.join("tampered.json");
        let body = serde_json::json!({
            "version": 1,
            "recorded_at_unix": 1,
            "node_count": 99,
            "edge_count": 0,
            "last_round": 0,
            "graph": { "nodes": {}, "edges": [], "last_round": 0 },
        });
        write_atomic(&path, &serde_json::to_string(&body)?)?;
        match GraphSnapshot::read(&path) {
            Err(Error::Format(msg)) => {
                if !msg.contains("declares node_count=99") {
                    return Err(Error::Format(format!(
                        "expected a mirror-mismatch diagnosis, got: {msg}"
                    )));
                }
                Ok(())
            }
            other => Err(Error::Format(format!(
                "expected Err(Format) for mirror mismatch, got {other:?}"
            ))),
        }
    }

    #[test]
    fn an_empty_graph_round_trips() -> Result<(), Error> {
        // The empty-graph case is a control: if the mirror checks were
        // only ever exercised against non-empty graphs, a bug where they
        // short-circuited on zero would stay silent.
        let snap = GraphSnapshot::capture(Graph::new())?;
        let dir = tempdir()?;
        let path = dir.join("empty.json");
        snap.write(&path)?;
        let back = GraphSnapshot::read(&path)?;
        if back.node_count != 0 || back.edge_count != 0 {
            return Err(Error::Format(format!(
                "empty graph did not round trip as empty: {back:?}"
            )));
        }
        Ok(())
    }

    // ---- tempdir helper, mirroring the one in omni-loop::record ---------

    struct TempDir {
        path: std::path::PathBuf,
    }
    impl TempDir {
        fn join(&self, name: &str) -> std::path::PathBuf {
            self.path.join(name)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            if let Err(e) = std::fs::remove_dir_all(&self.path) {
                omni_core::note_advisory(
                    &format!("removing temp dir {}", self.path.display()),
                    &Error::from(e),
                );
            }
        }
    }
    fn tempdir() -> Result<TempDir, Error> {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("omni-graph-snapshot-{pid}-{seq}"));
        std::fs::create_dir_all(&path)?;
        Ok(TempDir { path })
    }
}
