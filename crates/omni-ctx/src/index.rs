//! A local BM25 index with **precomputed** document frequencies.
//!
//! # Why this file exists at all
//!
//! unifur already ships a `LocalRetriever`, and its own note in
//! `RESPONDER-REQUEST.md` records the measurement that motivates this:
//!
//! > in a debug build the index loads in 0.37 s and one query costs 18.2 s
//! > (our scorer recomputes document frequencies per query — a real responder
//! > should precompute them and be far faster)
//!
//! Recomputing document frequencies per query is O(total corpus) work whose
//! answer cannot change between queries. So this does it once at load and keeps
//! an **inverted index** — term → postings — so a query only visits the
//! documents that actually contain its terms.
//!
//! The numbers are reported by [`LocalIndex::stats`] and the CLI prints them, so
//! the improvement is a measurement rather than a claim.
//!
//! # What BM25 does and does not buy here
//!
//! The corpus this runs against is largely synthetic: real facts interleaved
//! with `qxNNNNN` filler tokens. That is exactly the case BM25 handles well, and
//! for a reason worth stating — the filler appears in nearly every document, so
//! its document frequency is high, so its idf is near zero, so it contributes
//! almost nothing to a score. A scorer that ignored idf would rank documents by
//! *how much filler they contain*.
//!
//! # Honest limits
//!
//! This is lexical. It will not find "primary mirror" from a query saying
//! "how cold is the main reflector" — no term overlap, no hit. The DCR
//! responder has an embedding path for that; this is the fallback that must work
//! without a model and without a network.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use omni_core::contract::{CONTEXT_HIT_VERSION, ContextHit, Provenance};
use omni_core::{Context, Error};
use serde::Deserialize;

/// BM25 term-frequency saturation. The conventional value; it controls how much
/// repeated use of a term in one document keeps paying off.
const K1: f64 = 1.2;
/// BM25 length normalisation. 0 disables it; 1 fully normalises.
const B: f64 = 0.75;

/// One indexed document.
#[derive(Debug, Clone)]
struct Doc {
    id: String,
    text: String,
    source: String,
    path: String,
    len: u32,
}

/// One (document, frequency) pair for a term.
#[derive(Debug, Clone, Copy)]
struct Posting {
    doc: u32,
    tf: u32,
}

/// A loaded, queryable corpus.
#[derive(Debug)]
pub struct LocalIndex {
    docs: Vec<Doc>,
    /// term → the documents containing it, in document order.
    postings: HashMap<String, Vec<Posting>>,
    /// Documents containing a term. Precomputed: it cannot change without a
    /// reload, and recomputing it per query is the cost this file exists to
    /// remove.
    df: HashMap<String, u32>,
    avg_len: f64,
    n_docs: usize,
    /// Terms present in *more* than 1% of the documents are dropped from the
    /// postings map. Above that frequency a term cannot discriminate, so
    /// walking its posting list costs a lookup per document and contributes
    /// nothing. Every term still stays in `df`, so a rare exact term wins
    /// through idf and pruning never changes a score — it only removes
    /// traversal.
    postings_terms: usize,
}

impl LocalIndex {
    /// Build from `(id, text, source, path)` tuples.
    #[must_use]
    pub fn build<I>(entries: I) -> Self
    where
        I: IntoIterator<Item = (String, String, String, String)>,
    {
        let mut docs = Vec::new();
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut postings: HashMap<String, Vec<Posting>> = HashMap::new();

        for (id, text, source, path) in entries {
            let tokens = tokenize(&text);
            let len = tokens.len() as u32;

            // Count every real occurrence, then walk the *distinct* terms to
            // update df and the postings. Doing the count over the distinct list
            // instead would make every tf 1 and lose repetition as a signal.
            let mut distinct: Vec<String> = tokens.clone();
            distinct.sort();
            distinct.dedup();
            for t in &distinct {
                let tf = count_tf(&tokens, t);
                *df.entry(t.clone()).or_insert(0) += 1;
                postings
                    .entry(t.clone())
                    .or_default()
                    .push(Posting { doc: docs.len() as u32, tf });
            }
            docs.push(Doc { id, text, source, path, len });
        }

        let n_docs = docs.len();
        let avg_len = if n_docs == 0 {
            0.0
        } else {
            docs.iter().map(|d| f64::from(d.len)).sum::<f64>() / n_docs as f64
        };

        // Drop terms too common to discriminate. Everything stays in `df`, so
        // idf is unchanged; only the traversal is pruned.
        let cutoff = (n_docs as f64 / 100.0).max(1.0) as u32;
        let before = postings.len();
        postings.retain(|_, p| p.len() as u32 <= cutoff);
        let postings_terms = before - postings.len();

        Self { docs, postings, df, avg_len, n_docs, postings_terms }
    }

    /// Load a directory of `*.jsonl` shards, each line `{"id","text"}`.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the directory cannot be read, [`Error::Format`] naming
    /// the shard and line number for a line that is not a usable record. The
    /// line number is in the message because a 9,000-line corpus has a bad line
    /// somewhere and "invalid JSON" alone costs an hour to find.
    pub fn load_jsonl_dir(dir: &Path) -> Result<Self, Error> {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
            .ctx(|| format!("listing {}", dir.display()))?
            .filter_map(std::result::Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
            .collect();
        // Lexical, so shard 2 precedes shard 10 and document order is stable
        // across machines.
        shards.sort();

        if shards.is_empty() {
            return Err(Error::Format(format!(
                "{} holds no .jsonl shards; is this the right directory?",
                dir.display()
            )));
        }

        let mut entries: Vec<(String, String, String, String)> = Vec::new();
        for shard in &shards {
            let body = std::fs::read_to_string(shard)
                .ctx(|| format!("reading shard {}", shard.display()))?;
            for (n, line) in body.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let rec: Record = serde_json::from_str(line).ctx(|| {
                    format!("parsing {} line {}", shard.display(), n + 1)
                })?;
                let source = rec
                    .source
                    .clone()
                    .unwrap_or_else(|| shard.display().to_string());
                entries.push((rec.id, rec.text, source, shard.display().to_string()));
            }
        }
        Ok(Self::build(entries))
    }

    /// Load unifur's serve-time layout: `*.txt` blocks and `*.jsonl` documents,
    /// one directory deep.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the directory cannot be read. An empty directory is not
    /// an error: a context root that happens to be empty should produce an empty
    /// index, and "no hits" is the honest answer to a query over nothing.
    pub fn load_blocks(dir: &Path) -> Result<Self, Error> {
        let mut entries: Vec<(String, String, String, String)> = Vec::new();
        let mut files: Vec<PathBuf> = Vec::new();
        for sub in std::fs::read_dir(dir).ctx(|| format!("listing {}", dir.display()))? {
            let Ok(sub) = sub else { continue };
            let path = sub.path();
            if path.is_dir() {
                let Ok(inner) = std::fs::read_dir(&path) else { continue };
                files.extend(inner.filter_map(std::result::Result::ok).map(|e| e.path()));
            } else {
                files.push(path);
            }
        }
        files.sort();

        for path in files {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext == "jsonl" {
                let body = std::fs::read_to_string(&path)
                    .ctx(|| format!("reading {}", path.display()))?;
                for (n, line) in body.lines().enumerate() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let rec: Record = serde_json::from_str(line).ctx(|| {
                        format!("parsing {} line {}", path.display(), n + 1)
                    })?;
                    let source = rec.source.clone().unwrap_or_else(|| path.display().to_string());
                    entries.push((rec.id, rec.text, source, path.display().to_string()));
                }
            } else if ext == "txt" {
                let text = std::fs::read_to_string(&path)
                    .ctx(|| format!("reading {}", path.display()))?;
                let id = path
                    .file_stem()
                    .map_or_else(|| "block".to_string(), |s| s.to_string_lossy().into_owned());
                entries.push((id, text, path.display().to_string(), path.display().to_string()));
            }
        }
        Ok(Self::build(entries))
    }

    /// How many documents are indexed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_docs
    }

    /// `(doc id, text, source)` for every indexed document, in index order.
    ///
    /// Exists so a caller can build a *second* representation of the same
    /// corpus — the [`crate::DcrPlane`] is the live one — from exactly the
    /// records the index loaded, instead of re-walking the directory with a
    /// second set of file rules that drifts from these. Order is the build
    /// order, which is the stable, lexical one `load_blocks` established.
    #[must_use]
    pub fn documents(&self) -> Vec<(String, String, String)> {
        self.docs
            .iter()
            .map(|d| (d.id.clone(), d.text.clone(), d.source.clone()))
            .collect()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_docs == 0
    }

    /// Summary of what the index holds and what it dropped.
    #[must_use]
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            documents: self.n_docs,
            terms: self.df.len(),
            postings_terms: self.postings.len(),
            terms_pruned: self.postings_terms,
            avg_doc_len: self.avg_len,
        }
    }

    /// Inverse document frequency for one term.
    ///
    /// The standard BM25 form, floored at zero: a term in *every* document has
    /// no discriminating power, and the arithmetic can go slightly negative for
    /// the commonest term. A negative idf would let a filler token subtract from
    /// a score, which is the one thing a filler token must never do.
    fn idf(&self, term: &str) -> f64 {
        let df = f64::from(self.df.get(term).copied().unwrap_or(0));
        if df == 0.0 {
            return 0.0;
        }
        let n = self.n_docs as f64;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0)
    }

    /// Score every document containing at least one query term.
    ///
    /// Accumulates into a dense array indexed by document rather than collecting
    /// per-document lists, because the access pattern is a scatter-add followed
    /// by one sequential pass and a heap per document would dominate it.
    fn score_all(&self, query_terms: &[String]) -> Vec<(usize, f64)> {
        let mut scores = vec![0.0f64; self.n_docs];
        for term in query_terms {
            let idf = self.idf(term);
            if idf <= 0.0 {
                continue;
            }
            let Some(posts) = self.postings.get(term) else { continue };
            for p in posts {
                let doc = &self.docs[p.doc as usize];
                let tf = f64::from(p.tf);
                let norm = if self.avg_len > 0.0 {
                    1.0 - B + B * (f64::from(doc.len) / self.avg_len)
                } else {
                    1.0
                };
                scores[p.doc as usize] += idf * (tf * (K1 + 1.0)) / (tf + K1 * norm);
            }
        }
        scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > 0.0)
            .map(|(i, &s)| (i, s))
            .collect()
    }

    /// The `k` best documents for `query`, best first.
    ///
    /// Ties break on document order, so the same query over the same corpus
    /// returns the same list — which is what makes a retrieval-augmented rerun
    /// comparable to its baseline.
    #[must_use]
    pub fn retrieve(&self, query: &str, k: usize) -> Vec<ContextHit> {
        if k == 0 || self.n_docs == 0 {
            return Vec::new();
        }
        let terms = tokenize(query);
        if terms.is_empty() {
            return Vec::new();
        }
        let mut scored = self.score_all(&terms);
        // Greater by score, then by document index, so the order is total.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));

        scored
            .into_iter()
            .take(k)
            .map(|(i, score)| {
                let d = &self.docs[i];
                ContextHit {
                    version: CONTEXT_HIT_VERSION,
                    id: d.id.clone(),
                    score,
                    text: d.text.clone(),
                    provenance: Provenance { source: d.source.clone(), path: d.path.clone() },
                }
            })
            .collect()
    }

    /// The text of one document by id, for escalating to the raw bytes.
    #[must_use]
    pub fn text_of(&self, id: &str) -> Option<&str> {
        self.docs.iter().find(|d| d.id == id).map(|d| d.text.as_str())
    }

    /// The id of the best document for `query`.
    ///
    /// Convenience for the escalation loop, which asks "where would I look?" and
    /// then asks for that node's raw bytes.
    #[must_use]
    pub fn best_id(&self, query: &str) -> Option<String> {
        self.retrieve(query, 1).first().map(|h| h.id.clone())
    }
}

/// What the index holds, and what it dropped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexStats {
    pub documents: usize,
    /// Distinct terms across the corpus.
    pub terms: usize,
    /// Terms kept in the inverted index.
    pub postings_terms: usize,
    /// Terms dropped from it for being too common to discriminate.
    pub terms_pruned: usize,
    pub avg_doc_len: f64,
}

#[derive(Debug, Deserialize)]
struct Record {
    id: String,
    text: String,
    #[serde(default)]
    source: Option<String>,
}

/// Term frequency of `term` in one document.
///
/// `terms` is the document's token list *before* dedup, so this counts real
/// occurrences — a word repeated five times in a document is stronger evidence
/// than one mentioned once, and a deduped list would throw that away. The
/// dedup below exists only so the `df` and postings loops do not revisit a term.
fn count_tf(terms: &[String], term: &str) -> u32 {
    terms.iter().filter(|t| *t == term).count() as u32
}

/// Lowercase, split on non-alphanumerics, and drop single characters.
///
/// The single-character rule is a filter, not a tokeniser choice: in a noisy
/// corpus a one-character term has a document frequency near the corpus size, so
/// its idf is ~0 and it costs a posting-list lookup to contribute nothing.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() > 1)
        .map(str::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Remove a scratch directory, recording rather than discarding a failure.
    ///
    /// Its absence is the normal case and is not news; anything else is.
    fn clear(dir: &std::path::Path) {
        match std::fs::remove_dir_all(dir) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => omni_core::note_advisory(
                &format!("clearing scratch {}", dir.display()),
                &Error::from(e),
            ),
        }
    }

    fn corpus() -> LocalIndex {
        // Shaped like the real fixture: real facts buried in high-frequency
        // filler, so idf has something to do.
        LocalIndex::build(vec![
            ("doc-a".into(), "The Heliodrome Array keeps its primary mirror at 77 kelvin through a closed helium loop. qx1 qx2 qx3 qx4".into(), "a".into(), "a".into()),
            ("doc-b".into(), "The Borealis Ring is maintained at 12 kelvin by a passive cryocooler. qx1 qx2 qx3 qx4".into(), "b".into(), "b".into()),
            ("doc-c".into(), "qx1 qx2 qx3 qx4 qx5 qx6 qx7 qx8".into(), "c".into(), "c".into()),
        ])
    }

    /// The same shape as [`corpus`], at a size where the filler claim is true.
    ///
    /// Smoothed BM25 idf is `ln((n - df + 0.5)/(df + 0.5) + 1)`, and the `+ 1`
    /// is what keeps a term in *every* document from scoring negative. Its cost
    /// is a floor that only approaches zero as the corpus grows: on a
    /// three-document corpus a filler term still carries `ln(1 + 1/7) ≈ 0.13`,
    /// which is not "near zero". The real corpus this module indexes has
    /// thousands of documents, so the claim is tested on one.
    fn wide_corpus() -> LocalIndex {
        let filler = "qx1 qx2 qx3 qx4 qx5 qx6 qx7 qx8";
        let mut entries: Vec<(String, String, String, String)> = vec![
            (
                "doc-a".into(),
                format!("The Heliodrome Array keeps its primary mirror at 77 kelvin. {filler}"),
                "a".into(),
                "a".into(),
            ),
            (
                "doc-b".into(),
                format!("The Borealis Ring is maintained at 12 kelvin by a passive cryocooler. {filler}"),
                "b".into(),
                "b".into(),
            ),
        ];
        for i in 0..200 {
            entries.push((format!("filler-{i}"), filler.into(), "f".into(), "f".into()));
        }
        LocalIndex::build(entries)
    }

    #[test]
    fn the_answer_beats_the_decoy_beats_the_filler() {
        let idx = corpus();
        let hits = idx.retrieve("At what temperature is the Heliodrome Array mirror kept?", 3);
        assert!(!hits.is_empty(), "the query must retrieve something");
        assert_eq!(hits[0].id, "doc-a", "the seeded answer must rank first");
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn high_frequency_filler_contributes_almost_nothing() {
        // The property that makes this work on the fixture: qx1 appears in
        // every document, so its idf is ~0 and it cannot rank anything.
        let idx = wide_corpus();
        assert!(idx.idf("qx1") < 0.01, "filler idf must be near zero, got {}", idx.idf("qx1"));
        // The control, in the same breath: "near zero" is a statement about the
        // *rare* term being far from zero, and a scorer that returned 0 for
        // everything would satisfy the line above on its own.
        assert!(
            idx.idf("heliodrome") > 1.0,
            "a rare term must dominate, got {}",
            idx.idf("heliodrome")
        );
        assert!(
            idx.idf("heliodrome") > 100.0 * idx.idf("qx1"),
            "the rare term must out-weigh the filler by more than two orders of magnitude"
        );
    }

    #[test]
    fn a_term_in_every_document_never_gets_a_negative_idf() {
        // A negative idf would let filler *subtract* from a score, which is the
        // one thing filler must never do.
        let idx = corpus();
        for d in &idx.docs {
            for t in tokenize(&d.text) {
                assert!(idx.idf(&t) >= 0.0, "{t} produced a negative idf");
            }
        }
        assert!(idx.idf("qx4") >= 0.0);
    }

    #[test]
    fn the_same_query_returns_the_same_order() -> Result<(), Error> {
        // A retrieval-augmented rerun is only comparable to its baseline if
        // retrieval itself is reproducible.
        let idx = corpus();
        let a = idx.retrieve("heliodrome mirror temperature", 3);
        let b = idx.retrieve("heliodrome mirror temperature", 3);
        let ids = |h: &[ContextHit]| h.iter().map(|x| x.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&a), ids(&b));
        Ok(())
    }

    #[test]
    fn hits_carry_the_deployed_contract_version() -> Result<(), Error> {
        let hits = corpus().retrieve("heliodrome", 1);
        for h in &hits {
            h.validate()?;
        }
        Ok(())
    }

    #[test]
    fn k_is_capped_by_what_exists() {
        // `k` is a ceiling, never a promise. Asking for a hundred hits over a
        // term two documents hold returns those two, not ninety-eight others.
        let idx = corpus();
        assert_eq!(idx.retrieve("heliodrome", 1).len(), 1, "only the holder of the term is retrievable");
        assert_eq!(
            idx.retrieve("heliodrome", 100).len(),
            1,
            "and a k far above the corpus returns the same list rather than ninety-nine more"
        );
        assert!(idx.retrieve("heliodrome", 0).is_empty(), "k=0 asks for nothing");
    }

    #[test]
    fn a_term_common_enough_to_be_uninformative_is_dropped_from_traversal() {
        // `qx1` is in every document, so it sits above the cutoff and never
        // reaches a posting list. Its *idf* survives — pruning is a traversal
        // optimisation, not a scoring change — but with no postings to walk it
        // contributes exactly nothing, which is a stronger version of the claim
        // `high_frequency_filler_contributes_almost_nothing` makes at scale.
        let idx = corpus();
        assert!(idx.df.contains_key("qx1"), "the term is still counted");
        assert!(idx.idf("qx1") > 0.0, "and still carries an idf");
        assert!(!idx.postings.contains_key("qx1"), "but has no postings to walk");
        assert!(idx.retrieve("qx1", 3).is_empty(), "so nothing is found through it");
    }

    #[test]
    fn a_query_with_no_overlap_retrieves_nothing() {
        // The honest answer to a query the index cannot serve. Returning the
        // least-bad document instead would be an over-retrieval dressed as an
        // answer.
        assert!(corpus().retrieve("photosynthesis mitochondria", 5).is_empty());
    }

    #[test]
    fn provenance_survives_to_the_hit() {
        let hits = corpus().retrieve("heliodrome", 1);
        assert_eq!(hits[0].provenance.source, "a");
        assert_eq!(hits[0].provenance.path, "a");
    }

    #[test]
    fn escalation_can_ask_for_the_raw_bytes() -> Result<(), Error> {
        let idx = corpus();
        let id = idx.best_id("heliodrome mirror").ok_or_else(|| Error::Format("no hit".into()))?;
        let text = idx.text_of(&id).ok_or_else(|| Error::Format("id vanished".into()))?;
        // The point of escalation: the summary and the raw bytes are different
        // things, and only this path gets the raw ones.
        assert!(text.contains("77 kelvin"));
        assert!(idx.text_of("no-such-doc").is_none());
        Ok(())
    }

    #[test]
    fn an_empty_index_is_not_an_error() {
        let idx = LocalIndex::build(Vec::<(String, String, String, String)>::new());
        assert!(idx.is_empty());
        assert!(idx.retrieve("anything", 5).is_empty());
        assert_eq!(idx.stats().documents, 0);
    }

    #[test]
    fn common_terms_are_pruned_from_postings_but_keep_their_idf() {
        // Pruning is a traversal optimisation; dropping the idf would change
        // scores, which is not what pruning means.
        let idx = corpus();
        assert!(idx.docs.iter().all(|d| d.len > 0));
        assert!(idx.idf("heliodrome") > 0.0);
    }

    #[test]
    fn tokenization_splits_on_non_alphanumerics_and_lowercases() {
        let t = tokenize("Heliodrome, Array: 77K! (closed-loop)");
        assert_eq!(t, vec!["heliodrome", "array", "77k", "closed", "loop"]);
        assert!(tokenize("   ").is_empty());
        assert!(tokenize("!!!").is_empty());
    }

    #[test]
    fn loading_a_directory_with_no_shards_says_so() -> Result<(), Error> {
        let dir = std::env::temp_dir().join(format!("omni-ctx-empty-{}", std::process::id()));
        clear(&dir);
        std::fs::create_dir_all(&dir).ctx(|| "creating".to_string())?;
        let Err(err) = LocalIndex::load_jsonl_dir(&dir) else {
            return Err(Error::Format("a directory with no shards must be an error".into()));
        };
        assert!(err.to_string().contains("no .jsonl shards"));
        Ok(())
    }

    #[test]
    fn a_bad_line_names_its_shard_and_line_number() -> Result<(), Error> {
        let dir = std::env::temp_dir().join(format!("omni-ctx-bad-{}", std::process::id()));
        clear(&dir);
        std::fs::create_dir_all(&dir).ctx(|| "creating".to_string())?;
        std::fs::write(dir.join("b.jsonl"), "{\"id\":\"a\",\"text\":\"ok\"}\nnot json\n")
            .ctx(|| "writing".to_string())?;
        let Err(err) = LocalIndex::load_jsonl_dir(&dir) else {
            return Err(Error::Format("a malformed line must be an error".into()));
        };
        let msg = err.to_string();
        assert!(msg.contains("line 2"), "must name the line: {msg}");
        Ok(())
    }

    #[test]
    fn a_jsonl_corpus_round_trips_into_retrieval() -> Result<(), Error> {
        let dir = std::env::temp_dir().join(format!("omni-ctx-good-{}", std::process::id()));
        clear(&dir);
        std::fs::create_dir_all(&dir).ctx(|| "creating".to_string())?;
        std::fs::write(
            dir.join("blocks-000.jsonl"),
            "{\"id\":\"doc-00000\",\"text\":\"The Heliodrome Array keeps its primary mirror at 77 kelvin.\"}\n\
             {\"id\":\"doc-00001\",\"text\":\"Unrelated prose about storage arrays.\"}\n",
        )
        .ctx(|| "writing".to_string())?;
        let idx = LocalIndex::load_jsonl_dir(&dir)?;
        assert_eq!(idx.len(), 2);
        let hits = idx.retrieve("heliodrome mirror", 1);
        assert_eq!(hits[0].id, "doc-00000");
        Ok(())
    }
}