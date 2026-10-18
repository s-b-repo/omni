//! Tools: the schema the model sees, the thing that runs, and the registry that
//! joins them through a [`Gate`].

use std::collections::BTreeMap;

use serde_json::Value;

use crate::{Context, Error};
use crate::gate::{Decision, Gate, GateChain, GateKind};
use crate::message::ToolCall;

/// What a tool advertises to the model.
///
/// This is the contract *offered*. Whether it is honoured is a separate question
/// — the [`Gate`] answers that, and `omni-eval`'s cheat scan answers whether
/// the tool did what it said.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for the arguments object.
    pub input_schema: Value,
}

impl ToolSpec {
    /// Build a spec.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when the name is empty or the schema is not an
    /// object. A tool whose schema is not an object cannot be advertised, and
    /// advertising it anyway produces a model-side validation error at a point
    /// where the cause is three layers away.
    pub fn new(
        name: &str,
        description: &str,
        input_schema: Value,
    ) -> Result<Self, Error> {
        if name.trim().is_empty() {
            return Err(Error::Format("a tool needs a non-empty name".into()));
        }
        if !input_schema.is_object() {
            return Err(Error::Format(format!(
                "tool {name:?} declares a {} schema; JSON Schema must be an object",
                if input_schema.is_array() { "array" } else { "non-object" }
            )));
        }
        Ok(Self {
            name: name.to_string(),
            description: description.to_string(),
            input_schema,
        })
    }

    /// The OpenAI wire form: `{"type":"function","function":{...}}`.
    pub fn to_wire(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            }
        })
    }
}

/// What running a tool produced.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutput {
    /// The text handed back to the model.
    pub content: String,
    /// A structured form, kept for the audit trail and for scoring. Not sent.
    pub structured: Option<Value>,
}

impl ToolOutput {
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self { content: content.into(), structured: None }
    }

    #[must_use]
    pub fn with_structure(content: impl Into<String>, structured: Value) -> Self {
        Self { content: content.into(), structured: Some(structured) }
    }
}

/// A runnable tool.
pub trait Tool: Send {
    /// What the model is told about this tool.
    fn spec(&self) -> &ToolSpec;

    /// Run it.
    ///
    /// Implementations must be safe to call concurrently from the dispatcher
    /// and the frontier alike, and must assume their arguments have already
    /// been through the [`Gate`] — but must not rely on it, because a tool
    /// reachable by another path has no gate in front of it.
    fn call(&mut self, args: &Value) -> Result<ToolOutput, Error>;
}

/// The set of tools a session may use, and the policy in front of them.
///
/// The registry is the one place that answers "may this run?", so there is one
/// answer rather than one per call site. [`ToolRegistry::evaluate`] is the pure
/// half — it runs the gate and nothing else, which is what lets the graph
/// frontier ask about an action that does not exist yet.
#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Box<dyn Tool>>,
    gate: GateChain,
}

impl ToolRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self { tools: BTreeMap::new(), gate: GateChain::new() }
    }

    /// Register a tool.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when the name is already taken. A duplicate name is
    /// not a merge: two tools answering to one name means the model cannot
    /// address either of them, and the disagreement is invisible until it
    /// matters.
    pub fn register(&mut self, tool: Box<dyn Tool>) -> Result<(), Error> {
        let name = tool.spec().name.clone();
        if self.tools.contains_key(&name) {
            return Err(Error::Format(format!("tool {name:?} is already registered")));
        }
        self.tools.insert(name, tool);
        Ok(())
    }

    /// Install a gate. Appends; existing gates keep their position, so the
    /// reason a call was refused does not change when a gate is added.
    pub fn set_gate(&mut self, gate: Box<dyn Gate>) -> &mut Self {
        self.gate.push(gate);
        self
    }

    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(String::as_str).collect()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// The schemas to advertise, sorted by name.
    ///
    /// Sorted because the advertised list is part of the prompt prefix: a
    /// registry iterated in insertion order produces a different prefix when
    /// two tools are registered in a different order, which invalidates every
    /// cached prefix for no reason.
    #[must_use]
    pub fn schemas(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|t| t.spec().clone()).collect()
    }

    /// The pure question: *may this call proceed?*
    ///
    /// No dispatch, no mutation, no I/O. This is the function the frontier runs
    /// over speculative actions, and it must return the same answer
    /// [`ToolRegistry::call`] will.
    #[must_use]
    pub fn evaluate(&self, call: &ToolCall) -> Decision {
        self.evaluate_parts(&call.name, &call.arguments)
    }

    /// [`ToolRegistry::evaluate`] from raw parts, for callers that have not
    /// built a [`ToolCall`] yet.
    #[must_use]
    pub fn evaluate_parts(&self, tool: &str, args: &Value) -> Decision {
        // Registration is checked before policy, so "you do not have a tool
        // called X" is never reported as "the ROE refused X" — those call for
        // different fixes.
        if !self.tools.contains_key(tool) {
            return Decision::block(GateKind::Unknown, format!("no tool named {tool:?} is registered"));
        }
        self.gate.evaluate(tool, args)
    }

    /// Run a tool, through the gate.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when the gate refuses, with the gate's own reason
    /// preserved — the caller must be able to show the user why, not merely
    /// that something stopped.
    pub fn call(&mut self, call: &ToolCall) -> Result<ToolOutput, Error> {
        let decision = self.evaluate(call);
        if !decision.allowed {
            return Err(Error::Format(format!(
                "tool {} blocked by {}: {}",
                call.name,
                decision.kind.as_str(),
                decision.reason
            )));
        }
        let tool = self
            .tools
            .get_mut(&call.name)
            .ok_or_else(|| Error::Format(format!("no tool named {:?} is registered", call.name)))?;
        tool.call(&call.arguments).ctx(|| format!("running tool {}", call.name))
    }

    /// A stable description of the installed policy order, for the audit trail.
    #[must_use]
    pub fn policy_description(&self) -> String {
        if self.gate.is_empty() {
            "no gate installed — every registered tool is callable".to_string()
        } else {
            self.gate.describe()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A tool that owns its spec, because `Tool::spec` returns a reference and
    /// a leaked `Box` in a test is a leak that outlives the test.
    struct Echo {
        spec: ToolSpec,
    }

    impl Echo {
        fn new() -> Result<Self, Error> {
            Ok(Self { spec: ToolSpec::new("echo", "echo back", json!({"type":"object"}))? })
        }
    }

    impl Tool for Echo {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }
        fn call(&mut self, args: &Value) -> Result<ToolOutput, Error> {
            let text = args.get("text").and_then(Value::as_str).unwrap_or("");
            Ok(ToolOutput::text(text))
        }
    }

    struct Reverse {
        spec: ToolSpec,
    }

    impl Reverse {
        fn new() -> Result<Self, Error> {
            Ok(Self {
                spec: ToolSpec::new("aardvark", "sorts first alphabetically", json!({"type":"object"}))?,
            })
        }
    }

    impl Tool for Reverse {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }
        fn call(&mut self, _args: &Value) -> Result<ToolOutput, Error> {
            Ok(ToolOutput::text(""))
        }
    }

    struct BlockWrites;

    impl Gate for BlockWrites {
        fn evaluate(&self, tool: &str, _args: &Value) -> Decision {
            if tool == "echo" {
                Decision::block(GateKind::Roe, "the ROE window is closed")
            } else {
                Decision::allow()
            }
        }
        fn name(&self) -> &str {
            "block-writes"
        }
    }

    #[test]
    fn a_spec_needs_a_name_and_an_object_schema() {
        assert!(ToolSpec::new("", "d", json!({})).is_err());
        assert!(ToolSpec::new("x", "d", json!([])).is_err());
        assert!(ToolSpec::new("x", "d", json!({"type":"object"})).is_ok());
    }

    #[test]
    fn a_duplicate_registration_is_refused() -> Result<(), Error> {
        let mut r = ToolRegistry::new();
        r.register(Box::new(Echo::new()?))?;
        let Err(err) = r.register(Box::new(Echo::new()?)) else {
            return Err(Error::Format("duplicate tool names must be refused".into()));
        };
        assert!(err.to_string().contains("already registered"));
        Ok(())
    }

    #[test]
    fn an_unregistered_tool_is_reported_as_unknown_not_as_policy() -> Result<(), Error> {
        let mut r = ToolRegistry::new();
        r.register(Box::new(Echo::new()?))?;
        r.set_gate(Box::new(BlockWrites));
        let call = ToolCall::new("c", "no_such_tool", json!({}))?;
        // These need different fixes — "install the tool" vs "change the ROE" —
        // so they must not be reported as the same thing.
        assert_eq!(r.evaluate(&call).kind, GateKind::Unknown);
        Ok(())
    }

    #[test]
    fn the_pure_answer_matches_what_call_would_do() -> Result<(), Error> {
        let mut r = ToolRegistry::new();
        r.register(Box::new(Echo::new()?))?;
        r.set_gate(Box::new(BlockWrites));
        let call = ToolCall::new("c", "echo", json!({"text":"hi"}))?;
        let dry = r.evaluate(&call);
        assert!(!dry.is_allowed());
        let Err(err) = r.call(&call) else {
            return Err(Error::Format("a blocked tool must not run".into()));
        };
        // The frontier's prediction has to be the dispatcher's outcome, or a
        // dry-run is not a prediction of anything.
        assert!(
            err.to_string().contains(dry.reason.as_str()),
            "the gate's reason must survive to the caller"
        );
        Ok(())
    }

    #[test]
    fn an_allowed_tool_runs_and_returns_its_output() -> Result<(), Error> {
        let mut r = ToolRegistry::new();
        r.register(Box::new(Echo::new()?))?;
        let call = ToolCall::new("c", "echo", json!({"text":"hi"}))?;
        assert_eq!(r.call(&call)?.content, "hi");
        assert_eq!(r.len(), 1);
        assert_eq!(r.names(), vec!["echo"]);
        Ok(())
    }

    #[test]
    fn an_empty_registry_says_it_has_no_policy() {
        let r = ToolRegistry::new();
        assert!(r.policy_description().contains("no gate installed"));
        assert!(r.is_empty());
    }

    #[test]
    fn schemas_are_sorted_so_the_prompt_prefix_is_stable() -> Result<(), Error> {
        // Insertion order must not leak into the advertised list, or registering
        // two tools in a different order invalidates every cached prefix.
        let mut r = ToolRegistry::new();
        r.register(Box::new(Echo::new()?))?;
        r.register(Box::new(Reverse::new()?))?;
        let names: Vec<String> = r.schemas().into_iter().map(|s| s.name).collect();
        assert_eq!(names, vec!["aardvark", "echo"]);
        Ok(())
    }

    #[test]
    fn the_wire_form_is_the_openai_function_envelope() -> Result<(), Error> {
        let spec = ToolSpec::new("echo", "echo back", json!({"type":"object"}))?;
        let wire = spec.to_wire();
        assert_eq!(wire["type"], "function");
        assert_eq!(wire["function"]["name"], "echo");
        assert_eq!(wire["function"]["parameters"]["type"], "object");
        Ok(())
    }
}
