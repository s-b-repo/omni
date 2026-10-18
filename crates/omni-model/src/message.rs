//! Messages, and the tool calls that hang off them.
//!
//! The shapes here are the ones the OpenAI dialect carries, because that is the
//! dialect peregrine speaks and the one every client of it already knows. The
//! one place omni departs from a naive reading of the spec is recorded below,
//! because getting it wrong silently breaks multi-tool turns.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Context, Error};

/// One turn in a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    /// Prepended once, before anything else. peregrine's tool preamble lands
    /// here too, so a system turn is not necessarily the first thing the model
    /// sees if a backend renders its own preamble.
    System { content: String },

    User { content: String },

    /// An assistant turn. `content` is `None` when the turn is tool calls and
    /// nothing else — which is the normal shape of a tool-calling turn and is
    /// not an error, so it is modelled rather than stringified into `""`.
    Assistant {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },

    /// The result of running one tool, fed back to the model.
    ///
    /// **`tool_call_id` is carried but not trusted for correlation.** peregrine
    /// replays `role: "tool"` turns as observations *in arrival order* rather
    /// than matched to the call that produced them. omni mirrors that,
    /// deliberately: if omni's client starts matching on id and the server
    /// ignores id, two results get swapped and the model reasons from evidence
    /// that never happened. Ordering is the contract; the id is decoration.
    Tool {
        #[serde(default)]
        tool_call_id: String,
        name: String,
        content: String,
    },
}

impl Message {
    /// A system turn.
    pub fn system(content: impl Into<String>) -> Self {
        Message::System { content: content.into() }
    }

    /// A user turn.
    pub fn user(content: impl Into<String>) -> Self {
        Message::User { content: content.into() }
    }

    /// An assistant turn carrying text only.
    pub fn assistant(content: impl Into<String>) -> Self {
        Message::Assistant {
            content: Some(content.into()),
            tool_calls: Vec::new(),
        }
    }

    /// An assistant turn carrying tool calls only.
    pub fn assistant_tool_calls(calls: Vec<ToolCall>) -> Self {
        Message::Assistant { content: None, tool_calls: calls }
    }

    /// A tool result turn.
    pub fn tool(tool_call_id: impl Into<String>, name: impl Into<String>, content: impl Into<String>) -> Self {
        Message::Tool {
            tool_call_id: tool_call_id.into(),
            name: name.into(),
            content: content.into(),
        }
    }

    /// The role name, as it appears on the wire.
    pub fn role(&self) -> &'static str {
        match self {
            Message::System { .. } => "system",
            Message::User { .. } => "user",
            Message::Assistant { .. } => "assistant",
            Message::Tool { .. } => "tool",
        }
    }

    /// The text of this turn, if it has any.
    ///
    /// A tool-result turn and a text assistant turn both have some; a
    /// tool-calling assistant turn does not, and returning `""` for it would
    /// make "said nothing" indistinguishable from "said the empty string".
    pub fn text(&self) -> Option<&str> {
        match self {
            Message::System { content } | Message::User { content } | Message::Tool { content, .. } => {
                Some(content.as_str())
            }
            Message::Assistant { content, .. } => content.as_deref(),
        }
    }

    /// The tool calls this turn requests.
    pub fn tool_calls(&self) -> &[ToolCall] {
        match self {
            Message::Assistant { tool_calls, .. } => tool_calls,
            _ => &[],
        }
    }
}

/// A request from the model to run one tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Parsed arguments, which must be a JSON **object**.
    ///
    /// The OpenAI wire form carries arguments as a *string* containing JSON.
    /// Parsing at this boundary — once, here — is what lets every downstream
    /// consumer treat them as structured data instead of re-parsing and
    /// handling failure separately in five places.
    #[serde(default)]
    pub arguments: Value,
}

impl ToolCall {
    /// Build a call, parsing the wire-form argument string.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] when the argument string is not valid JSON, or parses
    /// to something other than an object. A non-object is rejected rather than
    /// coerced: `arguments: "[1,2]"` reaching a tool that expects fields is a
    /// crash at best and a wrong answer at worst.
    pub fn parse(id: &str, name: &str, arguments: &str) -> Result<Self, Error> {
        let arguments: Value = serde_json::from_str(arguments).ctx(|| {
            format!("parsing arguments for tool {name:?} (call {id}): {arguments}")
        })?;
        if !arguments.is_object() {
            return Err(Error::Format(format!(
                "tool {name:?} (call {id}) returned arguments that are {}, \
                 not an object; the OpenAI schema says they must be one",
                kind_of(&arguments)
            )));
        }
        Ok(Self { id: id.to_string(), name: name.to_string(), arguments })
    }

    /// Build a call from already-structured arguments.
    pub fn new(id: &str, name: &str, arguments: Value) -> Result<Self, Error> {
        if !arguments.is_object() {
            return Err(Error::Format(format!(
                "tool {name:?} (call {id}) was given {} arguments, not an object",
                kind_of(&arguments)
            )));
        }
        Ok(Self { id: id.to_string(), name: name.to_string(), arguments })
    }

    /// Read one argument.
    pub fn arg<'a>(&'a self, key: &str) -> Option<&'a Value> {
        self.arguments.get(key)
    }

    /// Read one argument as a string.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] naming the key and the type actually seen, because a
    /// tool receiving `"count": 5` where it wanted `"count": "5"` needs to be
    /// told the difference.
    pub fn arg_str(&self, key: &str) -> Result<&str, Error> {
        match self.arguments.get(key) {
            Some(Value::String(s)) => Ok(s.as_str()),
            Some(other) => Err(Error::Format(format!(
                "tool {}: argument {key:?} is {}, expected a string",
                self.name,
                kind_of(other)
            ))),
            None => Err(Error::Format(format!(
                "tool {}: argument {key:?} is missing",
                self.name
            ))),
        }
    }
}

fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn an_argument_string_parses_into_an_object() -> Result<(), Error> {
        let call = ToolCall::parse("call_1", "shell", r#"{"command":"ls"}"#)?;
        assert_eq!(call.arg_str("command")?, "ls");
        assert!(call.arg_str("nope").is_err(), "a missing key is an error, not an empty string");
        Ok(())
    }

    #[test]
    fn non_object_arguments_are_refused_not_coerced() {
        // The failure this prevents: a tool receiving `[1,2]` where it expected
        // fields, at best a panic and at worst a wrong answer.
        assert!(ToolCall::parse("c", "t", "[1,2]").is_err());
        assert!(ToolCall::parse("c", "t", "\"str\"").is_err());
        assert!(ToolCall::new("c", "t", Value::Array(vec![])).is_err());
    }

    #[test]
    fn a_wrong_argument_type_names_both_the_key_and_what_it_found() -> Result<(), Error> {
        let call = ToolCall::parse("c", "shell", r#"{"timeout":5}"#)?;
        let Err(err) = call.arg_str("timeout") else {
            return Err(Error::Format("a number must not pass as a string".into()));
        };
        let msg = err.to_string();
        assert!(msg.contains("timeout"), "must name the key: {msg}");
        assert!(msg.contains("a number"), "must name what it found: {msg}");
        Ok(())
    }

    #[test]
    fn malformed_argument_json_names_the_tool_and_the_call() -> Result<(), Error> {
        let Err(err) = ToolCall::parse("call_9", "shell", "{not json") else {
            return Err(Error::Format("malformed JSON must be refused".into()));
        };
        let msg = err.to_string();
        assert!(msg.contains("shell"), "must name the tool: {msg}");
        assert!(msg.contains("call_9"), "must name the call: {msg}");
        Ok(())
    }

    #[test]
    fn a_tool_calling_assistant_turn_has_no_text_and_says_so() -> Result<(), Error> {
        let m = Message::assistant_tool_calls(vec![ToolCall::new("c", "t", json!({}))?]);
        assert_eq!(m.text(), None, "said-nothing must differ from said-empty");
        assert_eq!(m.tool_calls().len(), 1);
        Ok(())
    }

    #[test]
    fn roles_are_the_wire_names() {
        assert_eq!(Message::system("s").role(), "system");
        assert_eq!(Message::user("u").role(), "user");
        assert_eq!(Message::assistant("a").role(), "assistant");
        assert_eq!(Message::tool("id", "n", "c").role(), "tool");
    }

    #[test]
    fn a_message_round_trips_through_the_wire_form() -> Result<(), Error> {
        let msgs = vec![
            Message::system("you are a harness"),
            Message::user("go"),
            Message::assistant_tool_calls(vec![ToolCall::new("c1", "shell", serde_json::json!({"command":"ls"}))?]),
            Message::tool("c1", "shell", "a.txt"),
        ];
        let text = serde_json::to_string(&msgs)?;
        let back: Vec<Message> = serde_json::from_str(&text)?;
        assert_eq!(msgs, back);
        Ok(())
    }
}