// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Dynamic, name-and-arguments dispatch to commands.
//!
//! This indirection is what lets a key press, a menu item, and a Python call all reach the
//! same operation. The argument types are deliberately a small, dynamically-typed set:
//! they have to survive the trip through a config file or a Python dict, so they can't be
//! arbitrary Rust types.

use crate::{Command, Document, OpError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A dynamically-typed operation argument. Kept small on purpose — every variant needs an
/// obvious representation in a config file and in Python.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Vec3([f64; 3]),
    IntList(Vec<i64>),
}

/// `x,y,z` as three numbers, or `None` for anything else.
pub fn parse_triple(s: &str) -> Option<[f64; 3]> {
    let parts: Vec<f64> = s.split(',').map(|p| p.trim().parse::<f64>().ok()).collect::<Option<_>>()?;
    match parts[..] {
        [x, y, z] => Some([x, y, z]),
        _ => None,
    }
}

impl Value {
    /// The text of a string argument, for an operation whose choice is a word rather than a
    /// number — a lattice cell, say. Only `Str`: a number that happens to parse as a name is
    /// far more likely to be a mistake than an intention.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The text of a value, whichever type it was read as.
    ///
    /// The console types each word by how it parses, so `tol=2` arrives as an `Int` and `tol=2%` a
    /// `Str`. A reader with a grammar of its own — a block's name, a comma list, a percentage —
    /// wants the words back whichever it was, which `as_str` deliberately does not give.
    pub fn text(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Vec3([x, y, z]) => format!("{x},{y},{z}"),
            Value::IntList(l) => l.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","),
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub(crate) fn as_vec3(&self) -> Option<[f64; 3]> {
        match self {
            Value::Vec3(v) => Some(*v),
            // The console types a word by how it parses, and `1,0,0` parses as nothing but a
            // string -- so a triple typed at the console arrived here as `Str` and was refused,
            // which is what made `cage.translate_vertices` unusable from a typed line.
            Value::Str(s) => parse_triple(s),
            _ => None,
        }
    }

    pub(crate) fn as_int_list(&self) -> Option<Vec<i64>> {
        match self {
            Value::IntList(v) => Some(v.clone()),
            Value::Int(i) => Some(vec![*i]),
            // `faces=3,7,12` from the console arrives as text, as a triple does.
            Value::Str(s) => s.split(',').map(|p| p.trim().parse::<i64>().ok()).collect(),
            _ => None,
        }
    }
}

/// Named arguments for an operation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Args(HashMap<String, Value>);

impl Args {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, key: &str, value: Value) -> Self {
        self.0.insert(key.to_string(), value);
        self
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key)
    }

    /// The same arguments without `key`.
    ///
    /// For a caller that reads one key itself and hands the rest on: the console reads `name=`
    /// as a name for what a command made, and the command must not then meet a key it does not
    /// declare and refuse the whole line.
    pub fn without(mut self, key: &str) -> Self {
        self.0.remove(key);
        self
    }

    /// Every argument, for an operation whose keys are data rather than a fixed set: `set`,
    /// whose keys name the control points it moves (`Line.p1=10,20,0`).
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Fetches a required argument, producing a precise error naming both the operation and
    /// the argument if it's missing or the wrong type.
    pub(crate) fn require<T>(
        &self,
        op: &str,
        key: &str,
        extract: impl Fn(&Value) -> Option<T>,
        expected: &str,
    ) -> Result<T, OpError> {
        let value = self.0.get(key).ok_or_else(|| OpError::MissingArg {
            op: op.to_string(),
            arg: key.to_string(),
        })?;
        extract(value).ok_or_else(|| OpError::BadArg {
            op: op.to_string(),
            arg: key.to_string(),
            detail: format!("expected {expected}, got {value:?}"),
        })
    }
}

/// Builds a command from arguments. Takes the document read-only so a command can capture
/// the "before" state it needs to be reversible.
///
/// A boxed closure rather than a bare `fn` pointer, and the difference is the whole extension
/// story. A `fn` pointer captures nothing, so an operation can only ever be a free function
/// with no context — which means a plugin cannot register an operation bound to its own
/// instance, its own configuration, or a handle to whatever it loaded at startup. That rules
/// out any non-trivial extension, and it is D5's cheapest item to fix now and an awkward one
/// to fix once operations exist in the wild.
///
/// `Send + Sync` is required up front deliberately. Adding a bound later breaks every
/// implementor; removing one breaks nobody, so the restrictive direction is the safe one to
/// guess in. A registry shared across threads is the expected shape for a host.
type Builder = Box<dyn Fn(&Args, &Document) -> Result<Box<dyn Command>, OpError> + Send + Sync>;

/// Maps operation names to their builders.
pub struct OpRegistry {
    ops: HashMap<String, Builder>,
}

impl Default for OpRegistry {
    /// A registry with all built-in operations registered.
    fn default() -> Self {
        let mut registry = Self {
            ops: HashMap::new(),
        };
        crate::ops::register_builtins(&mut registry);
        registry
    }
}

impl OpRegistry {
    /// Registers an operation under `name`, replacing any previous one.
    ///
    /// Generic over the closure so callers pass one directly and never name [`Builder`].
    pub(crate) fn register<F>(&mut self, name: &str, builder: F)
    where
        F: Fn(&Args, &Document) -> Result<Box<dyn Command>, OpError> + Send + Sync + 'static,
    {
        self.ops.insert(name.to_string(), Box::new(builder));
    }

    pub fn build(
        &self,
        name: &str,
        args: &Args,
        doc: &Document,
    ) -> Result<Box<dyn Command>, OpError> {
        let builder = self
            .ops
            .get(name)
            .ok_or_else(|| OpError::UnknownOp(name.to_string()))?;
        builder(args, doc)
    }

    /// Every registered operation name — for discoverability from a Python `dir()`-style
    /// listing, a command palette, or keymap-config validation.
    ///
    /// None of those three exists yet, so only the tests read it, and only the tests compile it
    /// (decision 1.4). `pub(crate)` rather than private because `ops`'s tests read it as well as
    /// this module's own.
    #[cfg(test)]
    pub(crate) fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.ops.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A command that reports whatever name it was built with, so a test can see which
    /// registered closure produced it.
    struct Named(String);

    impl Command for Named {
        fn name(&self) -> &str {
            &self.0
        }
        fn execute(&mut self, _doc: &mut Document) -> Result<(), OpError> {
            Ok(())
        }
        fn undo(&mut self, _doc: &mut Document) -> Result<(), OpError> {
            Ok(())
        }
    }

    /// The reason `Builder` is a boxed closure rather than a `fn` pointer.
    ///
    /// A plugin registering an operation needs it bound to something the plugin owns — its
    /// configuration, its instance, a handle it opened at startup. A `fn` pointer captures
    /// nothing, so none of that is expressible and every non-trivial extension is blocked.
    /// This will not compile against a `fn` pointer, which is the whole assertion.
    #[test]
    fn an_operation_can_capture_state_from_whatever_registered_it() {
        let plugin_context = Arc::new("acme-plugin-v3".to_string());

        let mut registry = OpRegistry::default();
        let captured = Arc::clone(&plugin_context);
        registry.register("acme.do_thing", move |_args, _doc| {
            Ok(Box::new(Named(captured.to_string())) as Box<dyn Command>)
        });

        let doc = Document::new();
        let built = registry
            .build("acme.do_thing", &Args::default(), &doc)
            .expect("the registered operation builds");
        assert_eq!(built.name(), "acme-plugin-v3");
    }

    /// Built-ins still register, so widening the type did not change how they are declared.
    #[test]
    fn the_built_in_operations_are_still_registered() {
        let registry = OpRegistry::default();
        assert!(registry.names().contains(&"cage.translate_vertices"));
    }
}
