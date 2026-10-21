// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Arithmetic on named numbers — `boss_r = radius * 5 - 2`.
//!
//! One text field, not an operator per block: `(a + b) * c` is one expression, the way other
//! CAD packages' expression fields read, rather than an `Add` feeding a `Multiply`.
//!
//! # References are ids; the text is a rendering
//!
//! A parsed [`Expression`] holds [`ParamId`]s, never names. If a reference were the *name*
//! `wall`, renaming that parameter to `wall_thickness` would either break every expression that
//! used it or need a text rewrite that gets string boundaries right — and `wall` is a substring
//! of `wall_thickness`, which is exactly how such a rewrite goes wrong. With ids a rename costs
//! nothing, and the text is re-rendered from the tree ([`Expression::rerender`]).
//!
//! # Nothing spatial
//!
//! An expression is one number for the whole document. It may read literals and other
//! parameters, and nothing else — not a node, a field, a ramp or a point's position. A value
//! that varied with position would stop the node it drives from being a distance field, and a
//! ramp is the one construction whose rate of change is known in closed form (see
//! `ParamRef::Field`). Positions get warps, amounts get ramps, constants get expressions.
//!
//! # No units
//!
//! A number is an `f64` and nothing checks dimensions: `wall * wall` is read as a length by
//! whatever consumes it. A literal means exactly what the same digits typed into a plain field
//! mean — canonical millimetres — because the fields never convert either.
//!
//! # Failure is not a number
//!
//! Division by zero, the root of a negative, an overflow: any step that is not finite makes the
//! whole expression fail ([`EvalError::NotFinite`]), and a failed expression has *no value*
//! rather than a NaN one. A NaN radius makes a field NaN everywhere, and the symptom is a blank
//! viewport a long way from the cause. Failing on the step rather than the final result is
//! deliberate: `min(1 / 0, 5)` is 5 arithmetically and a division by zero the person did not
//! mean, and saying so beats quietly picking 5.
//!
//! # The grammar
//!
//! ```text
//! expr    := term (('+' | '-') term)*
//! term    := power (('*' | '/') power)*
//! power   := unary ('^' power)?          -- right associative: 2 ^ 3 ^ 2 = 512
//! unary   := '-' unary | primary         -- binds tighter than '^': -2 ^ 2 = 4
//! primary := number | name | '(' expr ')' | function '(' expr (',' expr)* ')'
//! ```
//!
//! Functions are `abs`, `sqrt`, `floor`, `ceil`, `round` of one argument and `min`, `max` of
//! two or more. Deliberately absent: trigonometry, `log`/`exp`, comparisons, conditionals and
//! constants such as `pi`. When trig lands its arguments are in **degrees**, because
//! `Node::Transform` stores rotation in degrees on the grounds that it is a number a person
//! types and reads back, and a `sin()` disagreeing with the rotation field beside it would be
//! a nasty surprise; `pi` arrives with radians or not at all.

use crate::id::ParamId;

/// An expression as typed, parsed, with the text kept beside the tree for display.
///
/// The tree is the truth and the text is derived from it after a rename; between renames the
/// text is what was typed, spacing and all, because a person expects to see their own writing.
#[derive(Clone, Debug, PartialEq)]
pub struct Expression {
    source: String,
    expr: Expr,
}

/// The parsed tree. References are ids — see the module notes.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Number(f64),
    Param(ParamId),
    Neg(Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Call(Function, Vec<Expr>),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

/// The functions v1 knows. See the module notes for what is deliberately not here.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Function {
    Abs,
    Min,
    Max,
    Sqrt,
    Floor,
    Ceil,
    Round,
}

/// Why a text is not an expression, and where it stopped being one.
///
/// `offset` counts characters from zero, which is what a caret in a text field wants; the
/// message says "column" and counts from one, which is what a person wants.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message} at column {}", .offset + 1)]
pub struct ParseError {
    pub offset: usize,
    pub message: String,
}

/// Why an expression has no value.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub(crate) enum EvalError {
    /// The step named — `5 / 0`, `sqrt(-4)`, `10 ^ 400` — did not produce a finite number.
    #[error("{0} is not a number")]
    NotFinite(String),
    /// A parameter it reads has no value of its own.
    #[error("a parameter it reads has no value")]
    Unresolved(ParamId),
}

impl Expression {
    /// Parses `text`, resolving every name through `names`.
    ///
    /// A name `names` does not know is a parse error at that name, not a reference left to
    /// fail later: an expression that is stored is one whose every reference exists.
    pub fn parse(
        text: &str,
        names: &dyn Fn(&str) -> Option<ParamId>,
    ) -> Result<Self, ParseError> {
        let tokens = lex(text)?;
        let mut parser = Parser { tokens, at: 0, names };
        let expr = parser.expr()?;
        if !matches!(parser.peek(), Token::End) {
            return Err(ParseError {
                offset: parser.offset(),
                message: format!("unexpected {}", parser.peek().describe()),
            });
        }
        Ok(Self { source: text.trim().to_string(), expr })
    }

    /// The text, as typed or as last re-rendered.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The parsed tree. Evaluation and the reference walk read it from inside this module, so
    /// only the tests here look at its shape, and only the tests compile this (decision 1.4).
    #[cfg(test)]
    fn tree(&self) -> &Expr {
        &self.expr
    }

    /// Every parameter this reads, each once, in a stable order.
    pub(crate) fn references(&self) -> Vec<ParamId> {
        let mut out = Vec::new();
        self.expr.collect_references(&mut out);
        out.sort();
        out.dedup();
        out
    }

    /// The value, given a value for each parameter it reads.
    pub(crate) fn evaluate(&self, lookup: &dyn Fn(ParamId) -> Option<f64>) -> Result<f64, EvalError> {
        self.expr.evaluate(lookup)
    }

    /// The tree written out with the names `names` gives, parenthesised where the tree needs
    /// it and nowhere else. Parsing the result gives the same tree back.
    pub(crate) fn rendered(&self, names: &dyn Fn(ParamId) -> String) -> String {
        self.expr.render(names)
    }

    /// Replaces the text with a fresh rendering — what a rename does to every expression that
    /// read the renamed parameter.
    pub(crate) fn rerender(&mut self, names: &dyn Fn(ParamId) -> String) {
        self.source = self.rendered(names);
    }
}

impl Expr {
    fn collect_references(&self, out: &mut Vec<ParamId>) {
        match self {
            Expr::Number(_) => {}
            Expr::Param(id) => out.push(*id),
            Expr::Neg(x) => x.collect_references(out),
            Expr::Binary(_, l, r) => {
                l.collect_references(out);
                r.collect_references(out);
            }
            Expr::Call(_, args) => args.iter().for_each(|a| a.collect_references(out)),
        }
    }

    fn evaluate(&self, lookup: &dyn Fn(ParamId) -> Option<f64>) -> Result<f64, EvalError> {
        match self {
            Expr::Number(v) => Ok(*v),
            Expr::Param(id) => lookup(*id).ok_or(EvalError::Unresolved(*id)),
            Expr::Neg(x) => Ok(-x.evaluate(lookup)?),
            Expr::Binary(op, l, r) => {
                let a = l.evaluate(lookup)?;
                let b = r.evaluate(lookup)?;
                finite(op.apply(a, b), || format!("{} {} {}", number(a), op.symbol(), number(b)))
            }
            Expr::Call(function, args) => {
                let values = args
                    .iter()
                    .map(|a| a.evaluate(lookup))
                    .collect::<Result<Vec<f64>, EvalError>>()?;
                finite(function.apply(&values), || {
                    let shown: Vec<String> = values.iter().map(|v| number(*v)).collect();
                    format!("{}({})", function.name(), shown.join(", "))
                })
            }
        }
    }

    /// How tightly this binds, for deciding where a rendering needs parentheses. Higher binds
    /// tighter; a leaf or a call never needs them.
    fn precedence(&self) -> u8 {
        match self {
            Expr::Binary(op, ..) => op.precedence(),
            Expr::Neg(_) => 4,
            Expr::Number(_) | Expr::Param(_) | Expr::Call(..) => 5,
        }
    }

    fn render(&self, names: &dyn Fn(ParamId) -> String) -> String {
        match self {
            Expr::Number(v) => number(*v),
            Expr::Param(id) => name_in_expression(&names(*id)),
            Expr::Neg(x) => {
                let inner = x.render(names);
                // `-(a + b)` and `-(-a)`: anything that is not a leaf goes in parentheses, so a
                // minus reads as applying to the whole of it.
                if x.precedence() < 5 { format!("-({inner})") } else { format!("-{inner}") }
            }
            Expr::Binary(op, l, r) => {
                let p = op.precedence();
                // A looser child always needs them. The power is right-associative, so a power on
                // its *left* needs them — `(2 ^ 3) ^ 2` — and a negation there gets them for
                // legibility even though the grammar would read `-2 ^ 2` the same way. Every
                // other operator associates left, so an equal child on the *right* needs them:
                // `2 - (3 - 4)`, `2 / (3 / 4)`.
                let left_needs = l.precedence() < p
                    || (*op == BinOp::Pow && (l.precedence() == p || matches!(**l, Expr::Neg(_))));
                let right_needs = r.precedence() < p || (r.precedence() == p && *op != BinOp::Pow);
                let left = wrap(l.render(names), left_needs);
                let right = wrap(r.render(names), right_needs);
                format!("{left} {} {right}", op.symbol())
            }
            Expr::Call(function, args) => {
                let inner: Vec<String> = args.iter().map(|a| a.render(names)).collect();
                format!("{}({})", function.name(), inner.join(", "))
            }
        }
    }
}

fn wrap(text: String, needs: bool) -> String {
    if needs { format!("({text})") } else { text }
}

/// Letters, digits and underscores, not starting with a digit: a name an expression says
/// bare. Any other name it says in square brackets.
pub(crate) fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

/// A parameter's name as an expression writes it: bare when it is an identifier, in square
/// brackets when it is not -- `[cap-1] * 2` -- which the lexer reads back as one name. So a
/// name is a label and never arithmetic: `cap-1` the name and `cap - 1` the difference are
/// told apart by the brackets, not by refusing the dash.
pub(crate) fn name_in_expression(name: &str) -> String {
    if is_identifier(name) { name.to_string() } else { format!("[{name}]") }
}

fn finite(value: f64, step: impl FnOnce() -> String) -> Result<f64, EvalError> {
    if value.is_finite() { Ok(value) } else { Err(EvalError::NotFinite(step())) }
}

/// A number as it is written back: `23` rather than `23.0`, so a rendered expression reads
/// as something a person would have typed.
fn number(v: f64) -> String {
    format!("{v}")
}

impl BinOp {
    fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Pow => "^",
        }
    }

    fn precedence(self) -> u8 {
        match self {
            BinOp::Add | BinOp::Sub => 1,
            BinOp::Mul | BinOp::Div => 2,
            BinOp::Pow => 3,
        }
    }

    fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Div => a / b,
            BinOp::Pow => a.powf(b),
        }
    }
}

impl Function {
    fn named(name: &str) -> Option<Self> {
        Some(match name {
            "abs" => Function::Abs,
            "min" => Function::Min,
            "max" => Function::Max,
            "sqrt" => Function::Sqrt,
            "floor" => Function::Floor,
            "ceil" => Function::Ceil,
            "round" => Function::Round,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Function::Abs => "abs",
            Function::Min => "min",
            Function::Max => "max",
            Function::Sqrt => "sqrt",
            Function::Floor => "floor",
            Function::Ceil => "ceil",
            Function::Round => "round",
        }
    }

    fn check_arity(self, given: usize) -> Result<(), String> {
        match self {
            Function::Min | Function::Max if given < 2 => {
                Err(format!("{} takes at least two arguments", self.name()))
            }
            Function::Min | Function::Max => Ok(()),
            _ if given != 1 => Err(format!("{} takes one argument", self.name())),
            _ => Ok(()),
        }
    }

    fn apply(self, args: &[f64]) -> f64 {
        match self {
            Function::Abs => args[0].abs(),
            Function::Min => args.iter().copied().fold(f64::INFINITY, f64::min),
            Function::Max => args.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            Function::Sqrt => args[0].sqrt(),
            Function::Floor => args[0].floor(),
            Function::Ceil => args[0].ceil(),
            // Half away from zero, which is what `f64::round` does and what a person expects
            // of `round(2.5)`; banker's rounding would read as a bug here.
            Function::Round => args[0].round(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Open,
    Close,
    Comma,
    End,
}

impl Token {
    fn describe(&self) -> String {
        match self {
            Token::Number(v) => format!("`{}`", number(*v)),
            Token::Ident(s) => format!("`{s}`"),
            Token::Plus => "`+`".into(),
            Token::Minus => "`-`".into(),
            Token::Star => "`*`".into(),
            Token::Slash => "`/`".into(),
            Token::Caret => "`^`".into(),
            Token::Open => "`(`".into(),
            Token::Close => "`)`".into(),
            Token::Comma => "`,`".into(),
            Token::End => "the end".into(),
        }
    }
}

/// Splits the text into tokens, each with the character offset it starts at.
fn lex(text: &str) -> Result<Vec<(Token, usize)>, ParseError> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut at = 0;
    while at < chars.len() {
        let c = chars[at];
        let start = at;
        if c.is_whitespace() {
            at += 1;
            continue;
        }
        let single = match c {
            '+' => Some(Token::Plus),
            '-' => Some(Token::Minus),
            '*' => Some(Token::Star),
            '/' => Some(Token::Slash),
            '^' => Some(Token::Caret),
            '(' => Some(Token::Open),
            ')' => Some(Token::Close),
            ',' => Some(Token::Comma),
            _ => None,
        };
        if let Some(token) = single {
            out.push((token, start));
            at += 1;
            continue;
        }
        let starts_number =
            c.is_ascii_digit() || (c == '.' && chars.get(at + 1).is_some_and(|d| d.is_ascii_digit()));
        if starts_number {
            while at < chars.len() && (chars[at].is_ascii_digit() || chars[at] == '.') {
                at += 1;
            }
            // An exponent only if digits follow it, so `2e` is the number 2 and a stray `e`
            // rather than a malformed number.
            if at < chars.len() && (chars[at] == 'e' || chars[at] == 'E') {
                let mut probe = at + 1;
                if probe < chars.len() && (chars[probe] == '+' || chars[probe] == '-') {
                    probe += 1;
                }
                if probe < chars.len() && chars[probe].is_ascii_digit() {
                    at = probe;
                    while at < chars.len() && chars[at].is_ascii_digit() {
                        at += 1;
                    }
                }
            }
            let text: String = chars[start..at].iter().collect();
            let value: f64 = text.parse().map_err(|_| ParseError {
                offset: start,
                message: format!("`{text}` is not a number"),
            })?;
            // `1e999` parses as infinity, which is refused here rather than allowed to fail at
            // evaluation as "inf is not a number" — the mistake is in the text.
            if !value.is_finite() {
                return Err(ParseError { offset: start, message: format!("`{text}` is too large") });
            }
            out.push((Token::Number(value), start));
            continue;
        }
        if c == '_' || c.is_ascii_alphabetic() {
            while at < chars.len() && (chars[at] == '_' || chars[at].is_ascii_alphanumeric()) {
                at += 1;
            }
            out.push((Token::Ident(chars[start..at].iter().collect()), start));
            continue;
        }
        // A name an expression could not otherwise say -- `[cap-1]`, `[wall 2]` -- in square
        // brackets, which is what the renderer writes for one. Everything up to the closing
        // bracket is the name.
        if c == '[' {
            let Some(close) = chars[at + 1..].iter().position(|&d| d == ']').map(|p| at + 1 + p)
            else {
                return Err(ParseError {
                    offset: start,
                    message: "`[` without a closing `]`".into(),
                });
            };
            let name: String = chars[at + 1..close].iter().collect();
            if name.trim().is_empty() {
                return Err(ParseError { offset: start, message: "`[]` names nothing".into() });
            }
            out.push((Token::Ident(name), start));
            at = close + 1;
            continue;
        }
        return Err(ParseError {
            offset: start,
            message: format!("`{c}` is not part of an expression"),
        });
    }
    out.push((Token::End, chars.len()));
    Ok(out)
}

struct Parser<'a> {
    tokens: Vec<(Token, usize)>,
    at: usize,
    names: &'a dyn Fn(&str) -> Option<ParamId>,
}

impl Parser<'_> {
    fn peek(&self) -> &Token {
        &self.tokens[self.at].0
    }

    fn offset(&self) -> usize {
        self.tokens[self.at].1
    }

    /// Takes the current token. The end stays put, so a parser that runs off the text keeps
    /// seeing the end rather than indexing past it.
    fn bump(&mut self) -> (Token, usize) {
        let current = self.tokens[self.at].clone();
        if self.at + 1 < self.tokens.len() {
            self.at += 1;
        }
        current
    }

    fn expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.term()?;
        loop {
            let op = match self.peek() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => return Ok(lhs),
            };
            self.bump();
            let rhs = self.term()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
    }

    fn term(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.power()?;
        loop {
            let op = match self.peek() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                _ => return Ok(lhs),
            };
            self.bump();
            let rhs = self.power()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
    }

    fn power(&mut self) -> Result<Expr, ParseError> {
        let base = self.unary()?;
        if matches!(self.peek(), Token::Caret) {
            self.bump();
            let exponent = self.power()?;
            return Ok(Expr::Binary(BinOp::Pow, Box::new(base), Box::new(exponent)));
        }
        Ok(base)
    }

    fn unary(&mut self) -> Result<Expr, ParseError> {
        if matches!(self.peek(), Token::Minus) {
            self.bump();
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        self.primary()
    }

    fn primary(&mut self) -> Result<Expr, ParseError> {
        let (token, offset) = self.bump();
        match token {
            Token::Number(v) => Ok(Expr::Number(v)),
            Token::Open => {
                let inner = self.expr()?;
                self.expect_close()?;
                Ok(inner)
            }
            Token::Ident(name) => {
                if matches!(self.peek(), Token::Open) {
                    let Some(function) = Function::named(&name) else {
                        return Err(ParseError {
                            offset,
                            message: format!("`{name}` is not a function"),
                        });
                    };
                    self.bump();
                    let mut args = vec![self.expr()?];
                    while matches!(self.peek(), Token::Comma) {
                        self.bump();
                        args.push(self.expr()?);
                    }
                    self.expect_close()?;
                    function
                        .check_arity(args.len())
                        .map_err(|message| ParseError { offset, message })?;
                    return Ok(Expr::Call(function, args));
                }
                match (self.names)(&name) {
                    Some(id) => Ok(Expr::Param(id)),
                    None => Err(ParseError { offset, message: format!("`{name}` is not a parameter") }),
                }
            }
            Token::End => Err(ParseError { offset, message: "expected a number or a name".into() }),
            other => Err(ParseError { offset, message: format!("unexpected {}", other.describe()) }),
        }
    }

    fn expect_close(&mut self) -> Result<(), ParseError> {
        if matches!(self.peek(), Token::Close) {
            self.bump();
            Ok(())
        } else {
            Err(ParseError { offset: self.offset(), message: "expected `)`".into() })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A name table for the tests: each name is a fresh id.
    fn table(names: &[&str]) -> Vec<(String, ParamId)> {
        names.iter().map(|n| (n.to_string(), ParamId::new())).collect()
    }

    fn parse(text: &str, table: &[(String, ParamId)]) -> Result<Expression, ParseError> {
        Expression::parse(text, &|name| {
            table.iter().find(|(n, _)| n == name).map(|(_, id)| *id)
        })
    }

    /// Parses and evaluates against no parameters at all.
    fn value(text: &str) -> f64 {
        parse(text, &[]).expect(text).evaluate(&|_| None).expect(text)
    }

    #[test]
    fn precedence_and_associativity_follow_the_grammar() {
        assert_eq!(value("2 + 3 * 4"), 14.0);
        assert_eq!(value("(2 + 3) * 4"), 20.0);
        assert_eq!(value("2 ^ 3 ^ 2"), 512.0, "the power is right associative");
        assert_eq!(value("10 - 4 - 3"), 3.0, "subtraction is left associative");
        assert_eq!(value("12 / 4 / 3"), 1.0, "so is division");
        assert_eq!(value("2 * 3 ^ 2"), 18.0);
    }

    /// The convention, tested and therefore documented: the grammar reads `-2 ^ 2` as
    /// `(-2) ^ 2`, the way Excel does and Python does not. Writing `-(2 ^ 2)` gets the other.
    #[test]
    fn unary_minus_binds_tighter_than_the_power() {
        assert_eq!(value("-2 ^ 2"), 4.0);
        assert_eq!(value("-(2 ^ 2)"), -4.0);
        assert_eq!(value("2 * -3"), -6.0);
        assert_eq!(value("--2"), 2.0);
        assert_eq!(value("2 ^ -1"), 0.5);
    }

    #[test]
    fn every_function_computes_what_its_name_says() {
        assert_eq!(value("abs(-3)"), 3.0);
        assert_eq!(value("min(3, 1, 2)"), 1.0);
        assert_eq!(value("max(3, 1, 2)"), 3.0);
        assert_eq!(value("sqrt(16)"), 4.0);
        assert_eq!(value("floor(2.7)"), 2.0);
        assert_eq!(value("ceil(2.1)"), 3.0);
        assert_eq!(value("round(2.5)"), 3.0, "half away from zero");
        assert_eq!(value("round(-2.5)"), -3.0);
        assert_eq!(value("min(2, 3) + max(4, 5)"), 7.0);
    }

    #[test]
    fn numbers_take_decimals_and_exponents() {
        assert_eq!(value("1.5e2"), 150.0);
        assert_eq!(value(".5"), 0.5);
        assert_eq!(value("2e-1"), 0.2);
        assert_eq!(value("2E+1"), 20.0);
    }

    #[test]
    fn a_parse_error_says_what_and_where() {
        let table = table(&["a"]);
        let err = |text: &str| parse(text, &table).expect_err(text);
        assert_eq!(err("2 +"), ParseError { offset: 3, message: "expected a number or a name".into() });
        assert_eq!(err("(2 + 3"), ParseError { offset: 6, message: "expected `)`".into() });
        assert_eq!(err("2 + 3)"), ParseError { offset: 5, message: "unexpected `)`".into() });
        assert_eq!(err("foo + 1"), ParseError { offset: 0, message: "`foo` is not a parameter".into() });
        assert_eq!(err("a + foo(1)"), ParseError { offset: 4, message: "`foo` is not a function".into() });
        assert_eq!(err("abs(1, 2)"), ParseError { offset: 0, message: "abs takes one argument".into() });
        assert_eq!(err("min(1)"), ParseError { offset: 0, message: "min takes at least two arguments".into() });
        assert_eq!(err(""), ParseError { offset: 0, message: "expected a number or a name".into() });
        assert_eq!(err("2 x"), ParseError { offset: 2, message: "unexpected `x`".into() });
        assert_eq!(err("2 $ 3"), ParseError { offset: 2, message: "`$` is not part of an expression".into() });
        assert_eq!(err("1.2.3"), ParseError { offset: 0, message: "`1.2.3` is not a number".into() });
        assert_eq!(err("1e999"), ParseError { offset: 0, message: "`1e999` is too large".into() });
        assert_eq!(err("2 +").to_string(), "expected a number or a name at column 4");
    }

    #[test]
    fn names_resolve_to_ids_and_are_case_sensitive() {
        let table = table(&["wall"]);
        let wall = table[0].1;
        let expr = parse("wall * 2", &table).unwrap();
        assert_eq!(expr.tree(), &Expr::Binary(BinOp::Mul, Box::new(Expr::Param(wall)), Box::new(Expr::Number(2.0))));
        assert_eq!(expr.references(), vec![wall]);
        assert_eq!(parse("Wall * 2", &table).unwrap_err().message, "`Wall` is not a parameter");
        assert_eq!(expr.evaluate(&|id| (id == wall).then_some(2.4)).unwrap(), 4.8);
    }

    #[test]
    fn a_non_finite_step_is_an_error_not_a_number() {
        let fails = |text: &str, step: &str| {
            let err = parse(text, &[]).unwrap().evaluate(&|_| None).expect_err(text);
            assert_eq!(err, EvalError::NotFinite(step.into()), "{text}");
        };
        fails("1 / 0", "1 / 0");
        fails("sqrt(-4)", "sqrt(-4)");
        fails("10 ^ 400", "10 ^ 400");
        fails("min(1 / 0, 5)", "1 / 0");
        fails("-1 / 0", "-1 / 0");
        assert_eq!(
            parse("2 / 0", &[]).unwrap().evaluate(&|_| None).unwrap_err().to_string(),
            "2 / 0 is not a number"
        );
    }

    #[test]
    fn an_unresolved_reference_is_an_error() {
        let table = table(&["a"]);
        let a = table[0].1;
        let expr = parse("a + 1", &table).unwrap();
        assert_eq!(expr.evaluate(&|_| None), Err(EvalError::Unresolved(a)));
    }

    #[test]
    fn references_are_listed_once_each() {
        let table = table(&["a", "b"]);
        let (a, b) = (table[0].1, table[1].1);
        let expr = parse("a + a * b + min(a, b)", &table).unwrap();
        let mut expected = vec![a, b];
        expected.sort();
        assert_eq!(expr.references(), expected);
    }

    /// A name that is not an identifier is said in square brackets, and reads back as itself.
    #[test]
    fn a_name_beyond_an_identifier_is_said_in_brackets_and_reads_back() {
        let table = table(&["cap-1", "cap", "wall 2"]);
        let names = |id: ParamId| table.iter().find(|(_, i)| *i == id).unwrap().0.clone();
        let expr = parse("[cap-1] * 2 + cap - [wall 2]", &table).unwrap();
        assert_eq!(expr.rendered(&names), "[cap-1] * 2 + cap - [wall 2]");
        // Without the brackets the same characters are a difference, which is what the
        // brackets are for.
        assert_eq!(parse("cap-1", &table).unwrap().rendered(&names), "cap - 1");
        assert_eq!(parse("[cap", &table).unwrap_err().message, "`[` without a closing `]`");
        assert_eq!(parse("[] + 1", &table).unwrap_err().message, "`[]` names nothing");
        assert!(is_identifier("cap0") && is_identifier("_a2") && !is_identifier("cap-1"));
        assert!(!is_identifier("2x") && !is_identifier("boss r") && !is_identifier(""));
    }

    #[test]
    fn rendering_from_the_tree_reparses_to_the_same_tree() {
        let table = table(&["a", "b", "c"]);
        let names = |id: ParamId| table.iter().find(|(_, i)| *i == id).unwrap().0.clone();
        for source in [
            "2 + 3 * 4",
            "(2 + 3) * 4",
            "2 - (3 - 4)",
            "2 / (3 / 4)",
            "10 - 4 - 3",
            "(2 ^ 3) ^ 2",
            "2 ^ 3 ^ 2",
            "-2 ^ 2",
            "2 ^ -1",
            "-(a + b)",
            "--a",
            "a * (b + c)",
            "min(a, b + 1) - abs(-c)",
            "1.5e2 * a",
        ] {
            let once = parse(source, &table).expect(source);
            let rendered = once.rendered(&names);
            let twice = parse(&rendered, &table).unwrap_or_else(|e| panic!("{source} rendered as {rendered:?}: {e}"));
            assert_eq!(once.tree(), twice.tree(), "{source} rendered as {rendered:?}");
        }
    }

    #[test]
    fn rendering_uses_the_current_name_and_no_more_parentheses_than_the_tree_needs() {
        let table = table(&["wall", "inner"]);
        let (wall, inner) = (table[0].1, table[1].1);
        let mut expr = parse("inner+2*wall", &table).unwrap();
        assert_eq!(expr.source(), "inner+2*wall", "as typed until a rename");
        expr.rerender(&|id| if id == wall { "wall_thickness".into() } else if id == inner { "inner".into() } else { unreachable!() });
        assert_eq!(expr.source(), "inner + 2 * wall_thickness");
        assert_eq!(parse("(a)", &self::table(&["a"])).unwrap().rendered(&|_| "a".into()), "a");
    }
}
