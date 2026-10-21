// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! How a console line is read: its words, flags and values, and the verbs it can name.

use super::*;

/// Splits `curve3` into `curve` and a degree of three.
///
/// **`curve3` is `curve degree=3`.** A degree is the one number worth putting in the word itself:
/// it is what you change most often about a curve, and it is what you are already thinking about
/// when you reach for the console rather than the menu. The long form still works and this is
/// only a way of writing it, so there is no second command to keep in step and `help` still
/// lists one `curve`.
///
/// Only `curve`, deliberately. Splitting a trailing number off *any* word would quietly turn
/// `cube3` into a cube and swallow the typo, and a console that silently does nearly what you
/// asked is worse than one that says it does not know the word.
pub(crate) fn split_degree_suffix(name: &str) -> (String, Option<u32>) {
    let digits = name.trim_end_matches(|c: char| c.is_ascii_digit());
    if digits != "curve" || digits.len() == name.len() {
        return (name.to_string(), None);
    }
    (digits.to_string(), name[digits.len()..].parse().ok())
}

/// What a typed word means: every word in the command registry, each once.
///
/// **The registry is the one list** (`commands.rs`, `interaction-spec.md` §2). This was a table of
/// its own beside the primitives' derived words and the menu's hand-built items, and the plane
/// and the point once landed in the menu alone; now the menu, the keymap and this are all read
/// from one table, so a word cannot exist in one and not the others.
pub(crate) fn console_commands() -> Vec<(String, &'static str)> {
    crate::commands::console_words()
}

/// What a console `pattern` asked for beyond the count and the step: `symmetric` to centre the
/// copies on the body, `extent` to read the step as the whole run, `angle=180` to stop a ring short
/// of a turn, `rows=3 step2=60` to repeat a row into a grid, and `skip=2,5` to leave copies out.
///
/// Refused by name rather than read as the nearest thing: a skip list with a word in it is a typo,
/// and dropping the word would leave out the wrong copies.
pub(crate) fn pattern_options_from(args: &Args) -> Result<fieldform_scene::PatternOptions, String> {
    let mut options = fieldform_scene::PatternOptions::default();
    let flag = |key: &str| matches!(args.get(key), Some(Value::Bool(true)));
    options.symmetric = flag("symmetric");
    if flag("extent")
        || args.get("spacing").and_then(|v| v.as_str()).is_some_and(|s| s.eq_ignore_ascii_case("extent"))
    {
        options.spacing = fieldform_scene::Spacing::Extent;
    }
    if let Some(angle) = args.get("angle").and_then(|v| v.as_f64()) {
        options.angle = ParamRef::Literal(angle);
    }
    match args.get("skip") {
        None => {}
        Some(Value::Int(n)) => options.skip.push(
            u32::try_from(*n).map_err(|_| format!("skip: {n} is not a copy number"))?,
        ),
        Some(Value::Str(list)) => {
            for word in list.split(',').filter(|w| !w.trim().is_empty()) {
                options.skip.push(
                    word.trim()
                        .parse::<u32>()
                        .map_err(|_| format!("skip: {word:?} is not a copy number"))?,
                );
            }
        }
        Some(_) => return Err("skip takes copy numbers, like skip=2,5".to_string()),
    }
    // A second direction: `rows=3` repeats the row three times along `step2=` millimetres of Y, or
    // along 50 if that is left out. Along Y because a console word is one number, and the first
    // step's own direction is the one thing a second step cannot be.
    if let Some(rows) = args.get("rows").and_then(|v| v.as_f64()) {
        let along_y = args.get("step2").and_then(|v| v.as_f64()).unwrap_or(50.0);
        options.second = Some(fieldform_scene::PatternSecond {
            step: [ParamRef::Literal(0.0), ParamRef::Literal(along_y), ParamRef::Literal(0.0)],
            count: ParamRef::Literal(rows.round()),
        });
    }
    options.skip.sort_unstable();
    options.skip.dedup();
    Ok(options)
}

/// Reads a console argument as the narrowest type that fits.
pub(crate) fn parse_value(text: &str) -> Value {
    if let Ok(i) = text.parse::<i64>() {
        return Value::Int(i);
    }
    if let Ok(f) = text.parse::<f64>() {
        return Value::Float(f);
    }
    match text {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        other => Value::Str(other.to_string()),
    }
}

/// The older console spellings of an extrude's numbers, and the field each stands for. The
/// field names are what a file and the key table say; the arms below read the older words, so
/// each field name is turned into the word the arm reads (`console_keys::aliased`).
pub(crate) const EXTRUDE_ALIASES: &[(&str, &str)] =
    &[("height", "depth"), ("height_2", "two"), ("draft_2", "draft2"), ("thickness", "wall")];

/// The same for a revolve: `angle_2` is the second side's angle the arm reads as `two`.
pub(crate) const REVOLVE_ALIASES: &[(&str, &str)] = &[("angle_2", "two"), ("thickness", "wall")];

/// Whether a command was asked for a wall rather than a profile: by `wall`, or by a side or an
/// end, which mean nothing without one. `wall=false` is a way of saying no.
pub(crate) fn wall_wanted(args: &Args) -> bool {
    match args.get("wall") {
        Some(Value::Bool(false)) => false,
        Some(_) => true,
        None => args.get("side").is_some() || args.get("ends").is_some(),
    }
}

/// How thick a wall command asked for: the number after `wall`, or the one remembered from
/// the last wall where it was given none — `--wall` alone, or only a side.
pub(crate) fn wall_thickness_asked(args: &Args, remembered: f64) -> Result<f64, String> {
    match args.get("wall") {
        None | Some(Value::Bool(_)) => Ok(remembered),
        Some(value) => match value.as_f64() {
            Some(t) if t.is_finite() && t > 0.0 => Ok(t),
            Some(t) => Err(format!("A wall needs a thickness above zero, and {t} is not")),
            None => Err("`wall` takes a thickness in millimetres, as in `--wall 2`".to_string()),
        },
    }
}

/// The side a console's word names, on a curve whose inside is `inside` — `None` for one that
/// closes no loop.
///
/// Left and right are taken as written on a loop too: they are what the document holds, and a
/// script that says `side=left` means the same wall wherever it is run. Inside and outside are
/// refused on an open curve rather than guessed at, since it has neither.
pub(crate) fn side_from_word(
    word: &str,
    inside: Option<fieldform_scene::WallSide>,
) -> Result<fieldform_scene::WallSide, String> {
    use fieldform_scene::WallSide;
    match (word.to_ascii_lowercase().as_str(), inside) {
        ("centre" | "center", _) => Ok(WallSide::Centre),
        ("left", _) => Ok(WallSide::Left),
        ("right", _) => Ok(WallSide::Right),
        ("inside", Some(inside)) => Ok(inside),
        ("outside", Some(inside)) => Ok(across(inside)),
        ("inside" | "outside", None) => Err(
            "Inside and outside are a loop's, and this curve does not close one — say left \
             or right"
                .to_string(),
        ),
        (other, _) => Err(format!(
            "{other:?} is not a side — centre, left or right, or inside or outside on a loop"
        )),
    }
}

/// How a console's word says a wall stops at its curve's ends.
pub(crate) fn ends_from_word(word: &str) -> Result<fieldform_scene::WallEnds, String> {
    use fieldform_scene::WallEnds;
    match word.to_ascii_lowercase().as_str() {
        "square" => Ok(WallEnds::Square),
        "round" => Ok(WallEnds::Round),
        other => Err(format!("{other:?} is not a way to end a wall — square or round")),
    }
}

/// Splits a console line into words, keeping anything quoted together.
///
/// Returns each word and whether it arrived in quotes, because that is what tells a *name* from
/// an argument: `sphere 'left bearing'` is one name with a space in it, and `sphere radius=20`
/// is a key and a value. Without the flag a bare word would have to be guessed at, and the guess
/// would be wrong for the first command anybody wrote with a typo in it.
///
/// Either quote character, since a name with an apostrophe in it is likelier than one with a
/// double quote, and requiring the other one is a rule nobody would remember. An unclosed quote
/// runs to the end of the line rather than being refused: what was meant is obvious, and the
/// alternative is an error message about punctuation.
pub(crate) fn console_words(line: &str) -> Vec<(String, bool)> {
    let mut out: Vec<(String, bool)> = Vec::new();
    let mut word = String::new();
    let mut quoted = false;
    let mut inside: Option<char> = None;
    for c in line.chars() {
        match inside {
            Some(q) if c == q => inside = None,
            Some(_) => word.push(c),
            None if c == '\'' || c == '"' => {
                inside = Some(c);
                quoted = true;
            }
            None if c.is_whitespace() => {
                if !word.is_empty() || quoted {
                    out.push((std::mem::take(&mut word), quoted));
                    quoted = false;
                }
            }
            None => word.push(c),
        }
    }
    if !word.is_empty() || quoted {
        out.push((word, quoted));
    }
    out
}

/// The key a dash-word says yes to, and whether it was written with two dashes — or `None` for
/// a word that is not a flag.
///
/// A letter after the dashes, so a negative number typed by mistake is still met by the
/// key=value message rather than becoming a flag called "5".
fn flag_name(word: &str) -> Option<(&str, bool)> {
    let flag = word.strip_prefix('-')?;
    let long = flag.starts_with('-');
    let flag = flag.trim_start_matches('-');
    flag.starts_with(|c: char| c.is_ascii_alphabetic()).then_some((flag, long))
}

/// What the words after a command's own word say: its arguments, and a name for what it makes.
///
/// Free of the app so that the grammar can be held by a test that reads the grammar itself; the
/// test used to carry a copy of this loop, which is a copy to keep in step.
///
/// **`name=` is left as an ordinary key here**, and `run_command` decides what it means: a name
/// for whatever the command made, unless the verb declares a `name` of its own
/// (`console_keys::reads_its_own_name`). This used to be decided here, and swallowed `name=` for
/// every verb — so `environment name=zebra` reached the environment command with no name at all
/// and listed the choices instead of making one.
///
/// **A quoted word is a name for whatever the command makes.** Asked for:
/// `sphere 'left bearing'` should leave a sphere called that, through the same `set_body_name`
/// a row's rename uses, so a later line can type it. A name is a label, not a declaration
/// (`block-ownership-spec.md` §6): the block is adopted by the first block that reads it all the
/// same. One mechanism for every command that makes something, rather than a
/// `name=` argument each of them would have to read. Quoted *and* key=value is an argument
/// whose value has spaces in it -- `expr="radius * 5 - 2"` -- not a name; a name with an `=` in
/// it is the thing given up for that, and nobody calls a sphere `a=b`.
///
/// **A flag is a word that says yes to itself.** `boolean -u` is `boolean u=true`, which is
/// what every command line means by a flag and what was asked for: "make it so `boolean -u`
/// makes a union, similarly -s is subtract and -i is intersect". Handled here rather than in
/// the boolean, so every command that takes a switch gains the short form at once rather than
/// one at a time.
///
/// **And a long one may be followed by its value.** `extrude --wall 2 --side left` is
/// `extrude wall=2 side=left`: two dashes, then the next word if it is neither a flag nor a
/// key=value of its own. That is how `outline-wall-spec.md` §6 writes its commands and how a
/// command line is usually written, so both spellings reach the same arguments. Only after two
/// dashes -- `-u` stays a switch whatever follows it, so a stray word after one is still
/// reported rather than swallowed as its value.
pub(crate) fn console_arguments(
    words: impl IntoIterator<Item = (String, bool)>,
) -> Result<(Args, Option<String>), String> {
    let mut args = Args::new();
    let mut naming: Option<String> = None;
    let mut words = words.into_iter().peekable();
    while let Some((word, quoted)) = words.next() {
        if quoted && !word.contains('=') {
            naming = Some(word);
            continue;
        }
        if let Some((flag, long)) = flag_name(&word) {
            let follows = words
                .peek()
                .filter(|(next, quoted)| {
                    long && !quoted && !next.contains('=') && flag_name(next).is_none()
                })
                .map(|(next, _)| parse_value(next));
            if follows.is_some() {
                words.next();
            }
            // Lowered here, once, so the key table's refusal and every reader after it agree:
            // `cube Size=10` used to pass the refusal (which compared without case) and then
            // make a default cube, because the reader asked for `size`.
            args = args.with(&flag.to_ascii_lowercase(), follows.unwrap_or(Value::Bool(true)));
            continue;
        }
        let Some((key, value)) = word.split_once('=') else {
            return Err(format!(
                "{word:?} is not key=value — try `sphere`, `sphere 'a name'`, \
                 `boolean -s` or `mesh cells=96`"
            ));
        };
        args = args.with(&key.to_ascii_lowercase(), parse_value(value));
    }
    Ok((args, naming))
}
