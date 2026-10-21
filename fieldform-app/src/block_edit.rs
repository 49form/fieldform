// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! A block's own keys applied to the block it already is (`interaction-spec.md` §6.2).
//!
//! **What `set Sphere_0.radius=12` does, and what a create verb's keys do once its block exists.**
//! A create verb reads its keys onto a default block (`console_keys`); this reads the same keys,
//! by the same words, onto a block that is already there and keeps every key not given. So the
//! console has one word for one field whether it is making a block or changing one, and the echo
//! (`echo.rs`) can write any block as the keys that make it and any edit as the keys that changed.
//!
//! The curve-built blocks -- extrude, revolve, sweep, loft, pipe -- had no reader of their own:
//! their create arms set the numbers on the block after it was built, by `Value::as_f64`, which
//! read a literal and silently dropped a parameter's name. They read through here now, so
//! `extrude height=Height` drives the height by the parameter as `sphere radius=Height` always did.
//!
//! Free of the `App`, like the key table it extends: what only the app knows -- which curve a name
//! is, which side of a loop is inside -- is handed in as a [`Context`].

use crate::console_keys::{
    self, axis_triple, corners_named, flag, loft_smooth_named, operand_named, operands_named,
    param_one, param_triple, percent_pair, section_named, Lookup, Slot,
};
use fieldform_ops::{Args, Value};
use fieldform_scene::{
    AngleSides, Axis, Banding, BooleanOp, Continuity, CurveId, Fillet, Frame, LatticeCell,
    LatticePhase, Node, Operand, Outline, ParamRef, Place, RampShape, Sides, Spacing, Taper,
    WallEnds, WallSide,
};

/// What only the app can answer about the names a line uses.
pub struct Context<'a> {
    /// A parameter's or a block's name (`console_keys::Lookup`).
    pub lookup: &'a dyn Fn(&str) -> Lookup,
    /// A curve by its name.
    pub curve: &'a dyn Fn(&str) -> Option<CurveId>,
    /// A profile by its name, or any curve of one by the curve's.
    pub profile: &'a dyn Fn(&str) -> Option<CurveId>,
    /// Which side of a loop is its inside, read on the block as it will be; `None` for a curve
    /// that closes no loop.
    pub inside: &'a dyn Fn(&Node, CurveId) -> Option<WallSide>,
    /// The phases a lattice cell offers, and the phase a lattice is seeded with for one.
    pub offered: &'a dyn Fn(&LatticeCell) -> &'static [Banding],
    pub seed: &'a dyn Fn(Banding) -> LatticePhase,
}

/// The console verb that makes a block of this kind, and the operation it runs -- `None` for a
/// kind no line makes. A placement is `move`'s, whose `set` keys are the block's own fields since
/// three verbs share it.
pub fn verb_of(node: &Node) -> Option<(&'static str, &'static str)> {
    Some(match node {
        Node::Sphere { .. } => ("sphere", "create.sphere"),
        Node::Cuboid { .. } => ("cube", "create.cube"),
        Node::SkewedBox { .. } => ("skewedbox", "create.skewbox"),
        Node::Cylinder { .. } => ("cylinder", "create.cylinder"),
        Node::Cone { .. } => ("cone", "create.cone"),
        Node::Torus { .. } => ("torus", "create.torus"),
        Node::Capsule { .. } => ("capsule", "create.capsule"),
        Node::Plane { .. } => ("plane", "create.plane"),
        Node::Point { .. } => ("point", "create.point"),
        Node::Lattice { .. } => ("tpms", "create.tpms"),
        Node::Ramp { .. } => ("ramp", "create.ramp"),
        Node::Boolean { .. } | Node::Blend { .. } => ("boolean", "create.boolean"),
        Node::Morph { .. } => ("morph", "modify.morph"),
        Node::Offset { .. } => ("offset", "modify.offset"),
        Node::Shell { .. } => ("shell", "modify.shell"),
        Node::Twist { .. } => ("twist", "modify.twist"),
        Node::Mirror { .. } => ("mirror", "modify.mirror"),
        Node::Attract { .. } => ("attract", "modify.attract"),
        Node::Pattern { .. } => ("pattern", "modify.pattern"),
        Node::Transform { .. } => ("move", "modify.move"),
        Node::Extrude { .. } => ("extrude", "create.extrude"),
        Node::Revolve { .. } => ("revolve", "create.revolve"),
        Node::Sweep { .. } => ("sweep", "create.sweep"),
        Node::Loft { .. } => ("loft", "create.loft"),
        Node::Pipe { .. } => ("pipe", "create.pipe"),
        Node::Cage(_) => ("cage", "create.cage"),
        Node::Mesh(_) | Node::Unknown(_) => return None,
    })
}

/// The keys a placement takes from `set`: its own fields, since `move`, `turn` and `scale` each
/// spell one of them `by`.
pub const PLACEMENT_KEYS: &[&str] = &["target", "translation", "rotation", "scale", "about"];

/// Why `args` may not go to this block, if they may not: a key its verb does not read.
pub fn refusal(node: &Node, args: &Args) -> Option<String> {
    let Some((word, op)) = verb_of(node) else {
        return Some("this block is not made from the console, so it has no keys to set".to_string());
    };
    if matches!(node, Node::Transform { .. }) {
        let unknown: Vec<&str> = args.iter().map(|(k, _)| k).filter(|k| !PLACEMENT_KEYS.contains(k)).collect();
        return (!unknown.is_empty()).then(|| {
            format!("a placement doesn't read `{}` — it reads {}", unknown.join("`, `"), PLACEMENT_KEYS.join(", "))
        });
    }
    console_keys::refusal(word, op, args)
}

/// `node` with `args` applied, every key not given kept as it was.
pub fn node_with_keys(node: Node, args: &Args, cx: &Context) -> Result<Node, String> {
    if let Some(why) = refusal(&node, args) {
        return Err(why);
    }
    let lookup = cx.lookup;
    let (word, _) = verb_of(&node).expect("refusal answered for a kind no verb makes");
    match node {
        Node::Sphere { .. }
        | Node::Cuboid { .. }
        | Node::SkewedBox { .. }
        | Node::Cylinder { .. }
        | Node::Cone { .. }
        | Node::Torus { .. }
        | Node::Capsule { .. }
        | Node::Plane { .. }
        | Node::Point { .. } => console_keys::primitive_with_keys(node, word, args, lookup),
        Node::Lattice { .. } => console_keys::lattice_with_keys(node, args, cx.offered, cx.seed, lookup),
        Node::Ramp { .. } => console_keys::ramp_with_keys(node, args, lookup),
        Node::Boolean { .. } | Node::Blend { .. } => boolean_with_keys(node, args, lookup),
        Node::Morph { mut a, mut b, t } => {
            if let Some(o) = operand_named(word, "a", args, lookup)? {
                a = o;
            }
            if let Some(o) = operand_named(word, "b", args, lookup)? {
                b = o;
            }
            console_keys::morph_with_keys(Node::Morph { a, b, t }, args, lookup)
        }
        Node::Offset { .. } => console_keys::offset_with_keys(targeted(node, word, args, lookup)?, args, lookup),
        Node::Shell { .. } => console_keys::shell_with_keys(targeted(node, word, args, lookup)?, args, lookup),
        Node::Twist { .. } => console_keys::twist_with_keys(targeted(node, word, args, lookup)?, args, lookup),
        Node::Mirror { .. } => console_keys::mirror_with_keys(targeted(node, word, args, lookup)?, args, lookup),
        Node::Attract { .. } => {
            let mut node = targeted(node, word, args, lookup)?;
            if let Some(toward) = console_keys::toward_named(args, cx.curve, lookup)?
                && let Node::Attract { toward: slot, .. } = &mut node
            {
                *slot = toward;
            }
            console_keys::attract_with_keys(node, args, lookup)
        }
        Node::Pattern { .. } => {
            let mut node = targeted(node, word, args, lookup)?;
            if let Node::Pattern { options, .. } = &mut node {
                pattern_options_with_keys(options, args, lookup)?;
            }
            console_keys::pattern_with_keys(node, args, lookup)
        }
        Node::Transform { .. } => transform_with_keys(node, args, lookup),
        Node::Extrude { .. } => extrude_with_keys(node, args, cx),
        Node::Revolve { .. } => revolve_with_keys(node, args, cx),
        Node::Sweep { .. } => sweep_with_keys(node, args, cx),
        Node::Loft { .. } => loft_with_keys(node, args, cx),
        Node::Pipe { .. } => pipe_with_keys(node, args, lookup, cx),
        // A cage reads no key, and `refusal` has said so of any given.
        Node::Cage(_) => Ok(node),
        Node::Mesh(_) | Node::Unknown(_) => unreachable!("refused above"),
    }
}

/// `target=` onto the one operand a modifier reads.
fn targeted(mut node: Node, word: &str, args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<Node, String> {
    if let Some(o) = operand_named(word, "target", args, lookup)? {
        match &mut node {
            Node::Offset { input, .. }
            | Node::Shell { input, .. }
            | Node::Twist { input, .. }
            | Node::Mirror { input, .. }
            | Node::Attract { input, .. }
            | Node::Pattern { input, .. }
            | Node::Transform { input, .. } => *input = o,
            _ => return Err(format!("`{word}` has no `target=`")),
        }
    }
    Ok(node)
}

/// A pattern's options from the keys given, keeping the rest -- where `pattern_options_from`
/// builds them fresh, as a new pattern wants.
fn pattern_options_with_keys(
    options: &mut fieldform_scene::PatternOptions,
    args: &Args,
    lookup: &dyn Fn(&str) -> Lookup,
) -> Result<(), String> {
    if args.get("symmetric").is_some() {
        options.symmetric = flag(args, "symmetric");
    }
    if args.get("extent").is_some() {
        options.spacing = if flag(args, "extent") { Spacing::Extent } else { Spacing::Step };
    }
    if let Some(v) = args.get("spacing") {
        options.spacing = match v.text().trim().to_ascii_lowercase().as_str() {
            "extent" => Spacing::Extent,
            "step" => Spacing::Step,
            other => return Err(format!("`pattern spacing=` is step or extent, not {other:?}")),
        };
    }
    if let Some(v) = args.get("angle") {
        options.angle = param_one("pattern", "angle", v, Slot::Number, lookup)?;
    }
    if args.get("skip").is_some() {
        options.skip = crate::console::pattern_options_from(&Args::new().with("skip", args.get("skip").cloned().expect("given")))?
            .skip;
    }
    Ok(())
}

/// A boolean or a blend, which are one block to a person: `op=`, `in=`, and `radius=` with `g=`.
///
/// A radius on two operands is a blend and a radius of nought takes the fillet off, so a block
/// can turn from one kind to the other under `set`, as it does under the panel's Sharp and
/// Filleted.
fn boolean_with_keys(node: Node, args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<Node, String> {
    let verb = "boolean";
    let (mut op, mut inputs, mut fillet) = match node {
        Node::Boolean { op, inputs } => (op, inputs, None),
        Node::Blend { op, a, b, fillet } => (op, vec![a, b], Some(fillet)),
        _ => return Err("`boolean` sets a boolean".to_string()),
    };
    let asked_op = [
        (BooleanOp::Union, ["u", "union"]),
        (BooleanOp::Difference, ["s", "subtract"]),
        (BooleanOp::Intersection, ["i", "intersect"]),
    ]
    .into_iter()
    .find(|(_, names)| {
        names.iter().any(|n| flag(args, n))
            || args.get("op").is_some_and(|v| names.iter().any(|n| v.text().eq_ignore_ascii_case(n)))
    })
    .map(|(op, _)| op);
    if let Some(v) = args.get("op")
        && asked_op.is_none()
    {
        return Err(format!("`boolean op=` is union, subtract or intersect — not {:?}", v.text()));
    }
    if let Some(asked) = asked_op {
        op = asked;
    }
    if let Some(v) = args.get("in") {
        inputs = if v.text().trim().eq_ignore_ascii_case(console_keys::NOTHING) {
            vec![Operand::empty(), Operand::empty()]
        } else {
            operands_named(verb, "in", v, lookup)?
        };
        if inputs.len() < 2 {
            return Err("`boolean in=` names two or more blocks, like in=box,ball".to_string());
        }
    }
    if let Some(v) = args.get("radius").or(args.get("r")) {
        fillet = match param_one(verb, "radius", v, Slot::Number, lookup)? {
            ParamRef::Literal(r) if r <= 0.0 => None,
            radius => Some(Fillet {
                radius,
                continuity: fillet.map_or(Continuity::G1, |f| f.continuity),
            }),
        };
    }
    if let Some(v) = args.get("g") {
        let continuity = match v.text().trim().to_ascii_lowercase().as_str() {
            "1" | "g1" => Continuity::G1,
            "2" | "g2" => Continuity::G2,
            other => return Err(format!("`boolean g=` is 1 or 2, not {other:?}")),
        };
        match &mut fillet {
            Some(f) => f.continuity = continuity,
            None => return Err("`boolean g=` goes with a radius, `radius=`".to_string()),
        }
    }
    Ok(match fillet {
        Some(fillet) => {
            let [a, b] = inputs[..] else {
                return Err("a fillet joins two bodies; union the rest first".to_string());
            };
            Node::Blend { op, a, b, fillet }
        }
        None => Node::Boolean { op, inputs },
    })
}

/// A placement's own fields, which is what `set` takes for one (`PLACEMENT_KEYS`).
fn transform_with_keys(node: Node, args: &Args, lookup: &dyn Fn(&str) -> Lookup) -> Result<Node, String> {
    let word = "move";
    let Node::Transform { mut input, mut translation, mut rotation, mut scale, mut about } = node else {
        return Err("a placement's keys set a placement".to_string());
    };
    if let Some(o) = operand_named(word, "target", args, lookup)? {
        input = o;
    }
    if let Some(v) = args.get("translation") {
        translation = param_triple(word, "translation", v, Slot::Number, lookup)?;
    }
    if let Some(v) = args.get("rotation") {
        rotation = param_triple(word, "rotation", v, Slot::Number, lookup)?;
    }
    if let Some(v) = args.get("scale") {
        scale = param_one(word, "scale", v, Slot::Number, lookup)?;
        if let ParamRef::Literal(s) = scale
            && s <= 0.0
        {
            return Err(format!("a placement's scale must be above nought, not {s}"));
        }
    }
    if let Some(v) = args.get("about") {
        about = Place::At(param_triple(word, "about", v, Slot::Number, lookup)?);
    }
    Ok(Node::Transform { input, translation, rotation, scale, about })
}

// --------------------------------------------------------------------- the curve-built blocks

/// The older spellings a curve-built verb still reads, renamed to the field each stands for so
/// one reader serves both. Both given at once is one thing said twice, and refused.
fn field_names(verb: &str, args: &Args, older: &[(&str, &str)]) -> Result<Args, String> {
    let mut out = Args::new();
    for (k, v) in args.iter() {
        match older.iter().find(|(old, _)| *old == k) {
            Some((_, field)) if args.get(field).is_some() => {
                return Err(format!("`{verb} {k}=` and `{field}=` are one thing said twice — say it once"));
            }
            Some((_, field)) => out = out.with(field, v.clone()),
            None => out = out.with(k, v.clone()),
        }
    }
    Ok(out)
}

/// What a curve-built block is built on: `profile=` a closed loop, `curve=` a wall on a curve,
/// and the wall's `thickness=`, `side=` and `ends=`. `wall=true` turns a profile into a wall on
/// its own curve, and `wall=false` back, which is the block's Wall checkbox typed.
fn outline_with_keys(verb: &str, node: &mut Node, args: &Args, cx: &Context) -> Result<(), String> {
    let lookup = cx.lookup;
    let before = node.clone();
    let Some(outline) = node.outline_mut() else { return Ok(()) };
    let curve_of = |o: &Outline| match o {
        Outline::Closed { profile } => *profile,
        Outline::Wall { curve, .. } => *curve,
    };
    let as_wall = |o: &Outline, curve: CurveId| match o {
        Outline::Wall { thickness, side, ends, .. } => Outline::Wall { curve, thickness: *thickness, side: *side, ends: *ends },
        Outline::Closed { .. } => Outline::Wall {
            curve,
            thickness: ParamRef::Literal(2.0),
            side: WallSide::Centre,
            ends: WallEnds::Square,
        },
    };
    match (args.get("profile"), args.get("curve")) {
        (Some(_), Some(_)) => return Err(format!("`{verb}` builds on a profile or a curve — profile= or curve=, not both")),
        (Some(v), None) => {
            let name = v.text();
            let id = (cx.profile)(name.trim()).ok_or_else(|| format!("no curve or profile is called {name:?}"))?;
            *outline = Outline::Closed { profile: id };
        }
        (None, Some(v)) => {
            let name = v.text();
            let id = (cx.curve)(name.trim()).ok_or_else(|| format!("no curve is called {name:?}"))?;
            *outline = as_wall(outline, id);
        }
        (None, None) => {}
    }
    match args.get("wall") {
        None => {}
        Some(Value::Bool(true)) => *outline = as_wall(outline, curve_of(outline)),
        Some(Value::Bool(false)) => *outline = Outline::Closed { profile: curve_of(outline) },
        Some(v) => {
            let t = param_one(verb, "wall", v, Slot::Number, lookup)?;
            *outline = as_wall(outline, curve_of(outline));
            if let Outline::Wall { thickness, .. } = outline {
                *thickness = t;
            }
        }
    }
    if let Some(v) = args.get("thickness") {
        let t = param_one(verb, "thickness", v, Slot::Number, lookup)?;
        if let ParamRef::Literal(t) = t
            && t <= 0.0
        {
            return Err(format!("A wall needs a thickness above zero, and {t} is not"));
        }
        *outline = as_wall(outline, curve_of(outline));
        if let Outline::Wall { thickness, .. } = outline {
            *thickness = t;
        }
    }
    if let Some(v) = args.get("ends") {
        let ends = crate::console::ends_from_word(&v.text())?;
        match outline {
            Outline::Wall { ends: slot, .. } => *slot = ends,
            Outline::Closed { .. } => return Err(format!("`{verb} ends=` is a wall's, and this is built on a closed profile")),
        }
    }
    if let Some(v) = args.get("side") {
        let curve = curve_of(outline);
        let side = crate::console::side_from_word(&v.text(), (cx.inside)(&before, curve))?;
        match outline {
            Outline::Wall { side: slot, .. } => *slot = side,
            Outline::Closed { .. } => return Err(format!("`{verb} side=` is a wall's, and this is built on a closed profile")),
        }
    }
    Ok(())
}

/// An extrude's keys: what it is built on, and `start`, `height`, `draft`, `symmetric`,
/// `height_2`, `draft_2` and `corners`, read as `apply_extrude_options` read them and with a
/// parameter's name where a number goes.
pub fn extrude_with_keys(mut node: Node, args: &Args, cx: &Context) -> Result<Node, String> {
    let verb = "extrude";
    let args = field_names(verb, args, &[("depth", "height"), ("two", "height_2"), ("draft2", "draft_2")])?;
    outline_with_keys(verb, &mut node, &args, cx)?;
    let lookup = cx.lookup;
    let one = |k: &str| args.get(k).map(|v| param_one(verb, k, v, Slot::Number, lookup)).transpose();
    let Node::Extrude { start, height, sides, draft, corners, rims, .. } = &mut node else {
        return Err("`extrude` sets an extrude".to_string());
    };
    if let Some(asked) = rims_read(verb, &args, *rims, lookup)? {
        *rims = asked;
    }
    if let Some(v) = one("start")? {
        *start = v;
    }
    if let Some(v) = one("height")? {
        *height = v;
    }
    if let Some(v) = one("draft")? {
        *draft = v;
    }
    let abs = |r: ParamRef| match r {
        ParamRef::Literal(v) => ParamRef::Literal(v.abs()),
        other => other,
    };
    let draft_2 = one("draft_2")?;
    *sides = match (one("height_2")?, args.get("symmetric")) {
        (Some(_), Some(_)) => return Err("`height_2` and `symmetric` are two answers to which way the depth lies".to_string()),
        (Some(h), None) => Sides::Two {
            height: abs(h),
            draft: draft_2.or(match &*sides {
                Sides::Two { draft, .. } => Some(*draft),
                _ => None,
            })
            .unwrap_or(ParamRef::Literal(0.0)),
        },
        (None, Some(Value::Bool(false))) => Sides::One,
        (None, Some(_)) => Sides::Symmetric,
        (None, None) => match (sides.clone(), draft_2) {
            (Sides::Two { height, .. }, Some(d)) => Sides::Two { height, draft: d },
            (_, Some(_)) => return Err("`draft_2` is a second side's draft, and there is no second side without `height_2`".to_string()),
            (sides, None) => sides,
        },
    };
    if let Some(v) = args.get("corners") {
        *corners = corners_named(v)?;
    }
    Ok(node)
}

/// A profile solid's rims from `rims=` and `rims_continuity=`: [`console_keys::rims_asked`]'s
/// reading, with a parameter's name taken where the reach goes, as every other number here takes
/// one. The rims' row has a V, so the echo can write a name there, and its line has to read back.
pub fn rims_read(
    verb: &str,
    args: &Args,
    current: fieldform_scene::Rims,
    lookup: &dyn Fn(&str) -> Lookup,
) -> Result<Option<fieldform_scene::Rims>, String> {
    match args.get("rims").filter(|v| v.as_f64().is_none()) {
        None => console_keys::rims_asked(verb, args, current),
        Some(v) => {
            let mut rims = current;
            rims.radius = param_one(verb, "rims", v, Slot::Number, lookup)?;
            let rest = args.clone().without("rims");
            Ok(Some(console_keys::rims_asked(verb, &rest, rims)?.unwrap_or(rims)))
        }
    }
}

/// A revolve's keys: what it is built on, `axis=` (a curve's name, x/y/z or three numbers) with
/// `origin=`, `angle`, `symmetric` and `angle_2`.
pub fn revolve_with_keys(mut node: Node, args: &Args, cx: &Context) -> Result<Node, String> {
    let verb = "revolve";
    let args = field_names(verb, args, &[("two", "angle_2")])?;
    outline_with_keys(verb, &mut node, &args, cx)?;
    let lookup = cx.lookup;
    let Node::Revolve { frame, angle, sides, rims, .. } = &mut node else {
        return Err("`revolve` sets a revolve".to_string());
    };
    if let Some(asked) = rims_read(verb, &args, *rims, lookup)? {
        *rims = asked;
    }
    if args.get("axis").is_some() || args.get("origin").is_some() {
        let axis_text = args.get("axis").map(|v| v.text());
        let letter = axis_text.as_deref().is_some_and(|t| Axis::ALL.iter().any(|a| a.label().eq_ignore_ascii_case(t.trim())));
        match axis_text.as_deref().filter(|t| !letter && !t.contains(',')) {
            Some(name) => {
                let curve = (cx.curve)(name.trim()).ok_or_else(|| format!("no curve is called {name:?} for `revolve axis=`"))?;
                if args.get("origin").is_some() {
                    return Err("a revolve about a curve turns about that curve — `origin=` goes with axis=x|y|z or three numbers".to_string());
                }
                *frame = Frame::Along { curve };
            }
            None => {
                let (was_origin, was_axis) = match frame {
                    Frame::At { origin, axis } => (*origin, *axis),
                    _ => ([ParamRef::Literal(0.0); 3], [ParamRef::Literal(0.0), ParamRef::Literal(0.0), ParamRef::Literal(1.0)]),
                };
                let origin = args.get("origin").map(|v| param_triple(verb, "origin", v, Slot::Number, lookup)).transpose()?.unwrap_or(was_origin);
                let axis = args.get("axis").map(|v| axis_triple(verb, "axis", v, lookup)).transpose()?.unwrap_or(was_axis);
                *frame = Frame::At { origin, axis };
            }
        }
    }
    if let Some(v) = args.get("angle") {
        *angle = param_one(verb, "angle", v, Slot::Number, lookup)?;
    }
    *sides = match (args.get("angle_2"), args.get("symmetric")) {
        (Some(_), Some(_)) => return Err("`angle_2` and `symmetric` are two answers to which way the angle lies".to_string()),
        (Some(v), None) => AngleSides::Two { angle: param_one(verb, "angle_2", v, Slot::Number, lookup)? },
        (None, Some(Value::Bool(false))) => AngleSides::One,
        (None, Some(_)) => AngleSides::Symmetric,
        (None, None) => sides.clone(),
    };
    Ok(node)
}

/// A sweep's keys: what it is built on, `spine=`, `section=`, and the law's own fields --
/// `taper_scale` or `taper_draft`, `twist`, `law`, `extent` and `corners`.
pub fn sweep_with_keys(mut node: Node, args: &Args, cx: &Context) -> Result<Node, String> {
    let verb = "sweep";
    let args = field_names(verb, args, &[("taper-scale", "taper_scale")])?;
    outline_with_keys(verb, &mut node, &args, cx)?;
    let lookup = cx.lookup;
    let Node::Sweep { spine, section, law, rims, .. } = &mut node else {
        return Err("`sweep` sets a sweep".to_string());
    };
    if let Some(asked) = rims_read(verb, &args, *rims, lookup)? {
        *rims = asked;
    }
    if let Some(v) = args.get("spine") {
        let name = v.text();
        *spine = (cx.curve)(name.trim()).ok_or_else(|| format!("no curve is called {name:?} for `sweep spine=`"))?;
    }
    if let Some(v) = args.get("section") {
        *section = section_named(v)?;
    }
    match (args.get("taper_scale"), args.get("taper_draft")) {
        (Some(_), Some(_)) => {
            return Err("`taper_scale` and `taper_draft` are two answers to how the section grows along the path".to_string())
        }
        (Some(v), None) => law.taper = Taper::Scale(param_one(verb, "taper_scale", v, Slot::Number, lookup)?),
        (None, Some(v)) => law.taper = Taper::Draft(param_one(verb, "taper_draft", v, Slot::Number, lookup)?),
        (None, None) => {}
    }
    if let Some(v) = args.get("twist") {
        law.twist = param_one(verb, "twist", v, Slot::Number, lookup)?;
    }
    if let Some(v) = args.get("corners") {
        law.corners = corners_named(v)?;
    }
    if let Some(v) = args.get("law") {
        let name = v.text();
        law.shape = RampShape::ALL
            .into_iter()
            .find(|s| s.label().eq_ignore_ascii_case(name.trim()))
            .ok_or_else(|| format!("`{name}` is not a law; they are G0, G1 and G2"))?;
    }
    if let Some(v) = args.get("extent") {
        let [from, to] = percent_pair(verb, "extent", v)?;
        law.extent = [ParamRef::Literal(from), ParamRef::Literal(to)];
    }
    Ok(node)
}

/// A loft's keys: `sections=` in the order the path meets them, `spine=` a curve or `none` for
/// the one built from the end sections, and `smooth=`.
pub fn loft_with_keys(mut node: Node, args: &Args, cx: &Context) -> Result<Node, String> {

    let Node::Loft { sections, spine, smooth, rims } = &mut node else {
        return Err("`loft` sets a loft".to_string());
    };
    if let Some(asked) = rims_read("loft", args, *rims, cx.lookup)? {
        *rims = asked;
    }
    if let Some(v) = args.get("sections") {
        let text = v.text();
        let mut found = Vec::new();
        for name in text.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            found.push(Outline::closed((cx.profile)(name).ok_or_else(|| format!("no curve or profile is called {name:?}"))?));
        }
        if found.len() < 2 {
            return Err("`loft sections=` names two or more closed curves".to_string());
        }
        *sections = found;
    }
    if let Some(v) = args.get("spine") {
        let name = v.text();
        *spine = if name.trim().eq_ignore_ascii_case(console_keys::NOTHING) {
            None
        } else {
            Some((cx.curve)(name.trim()).ok_or_else(|| format!("no curve is called {name:?} for `loft spine=`"))?)
        };
    }
    if let Some(v) = args.get("smooth").or(args.get("g1")) {
        *smooth = loft_smooth_named(v)?;
    }
    Ok(node)
}

/// A pipe's keys: `curve=` and `radius=`, which a ramp may drive.
fn pipe_with_keys(mut node: Node, args: &Args, lookup: &dyn Fn(&str) -> Lookup, cx: &Context) -> Result<Node, String> {
    let Node::Pipe { curve, radius } = &mut node else { return Err("`pipe` sets a pipe".to_string()) };
    if let Some(v) = args.get("curve") {
        let name = v.text();
        *curve = (cx.curve)(name.trim()).ok_or_else(|| format!("no curve is called {name:?}"))?;
    }
    if let Some(v) = args.get("radius") {
        *radius = param_one("pipe", "radius", v, Slot::NumberOrField, lookup)?;
    }
    Ok(node)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A rim's reach takes a parameter's name, as every other number a line gives does**: the
    /// rims' row has a V, so the echo writes a name there, and the line must build what it says.
    /// A number and a shape still read as `console_keys::rims_asked` reads them.
    #[test]
    fn a_rim_s_reach_reads_a_parameter_s_name() {
        let wall = fieldform_scene::ParamId::new();
        let lookup = |t: &str| if t == "wall" { Lookup::Parameter(wall) } else { Lookup::Nothing };
        let named = Args::new().with("rims", Value::Str("wall".into())).with("rims_continuity", Value::Str("chamfer".into()));
        let rims = rims_read("extrude", &named, fieldform_scene::Rims::hard(), &lookup).unwrap().expect("asked");
        assert_eq!(rims.radius, ParamRef::Named(wall));
        assert_eq!(rims.continuity, Continuity::Chamfer);
        let number = Args::new().with("rims", Value::Float(2.0));
        assert_eq!(rims_read("extrude", &number, fieldform_scene::Rims::hard(), &lookup).unwrap().map(|r| r.radius), Some(ParamRef::Literal(2.0)));
        let unknown = Args::new().with("rims", Value::Str("nowhere".into()));
        assert!(rims_read("extrude", &unknown, fieldform_scene::Rims::hard(), &lookup).is_err(), "a name nothing has is refused");
        assert_eq!(rims_read("extrude", &Args::new(), fieldform_scene::Rims::hard(), &lookup), Ok(None), "nothing asked keeps the rims");
    }
}
