// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The GUI types the command language (`interaction-spec.md` §6).
//!
//! **Every block can be written as the line that makes it, and every edit as the keys that
//! changed.** [`keys_of`] is the inverse of the readers in `console_keys` and `block_edit`: what it
//! writes, they read back to the same block, and `every_block_written_reads_back_to_itself` holds
//! the pair to that over a sample of every kind with every field set. The echo is built on that
//! and on nothing else, so a line the GUI prints is a line the console would run to the same
//! effect -- the contract §6.3 states and T1 tests end to end.
//!
//! **Floats print as Rust's shortest round-trip form** (§6.3), so `0.1` is written `0.1` and reads
//! back to the same bits. **A parameter is written by its name**, so a field driven by an
//! expression echoes the expression's name and the expression its own source text, never the
//! value it happens to evaluate to.
//!
//! Free of the `App`: it reads a `Document` and nothing else.

use fieldform_scene::{
    AngleSides, Attractor, BooleanOp, CellMap, Continuity, Corners, CurveId, Document, Ends, Fillet,
    Frame, Graph, LatticePhase, LoftSmooth, Node, NodeId, Operand, Outline, ParamId, ParamRef,
    PatternKind, Place, Section, Sides, Spacing, Taper, Turn, WallEnds, WallSide,
};

/// What a block, a curve and a parameter are called, for a line that names them.
pub struct Names<'a> {
    pub doc: &'a Document,
}

impl Names<'_> {
    pub fn block(&self, id: NodeId) -> Result<String, String> {
        self.doc
            .node_meta
            .get(&id)
            .and_then(|m| m.name.clone())
            .ok_or_else(|| "it reads a block with no name, which no line can name".to_string())
    }

    pub fn curve(&self, id: CurveId) -> Result<String, String> {
        self.doc.curve(id).map(|c| c.name.clone()).ok_or_else(|| "it names a curve that is gone".to_string())
    }

    pub fn param(&self, id: ParamId) -> Result<String, String> {
        self.doc
            .parameters
            .get(&id)
            .map(|p| p.name.clone())
            .ok_or_else(|| "it reads a parameter that is gone".to_string())
    }
}

/// A number as the console reads it back exactly: the shortest form that round-trips.
pub fn number(v: f64) -> String {
    // `-0` reads as the integer nought, which is the same number; written plainly.
    if v == 0.0 { "0".to_string() } else { format!("{v}") }
}

fn value(r: &ParamRef, n: &Names) -> Result<String, String> {
    match r {
        ParamRef::Literal(v) => Ok(number(*v)),
        ParamRef::Named(id) => n.param(*id),
        ParamRef::Field(id) => n.block(*id),
    }
}

fn triple(t: &[ParamRef; 3], n: &Names) -> Result<String, String> {
    Ok(format!("{},{},{}", value(&t[0], n)?, value(&t[1], n)?, value(&t[2], n)?))
}

fn place(p: &Place, n: &Names) -> Result<String, String> {
    match p {
        Place::At(t) => triple(t, n),
        _ => Err("a place taken from another block has no console spelling yet".to_string()),
    }
}

fn turn(t: &Turn, n: &Names) -> Result<String, String> {
    match t {
        Turn::By(t) => triple(t, n),
        _ => Err("a turn taken from another block has no console spelling yet".to_string()),
    }
}

fn ends(e: &Ends, n: &Names) -> Result<(String, String), String> {
    match e {
        Ends::At { from, to } => Ok((place(from, n)?, place(to, n)?)),
        _ => Err("ends taken from a curve have no console spelling yet".to_string()),
    }
}

fn frame(f: &Frame, n: &Names) -> Result<(String, String), String> {
    match f {
        Frame::At { origin, axis } => Ok((triple(origin, n)?, triple(axis, n)?)),
        _ => Err("a frame taken from another block has no console spelling yet".to_string()),
    }
}

fn operand(o: &Operand, n: &Names) -> Result<String, String> {
    match o.source() {
        Some(id) => n.block(id),
        None => Ok(crate::console_keys::NOTHING.to_string()),
    }
}

fn op_word(op: BooleanOp) -> &'static str {
    match op {
        BooleanOp::Union => "union",
        BooleanOp::Difference => "subtract",
        BooleanOp::Intersection => "intersect",
    }
}

fn g_word(c: Continuity) -> &'static str {
    match c {
        Continuity::G1 => "1",
        Continuity::G2 => "2",
        Continuity::Chamfer => "chamfer",
    }
}

/// A profile solid's rims: the reach always, so an edit back to hard is a key that changed rather
/// than one that went quiet, and the shape only where there is a rim to shape -- `rims_continuity=`
/// on a hard rim is refused.
fn rims_keys(rims: &fieldform_scene::Rims, n: &Names, keys: &mut Vec<(&'static str, String)>) -> Result<(), String> {
    keys.push(("rims", value(&rims.radius, n)?));
    if !rims.is_plainly_hard() {
        keys.push(("rims_continuity", g_word(rims.continuity).to_string()));
        // Which ends, by the word `rims_at=` reads, so a rim on one end echoes back as one.
        keys.push((
            "rims_at",
            match rims.at {
                fieldform_scene::RimEnds::Both => "both",
                fieldform_scene::RimEnds::Start => "start",
                fieldform_scene::RimEnds::End => "end",
            }
            .to_string(),
        ));
        // A chamfer's second leg, only where there is one.
        if let Some(second) = &rims.second {
            keys.push(("rims_second", value(second, n)?));
        }
    }
    Ok(())
}

/// A box's or a cylinder's edge as `edge=` and `edge_continuity=`, and nothing for a sharp one --
/// which is what a line that never named them builds.
fn edge_keys(edge: &Fillet, n: &Names, keys: &mut Vec<(&'static str, String)>) -> Result<(), String> {
    if !edge.is_plainly_hard() {
        keys.push(("edge", value(&edge.radius, n)?));
        keys.push(("edge_continuity", g_word(edge.continuity).to_string()));
    }
    Ok(())
}

fn corners_word(c: Corners) -> &'static str {
    match c {
        Corners::Round => "round",
        Corners::Sharp => "sharp",
    }
}

/// A block written as keys: the verb that makes its kind, the keys that make it, and the keys
/// `set` gives it after, for a placement whose verb can say only one of its three amounts.
#[derive(Clone, Debug, PartialEq)]
pub struct Written {
    pub verb: &'static str,
    pub keys: Vec<(&'static str, String)>,
    pub after: Vec<(&'static str, String)>,
}

/// What a curve-built block is built on, as keys.
fn outline(o: &Outline, n: &Names, keys: &mut Vec<(&'static str, String)>) -> Result<(), String> {
    match o {
        Outline::Closed { profile } => keys.push(("profile", n.curve(*profile)?)),
        Outline::Wall { curve, thickness, side, ends } => {
            keys.push(("curve", n.curve(*curve)?));
            keys.push(("thickness", value(thickness, n)?));
            keys.push((
                "side",
                match side {
                    WallSide::Centre => "centre",
                    WallSide::Left => "left",
                    WallSide::Right => "right",
                }
                .to_string(),
            ));
            keys.push((
                "ends",
                match ends {
                    WallEnds::Square => "square",
                    WallEnds::Round => "round",
                }
                .to_string(),
            ));
        }
    }
    Ok(())
}

/// The keys that make `node`, as its create verb reads them, or why it cannot be written.
///
/// Every key is written whether or not it is the default, so a line says the whole block and does
/// not lean on a default that may change.
pub fn keys_of(node: &Node, n: &Names) -> Result<Written, String> {
    let mut keys: Vec<(&'static str, String)> = Vec::new();
    let mut after: Vec<(&'static str, String)> = Vec::new();
    let verb = crate::block_edit::verb_of(node).map(|(w, _)| w).ok_or("no line makes a block of this kind")?;
    match node {
        Node::Point { at } => keys.push(("at", triple(at, n)?)),
        Node::Plane { origin, flipped, turn: t } => {
            keys.push(("origin", place(origin, n)?));
            keys.push(("turn", turn(t, n)?));
            keys.push(("flipped", flipped.to_string()));
        }
        Node::Sphere { centre, radius } => {
            keys.push(("radius", value(radius, n)?));
            keys.push(("centre", place(centre, n)?));
        }
        Node::Cuboid { centre, size, turn: t, edge } => {
            keys.push(("size", triple(size, n)?));
            keys.push(("centre", place(centre, n)?));
            keys.push(("turn", turn(t, n)?));
            edge_keys(edge, n, &mut keys)?;
        }
        Node::SkewedBox { centre, edges } => {
            keys.push(("centre", place(centre, n)?));
            let rows: Result<Vec<String>, String> = edges.iter().map(|e| triple(e, n)).collect();
            keys.push(("edges", rows?.join(";")));
        }
        Node::Cylinder { ends: e, radius, edge } => {
            let (from, to) = ends(e, n)?;
            keys.push(("radius", value(radius, n)?));
            keys.push(("from", from));
            keys.push(("to", to));
            edge_keys(edge, n, &mut keys)?;
        }
        Node::Cone { ends: e, base_radius, top_radius, edge } => {
            let (from, to) = ends(e, n)?;
            keys.push(("base_radius", value(base_radius, n)?));
            keys.push(("top_radius", value(top_radius, n)?));
            keys.push(("from", from));
            keys.push(("to", to));
            edge_keys(edge, n, &mut keys)?;
        }
        Node::Torus { centre, major_radius, minor_radius, turn: t } => {
            keys.push(("major_radius", value(major_radius, n)?));
            keys.push(("minor_radius", value(minor_radius, n)?));
            keys.push(("centre", place(centre, n)?));
            keys.push(("turn", turn(t, n)?));
        }
        Node::Capsule { ends: e, start_radius, end_radius } => {
            let (from, to) = ends(e, n)?;
            keys.push(("start_radius", value(start_radius, n)?));
            keys.push(("end_radius", value(end_radius, n)?));
            keys.push(("from", from));
            keys.push(("to", to));
        }
        Node::Lattice { cell, frame: f, map, phase } => {
            keys.push(("cell", cell.name().to_string()));
            if let Some(jitter) = cell.jitter() {
                keys.push(("jitter", value(jitter, n)?));
            }
            if let Some(seed) = cell.seed() {
                keys.push(("seed", seed.to_string()));
            }
            match map {
                CellMap::Rectangular { cell } => {
                    keys.push(("map", "rectangular".to_string()));
                    keys.push(("cell_size", value(cell, n)?));
                }
                CellMap::Cylindrical { cell_radius, cell_height, arc_count } => {
                    keys.push(("map", "cylindrical".to_string()));
                    keys.push(("cell_radius", value(cell_radius, n)?));
                    keys.push(("cell_height", value(cell_height, n)?));
                    keys.push(("arc", arc_count.to_string()));
                }
                CellMap::Spherical { cell_radius, azimuth_count, polar_count } => {
                    keys.push(("map", "spherical".to_string()));
                    keys.push(("cell_radius", value(cell_radius, n)?));
                    keys.push(("azimuth", azimuth_count.to_string()));
                    keys.push(("polar", polar_count.to_string()));
                }
                CellMap::Conformal { .. } => {
                    return Err("a conformal map follows a quad mesh, which is picked in the panel".to_string())
                }
            }
            match phase {
                LatticePhase::Sheet { thickness, offset } => {
                    keys.push(("phase", "sheet".to_string()));
                    keys.push(("thickness", value(thickness, n)?));
                    keys.push(("offset", value(offset, n)?));
                }
                LatticePhase::Solid { density } => {
                    keys.push(("phase", "solid".to_string()));
                    keys.push(("density", value(density, n)?));
                }
                LatticePhase::Strut { radius } => {
                    keys.push(("phase", "strut".to_string()));
                    keys.push(("radius", value(radius, n)?));
                }
            }
            let (origin, axis) = frame(f, n)?;
            keys.push(("origin", origin));
            keys.push(("axis", axis));
        }
        Node::Ramp { input, in_min, in_max, out_min, out_max, shape } => {
            keys.push(("input", operand(input, n)?));
            keys.push(("in", format!("{},{}", value(in_min, n)?, value(in_max, n)?)));
            keys.push(("out", format!("{},{}", value(out_min, n)?, value(out_max, n)?)));
            keys.push(("shape", shape.label().to_string()));
        }
        Node::Boolean { op, inputs } => {
            keys.push(("op", op_word(*op).to_string()));
            let named: Result<Vec<String>, String> = inputs.iter().map(|o| operand(o, n)).collect();
            keys.push(("in", named?.join(",")));
        }
        Node::Blend { op, a, b, fillet: Fillet { radius, continuity, second } } => {
            keys.push(("op", op_word(*op).to_string()));
            keys.push(("in", format!("{},{}", operand(a, n)?, operand(b, n)?)));
            keys.push(("radius", value(radius, n)?));
            keys.push(("g", g_word(*continuity).to_string()));
            // A chamfer's second leg, only where there is one.
            if let Some(second) = second {
                keys.push(("second", value(second, n)?));
            }
        }
        Node::Morph { a, b, t } => {
            keys.push(("a", operand(a, n)?));
            keys.push(("b", operand(b, n)?));
            keys.push(("t", value(t, n)?));
        }
        Node::Offset { input, distance } => {
            keys.push(("target", operand(input, n)?));
            keys.push(("distance", value(distance, n)?));
        }
        Node::Shell { input, thickness } => {
            keys.push(("target", operand(input, n)?));
            keys.push(("thickness", value(thickness, n)?));
        }
        Node::Twist { input, degrees_per_mm, frame: f } => {
            let (origin, axis) = frame(f, n)?;
            keys.push(("target", operand(input, n)?));
            keys.push(("degrees_per_mm", value(degrees_per_mm, n)?));
            keys.push(("axis", axis));
            keys.push(("origin", origin));
        }
        Node::Mirror { input, frame: f, join } => {
            let (origin, axis) = frame(f, n)?;
            keys.push(("target", operand(input, n)?));
            keys.push(("axis", axis));
            keys.push(("origin", origin));
            keys.push(("join", join.to_string()));
        }
        Node::Attract { input, toward, strength, falloff } => {
            keys.push(("target", operand(input, n)?));
            keys.push((
                "toward",
                match toward {
                    Attractor::Curve(c) => n.curve(*c)?,
                    Attractor::Point(p) => n.block(*p)?,
                },
            ));
            keys.push(("strength", value(strength, n)?));
            keys.push(("falloff", value(falloff, n)?));
        }
        Node::Pattern { input, count, kind, join, fillet, options } => {
            keys.push(("target", operand(input, n)?));
            keys.push(("count", value(count, n)?));
            match kind {
                PatternKind::Linear { step } => {
                    keys.push(("kind", "linear".to_string()));
                    keys.push(("step", triple(step, n)?));
                    // Said either way, so an edit back to a step is written as one.
                    keys.push(("spacing", if options.spacing == Spacing::Extent { "extent" } else { "step" }.to_string()));
                    if let Some(second) = &options.second {
                        keys.push(("count2", value(&second.count, n)?));
                        keys.push(("step2", triple(&second.step, n)?));
                    }
                }
                PatternKind::Radial { frame: f } => {
                    let (origin, axis) = frame(f, n)?;
                    keys.push(("kind", "radial".to_string()));
                    keys.push(("axis", axis));
                    keys.push(("origin", origin));
                    keys.push(("angle", value(&options.angle, n)?));
                }
            }
            // The radius before the join: a radius joins, and the join after it says whether. All
            // of these are written whatever they hold, nought and an empty skip included, so an edit
            // back to the default is a key that changed rather than one that went quiet.
            keys.push(("radius", value(&fillet.radius, n)?));
            keys.push(("g", g_word(fillet.continuity).to_string()));
            keys.push(("join", join.to_string()));
            keys.push(("symmetric", options.symmetric.to_string()));
            keys.push(("skip", options.skip.iter().map(u32::to_string).collect::<Vec<_>>().join(",")));
        }
        // A placement: `move` makes it with its translation, and `set` gives it the rest, since
        // `turn` and `scale` fold into an existing placement only where the two commute.
        Node::Transform { input, translation, rotation, scale, about } => {
            keys.push(("target", operand(input, n)?));
            keys.push(("by", triple(translation, n)?));
            // A placement of its own, never folded into the one it may wrap.
            keys.push(("wrap", "true".to_string()));
            let zero = |r: &ParamRef| matches!(r, ParamRef::Literal(v) if *v == 0.0);
            if !rotation.iter().all(zero) {
                after.push(("rotation", triple(rotation, n)?));
            }
            if !matches!(scale, ParamRef::Literal(s) if *s == 1.0) {
                after.push(("scale", value(scale, n)?));
            }
            if *about != Place::origin() {
                after.push(("about", place(about, n)?));
            }
        }
        Node::Cage(_) => {}
        Node::Extrude { outline: o, sweep, start, height, sides, draft, corners, rims } => {
            if !sweep.is_exact() {
                return Err("which way an extrude travels, if not straight out, is chosen in the panel".to_string());
            }
            outline(o, n, &mut keys)?;
            keys.push(("height", value(height, n)?));
            keys.push(("start", value(start, n)?));
            keys.push(("draft", value(draft, n)?));
            match sides {
                Sides::One => keys.push(("symmetric", "false".to_string())),
                Sides::Symmetric => keys.push(("symmetric", "true".to_string())),
                Sides::Two { height, draft } => {
                    keys.push(("height_2", value(height, n)?));
                    keys.push(("draft_2", value(draft, n)?));
                }
            }
            keys.push(("corners", corners_word(*corners).to_string()));
            rims_keys(rims, n, &mut keys)?;
        }
        Node::Revolve { outline: o, frame: f, angle, sides, rims } => {
            outline(o, n, &mut keys)?;
            match f {
                Frame::Along { curve } => keys.push(("axis", n.curve(*curve)?)),
                other => {
                    let (origin, axis) = frame(other, n)?;
                    keys.push(("axis", axis));
                    keys.push(("origin", origin));
                }
            }
            keys.push(("angle", value(angle, n)?));
            match sides {
                AngleSides::One => keys.push(("symmetric", "false".to_string())),
                AngleSides::Symmetric => keys.push(("symmetric", "true".to_string())),
                AngleSides::Two { angle } => keys.push(("angle_2", value(angle, n)?)),
            }
            rims_keys(rims, n, &mut keys)?;
        }
        Node::Sweep { outline: o, spine, section, law, rims } => {
            outline(o, n, &mut keys)?;
            keys.push(("spine", n.curve(*spine)?));
            keys.push((
                "section",
                match section {
                    Section::Square => "square",
                    Section::Skewed => "skewed",
                    Section::Parallel => "parallel",
                }
                .to_string(),
            ));
            match &law.taper {
                Taper::None => {}
                Taper::Scale(s) => keys.push(("taper_scale", value(s, n)?)),
                Taper::Draft(d) => keys.push(("taper_draft", value(d, n)?)),
            }
            keys.push(("twist", value(&law.twist, n)?));
            keys.push(("law", law.shape.label().to_string()));
            keys.push(("extent", format!("{},{}", value(&law.extent[0], n)?, value(&law.extent[1], n)?)));
            keys.push(("corners", corners_word(law.corners).to_string()));
            rims_keys(rims, n, &mut keys)?;
        }
        Node::Loft { sections, spine, smooth, rims } => {
            let named: Result<Vec<String>, String> = sections
                .iter()
                .map(|s| match s {
                    Outline::Closed { profile } => n.curve(*profile),
                    Outline::Wall { .. } => Err("a loft through a wall has no console spelling".to_string()),
                })
                .collect();
            keys.push(("sections", named?.join(",")));
            keys.push(("spine", match spine {
                Some(c) => n.curve(*c)?,
                None => crate::console_keys::NOTHING.to_string(),
            }));
            keys.push((
                "smooth",
                match smooth {
                    LoftSmooth::G0 => "g0",
                    LoftSmooth::G1 => "g1",
                }
                .to_string(),
            ));
            rims_keys(rims, n, &mut keys)?;
        }
        Node::Pipe { curve, radius } => {
            keys.push(("curve", n.curve(*curve)?));
            keys.push(("radius", value(radius, n)?));
        }
        Node::Mesh(_) | Node::Unknown(_) => unreachable!("verb_of has no verb for these"),
    }
    Ok(Written { verb, keys, after })
}

/// One `key=value` word, quoted whole when it holds a space, so the console reads it as one.
fn word(key: &str, value: &str) -> String {
    let pair = format!("{key}={value}");
    if pair.contains(char::is_whitespace) {
        let mark = if pair.contains('"') { '\'' } else { '"' };
        format!("{mark}{pair}{mark}")
    } else {
        pair
    }
}

/// A name as the console reads it: always quoted, in whichever mark it does not hold.
pub fn quoted(name: &str) -> String {
    let mark = if name.contains('\'') { '"' } else { '\'' };
    format!("{mark}{name}{mark}")
}

/// The lines that make block `id` as it now is: its verb, every key, its name; then a `set` for
/// what the verb could not say.
pub fn create_lines(doc: &Document, graph: &Graph, id: NodeId) -> Result<Vec<String>, String> {
    let n = Names { doc };
    let node = graph.node(id).ok_or("the block is gone")?;
    let name = n.block(id)?;
    let written = keys_of(node, &n)?;
    let mut line = written.verb.to_string();
    for (k, v) in &written.keys {
        line.push(' ');
        line.push_str(&word(k, v));
    }
    line.push(' ');
    line.push_str(&quoted(&name));
    let mut lines = vec![line];
    if let Some(set) = set_line(&name, &written.after) {
        lines.push(set);
    }
    Ok(lines)
}

/// `set Name.key=value …` for every key given, or `None` for none.
pub fn set_line(name: &str, keys: &[(&'static str, String)]) -> Option<String> {
    if keys.is_empty() {
        return None;
    }
    let mut line = "set".to_string();
    for (k, v) in keys {
        line.push(' ');
        line.push_str(&word(&format!("{name}.{k}"), v));
    }
    Some(line)
}

/// The keys of block `id` that differ between two states of it, as a `set` line -- `None` when
/// none does, or when either state cannot be written.
///
/// A placement compares its own fields, which is what `set` takes for one (`block_edit`).
pub fn edit_line(doc: &Document, before: &Node, after: &Node, id: NodeId) -> Option<String> {
    let n = Names { doc };
    let name = n.block(id).ok()?;
    let (was, now) = (keys_of(before, &n).ok()?, keys_of(after, &n).ok()?);
    if std::mem::discriminant(before) != std::mem::discriminant(after)
        && !matches!((before, after), (Node::Boolean { .. } | Node::Blend { .. }, Node::Boolean { .. } | Node::Blend { .. }))
    {
        return None;
    }
    let all = |w: &Written| -> Vec<(&'static str, String)> {
        match after {
            // `by` is the translation's word only on the line that makes a placement.
            Node::Transform { .. } => {
                let mut keys: Vec<(&'static str, String)> = w
                    .keys
                    .iter()
                    .filter(|(k, _)| *k != "wrap")
                    .map(|(k, v)| (if *k == "by" { "translation" } else { *k }, v.clone()))
                    .collect();
                keys.extend(w.after.iter().cloned());
                keys
            }
            _ => w.keys.clone(),
        }
    };
    let (was, now) = (all(&was), all(&now));
    let changed: Vec<(&'static str, String)> = now
        .iter()
        .filter(|(k, v)| was.iter().find(|(wk, _)| wk == k).is_none_or(|(_, wv)| wv != v))
        .cloned()
        .collect();
    // A placement that went back to the identity on one amount drops that key from what it writes;
    // say the identity out loud so the replay moves it back too.
    let mut changed = changed;
    if matches!(after, Node::Transform { .. }) {
        for (k, identity) in [("rotation", "0,0,0"), ("scale", "1"), ("about", "0,0,0")] {
            if was.iter().any(|(wk, _)| *wk == k) && !now.iter().any(|(nk, _)| *nk == k) {
                changed.push((k, identity.to_string()));
            }
        }
    }
    set_line(&name, &changed)
}

/// A parameter as the line that makes or sets it: its expression's source text where it is
/// computed, its number where it is not.
pub fn param_line(doc: &Document, id: ParamId) -> Option<String> {
    let p = doc.parameters.get(&id)?;
    Some(match p.value.expression() {
        Some(e) => format!("param {} {}", word("name", &p.name), word("expr", e.source())),
        None => format!("param {} {}", word("name", &p.name), word("value", &number(p.value.as_literal()?))),
    })
}

// -------------------------------------------------------------------------------- the diff

/// What the echo compares between two moments: the part's graph and everything a line names
/// beside it.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub graph: Graph,
    pub parameters: std::collections::HashMap<ParamId, fieldform_scene::Parameter>,
    pub curves: std::collections::HashMap<CurveId, fieldform_scene::Curve>,
    /// The blocks declared variables, which `variable` and `inline` move between.
    pub variables: std::collections::HashSet<NodeId>,
    /// What each block was called, for the line that deletes one.
    pub names: std::collections::HashMap<NodeId, String>,
    /// What each block with no name showed instead -- "Sphere" -- which is what a line calls it
    /// by until it is named. The application's to fill, since a label is the list's.
    pub labels: std::collections::HashMap<NodeId, String>,
    /// Each profile's name, and what each loop there was answered to: its name or "Profile N".
    /// The second is the application's to fill, as the labels are.
    pub profiles: std::collections::HashMap<CurveId, String>,
    pub profile_labels: std::collections::HashMap<CurveId, String>,
    /// Each cage's positions, by their bits, and its face count -- so a vertex moved by hand is
    /// written as the line that moves it (R11). A cage whose face count changed was edited by a
    /// topology verb, which is a typed line and its own record, or by a tool that writes its own.
    pub cages: std::collections::HashMap<fieldform_scene::CageId, CageState>,
}

/// A cage as the echo remembers it: enough to say which vertices moved, and whether the mesh is
/// still the same mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CageState {
    pub positions: Vec<[u64; 3]>,
    pub face_count: usize,
}

impl CageState {
    pub fn of(mesh: &fieldform_cage::QuadMesh) -> Self {
        CageState {
            positions: mesh
                .vertex_ids()
                .map(|v| mesh.vertex_position(v).coords.map(f64::to_bits).into())
                .collect(),
            face_count: mesh.face_count(),
        }
    }
}

impl Snapshot {
    pub fn of(doc: &Document, graph: &Graph) -> Self {
        Snapshot {
            graph: graph.clone(),
            parameters: doc.parameters.clone(),
            curves: doc.curves.clone(),
            cages: doc.cages.iter().map(|(id, body)| (*id, CageState::of(body.mesh()))).collect(),
            variables: graph
                .nodes()
                .map(|(id, _)| id)
                .filter(|id| doc.ownership_nodes.get(id) == Some(&fieldform_scene::Ownership::Variable))
                .collect(),
            names: graph
                .nodes()
                .filter_map(|(id, _)| doc.node_meta.get(&id).and_then(|m| m.name.clone()).map(|n| (id, n)))
                .collect(),
            labels: Default::default(),
            profiles: doc.profile_names.clone().into_iter().collect(),
            profile_labels: Default::default(),
        }
    }
}

/// The renames between two moments, as `rename 'was' to=now` lines.
///
/// **Written before anything else**, because every other line calls a row by the name it has
/// *now*: an edit to a renamed block is `set ball.radius=…`, which the replay can only follow once
/// it has been told the sphere is `ball`. Each row is called by what it answered to before -- its
/// name, or the label an unnamed block or profile showed -- and a row that answered to nothing a
/// line can say is left out rather than guessed at. A row made or deleted in between is not
/// renamed: its making already carries its name and its delete the one it had.
pub fn rename_lines(before: &Snapshot, now: &Snapshot) -> Vec<String> {
    let line = |was: &str, to: &str| {
        // An empty name clears one; `to=` with nothing after it would read as a key with no value.
        let to = if to.is_empty() { "to=\"\"".to_string() } else { word("to", to) };
        format!("rename {} {to}", quoted(was))
    };
    let mut out: Vec<String> = Vec::new();
    let mut params: Vec<(String, String)> = now
        .parameters
        .iter()
        .filter_map(|(id, p)| before.parameters.get(id).filter(|was| was.name != p.name).map(|was| (was.name.clone(), p.name.clone())))
        .collect();
    params.sort();
    out.extend(params.iter().map(|(was, to)| line(was, to)));
    let mut curves: Vec<(String, String)> = now
        .curves
        .iter()
        .filter_map(|(id, c)| before.curves.get(id).filter(|was| was.name != c.name).map(|was| (was.name.clone(), c.name.clone())))
        .collect();
    curves.sort();
    out.extend(curves.iter().map(|(was, to)| line(was, to)));
    let mut profiles: Vec<(String, String)> = now
        .profile_labels
        .keys()
        .filter(|key| before.profiles.get(key) != now.profiles.get(key))
        .filter_map(|key| Some((before.profile_labels.get(key)?.clone(), now.profiles.get(key).cloned().unwrap_or_default())))
        .collect();
    profiles.sort();
    out.extend(profiles.iter().map(|(was, to)| line(was, to)));
    let mut blocks: Vec<(String, String)> = now
        .graph
        .nodes()
        .filter(|(id, _)| before.graph.node(*id).is_some() && before.names.get(id) != now.names.get(id))
        .filter_map(|(id, _)| {
            let was = before.names.get(&id).or_else(|| before.labels.get(&id))?;
            Some((was.clone(), now.names.get(&id).cloned().unwrap_or_default()))
        })
        .collect();
    blocks.sort();
    out.extend(blocks.iter().map(|(was, to)| line(was, to)));
    out
}

/// How deep a block sits: nought for one that reads no block, else one more than the deepest it
/// reads. Inputs always come shallower, so creating by depth creates what is read first.
pub fn depths(graph: &Graph) -> std::collections::HashMap<NodeId, usize> {
    fn depth(graph: &Graph, id: NodeId, memo: &mut std::collections::HashMap<NodeId, usize>, on: &mut Vec<NodeId>) -> usize {
        if let Some(d) = memo.get(&id) {
            return *d;
        }
        if on.contains(&id) {
            return 0;
        }
        on.push(id);
        let d = graph
            .node(id)
            .map(|n| n.inputs().into_iter().filter(|i| graph.node(*i).is_some()).map(|i| depth(graph, i, memo, on) + 1).max().unwrap_or(0))
            .unwrap_or(0);
        on.pop();
        memo.insert(id, d);
        d
    }
    let mut memo = std::collections::HashMap::new();
    for (id, _) in graph.nodes() {
        depth(graph, id, &mut memo, &mut Vec::new());
    }
    memo
}

/// Whether `id` is a loop's joint: a point a curve's end sits on. Closing a loop makes those
/// itself (`profile-group-spec.md` §4), so the lines that draw the loop make them again on replay,
/// and a line of their own would make each twice -- which is how T1 first found them.
pub fn is_joint(doc: &Document, graph: &Graph, id: NodeId) -> bool {
    matches!(graph.node(id), Some(Node::Point { .. }))
        && doc
            .curves
            .values()
            .any(|c| (0..c.control_points().len()).any(|i| c.placed_on(i) == Some(id)))
}

/// The blocks in `graph` that `before` did not hold, in the order they are made and named:
/// shallowest first, then by kind, then by what they say -- never by id, which is random, so the
/// same edits name the same blocks the same way every run.
pub fn new_blocks(before: &Graph, doc: &Document, graph: &Graph) -> Vec<NodeId> {
    let d = depths(graph);
    let n = Names { doc };
    let mut fresh: Vec<(usize, &'static str, String, NodeId)> = graph
        .nodes()
        .filter(|(id, _)| before.node(*id).is_none() && !is_joint(doc, graph, *id))
        .map(|(id, node)| {
            let said = keys_of(node, &n)
                .map(|w| format!("{w:?}"))
                .unwrap_or_else(|_| serde_json::to_string(node).unwrap_or_default());
            (d.get(&id).copied().unwrap_or(0), crate::names::operation_word(node), said, id)
        })
        .collect();
    fresh.sort_by(|a, b| (a.0, a.1, &a.2, a.3.raw()).cmp(&(b.0, b.1, &b.2, b.3.raw())));
    fresh.into_iter().map(|(_, _, _, id)| id).collect()
}

/// A curve as the line that makes it: its points, its degree and its weights, then its name.
/// `None` for a curve of more than one span, which no line can make yet.
fn curve_line(curve: &fieldform_scene::Curve) -> Option<String> {
    if !curve.is_single_span() {
        return None;
    }
    let points: Vec<String> = curve
        .control_points()
        .iter()
        .map(|p| format!("{},{},{}", number(p.x), number(p.y), number(p.z)))
        .collect();
    let mut line = format!("curve p={} degree={}", points.join(";"), curve.degree());
    if curve.weights().iter().any(|w| *w != 1.0) {
        let w: Vec<String> = curve.weights().iter().map(|w| number(*w)).collect();
        line.push_str(&format!(" w={}", w.join(",")));
    }
    line.push(' ');
    line.push_str(&quoted(&curve.name));
    Some(line)
}

/// The lines that take the document from `before` to how `doc` and `graph` are now.
///
/// `fresh` is [`new_blocks`]' order, named already. Parameters first, since blocks read them;
/// then curves, which blocks are built on; then each new block, shallowest first; then the keys
/// that changed on blocks that were there; then what was declared or put back; then what went,
/// deepest first, so nothing is deleted while something still reads it.
pub fn lines(before: &Snapshot, doc: &Document, graph: &Graph, fresh: &[NodeId]) -> Vec<String> {
    let mut out = Vec::new();
    // Parameters new or changed, each after those it reads. One that changed only its name is
    // [`rename_lines`]' to write: a `param` line would make a second parameter on replay.
    let mut pending: Vec<ParamId> = doc
        .parameters
        .iter()
        .filter(|(id, p)| match before.parameters.get(id) {
            None => true,
            Some(was) => {
                let mut was = was.clone();
                was.name = p.name.clone();
                was != **p
            }
        })
        .map(|(id, _)| *id)
        .collect();
    pending.sort_by_key(|id| doc.parameters.get(id).map(|p| p.name.clone()));
    let mut known: Vec<String> = doc
        .parameters
        .values()
        .filter(|p| !pending.contains(&p.id))
        .map(|p| p.name.clone())
        .collect();
    while !pending.is_empty() {
        let ready = pending.iter().position(|id| {
            doc.parameters.get(id).is_some_and(|p| match p.value.expression() {
                None => true,
                Some(e) => fieldform_scene::Expression::parse(e.source(), &|name| {
                    known.iter().any(|k| k == name).then(fieldform_scene::ParamId::new)
                })
                .is_ok(),
            })
        });
        let id = pending.remove(ready.unwrap_or(0));
        if let Some(line) = param_line(doc, id) {
            out.push(line);
        }
        if let Some(p) = doc.parameters.get(&id) {
            known.push(p.name.clone());
        }
    }
    // Curves new, then curves moved.
    let mut new_curves: Vec<&fieldform_scene::Curve> =
        doc.curves.values().filter(|c| !before.curves.contains_key(&c.id)).collect();
    new_curves.sort_by_key(|c| (doc.curves_created.get(&c.id).copied().unwrap_or(0), c.id.raw()));
    out.extend(new_curves.into_iter().filter_map(curve_line));
    let mut moved: Vec<&fieldform_scene::Curve> = doc
        .curves
        .values()
        .filter(|c| before.curves.get(&c.id).is_some_and(|was| was.control_points() != c.control_points()))
        .collect();
    moved.sort_by(|a, b| a.name.cmp(&b.name));
    for c in moved {
        let was = &before.curves[&c.id];
        if was.control_points().len() != c.control_points().len() {
            continue;
        }
        let keys: Vec<(String, String)> = c
            .control_points()
            .iter()
            .zip(was.control_points())
            .enumerate()
            .filter(|(_, (now, then))| now != then)
            .map(|(i, (p, _))| (format!("{}.p{i}", c.name), format!("{},{},{}", number(p.x), number(p.y), number(p.z))))
            .collect();
        if !keys.is_empty() {
            out.push(format!("set {}", keys.iter().map(|(k, v)| word(k, v)).collect::<Vec<_>>().join(" ")));
        }
    }
    // New blocks, then edited ones.
    for id in fresh {
        if let Ok(made) = create_lines(doc, graph, *id) {
            out.extend(made);
        }
    }
    let mut edited: Vec<(String, String)> = graph
        .nodes()
        .filter_map(|(id, node)| {
            let was = before.graph.node(id)?;
            if was == node {
                return None;
            }
            let line = edit_line(doc, was, node, id)?;
            Some((Names { doc }.block(id).ok()?, line))
        })
        .collect();
    edited.sort();
    out.extend(edited.into_iter().map(|(_, line)| line));
    // What was declared a variable, or put back.
    let now: std::collections::HashSet<NodeId> = graph
        .nodes()
        .map(|(id, _)| id)
        .filter(|id| doc.ownership_nodes.get(id) == Some(&fieldform_scene::Ownership::Variable))
        .collect();
    let mut declared: Vec<String> = now
        .difference(&before.variables)
        .filter(|id| before.graph.node(**id).is_some())
        .filter_map(|id| Names { doc }.block(*id).ok())
        .map(|name| format!("variable {}", quoted(&name)))
        .collect();
    declared.sort();
    out.extend(declared);
    let mut put_back: Vec<String> = before
        .variables
        .difference(&now)
        .filter(|id| graph.node(**id).is_some())
        .filter_map(|id| Names { doc }.block(*id).ok())
        .map(|name| format!("inline {}", quoted(&name)))
        .collect();
    put_back.sort();
    out.extend(put_back);
    // What went: deepest first, then curves.
    let d = depths(&before.graph);
    let mut gone: Vec<(usize, String)> = before
        .graph
        .nodes()
        .filter(|(id, _)| graph.node(*id).is_none())
        .filter_map(|(id, _)| before.names.get(&id).map(|n| (d.get(&id).copied().unwrap_or(0), n.clone())))
        .collect();
    gone.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    out.extend(gone.into_iter().map(|(_, name)| format!("delete {}", word("target", &name))));
    let mut curves_gone: Vec<String> =
        before.curves.values().filter(|c| !doc.curves.contains_key(&c.id)).map(|c| c.name.clone()).collect();
    curves_gone.sort();
    out.extend(curves_gone.into_iter().map(|name| format!("delete {}", word("target", &name))));
    out.extend(cage_lines(before, doc));
    out
}

/// The cage vertices moved by hand since `before`, as `setpositions` lines -- one per cage, in a
/// fixed order, naming the cage only where the document holds more than one. A cage whose vertex
/// or face count changed is left out: that was a topology edit, which writes its own line.
fn cage_lines(before: &Snapshot, doc: &Document) -> Vec<String> {
    let mut ids: Vec<_> = doc.cages.keys().copied().collect();
    ids.sort_by_key(|id| id.raw());
    let several = ids.len() > 1;
    ids.into_iter()
        .filter_map(|id| {
            let now = doc.cages.get(&id)?.mesh();
            let was = before.cages.get(&id)?;
            if was.face_count != now.face_count() || was.positions.len() != now.vertex_count() {
                return None;
            }
            let moved: Vec<_> = now
                .vertex_ids()
                .filter(|&v| {
                    let bits: [u64; 3] = now.vertex_position(v).coords.map(f64::to_bits).into();
                    was.positions[v.index()] != bits
                })
                .collect();
            if moved.is_empty() {
                return None;
            }
            let vertices: Vec<String> = moved.iter().map(|v| v.index().to_string()).collect();
            let to: Vec<String> = moved
                .iter()
                .map(|&v| {
                    let p = now.vertex_position(v);
                    format!("{},{},{}", number(p.x), number(p.y), number(p.z))
                })
                .collect();
            let mut line = String::from("setpositions ");
            if several {
                line.push_str(&format!("cage={id} "));
            }
            line.push_str(&format!("vertices={} to={}", vertices.join(","), to.join(";")));
            Some(line)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_edit::{node_with_keys, Context};
    use crate::console_keys::Lookup;
    use fieldform_ops::Args;
    use fieldform_scene::{Banding, LatticeCell};

    /// A document whose every block, curve and parameter a sample can name has a name: the
    /// samples' two blocks, their one curve, and a parameter for the test that writes one.
    fn named(samples: &[Node]) -> (Document, std::collections::HashMap<String, Lookup>, CurveId) {
        let mut doc = Document::new();
        let mut words = std::collections::HashMap::new();
        let mut curve = None;
        for node in samples {
            for id in node.inputs() {
                let k = format!("B{}", words.len());
                doc.node_meta.entry(id).or_default().name.get_or_insert_with(|| {
                    words.insert(k.clone(), Lookup::Block(id));
                    k
                });
            }
            if let Node::Attract { toward: Attractor::Point(id), .. } = node {
                let k = format!("B{}", words.len());
                doc.node_meta.entry(*id).or_default().name.get_or_insert_with(|| {
                    words.insert(k.clone(), Lookup::Block(*id));
                    k
                });
            }
            for c in node.curves() {
                curve.get_or_insert(c);
            }
        }
        let curve = curve.expect("the samples name a curve");
        let mut line = crate::curve_edits::single_span_bezier(curve, 1, 10.0).expect("a line");
        line.name = "C".to_string();
        doc.curves.insert(curve, line);
        (doc, words, curve)
    }

    /// A block of the same kind with every field somewhere else, which the keys must put back:
    /// what a create verb starts from, near enough, and nothing the sample holds.
    fn blank(node: &Node, curve: CurveId) -> Node {
        use fieldform_scene::{LatticeCell, PatternOptions, RampShape, SweepLaw};
        let one = ParamRef::Literal(1.0);
        let zero = ParamRef::Literal(0.0);
        let z3 = [zero; 3];
        let ends = Ends::between(Place::origin(), Place::at(0.0, 0.0, 1.0));
        let empty = Operand::empty();
        match node {
            Node::Point { .. } => Node::Point { at: z3 },
            Node::Plane { .. } => Node::Plane { origin: Place::origin(), flipped: false, turn: fieldform_scene::no_turn() },
            Node::Sphere { .. } => Node::Sphere { centre: Place::origin(), radius: one },
            Node::Cuboid { .. } => Node::Cuboid { edge: Default::default(), centre: Place::origin(), size: [one; 3], turn: fieldform_scene::no_turn() },
            Node::SkewedBox { .. } => Node::SkewedBox { centre: Place::origin(), edges: [z3; 3] },
            Node::Cylinder { .. } => Node::Cylinder { edge: Default::default(), ends, radius: one },
            Node::Cone { .. } => Node::Cone { ends, base_radius: one, top_radius: one, edge: Default::default() },
            Node::Torus { .. } => Node::Torus { centre: Place::origin(), major_radius: one, minor_radius: one, turn: fieldform_scene::no_turn() },
            Node::Capsule { .. } => Node::Capsule { ends, start_radius: one, end_radius: one },
            Node::Lattice { .. } => Node::Lattice {
                cell: LatticeCell::Gyroid,
                frame: Frame::default(),
                map: CellMap::Rectangular { cell: ParamRef::Literal(20.0) },
                phase: LatticePhase::Sheet { thickness: one, offset: zero },
            },
            Node::Ramp { .. } => Node::Ramp {
                input: empty,
                in_min: zero,
                in_max: ParamRef::Literal(10.0),
                out_min: zero,
                out_max: one,
                shape: RampShape::Linear,
            },
            Node::Boolean { .. } | Node::Blend { .. } => Node::Boolean { op: BooleanOp::Intersection, inputs: vec![empty, empty] },
            Node::Morph { .. } => Node::Morph { a: empty, b: empty, t: zero },
            Node::Offset { .. } => Node::Offset { input: empty, distance: zero },
            Node::Shell { .. } => Node::Shell { input: empty, thickness: ParamRef::Literal(0.1) },
            Node::Twist { .. } => Node::Twist { input: empty, degrees_per_mm: zero, frame: Frame::default() },
            Node::Mirror { .. } => Node::Mirror { input: empty, frame: Frame::default(), join: false },
            Node::Attract { .. } => Node::Attract { input: empty, toward: Attractor::Curve(curve), strength: zero, falloff: one },
            Node::Pattern { .. } => Node::Pattern {
                input: empty,
                count: ParamRef::Literal(2.0),
                kind: PatternKind::Linear { step: [ParamRef::Literal(50.0), zero, zero] },
                join: false,
                fillet: Fillet::hard(),
                options: PatternOptions::default(),
            },
            Node::Transform { .. } => Node::Transform { input: empty, translation: z3, rotation: z3, scale: one, about: Place::origin() },
            Node::Extrude { .. } => Node::Extrude {
                outline: Outline::closed(curve),
                sweep: fieldform_scene::Sweep::Normal,
                start: zero,
                height: one,
                sides: Sides::One,
                draft: zero,
                corners: Corners::Round,
                rims: fieldform_scene::Rims::hard(),
            },
            Node::Revolve { .. } => Node::Revolve { outline: Outline::closed(curve), frame: Frame::default(), angle: ParamRef::Literal(360.0), sides: AngleSides::One, rims: fieldform_scene::Rims::hard() },
            Node::Sweep { .. } => Node::Sweep { outline: Outline::closed(curve), spine: curve, section: Section::Square, law: SweepLaw::default(), rims: fieldform_scene::Rims::hard() },
            Node::Loft { .. } => Node::Loft { sections: vec![Outline::closed(curve); 2], spine: None, smooth: LoftSmooth::G0, rims: fieldform_scene::Rims::hard() },
            Node::Pipe { .. } => Node::Pipe { curve, radius: one },
            other => other.clone(),
        }
    }

    /// **Every block written reads back to itself** (§6.3): for a sample of every kind with every
    /// field set, the keys `keys_of` writes, read by the console's own readers onto a block of that
    /// kind, give the sample back -- so the echo of any block is a line that makes it.
    #[test]
    fn every_block_written_reads_back_to_itself() {
        let samples = crate::console_keys::tests::samples();
        let (doc, words, curve) = named(&samples);
        let n = Names { doc: &doc };
        let lookup = |t: &str| words.get(t).copied().unwrap_or(Lookup::Nothing);
        let curves = |t: &str| (t == "C").then_some(curve);
        let inside = |_: &Node, _: CurveId| Some(WallSide::Left);
        let offered = |c: &LatticeCell| fieldform_field::CellKind::from(*c).supported_bandings();
        let seed = |b: Banding| match b {
            Banding::Sheet => LatticePhase::Sheet { thickness: ParamRef::Literal(9.0), offset: ParamRef::Literal(0.0) },
            Banding::Solid => LatticePhase::Solid { density: ParamRef::Literal(0.9) },
            Banding::Strut => LatticePhase::Strut { radius: ParamRef::Literal(9.0) },
        };
        let cx = Context { lookup: &lookup, curve: &curves, profile: &curves, inside: &inside, offered: &offered, seed: &seed };
        let mut written = 0;
        for sample in &samples {
            let w = match keys_of(sample, &n) {
                Ok(w) => w,
                Err(why) => {
                    // The recorded gaps: what `console_keys`' first paragraph says has no spelling.
                    assert!(
                        why.contains("no console spelling") || why.contains("panel") || why.contains("no line"),
                        "{sample:?} cannot be written: {why}"
                    );
                    continue;
                }
            };
            let args = w.keys.iter().chain(&w.after).filter(|(k, _)| *k != "wrap").fold(Args::new(), |a, (k, v)| {
                let k = if matches!(sample, Node::Transform { .. }) && *k == "by" { "translation" } else { k };
                a.with(k, crate::console::parse_value(v))
            });
            let back = node_with_keys(blank(sample, curve), &args, &cx).unwrap_or_else(|why| panic!("{sample:?} read back refused: {why}\n{args:?}"));
            assert_eq!(&back, sample, "written as {:?}", w);
            written += 1;
        }
        assert!(written >= 30, "only {written} samples were written");
    }

    /// **A cage vertex moved by hand is written as the line that moves it, and the line puts the
    /// cage back bit for bit** (R11). A topology edit writes nothing here: it is a typed line, or
    /// a tool's own.
    #[test]
    fn a_moved_cage_vertex_echoes_as_setpositions_and_the_line_replays_to_the_same_cage() {
        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let id = *doc.cages.keys().next().unwrap();
        let graph = doc.parts.values().next().unwrap().graph.clone();
        let before = Snapshot::of(&doc, &graph);
        assert!(cage_lines(&before, &doc).is_empty(), "nothing moved, nothing written");

        let v = fieldform_cage::VertexId::from_index(3);
        doc.cage_mesh_mut(id).unwrap().set_vertex_position(v, nalgebra::Point3::new(1.25, -2.0, 0.1));
        let lines = cage_lines(&before, &doc);
        assert_eq!(lines, vec!["setpositions vertices=3 to=1.25,-2,0.1".to_string()], "one cage, so unnamed");

        // Replayed on the cage as it was: the same positions, to the bit.
        let moved = doc.cage_body(id).unwrap().geometry_fingerprint();
        doc.cage_mesh_mut(id).unwrap().set_vertex_position(v, before_position(&before, id, v));
        let (verb, rest) = lines[0].split_once(' ').unwrap();
        assert_eq!(verb, "setpositions");
        let args = rest.split(' ').fold(Args::new(), |a, kv| {
            let (k, val) = kv.split_once('=').unwrap();
            a.with(k, crate::console::parse_value(val))
        });
        let cmd = fieldform_ops::OpRegistry::default().build("cage.set_positions", &args, &doc).unwrap();
        fieldform_ops::CommandStack::default().execute(&mut doc, cmd).unwrap();
        assert_eq!(doc.cage_body(id).unwrap().geometry_fingerprint(), moved);

        // A face taken off is a topology edit, and not this line's to write.
        let after_move = Snapshot::of(&doc, &graph);
        let open = fieldform_cage::edit::delete_faces(doc.cage_body(id).unwrap().mesh(), &[fieldform_cage::FaceId::from_index(0)]).unwrap();
        doc.cage_body_mut(id).unwrap().set_mesh(open.mesh);
        assert!(cage_lines(&after_move, &doc).is_empty());
    }

    fn before_position(s: &Snapshot, id: fieldform_scene::CageId, v: fieldform_cage::VertexId) -> nalgebra::Point3<f64> {
        let b = s.cages[&id].positions[v.index()];
        nalgebra::Point3::new(f64::from_bits(b[0]), f64::from_bits(b[1]), f64::from_bits(b[2]))
    }

    /// A number is written as the shortest text that reads back to the same bits.
    #[test]
    fn a_number_is_written_so_it_reads_back_exactly() {
        for v in [0.1, 1.0 / 3.0, 100.0, -2.5, 1e-9, 123_456.789, f64::MIN_POSITIVE] {
            let text = number(v);
            assert_eq!(text.parse::<f64>().unwrap().to_bits(), v.to_bits(), "{v} as {text}");
        }
        assert_eq!(number(100.0), "100");
        assert_eq!(number(-0.0), "0");
    }

    /// A value with a space in it is one word to the console, quoted whole.
    #[test]
    fn a_value_with_a_space_is_one_word() {
        assert_eq!(word("profile", "Line 2"), "\"profile=Line 2\"");
        let words = crate::console::console_words(&format!("extrude {}", word("profile", "Line 2")));
        assert_eq!(words[1].0, "profile=Line 2");
        assert_eq!(quoted("Sphere_0"), "'Sphere_0'");
        assert_eq!(quoted("Bob's"), "\"Bob's\"");
    }
}
