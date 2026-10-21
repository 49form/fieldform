// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How fast a node's value can change, whether it is still a distance, and what it divides by.

use super::{
    Amount, CompileError, Op, blend_divisor, blend_radius_rate, loft_stretch, operand_extent,
    pattern_join_divisor, pattern_reach_in, reach_across_axis, resolve_frame, revolve_wedge,
    sweep_lean_degrees, sweep_stretch, taper_ratio,
};
use crate::{Normalization, PatternLayout};
use fieldform_scene::{Document, Graph, Node, NodeId, Operand};
#[cfg(doc)] use super::{GraphField, extent::extent_below};
#[cfg(doc)] use crate::ScalarField;

/// How fast a node's value can change with position, from the graph alone.
///
/// The same quantity [`ScalarField::lipschitz`] reports for a compiled field, computed without
/// compiling one. Two callers need it and neither can use the other's answer: shader generation
/// has a graph and no ops, and compilation needs a driven offset's divisor *while* the ops are
/// still being built. Sharing this is what keeps the CPU and the shader dividing by the same
/// number — a test checks the two against each other, because a divisor that differs moves no
/// surface and so would never show up as a residual.
///
/// Every leaf in this kernel reports 1.0: the primitives are exact distance fields, and a
/// lattice has already divided by its own gradient bound. So the only thing here that raises
/// the constant is a [`Node::Ramp`], which is the point.
pub fn graph_lipschitz(document: &Document, graph: &Graph, node: NodeId) -> f64 {
    lipschitz_of_node(document, graph, node, 0)
}

/// [`graph_lipschitz`] of what a slot reads, and nought for a slot with nothing in it.
///
/// **Nothing to read is a rate of nothing** — the answer [`Node::Ramp`]'s arm has given an unwired
/// input since `ramp-decision.md` §7.2, now that every slot can be one. A rate is a statement
/// about a value, and a block with a hole has no value to make one about; nought is the only
/// reading that cannot be mistaken for a gentle slope by anything above it.
///
/// One reader, shared with the recursion in `lipschitz_of_node`, so a caller holding an
/// [`Operand`] and the walk itself cannot answer differently about the same hole.
pub fn operand_lipschitz(document: &Document, graph: &Graph, slot: &Operand) -> f64 {
    match slot.source() {
        Some(id) => graph_lipschitz(document, graph, id),
        None => 0.0,
    }
}

fn lipschitz_of_node(document: &Document, graph: &Graph, node: NodeId, depth: usize) -> f64 {
    // The same guard the extent walk uses, and for the same reason: a graph that cycles is
    // refused at compile time, but this runs before that and must not recurse forever.
    if depth > 256 {
        return f64::INFINITY;
    }
    let Some(node) = graph.node(node) else {
        return 1.0;
    };
    // **Nought for an empty slot, which is not the `1.0` the missing node above gets.** That
    // default stands for a leaf this walk cannot see and whose value something downstream still
    // reads; an empty slot stands for no value at all, so the honest rate is nothing — see
    // [`operand_lipschitz`], which is this same answer for a caller outside the walk.
    let recurse = |slot: &Operand| match slot.source() {
        Some(id) => lipschitz_of_node(document, graph, id, depth + 1),
        None => 0.0,
    };
    let resolve = |r: &fieldform_scene::ParamRef| -> f64 { document.resolve(r).unwrap_or(0.0) };
    match node {
        Node::Blend { a, b, .. } => recurse(a).max(recurse(b)),
        Node::Boolean { inputs, .. } => inputs.iter().map(|i| recurse(i)).fold(0.0, f64::max),
        // The larger of the two, which is what a convex combination can reach: at `t = 0` it is
        // exactly `a` and at `t = 1` exactly `b`, and in between it cannot rise faster than the
        // steeper of them.
        Node::Morph { a, b, .. } => recurse(a).max(recurse(b)),
        // The chain rule, with both terms known — see [`RampShape::slope_factor`]. This is the
        // number that makes a driven parameter statable at all.
        Node::Ramp {
            input,
            in_min,
            in_max,
            out_min,
            out_max,
            shape,
        } => {
            let in_span = resolve(in_max) - resolve(in_min);
            if in_span.abs() < f64::EPSILON {
                // Refused at compile time; infinity here so that nothing downstream can read
                // an empty range as a gentle slope.
                return f64::INFINITY;
            }
            ((resolve(out_max) - resolve(out_min)) / in_span).abs()
                * shape.slope_factor()
                // Nought for an unwired ramp, which is what `recurse` answers for every empty
                // slot now rather than for this one by hand. Unreachable in a compiled graph,
                // since a consumer of an unwired ramp is refused before this is asked.
                * recurse(input)
        }
        // Each of these has already paid for whatever it does to the rate — an offset's
        // division, a twist's stretch — or does nothing to it at all, for the reasons
        // `lipschitz_of` states arm by arm.
        Node::Offset { input, .. }
        | Node::Shell { input, .. }
        | Node::Twist { input, .. }
        | Node::Attract { input, .. }
        | Node::Mirror { input, .. }
        | Node::Pattern { input, .. }
        | Node::Transform { input, .. } => recurse(input),
        // An exact distance field, or one that has already divided by its own bound.
        Node::Cage(_)
        | Node::Mesh(_)
        | Node::Sphere { .. }
        | Node::Cuboid { .. }
        | Node::SkewedBox { .. }
        | Node::Cylinder { .. }
        | Node::Cone { .. }
        | Node::Torus { .. }
        | Node::Capsule { .. }
        | Node::Plane { .. }
        | Node::Point { .. }
        | Node::Pipe { .. }
        // Divided by its own stretch already, as a twist and an attractor are.
        | Node::Sweep { .. }
        | Node::Loft { .. }
        // Both are built from a true 2D distance and placed by a rigid motion, so neither
        // changes the rate at which the value moves.
        | Node::Extrude { .. }
        | Node::Revolve { .. }
        | Node::Lattice { .. }
        | Node::Unknown(_) => 1.0,
    }
}

/// Whether this node's value is a true distance, answered from the graph without compiling it.
///
/// **The same question [`normalization_of`] answers, asked earlier.** That one reads compiled
/// ops, which do not exist until a compile has succeeded -- and the caller here wants to know
/// *before* offering an edit whether the edit would be refused. A panel cannot compile the
/// whole graph once per frame to grey out a button; that is the per-frame compile that made
/// the interface slow once already.
///
/// Two implementations of one rule is exactly the drift this codebase keeps being bitten by,
/// so `the_graph_walk_agrees_with_the_compiler_about_true_distance` stands on it: it builds
/// every shape of graph below and checks that the compiled ops say what this says, and that a
/// blend is reported approximate exactly when this says an operand is not a distance. If a new
/// node kind changes one and not the other, that test fails.
pub fn carries_true_distance(document: &Document, graph: &Graph, node: NodeId) -> bool {
    // **The rewrite is read first, as the compiler and the shader generator read it** (item 113,
    // cases 1 and 2). Bodies that meet face to face by construction are built as one, so the
    // mirror whose face would have been buried is no longer a mirror by the time anything is
    // asked about it. Without this the two answers part: `GraphField::build` would compile one
    // exact extrude while this walk still saw a joined mirror reaching its plane and called it
    // approximate, and `the_graph_walk_agrees_with_the_compiler_about_true_distance` is the test
    // that would say so. A document with no such mirror clones nothing and this is a scan.
    if let Some((merged, merged_graph)) = crate::with_shared_faces_merged(document, graph) {
        return true_distance_at(&merged, &merged_graph, node, 0);
    }
    true_distance_at(document, graph, node, 0)
}

/// [`carries_true_distance`] of what a slot reads, and `false` for a slot with nothing in it.
///
/// The third of the slot readers, beside [`operand_extent`] and [`operand_lipschitz`], and the one
/// whose answer is not the walk's own default: a *missing node* is `true`, because it stands for a
/// leaf this walk cannot see and nothing is drawn from it, while an empty slot is `false` because
/// there is no value a missing input could supply that does not overestimate a distance. Said once
/// here and once in `true_distance_at`'s recursion, which is one reading in two places rather than
/// two decisions.
pub fn operand_carries_true_distance(document: &Document, graph: &Graph, slot: &Operand) -> bool {
    match slot.source() {
        Some(id) => carries_true_distance(document, graph, id),
        None => false,
    }
}

fn true_distance_at(document: &Document, graph: &Graph, node: NodeId, depth: usize) -> bool {
    // The same guard the lipschitz walk uses, and for the same reason: a cyclic graph is
    // refused at compile time, but this runs before that and must not recurse forever.
    if depth > 256 {
        return false;
    }
    // Kept before the line below shadows it: the mirror arm asks a question about the *node*,
    // not about what it reads.
    let id = node;
    let Some(node) = graph.node(node) else {
        return true;
    };
    // **An empty slot is `false`, where the missing node above is `true`, and the difference is
    // the whole of this walk's part in the `ScalarField` contract.** That default is for a leaf
    // this walk cannot see, and claiming exactness for it is safe because nothing is drawn from
    // it. An empty slot is read by a *blend* that is in the document and on screen: if it claimed
    // true distance, [`approximate_blends`] would leave the blend out as though its radius were
    // honest, and the panel would quote a fillet radius the field has no way to produce. There is
    // no value a missing input can supply that does not overestimate a distance, so the weaker
    // answer is the only one available. `a_blend_with_an_empty_operand_is_never_called_exact`
    // holds it.
    let recurse = |slot: &Operand| match slot.source() {
        Some(id) => true_distance_at(document, graph, id, depth + 1),
        None => false,
    };
    // Whether a fillet rounds anything, asked of the document as the compiler asks it. One it
    // cannot resolve is a compile error there, and takes the weaker answer here.
    let rounds = |fillet: &fieldform_scene::Fillet| {
        document.resolve(&fillet.radius).is_none_or(|k| k > 0.0)
    };
    match node {
        // **Never, once it rounds** (blend-integrity-spec §3, ruled A). The blended value is
        // divided by 2^(1/n) so it cannot overstate across a shared face, and a divided
        // distance is a bound — so a blend of a blend is approximate and
        // [`approximate_blends`] names it. That is the known cost of A, and intended.
        Node::Blend { fillet, .. } if rounds(fillet) => false,
        // A radius of nought is a hard boolean: both operands, since min and max of a distance
        // and a bound is a bound.
        Node::Blend { a, b, .. } => recurse(a) && recurse(b),
        Node::Boolean { inputs, .. } => inputs.iter().all(|i| recurse(i)),
        // Never, whatever it is made of: where the two gradients oppose the value rises more
        // slowly than one per millimetre, so the reading is short. The same answer
        // `normalization_of` gives over the compiled ops, and the two are held together by
        // `the_graph_walk_agrees_with_the_compiler_about_true_distance`.
        Node::Morph { .. } => false,
        // A surface cell is a pseudo-distance whose magnitude is in no particular units — the
        // case this whole rule exists for. **A beam cell is not**: it is a union of capsules,
        // and a capsule is an exact signed distance, so on an even rectangular map the value is
        // the real distance and a blend against it is exact.
        //
        // The two conditions are both load-bearing and are the same two `Lattice::normalization`
        // tests for, which is what
        // `the_graph_walk_agrees_with_the_compiler_about_true_distance` holds together: a curved
        // map divides by the largest of three unequal frequencies and understates, and unequal
        // cell extents do the same on a rectangular one. The scene writes a single cell size into
        // all three, so the second cannot happen from here — said anyway, because this arm is
        // read as the statement of when it is true.
        Node::Lattice { cell, map, phase, .. } => {
            // A prism is a distance to its walls for the reason a beam is one to its struts.
            (cell.is_beam() || cell.is_prism())
                && matches!(map, fieldform_scene::CellMap::Rectangular { .. })
                // A graded beam is divided by its ramp's rate, so it is a bound like the rest.
                && phase.param().driver().is_none()
        }
        // The warp does not preserve distance -- that is what its stretch division is
        // compensating for -- so the honest answer is the weaker one.
        Node::Twist { .. } => false,
        // The other warp, and the same answer. A blend over an attracted body is allowed and
        // is listed by `approximate_blends`, which is what puts the note beside the radius.
        Node::Attract { .. } => false,
        // Not a distance to anything; it is reached only as an amount.
        Node::Ramp { .. } => false,
        // A lower bound always: the section planes converge on the inside of every bend and
        // the value is divided by that. A straight spine would be exact, and a straight spine
        // is what an extrude is for -- `sweep-spec.md` section 5.
        Node::Sweep { .. } => false,
        // Never: an interpolation of distances is not a distance, and a bent centerline
        // stretches the section planes besides. The leaf says `LowerBound` for both reasons.
        Node::Loft { .. } => false,
        // A fixed distance subtracts a constant and leaves the gradient alone; a driven one
        // does not, which is the cost the Ramp decision accepted.
        Node::Offset { input, distance } => distance.driver().is_none() && recurse(input),
        // A shell is two offsets by a fixed amount, so it carries through.
        Node::Shell { input, .. } => recurse(input),
        // A rounded fold divides exactly as a blend does, by W^(1/n), and answers as one.
        Node::Pattern { fillet, .. } if rounds(fillet) => false,
        // **A joined mirror is a union, and a union of touching operands buries their shared
        // face.** The reflection is a rigid motion and carries through; the join is
        // `min(f(p), f(mirror(p)))`, which reads nought along a face the body shares with its own
        // reflection while the solid there is deep (item 113, option 1). This arm used to sit
        // with the two below, which is how an undrafted extrude mirrored onto its own plane came
        // to claim an exact distance while reading nought 7.5 mm in — and a fillet against it
        // grooved with nothing said, where the project notes requires an approximate blend to announce
        // itself.
        //
        // Only a mirror no rewrite reaches gets here: `carries_true_distance` reads
        // `with_shared_faces_merged` first, as the compiler and the shader generator do, so the
        // mirrors built as one body are not mirrors by now.
        Node::Mirror { input, join, .. } => {
            if *join && crate::mirror_reaches_its_plane(document, graph, id) {
                false
            } else {
                recurse(input)
            }
        }
        // Rigid motions and similarities map distances to distances, only rescaled.
        Node::Pattern { input, .. } | Node::Transform { input, .. } => recurse(input),
        // Nothing this build could read, so nothing it can promise. It is refused at compile
        // time anyway; saying so here keeps the two answers the same.
        Node::Unknown(_) => false,
        // **Only straight out and upright.** A profile's distance is exact in its own plane,
        // and reading it in space is exact too: the `max` underestimates only at the rim, which
        // is the ordinary CSG crease this codebase has always counted as a distance.
        //
        // A lean and a draft each break that, and each says so for itself -- `SkewExtrusion`
        // and a drafted `Extrusion` both return `LowerBound`. This arm said `true` for every
        // extrude regardless, so the walk and the compiler disagreed the moment a sweep was
        // aimed or an angle typed: 32 degrees of lean reads as little as half the true
        // distance, and a fillet against it comes out about twice the radius asked for **with
        // nothing marking it approximate**. That is precisely the silent failure the surviving
        // half of D8 exists to prevent, and it survived because the agreement test had eight
        // scenes and not one of them was an extrude.
        Node::Extrude { sweep, draft, sides, rims, .. } => {
            let upright = |draft: &fieldform_scene::ParamRef| {
                document
                    .resolve(draft)
                    // Unresolvable means unknown, and unknown takes the weaker answer.
                    .is_some_and(|d| d == 0.0)
            };
            // Both sides upright, where there are two: the field divides by the steeper.
            let other_upright = match sides {
                fieldform_scene::Sides::Two { draft, .. } => upright(draft),
                _ => true,
            };
            // And a hard rim: a rounded one is the blend's bound across the section and the
            // slab, which reads short inside its reach as every fillet does. Resolved, as the
            // compile resolves it, so a reach named and nought agrees with the leaf.
            let hard_rims = document.resolve(&rims.radius).is_some_and(|r| r <= 0.0);
            sweep.is_exact() && upright(draft) && other_upright && hard_rims
        }
        // A full revolve is exact outright: the nearest point lies in the query's own
        // half-plane. A partial one is a bound inside its material, where the depth is held
        // against each cap's plane rather than the cap (`RevolvedWedge`), and says so.
        Node::Revolve { angle, sides, .. } => {
            matches!(revolve_wedge(document, angle, sides), Ok((_, total)) if total >= 360.0 - 1e-9)
        }
        // A box's or a cylinder's edge is exact rounded at G1 -- the inner solid grown by a ball
        // -- and a bound at G2 or chamfered, as `EdgedCuboid::normalization` says; hard is the
        // solid it always was.
        Node::Cuboid { edge, .. } | Node::Cylinder { edge, .. } => {
            edge.continuity == fieldform_scene::Continuity::G1
                || document.resolve(&edge.radius).is_some_and(|r| r <= 0.0)
        }
        // A cone's round *and* chamfer are exact, read as a convex polygon in its meridian
        // (`EdgedCone`); only its G2 is a bound.
        Node::Cone { edge, .. } => {
            edge.continuity != fieldform_scene::Continuity::G2
                || document.resolve(&edge.radius).is_some_and(|r| r <= 0.0)
        }
        Node::Cage(_)
        | Node::Mesh(_)
        | Node::Sphere { .. }
        | Node::SkewedBox { .. }
        | Node::Torus { .. }
        | Node::Capsule { .. }
        | Node::Plane { .. }
        | Node::Point { .. }
        | Node::Pipe { .. } => {
            let _ = document;
            true
        }
    }
}

/// The blends whose radius will not be exactly the radius asked for.
///
/// **Why these are allowed at all, having once been refused.** `smooth_min` mixes by
/// magnitude, so an operand whose value understates the true distance pulls the seam toward
/// itself and the fillet comes out *larger* than the number typed. That was reason enough to
/// refuse the combination outright: the geometry looks fine, is the wrong shape, and there is
/// nothing to notice later.
///
/// Two things changed the answer. The error is **bounded and one-directional** — a lattice
/// already divides by its own gradient bound (see `Lattice::distance`), so its value is a
/// lower bound in millimetres rather than an arbitrary isovalue, and understating a distance
/// can only make a fillet bigger, never smaller or absent. For a gyroid the bound is √3, so the
/// worst case is a fillet about 1.7 times the radius asked for, in the places where the
/// gradient falls furthest short. And "nothing to notice later" was a fact about the
/// *interface* rather than about the mathematics: a blend named here is one the panel marks as
/// approximate where the radius is typed.
///
/// So a caller must not quietly ignore this list. These blends are honest only while something
/// says so at the point the number is entered.
///
/// **A blend with a hole below it is left out, and that is not a claim that its radius is
/// honest.** It has no radius to speak of: the block cannot be evaluated at all, so what the panel
/// says about it is that the slot is empty (`notebook-spec.md` §7.2), and marking it
/// `≈ approximate` beside a number would be a second, wronger story about the same block —
/// a radius that will not be the radius asked for is still a radius, and this is not one.
/// [`carries_true_distance`] answers `false` for it either way, which is the answer the
/// [`crate::ScalarField`] contract needs and is what keeps the omission from reading as
/// exactness. `Graph::incomplete` is closed downstream, so this covers a blend fed by a hole
/// further below as well as one holed in its own slot.
pub fn approximate_blends(document: &Document, graph: &Graph) -> Vec<NodeId> {
    // The same rewrite [`carries_true_distance`] reads, once for the whole list rather than once
    // per operand below: a blend keeps its id through the merge, so the ids answered are the ids
    // asked about either way.
    if let Some((merged, merged_graph)) = crate::with_shared_faces_merged(document, graph) {
        return approximate_blends(&merged, &merged_graph);
    }
    // Derived from the edges each time it is asked, like `roots` and for the same reason.
    let holed = graph.incomplete();
    let mut out: Vec<NodeId> = graph
        .nodes()
        .filter(|(id, _)| !holed.contains(id))
        .filter_map(|(id, n)| match n {
            Node::Blend { a, b, .. }
                if !operand_carries_true_distance(document, graph, a)
                    || !operand_carries_true_distance(document, graph, b) =>
            {
                Some(id)
            }
            // **A rounded fold is a fillet and owes the same announcement.** The obligation
            // travels with `Fillet`: wherever a radius is typed, a radius that will not be the
            // radius asked for has to say so. A pattern rounding copies of a lattice is exactly
            // that case.
            Node::Pattern { input, fillet, .. }
                if !fillet.is_plainly_hard()
                    && !operand_carries_true_distance(document, graph, input) =>
            {
                Some(id)
            }
            _ => None,
        })
        .collect();
    // Sorted, because the graph is a hash map and a message that reshuffles itself between
    // frames reads as flicker rather than as information.
    out.sort_by_key(|id| id.raw());
    out
}

/// What a driven offset divides its value by, from the graph alone.
///
/// `|∇(F − d)| ≤ |∇F| + |∇d|`, and both terms come from [`graph_lipschitz`]. One for a fixed
/// distance, which is the same as not dividing at all.
pub fn driven_offset_divisor(document: &Document, graph: &Graph, offset: NodeId) -> f64 {
    let Some(Node::Offset { input, distance }) = graph.node(offset) else {
        return 1.0;
    };
    let Some(driver) = distance.driver() else {
        return 1.0;
    };
    (operand_lipschitz(document, graph, input) + graph_lipschitz(document, graph, driver)).max(1e-9)
}

/// What a pipe with a driven radius divides its value by, from the graph alone.
///
/// **A pipe of a varying radius is its spine, offset by that radius.** `F(p) = d(p) − R(p)`,
/// where `d` is the distance to the curve and is 1-Lipschitz by construction — so this is
/// `driven_offset_divisor` with the one term known in advance, and the field is built as
/// exactly that offset rather than as a second thing that happens to agree.
///
/// One for a fixed radius, which is the same as not dividing at all.
pub fn driven_pipe_divisor(document: &Document, graph: &Graph, pipe: NodeId) -> f64 {
    let Some(Node::Pipe { radius, .. }) = graph.node(pipe) else {
        return 1.0;
    };
    let Some(driver) = radius.driver() else {
        return 1.0;
    };
    (1.0 + graph_lipschitz(document, graph, driver)).max(1e-9)
}

/// What a blend with a driven radius divides its value by, from the graph alone.
///
/// The value is `h(a(p), b(p), k(p))`, the blend's reading at radius `k`. In `(a, b)` it rises
/// at most one per millimetre whatever `k` is ([`blend_bound`] is a max-norm distance, and the
/// other forms are divided to the same); in `k` it rises at most [`blend_radius_rate`] per
/// millimetre of radius. So the pair rises at `1 + c · L`, `L` the ramp's own rate, and that is
/// the divisor -- the offset's arithmetic with the radius's rate in place of the distance's.
/// One for a typed radius, which is the same as not dividing at all.
pub fn driven_blend_divisor(document: &Document, graph: &Graph, blend: NodeId) -> f64 {
    let Some(Node::Blend { fillet, .. }) = graph.node(blend) else {
        return 1.0;
    };
    let Some(driver) = fillet.radius.driver() else {
        return 1.0;
    };
    let tapers = taper_ratio(document, graph, blend) > 0.0;
    (1.0 + blend_radius_rate(fillet.exponent(), tapers) * graph_lipschitz(document, graph, driver))
        .max(1e-9)
}

/// The largest radius a fillet can have: its number, or the top of the ramp that drives it.
///
/// What every graph walk that reached for `resolve(&fillet.radius)` reads instead, because
/// `Document::resolve` answers `None` for a driven reference and `unwrap_or(0.0)` would have
/// made a graded fillet reach nothing -- an extent that misses the rounding, a bake read short
/// of it, a step budget that buys no steps for it. Nought for a ramp whose ends cannot be
/// resolved, which the compile refuses by name.
pub fn fillet_reach(document: &Document, graph: &Graph, fillet: &fieldform_scene::Fillet) -> f64 {
    match fillet.radius.driver() {
        None => document.resolve(&fillet.radius).unwrap_or(0.0),
        Some(driver) => match graph.node(driver) {
            Some(Node::Ramp { out_min, out_max, .. }) => {
                let lo = document.resolve(out_min).unwrap_or(0.0);
                let hi = document.resolve(out_max).unwrap_or(0.0);
                lo.max(hi).max(0.0)
            }
            _ => 0.0,
        },
    }
}

/// What a lattice with a driven phase divides its value by, from the graph alone.
///
/// The value is `V(p, t(p))`, the fixed lattice of size `t` read at `p`. Each fixed lattice is
/// already one-Lipschitz in `p`, so the pair can rise at `1 + sup|dV/dt| * L`, with `L` the
/// ramp's rate -- and `sup|dV/dt|` is the phase's own business:
///
/// - **Strut.** `d - R`: one. The offset's arithmetic, and the pipe's.
/// - **Sheet.** `(|F| - t/2 * g * w) / (B * w)`, so `g / 2B` -- the cell's typical gradient over
///   its bound, halved. Never above a half, and 0.44 for a gyroid: a graded wall costs less than
///   the offset that used to stand in for it.
/// - **Solid.** `(F - iso(density)) / (B * w)`, so `iso' / (B * w)` with `iso'` the steepest the
///   density table gets and `w` the slowest frequency the map is ever divided by. The one case
///   with a length in it, because a density is a fraction and the ramp's rate is per millimetre.
///
/// One for a typed number, which is the same as not dividing at all.
pub fn driven_lattice_divisor(document: &Document, graph: &Graph, lattice: NodeId) -> f64 {
    let Some(Node::Lattice { cell, map, phase, .. }) = graph.node(lattice) else {
        return 1.0;
    };
    let Some(driver) = phase.param().driver() else {
        return 1.0;
    };
    let rate = graph_lipschitz(document, graph, driver);
    let resolve = |r: &fieldform_scene::ParamRef| document.resolve(r).unwrap_or(0.0);
    let kind = crate::CellKind::from(*cell);
    let moves = match (phase, kind) {
        (fieldform_scene::LatticePhase::Sheet { .. }, crate::CellKind::Tpms(kind)) => {
            0.5 * kind.argument_gradient().norm() / kind.gradient_bound()
        }
        (fieldform_scene::LatticePhase::Solid { .. }, crate::CellKind::Tpms(kind)) => {
            // The frequency every map divides by at least: its constant axes. A cylindrical
            // map's circumferential term only ever adds to the largest of the three.
            let tau = std::f64::consts::TAU;
            let slowest = match map {
                fieldform_scene::CellMap::Rectangular { cell } => tau / resolve(cell),
                fieldform_scene::CellMap::Cylindrical { cell_radius, cell_height, .. } => {
                    (tau / resolve(cell_radius)).max(tau / resolve(cell_height))
                }
                fieldform_scene::CellMap::Spherical { cell_radius, .. } => {
                    tau / resolve(cell_radius)
                }
                fieldform_scene::CellMap::Conformal { depth, .. } => tau / resolve(depth),
            };
            if !(slowest.is_finite() && slowest > 0.0) {
                return f64::INFINITY;
            }
            kind.density_slope() / (kind.gradient_bound() * slowest)
        }
        // `d - t/2`: a prism's wall comes off a distance, half of it each side, and a Voronoi
        // foam's wall comes off its measure the same way. A Voronoi beam's radius, like every
        // other strut radius, comes off whole and falls through to one.
        (fieldform_scene::LatticePhase::Sheet { .. }, crate::CellKind::Prism(_))
        | (fieldform_scene::LatticePhase::Sheet { .. }, crate::CellKind::Stochastic { .. }) => 0.5,
        // A strut, and any pairing the compile refuses anyway.
        _ => 1.0,
    };
    (1.0 + moves * rate).max(1e-9)
}

/// How much longer a march through this graph takes than through one that reports true distance.
///
/// **Every divisor is paid for in steps.** A driven morph or offset divides its value so the
/// result never overestimates, which is what makes it traceable at all — but a value divided by
/// 8.5 is a value 8.5 times smaller, and a sphere-tracer steps by what it reads. So the same
/// number that makes the field honest makes each step shorter, and a fixed step budget then
/// covers 8.5 times less of the ray.
///
/// That is not a slower picture, it is a wrong one: a ray that runs out of steps is
/// indistinguishable from a ray that hit nothing, so the surface goes missing. It shows first
/// where a ray runs closest to the surface for longest — along the silhouette, at grazing
/// incidence — which is exactly where it was reported.
/// `fieldform-field/examples/grazing_rays.rs` measures it: at ×8.5 the ray 0.1 mm inside the
/// limb needs 2791 steps and had 512.
///
/// **The product along a chain, not the largest one.** Each divisor makes its own output
/// 1-Lipschitz again, so they do not compound in the *value*; they do compound in the *step*,
/// because each one is a further constant factor between what is read and how far it is safe to
/// go. A morph of two morphs is that much shorter-stepping again.
pub fn step_multiplier(document: &Document, graph: &Graph) -> f64 {
    step_multiplier_with(document, graph, &[])
}

/// The same, told what each driven morph was actually divided by.
///
/// The budget has to be bought against the divisor the field really uses, or it buys the wrong
/// number of steps: too few and the silhouette goes missing, too many and the viewport crawls.
/// See [`GraphField::morph_divisors`] — on the reported scene the box bound says ×42.5 and the
/// intervals say ×24.5, which is the difference between 43520 steps and 25088.
pub fn step_multiplier_with(
    document: &Document,
    graph: &Graph,
    morph_divisors: &[(NodeId, f64)],
) -> f64 {
    divided_along_chains(graph, &|id, node| {
        own_divisor(document, graph, id, node, morph_divisors)
    })
}

/// How far short of the distance a reading taken just outside the surface falls, over the
/// whole graph: what the shading multiplies a reading by before treating it as a distance
/// (`Camera::value_scale` in `fieldform-gpu`).
///
/// [`step_multiplier_with`]'s number but for one thing. **A sweep read in its tube** reads its
/// walls truly and divides only its caps read from inside ([`super::sweep_reads_true`],
/// `sweep-spec.md` §4). The step budget still buys steps for those -- a sweep subtracted from a
/// body is marched through its inside -- but the shading samples outside the surface and never
/// meets them, and scaling a true reading by the stretch would read every crease on the sweep
/// as open: at the glyph's stretch of 14 a crease would have to be fourteen times nearer to
/// darken at all. Within the frame's allowance of the surface -- 0.14 mm on the glyph walked at
/// two degrees -- the reading is the divided one again, which is far inside the first sample
/// the occlusion takes, 1.2 % of the scene's radius out.
/// Whether any block of `graph` is of a kind that can divide its reading -- whatever its numbers
/// are now.
///
/// **By structure alone, because a number is something a drag changes without regenerating the
/// shader.** The arms are [`own_divisor`]'s that can answer more than one: every extrude, sweep,
/// loft, blend, pattern, twist, morph and attractor, since each divides by a number a drag can
/// move (an extrude once its draft or lean is not nought, a pattern once its fillet is not); and an
/// offset, pipe or lattice only when something drives it, which is a node it names. What reads this
/// is the march's `SETTLES` (backlog item 114), which a value-scale test broke the rule for -- the
/// shader recompiled while a blend's radius was dragged.
pub fn may_divide(graph: &Graph) -> bool {
    graph.nodes().any(|(_, node)| match node {
        Node::Extrude { .. }
        | Node::Sweep { .. }
        | Node::Loft { .. }
        | Node::Blend { .. }
        | Node::Pattern { .. }
        | Node::Twist { .. }
        | Node::Morph { .. }
        | Node::Attract { .. } => true,
        Node::Offset { distance, .. } => distance.driver().is_some(),
        Node::Pipe { radius, .. } => radius.driver().is_some(),
        Node::Lattice { phase, .. } => phase.param().driver().is_some(),
        _ => false,
    })
}

pub fn value_scale_with(
    document: &Document,
    graph: &Graph,
    morph_divisors: &[(NodeId, f64)],
) -> f64 {
    divided_along_chains(graph, &|id, node| match node {
        // A rim's bound moves no faster than the readings it joins (`Rim::join`), so it costs
        // no steps the sweep did not.
        Node::Sweep { outline, spine, section, law, rims: _ }
            if super::sweep_reads_true(document, outline, *spine, section, law) =>
        {
            1.0
        }
        // A held section's reading is shortened by its lean as well as divided by its stretch
        // (`Swept::distance`), and the shading puts both back: see `sweep_lean`.
        Node::Sweep { outline, spine, section, .. } => {
            own_divisor(document, graph, id, node, morph_divisors)
                / super::sweep_lean(document, outline, *spine, section)
        }
        _ => own_divisor(document, graph, id, node, morph_divisors),
    })
}

/// The largest product of `own` along any chain from a root down, which is how divisions
/// compound in what is measured against a value. See [`step_multiplier`].
fn divided_along_chains(graph: &Graph, own: &dyn Fn(NodeId, &Node) -> f64) -> f64 {
    fn chain(
        graph: &Graph,
        id: NodeId,
        own: &dyn Fn(NodeId, &Node) -> f64,
        known: &mut std::collections::HashMap<NodeId, f64>,
    ) -> f64 {
        if let Some(seen) = known.get(&id) {
            return *seen;
        }
        // Standing in for the answer while it is being computed, so a cycle terminates. A cyclic
        // graph does not compile, but this runs before anything has checked.
        known.insert(id, 1.0);
        let Some(node) = graph.node(id) else {
            return 1.0;
        };
        let deepest = node
            .inputs()
            .iter()
            .map(|input| chain(graph, *input, own, known))
            .fold(1.0f64, f64::max);
        let total = (own(id, node) * deepest).max(1.0);
        known.insert(id, total);
        total
    }
    let mut known = std::collections::HashMap::new();
    graph
        .roots()
        .into_iter()
        .map(|root| chain(graph, root, own, &mut known))
        .fold(1.0f64, f64::max)
}

/// What this node alone divides its value by, its inputs' own divisions left out.
///
/// **One table, because two walks read it and a third was reading nothing.** The step budget
/// multiplies these along a chain ([`step_multiplier`]); the extent walk carries a threshold down
/// through them ([`extent_below`]), since a node that divides by `D` reads below `g` wherever its
/// input reads below `D · g`. They were one `match` inside the step walk, so the extent walk had
/// no way to ask and grew its boxes as though nothing divided —
/// `tile-mask-design-01941c5.md` §1.8 R2.
///
/// `morph_divisors` is what a compiled field measured, where there is one; a morph not named in it
/// falls back to the box bound, which is looser. See [`GraphField::morph_divisors`].
///
/// **Not a Lipschitz constant.** Every one of these outputs is 1-Lipschitz again — that is what
/// the division buys — so these do not compound in the *value*, which is what `graph_lipschitz`
/// reports and why it answers 1.0 for all of them. They compound in everything measured *against*
/// the value: the step it is safe to take, and the distance a box has to grow by.
pub(crate) fn own_divisor(
    document: &Document,
    graph: &Graph,
    id: NodeId,
    node: &Node,
    morph_divisors: &[(NodeId, f64)],
) -> f64 {
    match node {
            // **Never below two, even at a constant amount.** A morph understates distance
            // whatever its divisor: the gradient is `(1 − t)·∇A + t·∇B`, so where the two
            // operands' gradients partially oppose the sum is shorter than either, and at
            // `t = 0.5` with exactly opposed gradients it is nought — a field that has stopped
            // changing while the distance still does. No finite factor bounds that, so this is
            // a mitigation chosen by measurement rather than a bound derived: at 512 steps the
            // reported scene at a constant half loses eleven pixels and at 1024 it loses none,
            // and `driven_morph_budget` measures the extra headroom as free on a true-distance
            // scene (5.8 ms flat from 512 to 16384) and within noise on the morphs.
            Node::Morph { .. } => morph_divisors
                .iter()
                .find(|(node, _)| *node == id)
                .map(|(_, d)| *d)
                .unwrap_or_else(|| driven_morph_divisor(document, graph, id))
                .max(2.0),
            Node::Offset { .. } => driven_offset_divisor(document, graph, id),
            // A pipe of a varying radius divides by the same kind of bound an offset does, so
            // it costs the same steps. Left out, the tracer would step by what the undivided
            // field would have read and lose the surface at grazing incidence -- the one cause
            // that has already produced a lost silhouette, a black morph and an unstable
            // attractor.
            Node::Pipe { .. } => driven_pipe_divisor(document, graph, id),
            // A graded lattice divides by its ramp's rate, scaled by how hard the phase's
            // number moves the value; one when nothing drives it.
            Node::Lattice { .. } => driven_lattice_divisor(document, graph, id),
            // A sweep divides by the convergence of its section planes on the inside of its
            // tightest bend, so it costs steps the way the attractor below does.
            Node::Sweep { outline, spine, law, .. } => sweep_stretch(document, outline, *spine, law),
            // A loft divides by its interpolation's rate and its centerline's bend together,
            // so it costs steps the way a sweep does. One for a loft that cannot be built,
            // which the compile refuses by name.
            Node::Loft { sections, spine, smooth, rims: _ } => {
                loft_stretch(document, sections, *spine, *smooth)
            }
            // **An attractor divides by its stretch, so it costs steps like the rest.** `Op::
            // Attract` returns `inner / stretch(strength)`, which is what keeps the warped field
            // a lower bound -- and a value divided is a value that reads short, so the tracer
            // steps short and the shading reads the surface as crowded. Reported as the viewport
            // breaking up while an attractor's strength was being dragged, which is the third
            // consumer of one cause: the morph lost its silhouette, then shaded itself black,
            // and this is the same arithmetic in a third place.
            Node::Attract { strength, .. } => {
                let s = document.resolve(strength).unwrap_or(0.0);
                crate::attract::stretch(s).max(1.0)
            }
            // **A rounded join divides too, and was the fourth consumer of the one cause.**
            // Blend integrity made a rounded blend divide by 2^(1/n) and a rounded pattern by
            // W^(1/n) -- 2.45 for six copies at G1 -- and nothing here was told. Reported as
            // "if you turn on fillet the shading goes dark": a pattern's fillet typed as
            // 0.001 mm, a join nobody could see, darkened the whole body, because the shading
            // march was scaling a value 2.45 times shorter than it believed and read every
            // surface as crowded. The same number is owed to the step budget, where the
            // symptom would have been a lost silhouette instead.
            // At the largest radius a ramp can give, where one drives it, and by the driven
            // divisor on top: the two divisions compound in the step exactly as a morph of a
            // morph's do.
            Node::Blend { fillet, .. } => {
                let k = fillet_reach(document, graph, fillet);
                blend_divisor(k, fillet.exponent(), taper_ratio(document, graph, id))
                    * driven_blend_divisor(document, graph, id)
            }
            Node::Pattern { input, fillet, .. } => {
                let k = document.resolve(&fillet.radius).unwrap_or(0.0);
                if k > 0.0 {
                    // **One copy when the layout will not resolve, because this answer is about
                    // to be discarded.** A pattern that fails is refused by `build`, so the field
                    // never exists and no divisor is read off it. What matters is that this cannot
                    // return a NaN into a step multiplier, which would blank the march a long way
                    // from the cause — so the fallback is the one layout that is always safe.
                    // **And one copy for an empty slot, which is the same case said twice.** A
                    // fold of nothing is refused by `build` as a fold with no layout is, so this
                    // answer is discarded either way; what matters is the NaN that must not
                    // reach a step multiplier.
                    match (input.source(), PatternLayout::of(document, graph, id)) {
                        (Some(input), Ok(layout)) => pattern_join_divisor(
                            pattern_reach_in(document, graph, input, &layout, k),
                            layout.total as usize,
                            layout.left_out(),
                            fillet.exponent(),
                            taper_ratio(document, graph, id),
                        ),
                        _ => pattern_join_divisor(
                            0,
                            1,
                            0,
                            fillet.exponent(),
                            taper_ratio(document, graph, id),
                        ),
                    }
                } else {
                    1.0
                }
            }
            // **A twist divides, and this walk was not told for as long as the twist existed.**
            // `Op::Twist` returns `inner / stretch`, exactly as `Op::Attract` returns
            // `inner / stretch(strength)` -- and the arm above it was written for the attractor
            // while this one fell through to the `1.0` below. So the step budget was bought
            // against a value up to five times shorter than the field reads on the twist cases in
            // `extent_holds_combiners.rs`, which is the same arithmetic that lost the morph's
            // silhouette and darkened the pattern's shading, in a fifth place.
            Node::Twist { .. } => twist_stretch(document, graph, id),
            // **An extrude divides by the secant of its lean and the slope of its draft, and
            // this walk was not told** -- the sixth place for the one arithmetic. A leaning,
            // drafted extrude bought its steps against a value up to twice what the field
            // reads, and lost rays at grazing incidence: found when a more honest bound on it
            // took six per cent of a scene's hits away in `gpu_matches_cpu`, with the budget
            // still bought as if it divided by nothing.
            Node::Extrude { .. } => extrude_stretch(document, graph, id),
            // Everything else either divides by nothing or has already paid for what it does,
            // which is the same list `graph_lipschitz` states arm by arm.
        _ => 1.0,
    }
}

/// What an extrude's value is divided by to stay a lower bound: `extrude_divisor` of its
/// lean and the steeper of its drafts, asked of the graph so that the step budget is bought
/// against the number the field divides by. One for anything that is not an extrude, and for
/// an extrude whose numbers cannot be read, which the compile refuses by name.
pub(crate) fn extrude_stretch(document: &Document, graph: &Graph, id: NodeId) -> f64 {
    let Some(Node::Extrude { draft, sides, .. }) = graph.node(id) else { return 1.0 };
    let resolve = |r: &fieldform_scene::ParamRef| document.resolve(r).unwrap_or(0.0);
    let slope = |draft: &fieldform_scene::ParamRef| resolve(draft).to_radians().tan().abs();
    let steepest = match sides {
        fieldform_scene::Sides::Two { draft: other, .. } => slope(draft).max(slope(other)),
        _ => slope(draft),
    };
    let along = sweep_lean_degrees(document, graph, id).unwrap_or(0.0).to_radians().cos();
    crate::profile::extrude_divisor(along, steepest).max(1.0)
}

/// What a twist's value is divided by to stay a lower bound.
///
/// **One reader, because three places need this number and two of them did not have it.** A twist
/// turns a point by `rate · z`, so two points at the same height and radius `r` are carried apart
/// by `r · rate` per millimetre of height. That grows without limit with radius, so the bound is
/// taken from the body's *own* extent rather than from the rate alone — which is why this reads
/// the graph and is not arithmetic at the call site.
///
/// `Op::Twist` divides by it, and that division is what keeps the warped field a lower bound.
/// Neither of the two walks that have to know about a division knew about this one:
/// `step_multiplier` bought the step budget against the undivided value, and `extent_of` grew a
/// blend's box by the radius typed where the operand's value reads five times short of its
/// distance (`tile-mask-design-01941c5.md` §1.8 R2).
///
/// **One for an unbounded body**, which is the case `CompileError::TwistNeedsBounds` refuses, so
/// nothing is ever drawn from the answer; one for an empty slot, which has no body to bound and is
/// refused for that instead; and one for anything that is not a twist.
pub(crate) fn twist_stretch(document: &Document, graph: &Graph, twist: NodeId) -> f64 {
    let Some(Node::Twist { input, degrees_per_mm, frame }) = graph.node(twist) else {
        return 1.0;
    };
    let numbers = |r: &fieldform_scene::ParamRef| Some(document.resolve(r).unwrap_or(0.0));
    let rate = numbers(degrees_per_mm).unwrap_or(0.0).to_radians();
    let Some((lo, hi)) = operand_extent(document, graph, input).as_box() else {
        return 1.0;
    };
    let Ok((translation, rotation)) = resolve_frame(document, graph, frame, &numbers) else {
        return 1.0;
    };
    let r_max = reach_across_axis((lo, hi), translation, rotation);
    (1.0 + (r_max * rate).powi(2)).sqrt().max(1.0)
}

/// What a driven morph's value has to be divided by to stay a distance.
///
/// `∇M` is `(1 − t)·∇A + t·∇B + (B − A)·∇t`, so the value can rise at `1 + sup|B − A|·L_t` per
/// millimetre. Both operands are already 1-Lipschitz, which is the `1`; the rest is what the
/// ramp does to the gap between them.
///
/// **`sup|B − A|` from the extents, which is looser than `morph-spec.md` §4.3 wants and is the
/// bound both paths can actually share.** The spec prefers the two operands' intervals over the
/// enclosing box, and that is tighter — but it is read over a compiled op list, and `generate`
/// has only the graph. Compiling a field there to ask would bake every profile and cage in it,
/// once per shader build. A number the CPU and the shader compute differently is a surface in
/// two places, so agreement wins and the tighter bound is named rather than taken.
///
/// The bound: over the box `U` enclosing both bodies, each field is 1-Lipschitz and has its zero
/// set inside its own extent, which is inside `U` — so no point of `U` is further from that zero
/// set than `U`'s own diagonal, and `|A| ≤ diam(U)`. The same for `B`, so `|B − A| ≤ 2·diam(U)`.
///
/// Returns 1 for a morph whose `t` is constant, and for anything that is not a morph.
pub fn driven_morph_divisor(document: &Document, graph: &Graph, morph: NodeId) -> f64 {
    let Some(Node::Morph { a, b, t }) = graph.node(morph) else {
        return 1.0;
    };
    let Some(driver) = t.driver() else {
        return 1.0;
    };
    let Some((lo, hi)) = operand_extent(document, graph, a)
        .union(operand_extent(document, graph, b))
        .as_box()
    else {
        // Refused at compile by `MorphNeedsBounds`, so this is the belt to those braces. An empty
        // slot lands here as well, having no box of its own, and is refused by name instead.
        return 1.0;
    };
    let spread = 2.0 * (hi - lo).norm();
    (1.0 + spread * graph_lipschitz(document, graph, driver)).max(1.0)
}

/// How fast a ramp's output can change per millimetre of space, read over the compiled ops.
///
/// The same number `graph_lipschitz` reports for a `Node::Ramp`, reached the other way: this is
/// used while the op list is being bounded, where there is no graph node to hand.
pub(crate) fn ramp_lipschitz(ops: &[Op], index: usize) -> f64 {
    let Op::Ramp { input, in_span, out_span, shape, .. } = &ops[index] else {
        return 0.0;
    };
    if in_span.abs() < 1e-12 {
        return f64::INFINITY;
    }
    (out_span / in_span).abs() * shape.slope_factor() * lipschitz_of(ops, *input)
}

/// Why this node's driven parameter is refused, if it has one that is not allowed.
///
/// Named per node kind rather than per parameter, because the message a person needs is which
/// block would not take it — the block has one driveable input at most today, so naming the
/// block names the input.
pub(crate) fn driven_refusal(node: &Node) -> Option<CompileError> {
    let named = match node {
        // The one that is allowed. `Node::Offset`'s distance is the case the mechanism was
        // built against: `F(p) − d(p)` is the simplest arithmetic there is, so the bound is
        // `1 + L` and nothing else is happening to obscure whether it holds.
        // Allowed. An offset was the case the mechanism was built against; a morph's amount is
        // the second, and it is the mode that makes a morph useful on bodies that are not in the
        // same place — nowhere is `t` near a half while both bodies are far away.
        // And a pipe's radius, which is the third. `d(p) − R(p)` is the offset's own
        // arithmetic with the spine standing in for the body: a pipe of a varying radius *is*
        // its own curve offset by that radius, which is why it costs one bound and no new
        // machinery. Backlog item 2.
        // And a blend's radius, the fourth (`round-chamfer-spec.md` §9.1): the blend's zero set
        // moves by at most the radius's own change, so the bound is `1 + s` again and the
        // reading is divided by it -- see [`driven_blend_divisor`].
        Node::Offset { .. }
        | Node::Ramp { .. }
        | Node::Morph { .. }
        | Node::Pipe { .. }
        | Node::Blend { .. } => return None,
        Node::Sphere { .. } => "a sphere's centre or radius",
        Node::Cuboid { .. } => "a cuboid's centre or size",
        Node::SkewedBox { .. } => "a skewed box's centre or edges",
        Node::Cylinder { .. } => "a cylinder's ends or radius",
        Node::Cone { .. } => "a cone's ends or radii",
        Node::Torus { .. } => "a torus's centre or radii",
        Node::Capsule { .. } => "a capsule's ends or radii",
        Node::Plane { .. } => "a plane's origin",
        Node::Extrude { .. } => "an extrude's height or sweep",
        Node::Revolve { .. } => "a revolve's frame",
        Node::Point { .. } => "a point's position",
        // The phase's own number may follow a ramp -- a graded wall, density or strut -- and
        // nothing else on the block may. A cell size that varies is not a number that varies:
        // it is a coordinate warp, and `graded-lattice-spec.md` section 5 says why that is a
        // different problem rather than a bigger one.
        Node::Lattice { phase, map, .. } => {
            // The map's own list, which `Node::param_refs` also reads, rather than a second copy of
            // which numbers a map has.
            if map.params().any(|r| r.driver().is_some()) {
                return Some(CompileError::ParameterCannotBeDriven("a lattice's cell size"));
            }
            let own = phase.param();
            return node
                .param_refs()
                .iter()
                .any(|r| !std::ptr::eq(*r, own) && r.driver().is_some())
                .then_some(CompileError::ParameterCannotBeDriven(
                    "a lattice's mid-surface offset, jitter or frame",
                ));
        }
        Node::Shell { .. } => "a shell's thickness",
        Node::Twist { .. } => "a twist's rate or origin",
        Node::Attract { .. } => "an attractor's strength or falloff radius",
        Node::Transform { .. } => "a placement's translation, rotation or scale",
        Node::Mirror { .. } => "a mirror's origin",
        // **"or count", because the count is in `param_refs` now and this is what catches it.** The
        // blanket test below asks whether any of a node's parameters has a driver, so a ramp wired
        // into a copy count is already refused here — correctly, since a ramp's value varies with the
        // evaluation point and a number of copies cannot. What was wrong was the wording: a refusal
        // reading "a pattern's step or origin" sends somebody to look at the step when it was the
        // count. `fieldform_scene::CountError::Driven` says the same thing at more length wherever a
        // count is resolved on its own.
        Node::Pattern { .. } => "a pattern's step, origin or count",
        // The law's numbers are amounts, and amounts get ramps by the 2026-09-02 ruling -- but
        // a ramp is a *spatial* field and the law runs along the *path*, so a driven taper or
        // twist would be two different things at one station. Refused in V1; a law driven by a
        // curve is the guide rail (`sweep-v2-spec.md` §6). A wall's thickness goes through here
        // too, as an extrude's does.
        Node::Sweep { .. } => "a sweep's taper, twist, extent or thickness",
        Node::Loft { .. } => "a loft's sections",
        Node::Boolean { .. } | Node::Cage(_) | Node::Mesh(_) | Node::Unknown(_) => return None,
    };
    node.param_refs()
        .iter()
        .any(|r| r.driver().is_some())
        .then_some(CompileError::ParameterCannotBeDriven(named))
}

/// The largest Lipschitz constant of one op's subtree.
///
/// Free rather than a method because compilation needs it *while* the op list is being built —
/// a driven offset has to know its divisor before it can be constructed, and at that moment
/// there is no `GraphField` to ask.
pub(crate) fn lipschitz_of(ops: &[Op], index: usize) -> f64 {
    match &ops[index] {
        Op::Leaf(field) => field.lipschitz(),
        // Already divided by the sum of both rates, as a driven offset is.
        Op::Lattice { .. } => 1.0,
        // The steeper operand's, and that is correct only *because* `blend` divides. Inside
        // the zone the raw form's gradient is the two operand gradients weighted by a unit
        // vector in the conjugate norm, which reaches 2^(1/n) times the steeper one where they
        // are parallel; `blend_divisor` takes exactly that factor back out. Without the
        // division this arm understated the rate — blend-integrity-spec §2.2. The rounded
        // pattern below is the same argument with W^(1/n).
        // The max-norm distance to the zero set rises no faster than the faster operand; a
        // radius that varies moves the zero set too, by at most its own change, and the
        // divisor is `1 +` that rate -- so the sum over the divisor, which is the operands'
        // own rate again when the radius is fixed.
        Op::Blend { a, b, divisor, .. } => {
            (lipschitz_of(ops, *a).max(lipschitz_of(ops, *b)) + (divisor - 1.0)) / divisor
        }
        Op::Boolean { inputs, .. } => inputs
            .iter()
            .map(|i| lipschitz_of(ops, *i))
            .fold(0.0, f64::max),
        // A fixed distance has no gradient, so the value moves at exactly the rate the body
        // does. A driven one has already been divided by the sum of both rates, which is what
        // makes reporting the inner constant safe here rather than optimistic.
        // One: the combination cannot rise faster than its steeper operand, and both operands
        // here have already been divided by their own bound.
        Op::Morph { .. } => 1.0,
        Op::Offset { input, .. } => lipschitz_of(ops, *input),
        // The chain rule, with both terms known: the mapping's own slope is at most
        // `|Δout| / |Δin|` times the shape's peak, and the input contributes its own constant.
        // This is the number the whole decision rests on — an arbitrary field driving a
        // parameter would have nothing to put here.
        Op::Ramp {
            input,
            in_span,
            out_span,
            shape,
            ..
        } => (out_span / in_span).abs() * shape.slope_factor() * lipschitz_of(ops, *input),
        // Rigid motions do not stretch, so the copies move at exactly the rate the body does
        // and the union of them at the same.
        Op::Pattern { input, .. } => lipschitz_of(ops, *input),
        // Already divided by the stretch, so the composed value moves at most as fast as the
        // inner one did. Reporting the inner constant is therefore safe and is what the
        // division was for.
        Op::Twist { input, .. } => lipschitz_of(ops, *input),
        // Divided by its own stretch, so the rate the value moves at is the input's again.
        Op::Attract { input, .. } => lipschitz_of(ops, *input),
        // Unchanged: an isometry moves points without stretching them, so the field moves at
        // exactly the rate the inner one does.
        Op::Mirror { input, .. } => lipschitz_of(ops, *input),
        // Unchanged by the placement. The argument is divided by `s` and the result multiplied
        // by it, so the two cancel in the gradient and the field moves at exactly the rate the
        // inner one does — which is the whole reason to restrict this node to a uniform scale.
        Op::Transform { input, .. } => lipschitz_of(ops, *input),
    }
}

/// [`GraphField::normalization_at`] over a bare slice, so it can be called during compilation
/// while the op list is still being built and no `GraphField` exists yet.
pub(crate) fn normalization_of(ops: &[Op], index: usize) -> Normalization {
    match &ops[index] {
        Op::Leaf(field) => field.normalization(),
        // A bound whatever the cell: a graded beam is no longer a distance to anything, so a
        // fillet against one has to say so.
        Op::Lattice { .. } => Normalization::LowerBound,
        // **A rounded blend is a bound even over two distances** (blend-integrity-spec §3).
        // What it reports never overstates, and inside the join's reach it understates -- by
        // up to 2^(1/n) across the middle of a fillet between square walls -- so it is not a
        // distance, the same declaration a lattice makes for the same reason. That holds for
        // [`blend_bound`] exactly as it held for the division it replaced; what changed is
        // that a radius clear of the join the value *is* the operand's, so a bound over two
        // distances is a distance again there. A radius of zero is a hard boolean and is
        // judged as one.
        Op::Blend { radius: super::compile::Amount::Fixed(k), .. } if *k > 0.0 => {
            Normalization::LowerBound
        }
        // A radius following a ramp rounds somewhere, and is divided besides.
        Op::Blend { radius: super::compile::Amount::Driven(_), .. } => Normalization::LowerBound,
        Op::Blend { a, b, .. } => {
            if normalization_of(ops, *a) == Normalization::Distance
                && normalization_of(ops, *b) == Normalization::Distance
            {
                Normalization::Distance
            } else {
                Normalization::LowerBound
            }
        }
        // Distance only if every operand is one: the min or max of a distance and a bound is
        // a bound.
        Op::Boolean { inputs, .. } => {
            if inputs
                .iter()
                .all(|i| normalization_of(ops, *i) == Normalization::Distance)
            {
                Normalization::Distance
            } else {
                Normalization::LowerBound
            }
        }
        // Carried through unchanged when the distance is fixed: subtracting a constant leaves
        // the gradient alone, so a true distance stays one and a lower bound stays a lower
        // bound of the same kind — which is what lets an offset body still be blended.
        //
        // A *driven* distance is the whole point of the decision this implements, and it costs
        // exactly this. `∇(F − d) = ∇F − ∇d`, whose magnitude lies in `[1 − L, 1 + L]`, so the
        // value is no longer a distance however small `L` is. Divided by `1 + L` at evaluation
        // it is an honest lower bound, and declaring it as one is what makes a blend over a
        // variable-thickness wall refuse at compile time rather than come out the wrong shape.
        // **A bound, not a distance, even from two exact operands.** The value is 1-Lipschitz
        // so it never overstates, but where the two gradients oppose it rises more slowly than
        // one per millimetre and the reading is short. Which is a lower bound, and saying so is
        // what lets a blend against a morph announce itself.
        Op::Morph { .. } => Normalization::LowerBound,
        Op::Offset {
            input, distance, ..
        } => match distance {
            Amount::Fixed(_) => normalization_of(ops, *input),
            Amount::Driven(_) => Normalization::LowerBound,
        },
        // Not a distance to anything, and nothing treats it as one — it is reached only as an
        // amount. Declared the weaker way so that a future consumer which does compose it
        // cannot inherit a claim this never made.
        Op::Ramp { .. } => Normalization::LowerBound,
        // Carried through: every instance is a rigid motion, which maps distances to
        // distances, so a pattern of true distance fields is one — and stays blendable.
        // Unless the copies are rounded into each other, which divides the value exactly as a
        // blend does and so claims exactly what a blend claims.
        Op::Pattern { radius, .. } if *radius > 0.0 => Normalization::LowerBound,
        Op::Pattern { input, .. } => normalization_of(ops, *input),
        // Always a lower bound, whatever the input was. The warp does not preserve distance —
        // that is what the stretch division is compensating for — so the honest declaration is
        // the weaker one, and a blend over a twisted body is refused at compile time.
        Op::Twist { .. } => Normalization::LowerBound,
        // The same, and for the same reason: a warp that varies with position is not an
        // isometry, so the magnitude stops being a distance however small the strength is. The
        // stretch division is what keeps it an honest lower bound rather than an overstatement.
        Op::Attract { .. } => Normalization::LowerBound,
        // Carried through exactly. A reflection maps distances to distances, so a true distance
        // stays one — which is what keeps a mirrored body blendable.
        // **A reflection is a rigid motion; the join is a union.** Mirroring maps distances to
        // distances, so the transform carries its body's answer through. Joining asks the body
        // twice and takes the `min`, and a union of operands that touch keeps their shared face
        // buried inside the solid — nought where the solid is deep (item 113). A blend reads
        // magnitudes, so it grooves along that face. `reaches_its_plane` is asked of the graph at
        // compile time, since the ops hold no extent to ask here.
        Op::Mirror { input, join, reaches_its_plane, .. } => {
            if *join && *reaches_its_plane {
                Normalization::LowerBound
            } else {
                normalization_of(ops, *input)
            }
        }
        // A similarity carries exactness through: it maps distances to distances, only
        // rescaled. This is what a per-axis scale would have forfeited, and why a transformed
        // body can still be blended.
        Op::Transform { input, .. } => normalization_of(ops, *input),
    }
}
