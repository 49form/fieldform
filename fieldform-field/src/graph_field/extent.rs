// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where a node's surface can be, walked from what it is made of without compiling it.

use super::{
    build_loft, corner_reach, draft_offset, ends_at, extrude_section, loft_parts, own_divisor,
    pattern_fillet, pattern_reach_in, placement_similarity, resolve_frame, resolve_place,
    rotation_from_z, sweep_direction, sweep_law, sweep_offset, sweep_section, turn_rotation,
    wall_about_axis, wall_in_plane,
};
use crate::{Extent, PatternLayout};
use fieldform_scene::{BooleanOp, Document, Graph, Node, NodeId, Operand};
use nalgebra::{Point3, Vector3};

/// Where one node's surface can be, walking what it is made of.
///
/// Derived rather than guessed, and derived *here* rather than in each consumer: the viewport
/// wants it for framing and picking, a lattice wants it to know where to tile, and a scatter
/// wants its target's box. Three copies of this walk would be three chances to disagree about
/// the same graph, which is the failure this codebase keeps meeting under other names.
///
/// See [`Extent`] for why the answer has three states rather than being an `Option`, and why a
/// half-space is usable despite having no box of its own.
///
/// Unresolvable parameters resolve to zero rather than refusing. An extent is a hint used to
/// size a view or a bake, so a wrong box costs resolution; refusing would cost the picture.
/// Compilation is where an unresolvable parameter is an error, and it already says so.
pub fn graph_extent(document: &Document, graph: &Graph, node: NodeId) -> Extent {
    extent_below(document, graph, node, 0.0)
}

/// [`graph_extent`] of what a slot reads, and [`Extent::Empty`] for a slot with nothing in it.
///
/// **One reader for the empty slot's extent**, shared with the recursion inside `extent_of`, so
/// a caller that holds an [`Operand`] cannot come to a different answer from the walk about the
/// same hole.
///
/// `Empty` for the reason [`Node::Ramp`]'s own arm is `Empty`: a block waiting to be told what to
/// read has no surface anywhere, at any threshold, and a box around nothing would hand the view's
/// framing, a bake's bounds and the per-tile cull material that is not there. It is the one answer
/// the [`crate::ScalarField`] contract leaves available — there is no value a missing input could
/// supply that does not overstate something.
pub fn operand_extent(document: &Document, graph: &Graph, slot: &Operand) -> Extent {
    match slot.source() {
        Some(id) => graph_extent(document, graph, id),
        None => Extent::Empty,
    }
}

/// Where one node's field can read **below `g`** — the box of `{p : f(p) < g}`.
///
/// [`graph_extent`] is this at `g = 0`, which is the box of the solid. Everything else here reads
/// its operands somewhere other than their zero set, and *that* is why the box of the solid was
/// not a guarantee: a fillet of radius `k` makes material wherever its operands read below about
/// `0.29·k`, an offset of `d` wherever its input reads below `d`, and a box grown after the fact
/// cannot answer either — an `Empty` operand grown by anything is still `Empty`, while two
/// disjoint solids intersected have a small field in the gap where a fillet goes negative
/// (`tile-mask-design-01941c5.md` §1.8 R3).
///
/// So the threshold is carried *down*. Each node asks its inputs for the threshold at which its
/// own value could reach `g`: a node that divides by `D` ([`own_divisor`]) asks for `D·g`, a
/// rounded join adds its own reach, and a leaf answers with its box grown by `g` times its own
/// division. A `min` passes `g` through untouched, which is why a plain union of exact primitives
/// is as tight as it ever was.
///
/// **What this still does not promise**, each left as `#[ignore]`d tests naming their cause:
/// a leaning drafted extrude (R4) and a revolve of a multi-curve loop (R6) are too tight in the
/// leaf arm itself, which no threshold can repair; and R7 is sub-millimetre drawn-surface slack
/// that step 11's world-space pad is for.
pub(crate) fn extent_below(document: &Document, graph: &Graph, node: NodeId, g: f64) -> Extent {
    // Through the document, so a computed parameter sizes the box the way a typed one does.
    // This used to read only a typed literal, and a sphere whose radius was `51 + 2` got an
    // extent of a point: the view was sized to nothing while the panel said 53.
    let resolve = |r: &fieldform_scene::ParamRef| -> f64 { document.resolve(r).unwrap_or(0.0) };
    let at = |c: &[fieldform_scene::ParamRef; 3]| {
        Point3::new(resolve(&c[0]), resolve(&c[1]), resolve(&c[2]))
    };
    extent_of(document, graph, node, &resolve, &at, 0, g.max(0.0))
}

/// The extent of every root together — the whole part, since a part's field is their union.
pub fn part_extent(document: &Document, graph: &Graph) -> Extent {
    graph.roots().into_iter().fold(Extent::Empty, |acc, id| {
        acc.union(graph_extent(document, graph, id))
    })
}

/// Guards against a graph that references itself. A cycle is not constructible through the
/// public API, but an extent is asked for on every frame and a hang there is a hang in the
/// viewport rather than a failed operation.
const MAX_EXTENT_DEPTH: usize = 256;

/// Where a place is, for the extent walk.
///
/// Substitutes the origin for anything unresolvable, which is the same latitude the `at` closure
/// beside it takes and for the same reason: an extent has nowhere to report a refusal to, and a
/// box around the origin is a safe answer where no box at all is not.
pub(crate) fn place_at(
    document: &Document,
    graph: &Graph,
    place: &fieldform_scene::Place,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> f64,
) -> Point3<f64> {
    let numbers = |r: &fieldform_scene::ParamRef| Some(resolve(r));
    resolve_place(document, graph, place, &numbers).unwrap_or_else(|_| Point3::origin())
}

fn extent_of(
    document: &Document,
    graph: &Graph,
    node: NodeId,
    resolve: &dyn Fn(&fieldform_scene::ParamRef) -> f64,
    at: &dyn Fn(&[fieldform_scene::ParamRef; 3]) -> Point3<f64>,
    depth: usize,
    // The threshold: this answers for `{f < g}`, so `0` is the solid. See `extent_below`.
    g: f64,
) -> Extent {
    if depth > MAX_EXTENT_DEPTH {
        return Extent::Unbounded;
    }
    // `resolve_frame` is the one reader of a frame (D9), and it answers with `Option` because
    // a compile has to refuse a parameter it cannot resolve. An extent has no such option and
    // substitutes zero, which is what `resolve` here already does — so this adapts rather than
    // duplicating the reader for the sake of a signature.
    let numbers = |r: &fieldform_scene::ParamRef| Some(resolve(r));
    // Kept before the shadowing below, because a placement's pivot is asked for by id.
    let here = node;
    let Some(node) = graph.node(node) else {
        return Extent::Empty;
    };
    // An input read at a threshold this node works out for it — the whole of the guarantee.
    //
    // **An empty slot answers `Empty`, and this is the single place that is decided.** Every arm
    // below reads its operands through here, so none of them needs a guard of its own: a union
    // skips the hole, an intersection and a difference come out `Empty` with it, and a modifier
    // above one bounds nothing. The same answer [`operand_extent`] gives outside the recursion,
    // and the same one `Node::Ramp` has given an unwired input since `ramp-decision.md` §7.2.
    let below = |slot: &Operand, g: f64| match slot.source() {
        Some(id) => extent_of(document, graph, id, resolve, at, depth + 1, g.max(0.0)),
        None => Extent::Empty,
    };
    // An input's own solid, which is what every leaf arm below is written against.
    let recurse = |slot: &Operand| below(slot, 0.0);
    // What this node divides its own value by, so a threshold crossing it is scaled by it.
    let divisor = own_divisor(document, graph, here, node, &[]);
    // How far a rounded join of `summed` operands reaches past them at radius `k`.
    //
    // The form is `max(k, m) − (Σ (k − a_i)₊ⁿ)^(1/n)` with `m` the least operand, and each term is
    // at most `(k − m)₊`, so the subtracted part is at most `W^(1/n)·(k − m)₊`. Below `k` that
    // gives `raw ≥ k − W^(1/n)(k − m)`, so `raw < G` needs `m < k(1 − W^(−1/n)) + G·W^(−1/n)`,
    // and `W^(−1/n) ≤ 1`. At G1 with two operands that is `0.293·k`, which is the figure §1.8 R2
    // quotes for how far a fillet reaches.
    let join_reach = |k: f64, summed: usize, exponent: f64| {
        k.max(0.0) * (1.0 - (summed.max(1) as f64).powf(-1.0 / exponent))
    };

    let bare = match node {
        // The limit surface lies inside the control hull, so the cage's vertices bound it.
        Node::Cage(id) => match document.cages.get(id) {
            Some(cage) => Extent::around(cage.vertex_ids().map(|v| cage.vertex_position(v)), 0.0),
            None => Extent::Empty,
        },
        Node::Mesh(id) => match document.meshes.get(id).and_then(|m| m.bounds()) {
            Some((lo, hi)) => Extent::Bounded {
                lo: Point3::new(lo[0], lo[1], lo[2]),
                hi: Point3::new(hi[0], hi[1], hi[2]),
            },
            None => Extent::Empty,
        },
        Node::Sphere { centre, radius } => {
            Extent::ball(place_at(document, graph, centre, resolve), resolve(radius))
        }
        // The two opposite corners, which is the box exactly — its own extent, with nothing
        // to spare. Growing a single radius around the centre would have to use the
        // half-*diagonal* to enclose the corners, and would be loose by that much on every
        // axis; a cube only looks right that way because its diagonal box and its faces
        // coincide when the ball is read as an axis-aligned span.
        // **Turned as the field turns it**, about its centre. This arm matched `turn` into `..`
        // and boxed the bar as though it were axis-aligned, while compiling lowers the turn into
        // a placement the field does honour — so a bar turned 30 degrees had half its solid
        // outside its box, by up to ten millimetres. Found by the extent sweep
        // (`fieldform-field/tests/extent_holds_primitives.rs`). The box of a rotated box is exact
        // through the absolute rotation matrix: each world half-extent is `sum_j |R_ij| half_j`.
        Node::Cuboid { centre, size, turn, edge: _ } => {
            let c = place_at(document, graph, centre, resolve);
            let half = Vector3::new(resolve(&size[0]), resolve(&size[1]), resolve(&size[2])) * 0.5;
            let reach = turn_rotation(document, graph, turn, &numbers).matrix().abs() * half.abs();
            Extent::around([c - reach, c + reach], 0.0)
        }
        // The box of a parallelepiped is the box of its eight corners, exactly: its extreme in any
        // axis direction is attained at a vertex, which is what makes this tight rather than a bound.
        Node::SkewedBox { centre, edges } => {
            let c = place_at(document, graph, centre, resolve);
            let e = edges.map(|v| Vector3::new(resolve(&v[0]), resolve(&v[1]), resolve(&v[2])));
            let corners = (0..8).map(|i: usize| {
                let side = |bit: usize| if i >> bit & 1 == 1 { 0.5 } else { -0.5 };
                c + e[0] * side(0) + e[1] * side(1) + e[2] * side(2)
            });
            Extent::around(corners, 0.0)
        }
        Node::Cylinder { ends, radius, edge: _ } => {
            let (base, top) = ends_at(ends, document, graph, resolve);
            let r = resolve(radius);
            Extent::between(base, top, r, r)
        }
        Node::Cone {
            ends,
            base_radius,
            top_radius,
            edge: _,
        } => {
            let (base, top) = ends_at(ends, document, graph, resolve);
            Extent::between(base, top, resolve(base_radius), resolve(top_radius))
        }
        Node::Capsule {
            ends,
            start_radius,
            end_radius,
        } => {
            let (start, end) = ends_at(ends, document, graph, resolve);
            Extent::between(start, end, resolve(start_radius), resolve(end_radius))
        }
        // **Turned, the same way and for the same reason as the cuboid above**: this arm too
        // matched `turn` into `..`, and a torus turned 40 degrees had its tube nearly ten
        // millimetres above its box. The ring lies in the plane normal to its axis `a`, so along
        // world axis `i` it reaches `major · sqrt(1 − a_i²)` — the ring's circle projected onto
        // that axis — and the tube adds `minor` on every axis. Unturned, `a` is `+Z` and this is
        // exactly the box it always was.
        Node::Torus {
            centre,
            major_radius,
            minor_radius,
            turn,
        } => {
            let (major, minor) = (resolve(major_radius), resolve(minor_radius).abs());
            let c = place_at(document, graph, centre, resolve);
            let axis = turn_rotation(document, graph, turn, &numbers) * Vector3::z();
            let reach = axis.map(|a| major.abs() * (1.0 - a * a).max(0.0).sqrt() + minor);
            Extent::around([c - reach, c + reach], 0.0)
        }
        // A half-space: the one leaf with no box of its own, and the reason this whole type
        // has three states. It is still usable, because an intersection bounds it.
        // **A conformal lattice confined to a shell is the one lattice with a box**, and it
        // needs one for the same reason every other body does: a body with no extent cannot be
        // framed, cannot be traced on its own, and cannot bound a boolean above it. The box is
        // the surface's own, grown by half the shell — which is exactly how far the band
        // reaches, since the offset is a distance to that surface.
        //
        // Every other lattice tiles through all of space and bounds nothing, which is fine for
        // the reason a plane's is: it is meant to be intersected with a part.
        Node::Lattice { map, .. } => match map {
            fieldform_scene::CellMap::Conformal {
                surface,
                thickness: Some(thickness),
                ..
            } => match document.quad_meshes.get(surface) {
                Some(mesh) => Extent::around(
                    mesh.vertex_ids().map(|v| mesh.vertex_position(v)),
                    resolve(thickness).abs() * 0.5,
                ),
                None => Extent::Unbounded,
            },
            _ => Extent::Unbounded,
        },
        Node::Plane { .. } => Extent::Unbounded,
        // A zero-radius source. Its surface is the single point it sits at, and it encloses no
        // volume at all — so it draws nothing as a body, and bounds tightly as an operand.
        Node::Point { at: p } => Extent::at(at(p)),
        // The profile's own reach, carried into place by the frame. Taken from the curve's
        // points rather than its control polygon, which is what the solid is actually built
        // from — a control point can sit well outside the curve it shapes.
        // A rim only takes material away, so the box without it holds the solid with it.
        Node::Extrude { outline, sweep, start, height, sides, draft, corners, rims: _ } => {
            let Ok(section) = extrude_section(
                document,
                outline,
                sweep,
                crate::profile::PROFILE_TURN_DEGREES,
            ) else {
                return Extent::Empty;
            };
            let rotation = rotation_from_z(section.normal);
            let translation = section.origin.coords;
            // What is carried to both ends, in the plane's own coordinates: a profile's points,
            // and for a wall the corners of the box its material lies in. A wall's curve is
            // not its reach — the band goes a thickness past it, to one side or both.
            let reach: Vec<(f64, f64)> = match outline {
                fieldform_scene::Outline::Closed { .. } => section
                    .points
                    .iter()
                    .map(|p| {
                        let local = rotation.inverse_transform_vector(&(p.coords - translation));
                        (local.x, local.y)
                    })
                    .collect(),
                fieldform_scene::Outline::Wall { curve, thickness, side: across, ends: stops } => {
                    let built = wall_in_plane(
                        &section,
                        *curve,
                        resolve(thickness),
                        *across,
                        *stops,
                        f64::INFINITY,
                    );
                    let Ok(wall) = built else { return Extent::Empty };
                    let (lo, hi) = wall.bounds();
                    vec![(lo.x, lo.y), (hi.x, lo.y), (hi.x, hi.y), (lo.x, hi.y)]
                }
            };
            // The direction the ends are offset along: the normal, or the lean.
            let along = match sweep {
                fieldform_scene::Sweep::Normal => Vector3::z(),
                other => sweep_direction(document, other, &numbers)
                    .map(|d| rotation.inverse_transform_vector(&d))
                    .and_then(|d| d.try_normalize(1e-12))
                    .unwrap_or_else(Vector3::z),
            };
            // Along the sweep, so a leaning solid is bounded where it actually reaches rather
            // than where a straight one would have.
            //
            // **By the depth itself, which is measured along the sweep and not along the
            // normal.** `SkewExtrusion` ends where a point has gone `h` *along the direction*,
            // so its far end is the section carried by `along · h`. This carried it by
            // `along · h / |along.z|` — to a height of `h` above the plane, which is further
            // along the same line. For a plain lean that was a box too long and nothing worse.
            // With a draft it was a box too short: the far section is where the flare is
            // widest, and padding it in the wrong place left the real one's trailing wall
            // outside, by `h · tan(draft) − h · sin(lean)` where the draft outruns the lean
            // (`tile-mask-design-01941c5.md` §1.8 R4, five cases, 1.9 mm).
            //
            // **And the two ends are enough.** In any direction, how far the section at `t`
            // along the sweep reaches is its own reach, plus `t · tan(draft)` for the flare,
            // plus `t` times the direction's share of the sweep: linear in `t`, so greatest at
            // one end or the other and never between.
            let step = if along.z.abs() < 1e-6 { Vector3::z() } else { along };
            // **Where the solid actually is.** This was `[-h/2, +h/2]` for every extrude, which
            // is right only for a symmetric one — an extrude that starts at its plane spans
            // `[0, h]`, so the box was the right size in the wrong place, missing the far half
            // of the solid and enclosing empty space on the near side. A leftover from when
            // every extrude straddled its plane, and invisible because it is the right size.
            //
            // **Each end with the flare it has**, and from the start plane. The material lies
            // along the sweep from `start`: one way by `h`, half each way, or `h` one way and
            // side two's depth the other. A draft opens from the start plane, so the section
            // at an end is the profile grown by that end's depth times that side's slope --
            // nought at the start of a one-sided extrude, and its own at each end otherwise.
            // The walls a draft moved out used to be left out of this box altogether, so a
            // frustum flaring by |h| · tan(draft) on every side had a box of its base alone;
            // the scene sphere, the view's framing and a bake's bounds all read it. Found by
            // the step-11 design review; `a_drafted_extrudes_extent_holds_the_walls_the_
            // draft_moved_out` holds it.
            let (h, at) = (resolve(height), resolve(start));
            let ends: Vec<(f64, f64)> = match sides {
                fieldform_scene::Sides::One => {
                    vec![
                        (at, 0.0),
                        (at + h, corner_reach(draft_offset(h, resolve(draft)), *corners)),
                    ]
                }
                fieldform_scene::Sides::Symmetric => {
                    let (half, flare) = (
                        h.abs() * 0.5,
                        corner_reach(draft_offset(h * 0.5, resolve(draft)), *corners),
                    );
                    vec![(at - half, flare), (at + half, flare)]
                }
                fieldform_scene::Sides::Two { height: other, draft: other_draft } => {
                    let forward = if h < 0.0 { -1.0 } else { 1.0 };
                    let (two, two_draft) = (resolve(other).abs(), resolve(other_draft));
                    vec![
                        (at + h, corner_reach(draft_offset(h, resolve(draft)), *corners)),
                        (at - forward * two, corner_reach(draft_offset(two, two_draft), *corners)),
                    ]
                }
            };
            let (mut x0, mut y0) = (f64::INFINITY, f64::INFINITY);
            let (mut x1, mut y1) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for &(x, y) in &reach {
                (x0, y0) = (x0.min(x), y0.min(y));
                (x1, y1) = (x1.max(x), y1.max(y));
            }
            let mut out = Extent::Empty;
            for (side, grow) in ends {
                let carry = |x: f64, y: f64| {
                    Point3::from(rotation * (Vector3::new(x, y, 0.0) + step * side) + translation)
                };
                // The profile's own points at this end, as drawn.
                for &(x, y) in &reach {
                    out = out.union(Extent::at(carry(x, y)));
                }
                // And the plane-local box of the profile padded by the flare there, where there
                // is one -- so an undrafted extrude keeps the box it always had.
                let grow = grow.max(0.0);
                if grow > 0.0 {
                    for (x, y) in [
                        (x0 - grow, y0 - grow),
                        (x1 + grow, y0 - grow),
                        (x1 + grow, y1 + grow),
                        (x0 - grow, y1 + grow),
                    ] {
                        out = out.union(Extent::at(carry(x, y)));
                    }
                }
            }
            out
        }
        // A ring about the axis: turned all the way round, the profile reaches its own furthest
        // radius in every direction across the axis, and no further along it than it already
        // did.
        //
        // **Of the whole section, which is not the one curve the node names.** A profile is the
        // loop that curve is part of, and a wall the chain it is part of, so this walks what the
        // compile walks. It used to read the named curve alone, and a revolve of a loop drawn as
        // two curves was boxed as the revolve of one of them: 11.7 mm of solid outside its box
        // (`tile-mask-design-01941c5.md` §1.8 R6). A wall adds its band, which reaches a
        // thickness past the curve and, across the axis, comes back as a radius on this side.
        Node::Revolve { outline, frame, .. } => {
            let Ok((translation, rotation)) = resolve_frame(document, graph, frame, &numbers) else {
                return Extent::Empty;
            };
            let into = |p: Point3<f64>| {
                Point3::from(rotation.inverse_transform_vector(&(p.coords - translation)))
            };
            // The section as `(r, z)` points it does not reach past.
            let reach: Vec<(f64, f64)> = match outline {
                fieldform_scene::Outline::Closed { profile } => {
                    let Ok(found) = crate::profile_loop(document, *profile, frame.curve_source())
                    else {
                        return Extent::Empty;
                    };
                    found
                        .points
                        .iter()
                        .map(|p| {
                            let local = into(*p);
                            (local.x.hypot(local.y), local.z)
                        })
                        .collect()
                }
                fieldform_scene::Outline::Wall { curve, thickness, side, ends } => {
                    let Ok(chain) = crate::outline_chain(document, *curve, frame.curve_source())
                    else {
                        return Extent::Empty;
                    };
                    let Ok(wall) =
                        wall_about_axis(&chain, *curve, resolve(thickness), *side, *ends, into)
                    else {
                        return Extent::Empty;
                    };
                    let (lo, hi) = wall.bounds();
                    let r = lo.x.abs().max(hi.x.abs());
                    vec![(r, lo.y), (r, hi.y)]
                }
            };
            let (mut r_max, mut z_lo, mut z_hi) = (0.0f64, f64::INFINITY, f64::NEG_INFINITY);
            for &(r, z) in &reach {
                r_max = r_max.max(r);
                z_lo = z_lo.min(z);
                z_hi = z_hi.max(z);
            }
            if !z_lo.is_finite() {
                return Extent::Empty;
            }
            let mut out = Extent::Empty;
            for &z in &[z_lo, z_hi] {
                for (dx, dy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    let at = Point3::from(
                        rotation * Vector3::new(dx * r_max, dy * r_max, z) + translation,
                    );
                    out = out.union(Extent::around([at, at], 0.0));
                }
            }
            out
        }
        // The spine grown by the section's reach from the point that carries it, at its
        // largest under the law -- the most the section can cover anywhere along the path,
        // since it turns about that point and no point of it is further from the spine than it
        // is from there. The law's own number, so a section doubled on the way does not grow
        // out of its box.
        Node::Sweep { outline, spine, law, .. } => {
            let Some(curve) = document.curves.get(spine) else { return Extent::Empty };
            let Some(section) = sweep_section(document, outline, *spine) else {
                return Extent::Empty;
            };
            let reach = match sweep_law(document, law) {
                Ok(law) => {
                    let path = crate::Spine::of_curve(curve, crate::DEFAULT_TOLERANCE_MM);
                    law.reach_under(section.reach, path.length() * (law.window.1 - law.window.0))
                }
                Err(_) => section.reach,
            };
            // Around the path as the sweep actually lays it: moved to start at the profile.
            let offset = sweep_offset(document, outline, *spine).unwrap_or_else(Vector3::zeros);
            let along: Vec<Point3<f64>> =
                curve.tessellate(0.1).into_iter().map(|p| p + offset).collect();
            Extent::around(along, reach)
        }
        // **The centerline grown by the furthest any section reaches from its own station.**
        // Measured in space rather than projected: a section read square to the spine is the
        // projection of what was drawn, and a projection is never longer than the thing
        // projected, so the drawn distance bounds the read one. Cheap, which matters here —
        // the extent is asked far more often than the loft is built.
        Node::Loft { sections, spine, smooth, rims: _ } => {
            let Ok(parts) = loft_parts(
                document,
                sections,
                *spine,
                crate::profile::PROFILE_TURN_DEGREES,
                crate::DEFAULT_TOLERANCE_MM,
            ) else {
                return Extent::Empty;
            };
            let path = crate::Spine::new(parts.spine.clone());
            let mut reach: f64 = 0.0;
            for section in &parts.sections {
                let centroid = Point3::from(
                    section.iter().fold(Vector3::zeros(), |sum, p| sum + p.coords)
                        / section.len().max(1) as f64,
                );
                let Some(station) = path.nearest(centroid) else { continue };
                for p in section {
                    reach = reach.max((p - station.point).norm());
                }
            }
            // **Grown by how far a smooth interpolation can bulge past its sections**, which a
            // linear one cannot and a Hermite can -- `Lofted::overshoot`. Asked of the build,
            // which is shared with the compile and costs nothing the second time; a loft that
            // cannot be built has no bulge to add and is refused by the compile anyway.
            let bulge = build_loft(&parts, *smooth).map(|built| built.overshoot()).unwrap_or(0.0);
            Extent::around(parts.spine, reach + bulge)
        }
        Node::Pipe { curve, radius } => match document.curves.get(curve) {
            // The widest the radius ever gets, when a ramp drives it — the same reading the
            // offset's arm below makes, and for the same reason: a box drawn to the *current*
            // radius of a varying one clips the body wherever it is fatter than that.
            Some(curve) => {
                let widest = match radius.driver().and_then(|id| graph.node(id)) {
                    Some(Node::Ramp { out_min, out_max, .. }) => {
                        resolve(out_min).max(resolve(out_max))
                    }
                    _ => resolve(radius),
                };
                Extent::around(curve.tessellate(0.1), widest.max(0.0))
            }
            None => Extent::Empty,
        },
        // **The union of the two boxes, and not the space between them.** Outside `a`'s extent
        // `a > 0`, outside `b`'s `b > 0`, so outside both the combination is positive for every
        // `t` and there is no surface. The morphed shape never leaves the two bodies' own
        // boxes — which is the tempting intuition inverted, since a morph looks like it should
        // travel between them and the emptiness result says it does not.
        //
        // **And the threshold passes through a `min`, because a convex combination is never
        // below both operands.** `mix` lies between `a` and `b`, so `mix < G` needs one of them
        // below `G` — which is the union of the two at `G`, and at `G = 0` is exactly the
        // emptiness argument above.
        Node::Morph { a, b, .. } => {
            let carried = g * divisor;
            return below(a, carried).union(below(b, carried));
        }
        // **Every arm passes the threshold straight through**, which is what makes a boolean of
        // exact primitives cost the guarantee nothing: `min f_i < g` holds exactly where some
        // `f_i < g`, and `max f_i < g` exactly where all of them are.
        Node::Boolean { op, inputs } => {
            let mut extents = inputs.iter().map(|i| below(i, g));
            let first = extents.next().unwrap_or(Extent::Empty);
            return match op {
                BooleanOp::Union => extents.fold(first, |acc, e| acc.union(e)),
                // Subtracting can only take material away, so the kept body's box holds.
                BooleanOp::Difference => first.difference(),
                BooleanOp::Intersection => extents.fold(first, |acc, e| acc.intersection(e)),
            };
        }
        // **A fillet asks its operands for a threshold; it does not grow the boolean's box.**
        // The two look the same over exact operands and are not the same thing at all: this used
        // to be the boolean's box grown by the radius, and a radius is a distance, so over an
        // operand whose value *reads short of its distance* the growth was short by that
        // operand's divisor. Two bodies wound at ten degrees a millimetre read about a fifth of
        // their distance, and 13.4 mm of solid sat outside the box
        // (`tile-mask-design-01941c5.md` §1.8 R2). Asked as a threshold instead, each operand
        // grows by its own division and the arithmetic is the operand's rather than the join's.
        Node::Blend { op, a, b, fillet } => {
            // The largest radius a ramp can give it, where one drives it: the reach is what the
            // rounding can add, and a graded fillet adds its most somewhere.
            let k = super::lipschitz::fillet_reach(document, graph, fillet);
            // **Only a union reaches.** A rounded intersection and a rounded difference are both
            // a smooth *maximum*, and a smooth maximum is at least the hard one -- it is
            // `−smooth_min(−a, −b)`, and a smooth minimum undershoots -- so the rounding there can
            // only take material away and the hard boolean's own box still holds. Charging them
            // the union's reach turned an intersection of two disjoint bodies from `Empty` into a
            // box, which is sound and is not the answer:
            // `a_blend_intersection_of_disjoint_bodies_is_empty_and_nothing_is_solid` is what
            // says so, and it caught this.
            let reach = match op {
                BooleanOp::Union => join_reach(k, 2, fillet.exponent()),
                BooleanOp::Difference | BooleanOp::Intersection => 0.0,
            };
            let carried = g * divisor + reach;
            return match op {
                BooleanOp::Union => below(a, carried).union(below(b, carried)),
                BooleanOp::Difference => below(a, carried).difference(),
                BooleanOp::Intersection => below(a, carried).intersection(below(b, carried)),
            };
        }
        // The input read at `g + d`, which is what `f − d < g` says. An eroded body asks for less
        // and never leaves the box the original occupied, so the threshold is clamped at nought
        // rather than running negative.
        //
        // A driven distance grows by the largest the ramp can reach, which is the same argument
        // with the constant replaced by the end of its range — and divides, which is what
        // `divisor` carries. Written as a growth of the input's own box before, which is the same
        // number over an exact input and short by the input's divisor over any other.
        Node::Offset { input, distance } => {
            let grow = match distance.driver().and_then(|id| graph.node(id)) {
                Some(Node::Ramp {
                    out_min, out_max, ..
                }) => resolve(out_min).max(resolve(out_max)),
                _ => resolve(distance),
            };
            return below(input, g * divisor + grow.max(0.0));
        }
        // A wall is inside the body it came from, so the body's own box holds it — and reads
        // below `g` only where the body does, since the wall's value is the larger of the body's
        // and the inward face's.
        Node::Shell { input, .. } => return below(input, g),
        // Not a shape, so it has no extent — and nothing asks, since a ramp is never a body.
        // `Empty` rather than `Unbounded` so that a stray one cannot widen a part's box.
        Node::Ramp { .. } => Extent::Empty,
        // The body's own box, grown by the furthest anything can move.
        //
        // **Tighter than the curve, deliberately.** The obvious bound is the box grown by the
        // whole falloff radius, which is what the spec suggested; nothing ever moves that far.
        // A point is displaced by `strength · falloff · x·A(x)`, which peaks at about 0.26 of
        // that product — so at full strength this box is roughly a quarter of the size in each
        // direction, and every ray that starts outside it still starts outside the geometry.
        //
        // It holds in both directions, which is the part worth stating: the material is drawn
        // *toward* the curve, so a body whose attractor sits outside it bulges out past its own
        // box — by that same displacement and no more, since every point of the warped body is
        // a point of the original one moved by at most that much.
        Node::Attract { input, strength, falloff, .. } => {
            // The warped value is the body's at the warped point, divided by the stretch — so a
            // threshold crosses the division and then the displacement, in that order.
            return below(input, g * divisor)
                .grown(crate::attract::max_pull(resolve(strength), resolve(falloff)));
        }
        // The cylinder the winding sweeps the body through: the same height, and a radius that
        // covers every corner of the untwisted box. Turning cannot carry a point further from
        // the axis than it already was, so this holds whatever the rate is.
        // The threshold crosses the twist's own division first — `twist_stretch`, which is the
        // number this walk had no way to ask for until `own_divisor` existed.
        Node::Twist { input, frame, .. } => match below(input, g * divisor).as_box() {
            Some((lo, hi)) => {
                // A sphere about the frame's origin rather than a cylinder about one of three
                // axes. Turning cannot carry a point further from the axis than it already was,
                // so a ball containing the untwisted box contains the twisted one — true for
                // any axis, where the cylinder had to know which of three it was.
                //
                // Looser than the cylinder was, by the box's half-extent along the axis. That
                // is the price of an axis that can point anywhere, and it is a bound rather
                // than a shape: nothing is drawn from it, only sized.
                let Ok((origin, _)) = resolve_frame(document, graph, frame, &numbers) else {
                    return Extent::Empty;
                };
                let c = Point3::from(origin);
                let across =
                    |i: usize| (lo[i] - c[i]).abs().max((hi[i] - c[i]).abs());
                let r = (across(0).powi(2) + across(1).powi(2) + across(2).powi(2)).sqrt();
                Extent::around(
                    [
                        Point3::new(c.x - r, c.y - r, c.z - r),
                        Point3::new(c.x + r, c.y + r, c.z + r),
                    ],
                    0.0,
                )
            }
            // Unbounded stays unbounded, and the twist itself refuses to compile against one.
            // An empty slot arrives here too, having answered `Empty` rather than a box, and
            // `recurse` answers `Empty` again — a twist of nothing bounds nothing.
            None => recurse(input),
        },
        Node::Pattern { input, .. } => {
            // The body where it is when the layout will not resolve, for the reason the divisor
            // gives: `build` refuses such a pattern, so this box is never the one anything is
            // framed or baked from, and bounding the body that is actually there is the honest
            // fallback.
            let Ok(layout) = PatternLayout::of(document, graph, here) else {
                return below(input, g * divisor);
            };
            let mut all = Extent::Empty;
            // **A rounded fold reaches past its instances, and the box did not grow for it**:
            // 5.5 mm of solid outside (`tile-mask-design-01941c5.md` §1.8 R5). The same argument
            // the blend above makes, with the window the fold actually sums rather than two —
            // `pattern_join_divisor`'s own `summed`, so the two read one number.
            let k = resolve(&pattern_fillet(node).radius).max(0.0);
            // Opened through `source` rather than guarded, because the reach is a reading of the
            // *body* being folded and `pattern_reach_in` wants a block to take the extent of.
            // There is nothing to ask for an empty slot and nothing to ask it for: each copy
            // below answers `Empty` whatever threshold it is given.
            let carried = match input.source().filter(|_| k > 0.0) {
                Some(id) => {
                    let summed = (2 * pattern_reach_in(document, graph, id, &layout, k) + 1)
                        .min(layout.held())
                        .max(1);
                    g * divisor + join_reach(k, summed, pattern_fillet(node).exponent())
                }
                None => g * divisor,
            };
            // The same rule the evaluation folds by: a copy the block leaves out does not bound
            // anything either. A box that covered a copy the field never evaluates would size the
            // view and the bake to material that is not there.
            for i in 0..layout.total {
                if layout.excluded[i as usize] {
                    continue;
                }
                let place = layout.placement(i);
                all = all.union(below(input, carried).placed(|p| place * p));
            }
            return all;
        }
        Node::Mirror { input, frame, join } => {
            // Not a plane, or an unreadable frame: the graph refuses to compile, so nothing
            // will be drawn from this box anyway.
            let Ok((origin, rotation)) = resolve_frame(document, graph, frame, &numbers) else {
                return Extent::Empty;
            };
            let point = Point3::from(origin);
            let normal = rotation * Vector3::z();
            // A reflection is rigid, so the threshold crosses it unchanged.
            let reflected =
                below(input, g).placed(|q| q - normal * (2.0 * (q - point).dot(&normal)));
            return if *join {
                reflected.union(below(input, g))
            } else {
                reflected
            };
        }
        Node::Transform { input, .. } => {
            // Through the same similarity the compile builds, so a pivot cannot move the solid
            // and leave its box behind.
            //
            // **A scale scales the value**, which `Op::Transform` does by multiplying what its
            // input answered — so a threshold of `g` out here is `g / scale` in there, and a
            // body scaled to half size reads below `g` where its original reads below `2g`.
            let map = placement_similarity(document, graph, here, node, &numbers);
            let scale = map.scaling();
            let carried = if scale > 1e-12 { g / scale } else { g };
            return below(input, carried).placed(|p| map * p);
        }
        // A node this build cannot read compiles to nothing, so it bounds nothing. Saying
        // `Unbounded` would be safer for a *drawing*, but the graph refuses to compile at all,
        // so there is nothing to draw and nothing to be conservative about.
        Node::Unknown(_) => Extent::Empty,
    };
    // **A leaf answers with its own box grown by the threshold, times its own division.** Every
    // arm above this point returned already; what is left is a source whose value is a distance to
    // the shape it describes, or a stated fraction of one. Over a true distance `{f < g}` is the
    // solid grown by `g`; over a value divided by `D` it is the solid grown by `g·D`, since such
    // a value reads at least `d/D`.
    //
    // At `g = 0` this is the identity, which is the point: [`graph_extent`] answers byte for byte
    // what it always did, and only a consumer asking for a positive threshold sees any of this.
    if g <= 0.0 {
        return bare;
    }
    bare.grown(g * own_divisor(document, graph, here, node, &[]))
}
