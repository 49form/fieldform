// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the viewport's overlay and panels read off a part every frame, as functions of the
//! document and the part's graph.
//!
//! **Out of the application so it can be measured.** Each frame paints every point and plane
//! block over the model, sizes the box picking aims into, and hands the trace every parameter's
//! value, and each of those was worked out again from the document: the points twice, since the
//! pick box asked for its own, and the points, the planes and the pick box each by cloning the
//! part's graph first. None of it could be timed while it ran only inside a window, so none of it
//! was (`fieldform-io/examples/overlay_cost.rs` does now). [`OverlayReadings`] is all of it at
//! once, and [`KeptReadings`] keeps it from one frame to the next, which is how the application
//! asks for it; the functions are its pieces.
//!
//! Here rather than in `fieldform-scene`, which holds the rule they draw by
//! ([`fieldform_scene::is_visible_in`]), because every piece but the parameter table needs this
//! crate: the pick box and a lattice's claim to be reference geometry are both
//! [`graph_extent`], and where a point or a plane is drawn is [`placement_above`] over the graph
//! [`with_turns_flattened`] makes.

use std::collections::HashMap;
use std::rc::Rc;

use fieldform_scene::{
    drawn_graph, is_visible_among, is_visible_in, shown_ingredients_in, Document, Graph, Node,
    NodeId, ParamId, ParamRef,
};
use nalgebra::{Point3, Rotation3};

use crate::{graph_extent, placement_above, resolve_place, with_turns_flattened};

/// What a frame needs that the document does not say: the application's own view of it.
pub struct OverlayInputs<'a> {
    /// Whether a body is drawn as its tessellated mesh rather than traced, which only the
    /// application knows, since it is the one holding the meshes.
    pub drawn_as_mesh: &'a dyn Fn(NodeId) -> bool,
    /// The joint points that go unseen with their profile row, by the rule the row's eye reads.
    /// The rows are the application's list, so it works these out.
    pub hidden_with_rows: &'a [NodeId],
    /// The meshing resolution, one cell of which is the margin round the pick box.
    pub resolution: usize,
    /// The size of the box given to a scene with nothing in it, or nothing but one spot: the
    /// size a new body starts at.
    pub empty_mm: f64,
    /// What a parameter reference is worth for drawing and sizing. The application's reads an
    /// unresolvable one as zero, which is right only for that.
    pub numbers: &'a dyn Fn(&ParamRef) -> Option<f64>,
}

/// Everything the overlay and the panels read off a part in a frame, built in one go.
#[derive(Clone, Debug)]
pub struct OverlayReadings {
    /// The part's graph with every turn rewritten as the placement it means
    /// ([`with_turns_flattened`]), which the points and the planes are placed by.
    pub placed: Graph,
    /// Every point block that draws, with where it is ([`point_positions`]).
    pub points: Vec<(NodeId, Point3<f64>)>,
    /// Every plane block that draws, as a point on it and its facing ([`plane_frames`]).
    pub planes: Vec<(NodeId, Point3<f64>, Rotation3<f64>)>,
    /// The box picking is sized to ([`pick_bounds`]).
    pub pick_bounds: (Point3<f64>, Point3<f64>),
    /// Every parameter's value ([`Document::parameter_values`]).
    pub values: HashMap<ParamId, f64>,
}

impl OverlayReadings {
    /// Every reading, each worked out once: one flattened graph for the points and the planes,
    /// and the points the pick box is grown by are those same points.
    pub fn build(document: &Document, part: &Graph, inputs: &OverlayInputs<'_>) -> Self {
        let placed = with_turns_flattened(document, part, inputs.numbers);
        let points =
            point_positions(document, part, &placed, inputs.hidden_with_rows, inputs.numbers);
        let planes = plane_frames(document, part, &placed, inputs.numbers);
        let (display, _) = display_graph_with_stand_ins(document, part, inputs.drawn_as_mesh);
        let pick_bounds =
            pick_bounds(document, &display, &points, inputs.resolution, inputs.empty_mm);
        let values = document.parameter_values();
        Self { placed, points, planes, pick_bounds, values }
    }
}

/// [`OverlayReadings`] kept from one frame to the next, and read again only when the key they were
/// read for moves.
///
/// **Kept because nothing they read moves in most frames.** An orbit moves the camera and nothing
/// else, and the pick box alone asked every drawn body for its extent each frame of it: 2.0 ms for
/// one sweep, 17 ms for a wing of nine blocks (`fieldform-io/examples/overlay_cost.rs`).
///
/// **The key is the caller's, and it is the whole of what can go wrong here.** A document is too big
/// to compare every frame, so the application counts the moments it may have changed and keys the
/// readings on that count, beside its own view of the part ([`OverlayInputs`]). Whatever the
/// readings read that the key leaves out is a reading left on screen after what it was read from has
/// moved: a point drawn where it was, a pick box sized to where a body used to be. One reading is
/// kept rather than one per key: the count only goes forward, so a reading for an earlier key is
/// rarely asked for again, and when it is, a build is all it costs.
///
/// **Whatever is read, not only the overlay's**, so that every reading kept against the count is
/// kept the same way: the block list keeps [`fieldform_scene::ListReadings`] in one of these too
/// (refactor plan step 3.4).
pub struct KeptReadings<K, R = OverlayReadings> {
    kept: Option<(K, Rc<R>)>,
}

impl<K, R> Default for KeptReadings<K, R> {
    fn default() -> Self {
        Self { kept: None }
    }
}

impl<K: PartialEq, R> KeptReadings<K, R> {
    /// The readings for `key`: those kept, when they were read for it, or else `read`'s, kept in
    /// their place. Shared rather than copied, so a frame can hand them on — the trace's parameter
    /// table outlives the borrow it is read under — and reading them costs a count.
    pub fn get(&mut self, key: K, read: impl FnOnce(&K) -> R) -> Rc<R> {
        if let Some((kept, readings)) = &self.kept
            && *kept == key
        {
            return Rc::clone(readings);
        }
        let readings = Rc::new(read(&key));
        self.kept = Some((key, Rc::clone(&readings)));
        readings
    }
}

/// Whether this body is reference geometry rather than a part: something infinite, whose
/// job is to be an operand.
///
/// A plane and a lattice are both true to their own definition and useless to look at on
/// their own — one is half of space and the other fills all of it, so tracing either as a
/// root turns the viewport into a solid that hides everything including itself. They become
/// visible by being booleaned into something, which is the same status a curve has: a curve
/// is a line on screen until a pipe consumes it.
///
/// Deliberately a check on the *leaf*, not on whether the root's extent is unbounded.
/// Unioning a plane into a part makes that union genuinely infinite too, and hiding it on
/// that ground would take the part with it — a body the user built and expects to see.
pub fn is_reference_body(document: &Document, graph: &Graph, id: NodeId) -> bool {
    let mut id = id;
    loop {
        match graph.node(id) {
            Some(Node::Plane { .. }) => return true,
            // **By its extent, not by its kind.** Every lattice used to tile all of space,
            // so "a lattice" and "infinite" were the same answer and this matched the node.
            // A conformal lattice confined to a shell is bounded — it is a part on its own,
            // which is the whole point of the shell — and matching the kind hid it: the
            // owner built one and saw "the part has no bodies" beside a row calling it a
            // body. What this rule is actually about is a solid that fills the viewport,
            // and that is a question the extent answers.
            Some(Node::Lattice { .. }) => {
                return graph_extent(document, graph, id).is_unbounded();
            }
            // An imported mesh with holes cannot be a signed field — there is no inside
            // for a sign to refer to — so it is reference geometry rather than a solid.
            // Excluded here rather than left to fail at compile time, because one body
            // that cannot compile used to take the whole viewport with it: after a partial
            // import, a cube created afterwards did not appear either.
            Some(Node::Mesh(mesh)) => {
                return document.mesh(*mesh).is_none_or(|m| !m.is_closed());
            }
            // **A hole ends the walk**, which is the right default here: this is looking for a
            // mesh leaf under a chain of placements, and an empty slot means there is no leaf
            // down there to find.
            Some(Node::Transform { input, .. }) => match input.source() {
                Some(below) => id = below,
                None => return false,
            },
            _ => return false,
        }
    }
}

/// The part's graph with the hidden bodies taken out — what the viewport draws, picks
/// against, and sizes its bake to — and which added root stands in for which block.
///
/// A whole body goes, interior and all, rather than being skipped at the last moment: a
/// hidden body must not contribute a surface, must not be pickable, and must not stretch
/// the box the bake spends its resolution on.
pub fn display_graph_with_stand_ins(
    document: &Document,
    part: &Graph,
    drawn_as_mesh: &dyn Fn(NodeId) -> bool,
) -> (Graph, HashMap<NodeId, NodeId>) {
    // Once for every root the closure is asked about (`is_visible_among`).
    let consumed = part.consumed();
    // **And once for the empty slots, for the same reason**: `incomplete` walks the whole graph,
    // and the closure below is asked per root. Of the document, so a body that is or reads an
    // open cage is pruned too and the part's other bodies draw (R12).
    let holed = document.incomplete_in(part);
    drawn_graph(part, &shown_ingredients_in(document, part), &|id| {
        !is_visible_among(document, part, &consumed, id)
            || is_reference_body(document, part, id)
            || drawn_as_mesh(id)
            // **A body with an empty slot is dropped**, the fourth term beside the three above
            // (`notebook-spec.md` §7.2). The precedent is `is_reference_body` two lines up,
            // excluded for exactly this reason: one body that cannot compile used to take the
            // whole viewport with it, so after a partial import a cube made afterwards did not
            // appear either. A holed body cannot compile, so left in it would blank everything.
            //
            // Closed downstream by `incomplete`, so a block reading a holed one goes with it.
            // The picking and the framing follow from here for free, because both read the graph
            // this returns; what does NOT follow is measurement, which refuses instead of
            // answering about a part with a body missing (§7.2, and `App::measure`'s own comment).
            || holed.contains(&id)
    })
}

/// Every point block in the part that is not hidden, with where it is.
///
/// **Because a point is in the scene whether or not it is selected.** Only the selected one
/// was drawn, so a document with six points showed one dot and five blanks, and the way to
/// find a point was to click its row and watch for something to appear. Reported as exactly
/// that. The eye beside the row is what takes one off screen, the same as every other body,
/// and a joint whose row is hidden goes with it (`hidden_with_rows`).
///
/// `placed` is `part` through [`with_turns_flattened`], made once for the list by the caller
/// rather than once per point, since making it clones the graph.
pub fn point_positions(
    document: &Document,
    part: &Graph,
    placed: &Graph,
    hidden_with_rows: &[NodeId],
    numbers: &dyn Fn(&ParamRef) -> Option<f64>,
) -> Vec<(NodeId, Point3<f64>)> {
    let mut ids: Vec<NodeId> = part
        .nodes()
        .filter(|(_, n)| matches!(n, Node::Point { .. }))
        .map(|(id, _)| id)
        .collect();
    ids.sort_by_key(|id| id.raw());
    ids.into_iter()
        .filter(|id| is_visible_in(document, part, *id) && !hidden_with_rows.contains(id))
        .filter_map(|id| {
            // From the extent rather than from the node's own coordinates, because a point's
            // extent is the single place it sits and a placement moves that extent with it.
            // Reading the node's numbers directly would draw a placed point where it was built
            // rather than where it is — the mistake the cage's placement exists to stop the
            // cage making, and the reason this asks the same walk everything else asks.
            let at = graph_extent(document, part, id).centre()?;
            Some((id, placement_above(document, placed, id, None, numbers) * at))
        })
        .collect()
}

/// Every plane block in the part that is not hidden, as a point on it and the way it faces.
///
/// The same generalisation [`point_positions`] is, for the same reason and with the same rule
/// about the eye, and `placed` is made once by the caller for the same reason. The application's
/// selected plane stays a question of its own: the gizmo needs to know which one is being
/// edited, which is a different question from which ones exist.
pub fn plane_frames(
    document: &Document,
    part: &Graph,
    placed: &Graph,
    numbers: &dyn Fn(&ParamRef) -> Option<f64>,
) -> Vec<(NodeId, Point3<f64>, Rotation3<f64>)> {
    let mut ids: Vec<NodeId> = part
        .nodes()
        .filter(|(_, n)| matches!(n, Node::Plane { .. }))
        .map(|(id, _)| id)
        .collect();
    ids.sort_by_key(|id| id.raw());
    ids.into_iter()
        .filter(|id| is_visible_in(document, part, *id))
        .filter_map(|id| {
            let Some(Node::Plane { origin, flipped, .. }) = part.node(id) else {
                return None;
            };
            let (origin, flipped) = (*origin, *flipped);
            // The same walk the selected one uses, which composes the block's own turn as
            // well as the placements above it.
            let place = placement_above(document, placed, id, None, numbers);
            // **The whole rotation, not just the normal.** A plane turned about its own
            // normal is the same plane and the same normal -- which is true, and is why
            // turning one in Z looked as though nothing had happened: the square was built
            // from an *invented* pair of in-plane axes, so the one thing that changed was
            // the one thing it did not show. Handing back the rotation lets the square be
            // drawn in the plane's own X and Y, and a roll turns it.
            let mut facing = place.isometry.rotation.to_rotation_matrix();
            if flipped {
                // Flipped is a half turn about the plane's own X: the normal reverses and
                // the square stays where it is, which is what flipping a half-space means.
                facing = facing * Rotation3::from_euler_angles(std::f64::consts::PI, 0.0, 0.0);
            }
            let at = resolve_place(document, part, &origin, numbers)
                .unwrap_or_else(|_| Point3::origin());
            Some((id, place * at, facing))
        })
        .collect()
}

/// The box the model occupies, for sizing picking tolerances.
///
/// `display` is the graph the viewport draws ([`display_graph_with_stand_ins`]) and `points`
/// the points it paints ([`point_positions`]).
pub fn pick_bounds(
    document: &Document,
    display: &Graph,
    points: &[(NodeId, Point3<f64>)],
    resolution: usize,
    empty_mm: f64,
) -> (Point3<f64>, Point3<f64>) {
    let mut lo = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut hi = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    // The *display* graph, which is what the viewport draws — this used to walk the part's
    // roots and skip the hidden ones, which is the same question answered twice and the
    // two disagreeing. Hiding the only body left this with nothing, so it returned a
    // zero-size box, the camera got a scene radius of 1e-6, and the tracer was sized to a
    // one-micron scene: the viewport filled with speckle.
    for body in display.roots() {
        // An unbounded body — a half-space, or anything unioned with one — contributes
        // no box, and deliberately does not veto the others'. Picking wants somewhere to
        // size a tolerance against, and the finite bodies are the ones with surfaces
        // anybody is aiming at; letting an infinite plane widen this to nothing would make
        // every other body unpickable.
        if let Some((blo, bhi)) = graph_extent(document, display, body).as_box() {
            lo = Point3::new(lo.x.min(blo.x), lo.y.min(blo.y), lo.z.min(blo.z));
            hi = Point3::new(hi.x.max(bhi.x), hi.y.max(bhi.y), hi.z.max(bhi.z));
        }
    }
    // **And the points**, which are on screen without being bodies (`Node::is_body`). A
    // document of nothing but a profile's joints is the first minute of drawing one, and
    // it used to be framed on them because they were bodies; it still is.
    for &(_, at) in points {
        lo = Point3::new(lo.x.min(at.x), lo.y.min(at.y), lo.z.min(at.z));
        hi = Point3::new(hi.x.max(at.x), hi.y.max(at.y), hi.z.max(at.z));
    }
    // A box with no size -- one point, or several on one spot -- would size the camera to
    // a micron, which is the speckle described below; it gets the default size around it.
    if lo.x.is_finite() && (hi - lo).norm() < 1e-3 {
        let half = empty_mm * 0.5;
        let centre = Point3::from((lo.coords + hi.coords) * 0.5);
        return (
            Point3::new(centre.x - half, centre.y - half, centre.z - half),
            Point3::new(centre.x + half, centre.y + half, centre.z + half),
        );
    }
    if !lo.x.is_finite() {
        // Nothing visible — everything hidden, or a part with no bodies yet. A default-
        // sized box rather than a zero one: this sizes the camera's clip planes and the
        // marcher's reach, and both go wrong in the same visible way when handed nothing.
        // An empty viewport is the right answer here; a viewport full of noise is not.
        let half = empty_mm * 0.5;
        return (Point3::new(-half, -half, -half), Point3::new(half, half, half));
    }
    // A margin of one cell keeps the surface off the sampling boundary, where a cell with
    // no outside corner would leave a hole.
    let margin = ((hi - lo).norm() / resolution as f64).max(1e-6) * 2.0;
    (
        Point3::new(lo.x - margin, lo.y - margin, lo.z - margin),
        Point3::new(hi.x + margin, hi.y + margin, hi.z + margin),
    )
}

#[cfg(test)]
mod tests {
    //! Each reading against the application's own code for it, as that code stood before it was
    //! lifted out (`App::placed_graph`, `point_positions`, `plane_frames`, `pick_bounds`,
    //! `display_graph_with_stand_ins`, `is_reference_body` and `draw_traced`'s table), written
    //! out here with the application's state as arguments. The readings are built once and share
    //! what the application worked out again per piece, so this is what says the sharing changed
    //! no answer.

    use super::*;
    use fieldform_scene::{NodeMeta, Place, Turn};
    use nalgebra::Similarity3;

    const RESOLUTION: usize = 64;
    const EMPTY_MM: f64 = 100.0;

    /// `App::resolve`: an unresolvable reference reads as zero, for drawing and sizing.
    fn sizing(document: &Document, r: &ParamRef) -> Option<f64> {
        Some(document.resolve(r).unwrap_or(0.0))
    }

    fn old_placed_graph(doc: &Document, part: &Graph) -> Graph {
        let numbers = |r: &ParamRef| sizing(doc, r);
        with_turns_flattened(doc, part, &numbers)
    }

    fn old_placement_in(doc: &Document, placed: &Graph, target: NodeId) -> Similarity3<f64> {
        let numbers = |r: &ParamRef| sizing(doc, r);
        placement_above(doc, placed, target, None, &numbers)
    }

    fn old_is_reference_body(doc: &Document, part: &Graph, id: NodeId) -> bool {
        let mut id = id;
        loop {
            match part.node(id) {
                Some(Node::Plane { .. }) => return true,
                Some(Node::Lattice { .. }) => return graph_extent(doc, part, id).is_unbounded(),
                Some(Node::Mesh(mesh)) => return doc.mesh(*mesh).is_none_or(|m| !m.is_closed()),
                Some(Node::Transform { input, .. }) => match input.source() {
                    Some(below) => id = below,
                    None => return false,
                },
                _ => return false,
            }
        }
    }

    fn old_display_graph(
        doc: &Document,
        part: &Graph,
        drawn_as_mesh: &dyn Fn(NodeId) -> bool,
    ) -> (Graph, HashMap<NodeId, NodeId>) {
        let consumed = part.consumed();
        // The empty-slot term is here as well as in the real predicate, unlike the other three.
        // This oracle stands for the application's code *before it was lifted out*, and the hole
        // came after the lift — so carrying it is not describing history, it is keeping the
        // agreement assertion about the sharing rather than about this one rule. Without it, the
        // first holed fixture anybody adds reads as the lift having changed an answer.
        let holed = doc.incomplete_in(part);
        drawn_graph(part, &shown_ingredients_in(doc, part), &|id| {
            !is_visible_among(doc, part, &consumed, id)
                || old_is_reference_body(doc, part, id)
                || drawn_as_mesh(id)
                || holed.contains(&id)
        })
    }

    /// **A body with an empty slot is not drawn, and the bodies beside it still are.**
    ///
    /// Half of the owner's §7.2 ruling, and the half that is easy to get backwards: the holed
    /// body is left out so that everything else survives, which is the `is_reference_body`
    /// precedent — "one body that cannot compile used to take the whole viewport with it".
    /// The other half, that a *measurement* refuses rather than answering about what is left,
    /// is asserted in the application, where the measuring is.
    #[test]
    fn a_body_with_an_empty_slot_is_not_drawn_and_its_siblings_still_are() {
        let doc = Document::new();
        let mut part = Graph::empty();
        let fine = part.add(Node::Sphere {
            centre: Place::At([ParamRef::Literal(0.0); 3]),
            radius: ParamRef::Literal(10.0),
        });
        let holed =
            part.add(Node::Shell { input: Default::default(), thickness: ParamRef::Literal(1.0) });
        let above = part.add(Node::Offset {
            input: holed.into(),
            distance: ParamRef::Literal(1.0),
        });

        let (display, _) = display_graph_with_stand_ins(&doc, &part, &|_| false);
        let drawn: Vec<NodeId> = display.nodes().map(|(id, _)| id).collect();
        assert!(drawn.contains(&fine), "the body with nothing wrong with it is still drawn");
        assert!(!drawn.contains(&holed), "the body with the empty slot is not");
        assert!(
            !drawn.contains(&above),
            "nor the one reading it, which is what closed downstream buys here",
        );
    }

    fn old_point_positions(
        doc: &Document,
        part: &Graph,
        with_rows: &[NodeId],
    ) -> Vec<(NodeId, Point3<f64>)> {
        let mut ids: Vec<NodeId> = part
            .nodes()
            .filter(|(_, n)| matches!(n, Node::Point { .. }))
            .map(|(id, _)| id)
            .collect();
        ids.sort_by_key(|id| id.raw());
        let placed = old_placed_graph(doc, part);
        ids.into_iter()
            .filter(|id| is_visible_in(doc, part, *id) && !with_rows.contains(id))
            .filter_map(|id| {
                let at = graph_extent(doc, part, id).centre()?;
                Some((id, old_placement_in(doc, &placed, id) * at))
            })
            .collect()
    }

    fn old_plane_frames(
        doc: &Document,
        part: &Graph,
    ) -> Vec<(NodeId, Point3<f64>, Rotation3<f64>)> {
        let mut ids: Vec<NodeId> = part
            .nodes()
            .filter(|(_, n)| matches!(n, Node::Plane { .. }))
            .map(|(id, _)| id)
            .collect();
        ids.sort_by_key(|id| id.raw());
        let placed = old_placed_graph(doc, part);
        let numbers = |r: &ParamRef| sizing(doc, r);
        ids.into_iter()
            .filter(|id| is_visible_in(doc, part, *id))
            .filter_map(|id| {
                let Some(Node::Plane { origin, flipped, .. }) = part.node(id) else {
                    return None;
                };
                let (origin, flipped) = (*origin, *flipped);
                let place = old_placement_in(doc, &placed, id);
                let mut facing = place.isometry.rotation.to_rotation_matrix();
                if flipped {
                    facing = facing * Rotation3::from_euler_angles(std::f64::consts::PI, 0.0, 0.0);
                }
                let at = resolve_place(doc, part, &origin, &numbers)
                    .unwrap_or_else(|_| Point3::origin());
                Some((id, place * at, facing))
            })
            .collect()
    }

    fn old_pick_bounds(
        doc: &Document,
        part: &Graph,
        with_rows: &[NodeId],
        drawn_as_mesh: &dyn Fn(NodeId) -> bool,
    ) -> (Point3<f64>, Point3<f64>) {
        let mut lo = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        let graph = old_display_graph(doc, part, drawn_as_mesh).0;
        for body in graph.roots() {
            if let Some((blo, bhi)) = graph_extent(doc, &graph, body).as_box() {
                lo = Point3::new(lo.x.min(blo.x), lo.y.min(blo.y), lo.z.min(blo.z));
                hi = Point3::new(hi.x.max(bhi.x), hi.y.max(bhi.y), hi.z.max(bhi.z));
            }
        }
        for (_, at) in old_point_positions(doc, part, with_rows) {
            lo = Point3::new(lo.x.min(at.x), lo.y.min(at.y), lo.z.min(at.z));
            hi = Point3::new(hi.x.max(at.x), hi.y.max(at.y), hi.z.max(at.z));
        }
        if lo.x.is_finite() && (hi - lo).norm() < 1e-3 {
            let half = EMPTY_MM * 0.5;
            let centre = Point3::from((lo.coords + hi.coords) * 0.5);
            return (
                Point3::new(centre.x - half, centre.y - half, centre.z - half),
                Point3::new(centre.x + half, centre.y + half, centre.z + half),
            );
        }
        if !lo.x.is_finite() {
            let half = EMPTY_MM * 0.5;
            return (Point3::new(-half, -half, -half), Point3::new(half, half, half));
        }
        let margin = ((hi - lo).norm() / RESOLUTION as f64).max(1e-6) * 2.0;
        (
            Point3::new(lo.x - margin, lo.y - margin, lo.z - margin),
            Point3::new(hi.x + margin, hi.y + margin, hi.z + margin),
        )
    }

    fn old_values(doc: &Document) -> HashMap<ParamId, f64> {
        doc.parameters
            .keys()
            .filter_map(|id| doc.parameter_value(*id).map(|v| (*id, v)))
            .collect()
    }

    /// Builds the readings once and holds every one of them to the application's own code.
    fn assert_read_as_the_application_read_it(
        what: &str,
        doc: &Document,
        part: &Graph,
        with_rows: &[NodeId],
        drawn_as_mesh: &dyn Fn(NodeId) -> bool,
    ) -> OverlayReadings {
        let numbers = |r: &ParamRef| sizing(doc, r);
        let inputs = OverlayInputs {
            drawn_as_mesh,
            hidden_with_rows: with_rows,
            resolution: RESOLUTION,
            empty_mm: EMPTY_MM,
            numbers: &numbers,
        };
        let readings = OverlayReadings::build(doc, part, &inputs);

        // A flattened turn is a placement over a copy of the block under a freshly minted id, so
        // two flattenings of one graph never share every id: the graph is held to what it is
        // for, which is where it puts each block.
        let old_placed = old_placed_graph(doc, part);
        assert_eq!(readings.placed.nodes().count(), old_placed.nodes().count(), "{what}: placed");
        for (id, _) in part.nodes() {
            assert_eq!(
                placement_above(doc, &readings.placed, id, None, &numbers),
                old_placement_in(doc, &old_placed, id),
                "{what}: where {:?} is placed",
                part.node(id),
            );
        }
        assert_eq!(readings.points, old_point_positions(doc, part, with_rows), "{what}: points");
        assert_eq!(readings.planes, old_plane_frames(doc, part), "{what}: planes");
        assert_eq!(
            readings.pick_bounds,
            old_pick_bounds(doc, part, with_rows, drawn_as_mesh),
            "{what}: the pick box",
        );
        assert_eq!(readings.values, old_values(doc), "{what}: the parameter table");

        let (display, stand_ins) = display_graph_with_stand_ins(doc, part, drawn_as_mesh);
        let (old_display, old_stand_ins) = old_display_graph(doc, part, drawn_as_mesh);
        let sorted = |graph: &Graph| {
            let mut ids: Vec<NodeId> = graph.nodes().map(|(id, _)| id).collect();
            ids.sort();
            ids
        };
        assert_eq!(sorted(&display), sorted(&old_display), "{what}: the drawn graph's blocks");
        assert_eq!(display.roots(), old_display.roots(), "{what}: the drawn graph's bodies");
        assert_eq!(stand_ins, old_stand_ins, "{what}: the stand-ins");
        for (id, _) in part.nodes() {
            assert_eq!(
                is_reference_body(doc, part, id),
                old_is_reference_body(doc, part, id),
                "{what}: whether {:?} is reference geometry",
                part.node(id),
            );
        }
        readings
    }

    #[test]
    fn points_and_planes_read_at_once_are_where_the_application_drew_them() {
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let r = doc.add_parameter("r", 12.0);
        let twice = doc.add_parameter("twice", 0.0);
        doc.set_parameter_expression(twice, "r * 2").expect("an expression");
        let broken = doc.add_parameter("broken", 0.0);
        doc.set_parameter_expression(broken, "r / 0").expect("stored, and fails to evaluate");

        let mut part = Graph::empty();
        part.add(Node::Sphere { centre: Place::origin(), radius: ParamRef::Named(r) });
        // A point under a placement that moves, turns and scales it, by a named number.
        let pin = part.add(Node::Point { at: [lit(30.0), lit(0.0), lit(0.0)] });
        part.add(Node::Transform {
            input: pin.into(),
            translation: [ParamRef::Named(twice), lit(0.0), lit(10.0)],
            rotation: [lit(0.0), lit(0.0), lit(90.0)],
            scale: lit(2.0),
            about: Place::origin(),
        });
        // A plane turned by its own numbers and flipped, and one sitting on a point.
        part.add(Node::Plane {
            origin: Place::at(5.0, 5.0, 5.0),
            flipped: true,
            turn: Turn::By([lit(30.0), lit(0.0), lit(45.0)]),
        });
        let seat = part.add(Node::Point { at: [lit(0.0), lit(-40.0), lit(0.0)] });
        part.add(Node::Plane {
            origin: Place::From { node: seat },
            flipped: false,
            turn: Turn::By([lit(90.0), lit(0.0), lit(0.0)]),
        });
        // One point hidden by its eye, one gone with its row, one at a number that fails.
        let hidden = part.add(Node::Point { at: [lit(70.0); 3] });
        doc.node_meta.insert(hidden, NodeMeta { hidden: true, ..NodeMeta::default() });
        let joint = part.add(Node::Point { at: [lit(-70.0); 3] });
        part.add(Node::Point { at: [ParamRef::Named(broken), lit(5.0), lit(5.0)] });

        let readings =
            assert_read_as_the_application_read_it("placed", &doc, &part, &[joint], &|_| false);
        assert!(readings.points.iter().all(|(id, _)| *id != hidden && *id != joint));
        assert_eq!(readings.planes.len(), 2, "both planes draw");
        assert_eq!(readings.values.len(), 2, "the failure is absent: {:?}", readings.values);
        assert_eq!(readings.values.get(&twice), Some(&24.0));
    }

    #[test]
    fn the_pick_box_read_at_once_is_the_one_the_application_sized_to_whatever_is_hidden_or_shown() {
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut part = Graph::empty();
        // A ball inside the offset that consumes it, asked to be drawn as well.
        let ball = part.add(Node::Sphere { centre: Place::origin(), radius: lit(10.0) });
        part.add(Node::Offset { input: ball.into(), distance: lit(2.0) });
        doc.node_meta.insert(ball, NodeMeta { shown: true, ..NodeMeta::default() });
        // A plane on its own, which is reference geometry and stretches nothing.
        part.add(Node::Plane { origin: Place::origin(), flipped: false, turn: Turn::default() });
        // A body hidden by its eye, and one the application is drawing as a mesh instead.
        let far = |x: f64| Node::Sphere { centre: Place::at(x, 0.0, 0.0), radius: lit(5.0) };
        let hidden = part.add(far(-300.0));
        doc.node_meta.insert(hidden, NodeMeta { hidden: true, ..NodeMeta::default() });
        let meshed = part.add(far(500.0));
        part.add(Node::Point { at: [lit(0.0), lit(0.0), lit(60.0)] });

        let drawn_as_mesh = |id: NodeId| id == meshed;
        let readings =
            assert_read_as_the_application_read_it("hidden", &doc, &part, &[], &drawn_as_mesh);
        let (lo, hi) = readings.pick_bounds;
        assert!(lo.x > -300.0 && hi.x < 500.0, "neither the hidden body nor the meshed one");
        assert!(hi.z > 60.0, "the point is in it");

        // And with nothing drawn as a mesh, the far body is back in the box.
        let readings =
            assert_read_as_the_application_read_it("traced", &doc, &part, &[], &|_| false);
        assert!(readings.pick_bounds.1.x > 500.0);
    }

    #[test]
    fn an_empty_part_or_one_spot_gets_the_box_the_application_gave_it() {
        let lit = ParamRef::Literal;
        let doc = Document::new();
        let empty = assert_read_as_the_application_read_it(
            "empty",
            &doc,
            &Graph::empty(),
            &[],
            &|_| false,
        );
        let half = EMPTY_MM * 0.5;
        let around_the_origin = (Point3::new(-half, -half, -half), Point3::new(half, half, half));
        assert_eq!(empty.pick_bounds, around_the_origin);

        let mut spot = Graph::empty();
        spot.add(Node::Point { at: [lit(20.0), lit(0.0), lit(0.0)] });
        spot.add(Node::Point { at: [lit(20.0), lit(0.0), lit(0.0)] });
        let one = assert_read_as_the_application_read_it("one spot", &doc, &spot, &[], &|_| false);
        assert_eq!(one.points.len(), 2);
        assert_eq!(one.pick_bounds.0, Point3::new(20.0 - half, -half, -half));
    }

    /// **Kept readings are read once for a key, and again whenever it moves** (refactor plan step
    /// 3.3). Four frames at one revision read the part once, where the application used to work
    /// every reading out again each frame; and a key that moves back is a move all the same, since
    /// only the last reading is kept.
    #[test]
    fn kept_readings_are_read_once_a_key_and_again_whenever_it_moves() {
        let lit = ParamRef::Literal;
        let doc = Document::new();
        let mut part = Graph::empty();
        part.add(Node::Sphere { centre: Place::origin(), radius: lit(10.0) });
        part.add(Node::Point { at: [lit(0.0), lit(0.0), lit(40.0)] });
        let numbers = |r: &ParamRef| sizing(&doc, r);
        let inputs = OverlayInputs {
            drawn_as_mesh: &|_| false,
            hidden_with_rows: &[],
            resolution: RESOLUTION,
            empty_mm: EMPTY_MM,
            numbers: &numbers,
        };
        let reads = std::cell::Cell::new(0);
        let mut kept = KeptReadings::default();
        let mut frame = |key: u64| {
            kept.get(key, |_| {
                reads.set(reads.get() + 1);
                OverlayReadings::build(&doc, &part, &inputs)
            })
        };

        for _ in 0..4 {
            frame(1);
        }
        assert_eq!(reads.get(), 1, "four frames at one revision");
        let moved = frame(2);
        frame(2);
        assert_eq!(reads.get(), 2, "the key moved once");
        frame(1);
        assert_eq!(reads.get(), 3, "and moving back is a move");

        let fresh = OverlayReadings::build(&doc, &part, &inputs);
        assert_eq!(moved.points, fresh.points);
        assert_eq!(moved.pick_bounds, fresh.pick_bounds);
    }
}
