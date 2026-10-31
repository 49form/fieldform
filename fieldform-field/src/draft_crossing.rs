// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How far a draft can move a section's walls before the section stops being a section.
//!
//! A draft is an offset of the section, growing with the distance from the start plane. Two
//! things can go wrong with an offset, and only one of them depends on how the corners are cut:
//!
//! - **The section closes.** An inward offset by the section's depth -- its inradius, the
//!   radius of the largest circle inside it -- leaves nothing, so the solid ends short of the
//!   depth that was typed. True of either corner, since the walls move the same way.
//! - **The mitred outline crosses itself.** A sharp draft walks the section's own polygon with
//!   each vertex moved along its miter vector. That *is* the mitred offset only while the moved
//!   polygon stays simple: once an edge shrinks through nothing, or a corner runs into an edge
//!   across the section, the walk reads a self-crossing polygon -- whose parity puts material
//!   where none should be, a solid that regrows above the point it closed, with no message. A
//!   round draft reads `d(q) − t`, the offset of a distance field, which never crosses itself.
//!
//! So a round draft is held to the depth and a sharp one to the first crossing, which for a
//! sharp draft also comes no later than the depth: a polygon that closes has crossed itself on
//! the way.

use crate::{CompileError, Footprint};
use nalgebra::{Point2, Vector2};

/// The smallest `τ` in `(0, limit]` at which the polygon with vertices `p + sign·τ·m` stops
/// being simple, or `None` if it stays simple over the whole range.
///
/// **Exact, not sampled.** Every vertex moves on a straight line, so whether vertex `j` lies on
/// the line of edge `(a, b)` is a quadratic in `τ` -- `cross(b − a, j − a)` with each point
/// linear in `τ` -- and whether it lies *on the segment* is a dot product at the root. The first
/// contact of two segments that start apart always has an endpoint of one on the other: a
/// transversal crossing of two interiors at `τ*` would already be a crossing just before it.
/// So the first `τ` at which any vertex touches an edge it is not an end of is the first `τ` at
/// which the outline meets itself. An edge shrinking through nothing is one of these too -- its
/// far end lands on the edge before it -- and a whole outline turning inside out, which a
/// triangle does with no edge left to touch, is caught by its area passing nought.
///
/// Quadratic in the number of vertices, pruned by the box each vertex and edge sweeps over the
/// range, so a pair that cannot meet costs a box test.
pub(crate) fn mitre_crossing(
    points: &[Point2<f64>],
    miters: &[Vector2<f64>],
    sign: f64,
    limit: f64,
) -> Option<f64> {
    let n = points.len();
    if n < 3 || miters.len() != n || !(limit > 0.0) {
        return None;
    }
    let moved = |i: usize, tau: f64| points[i] + miters[i] * (sign * tau);
    let velocity = |i: usize| miters[i] * sign;
    // A scale for the tolerances, so a profile in microns and one in metres are judged alike.
    let (lo, hi) = points.iter().fold(
        (Point2::new(f64::INFINITY, f64::INFINITY), Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY)),
        |(lo, hi), p| (Point2::new(lo.x.min(p.x), lo.y.min(p.y)), Point2::new(hi.x.max(p.x), hi.y.max(p.y))),
    );
    let size = (hi - lo).norm().max(1e-12);
    let pad = size * 1e-9;
    // The box a vertex sweeps over the range.
    let swept: Vec<(Point2<f64>, Point2<f64>)> = (0..n)
        .map(|i| {
            let (a, b) = (points[i], moved(i, limit));
            (
                Point2::new(a.x.min(b.x) - pad, a.y.min(b.y) - pad),
                Point2::new(a.x.max(b.x) + pad, a.y.max(b.y) + pad),
            )
        })
        .collect();
    let overlaps = |a: (Point2<f64>, Point2<f64>), b: (Point2<f64>, Point2<f64>)| {
        a.0.x <= b.1.x && b.0.x <= a.1.x && a.0.y <= b.1.y && b.0.y <= a.1.y
    };
    let cross = |u: Vector2<f64>, v: Vector2<f64>| u.x * v.y - u.y * v.x;

    let mut first: Option<f64> = None;
    let mut consider = |tau: f64| {
        if tau > 0.0 && tau <= limit && first.is_none_or(|f| tau < f) {
            first = Some(tau);
        }
    };

    for i in 0..n {
        let k = (i + 1) % n;
        let edge_box = {
            let (a, b) = (swept[i], swept[k]);
            (Point2::new(a.0.x.min(b.0.x), a.0.y.min(b.0.y)), Point2::new(a.1.x.max(b.1.x), a.1.y.max(b.1.y)))
        };
        let (e0, de) = (points[k] - points[i], velocity(k) - velocity(i));
        for j in 0..n {
            if j == i || j == k || !overlaps(swept[j], edge_box) {
                continue;
            }
            let (w0, dw) = (points[j] - points[i], velocity(j) - velocity(i));
            let c0 = cross(e0, w0);
            let c1 = cross(e0, dw) + cross(de, w0);
            let c2 = cross(de, dw);
            for tau in quadratic_roots(c2, c1, c0, size) {
                if !(tau > 0.0 && tau <= limit) {
                    continue;
                }
                let e = e0 + de * tau;
                let w = w0 + dw * tau;
                let length2 = e.norm_squared();
                let on_segment = if length2 <= (size * 1e-12).powi(2) {
                    // The edge has shrunk to a point here, which every line passes through:
                    // the vertex touches it only if it is at that point.
                    w.norm() <= size * 1e-9
                } else {
                    let s = w.dot(&e) / length2;
                    (-1e-9..=1.0 + 1e-9).contains(&s)
                };
                if on_segment {
                    consider(tau);
                }
            }
        }
    }

    // **Turned inside out.** Twice the signed area is a quadratic in `τ` too; where it passes
    // nought the outline has no inside left, whichever way it was wound.
    let (mut a0, mut a1, mut a2) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let k = (i + 1) % n;
        let (p, q) = (points[i].coords, points[k].coords);
        let (u, v) = (velocity(i), velocity(k));
        a0 += cross(p, q);
        a1 += cross(p, v) + cross(u, q);
        a2 += cross(u, v);
    }
    for tau in quadratic_roots(a2, a1, a0, size * size) {
        consider(tau);
    }
    first
}

/// The real roots of `a·τ² + b·τ + c`, in either order. `scale` is the size of the coefficients'
/// units, which is what says when `a` is nought for a quadratic that has gone linear.
fn quadratic_roots(a: f64, b: f64, c: f64, scale: f64) -> Vec<f64> {
    let tiny = scale.abs().max(1e-300) * 1e-14;
    if a.abs() <= tiny {
        if b.abs() <= tiny {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return Vec::new();
    }
    // The form that does not cancel: one root from `q`, the other from the product `c / a`.
    let q = -0.5 * (b + if b >= 0.0 { disc.sqrt() } else { -disc.sqrt() });
    let mut roots = Vec::with_capacity(2);
    if q != 0.0 {
        roots.push(q / a);
        roots.push(c / q);
    } else {
        roots.push(0.0);
    }
    roots
}

/// One side of a drafted solid: how long it runs from the start plane, and its draft's tangent.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DraftedSide {
    pub length: f64,
    pub tangent: f64,
}

/// Refuses a draft that moves the section's walls further than the section has room for, on
/// any side of the solid, naming how far they move, how far they may, and the steepest draft
/// that would fit.
///
/// **Held to the far end of each side**, which is where a draft has moved its walls furthest.
/// The section is read past the ends too, but there the slab is what bounds the reading.
///
/// A round draft is held to the section's depth, which [`Footprint::depth`] gives as a floor
/// under the inradius -- conservative by at most the spacing of its samples, so it may refuse
/// a draft that just fits, and never passes one that closes the section. A sharp one is held to
/// [`mitre_crossing`], which is exact.
pub(crate) fn check_draft_room(
    node: fieldform_scene::NodeId,
    footprint: &Footprint,
    sides: &[DraftedSide],
    corners: fieldform_scene::Corners,
) -> Result<(), CompileError> {
    let sharp = !corners.is_default();
    // The depth is sampled, so it is asked only when some side narrows.
    let mut depth: Option<f64> = None;
    for side in sides {
        let moved = side.length.abs() * side.tangent;
        if moved == 0.0 || !moved.is_finite() {
            continue;
        }
        let refuse = |room: f64, what: &'static str| {
            let room = room.max(0.0);
            Err(CompileError::DraftCrossesTheSection {
                node,
                degrees: side.tangent.atan().to_degrees().abs(),
                way: if moved < 0.0 { "inward" } else { "outward" },
                what,
                room,
                along: room / side.tangent.abs(),
                length: side.length.abs(),
                fits: (room / side.length.abs()).atan().to_degrees(),
                or_round: if sharp { ", or round the corners" } else { "" },
            })
        };
        let closed = match (sharp, footprint) {
            (true, Footprint::Closed(profile)) => {
                let sign = moved.signum();
                mitre_crossing(profile.points(), profile.miters(), sign, moved.abs())
                    .map(|tau| (tau, "makes the mitred outline cross itself"))
            }
            // Rounding never crosses; it can only close the section, which only narrowing does.
            _ if moved < 0.0 => {
                let room = *depth.get_or_insert_with(|| footprint.depth());
                (-moved >= room).then_some((room, "closes the section to nothing"))
            }
            _ => None,
        };
        if let Some((room, what)) = closed {
            return refuse(room, what);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Depth, Extrusion, Profile, ScalarField};
    use fieldform_scene::Corners;
    use nalgebra::Point3;

    /// A slot 40 mm long and 4 mm wide: its depth is 2 mm, so a draft that moves its walls
    /// 2 mm in has closed it, round or sharp.
    fn slot() -> Profile {
        Profile::new(vec![
            Point2::new(-20.0, -2.0),
            Point2::new(20.0, -2.0),
            Point2::new(20.0, 2.0),
            Point2::new(-20.0, 2.0),
        ])
        .unwrap()
    }

    /// An L: 30 by 30 with arms 6 mm thick, so five convex corners and one reflex. Its depth is
    /// 3 mm, at the middle of either arm.
    fn l_shape() -> Profile {
        Profile::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(30.0, 0.0),
            Point2::new(30.0, 6.0),
            Point2::new(6.0, 6.0),
            Point2::new(6.0, 30.0),
            Point2::new(0.0, 30.0),
        ])
        .unwrap()
    }

    fn extrude(profile: Profile, corners: Corners, tangent: f64, height: f64) -> Extrusion {
        Extrusion {
            rims: Default::default(),
            footprint: profile.into(),
            start: 0.0,
            height,
            depth: Depth::One,
            draft_tangent: tangent,
            corners,
        }
    }

    /// **The fault itself.** A slot drafted in by 3 mm over 20 mm has closed by 13.3 mm up, so
    /// at 19 mm up there is nothing -- and the sharp draft's walk read a solid there, because
    /// the moved outline has turned inside out and its parity counts that as an inside. The
    /// round one reads the empty section it is.
    #[test]
    fn a_sharp_draft_past_the_crossing_reads_material_where_the_section_has_closed() {
        let tangent = -3.0 / 20.0;
        let round = extrude(slot(), Corners::Round, tangent, 20.0);
        let sharp = extrude(slot(), Corners::Sharp, tangent, 20.0);
        let above_the_close = Point3::new(0.0, 0.0, 19.0);
        assert!(round.distance(above_the_close) > 0.0, "the round section has closed");
        assert!(
            sharp.distance(above_the_close) < 0.0,
            "the walk reads the inverted outline as material, which is why it must be refused"
        );
        // And so the compile does refuse it, with the room the section had.
        let err = check_draft_room(
            fieldform_scene::NodeId::new(),
            &slot().into(),
            &[DraftedSide { length: 20.0, tangent }],
            Corners::Sharp,
        )
        .expect_err("a sharp draft past its crossing");
        assert!(matches!(err, CompileError::DraftCrossesTheSection { room, .. } if (room - 2.0).abs() < 1e-9));
    }

    /// **The crossing is where the outline first meets itself, to the root.** An L's arms are
    /// 6 mm thick, so an inward mitre closes each at 3 mm; outward, nothing ever meets, since
    /// the L's one reflex corner moves into its own notch and no edge faces it.
    #[test]
    fn an_inward_mitre_crosses_where_the_thinnest_arm_closes() {
        let l = l_shape();
        let inward = mitre_crossing(l.points(), l.miters(), -1.0, 10.0).expect("the arms close");
        assert!((inward - 3.0).abs() < 1e-9, "crossed at {inward}");
        assert_eq!(mitre_crossing(l.points(), l.miters(), 1.0, 50.0), None);
        // A range that stops short of it finds nothing.
        assert_eq!(mitre_crossing(l.points(), l.miters(), -1.0, 2.999), None);
    }

    /// **Outward can cross too.** A U whose notch is 4 mm wide closes the notch when its walls
    /// have moved 2 mm out, and a mitre then folds the outline over itself there; a round draft
    /// unions the two prongs, which is what an offset is.
    #[test]
    fn an_outward_mitre_crosses_where_a_notch_closes() {
        let u = Profile::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(24.0, 0.0),
            Point2::new(24.0, 30.0),
            Point2::new(14.0, 30.0),
            Point2::new(14.0, 10.0),
            Point2::new(10.0, 10.0),
            Point2::new(10.0, 30.0),
            Point2::new(0.0, 30.0),
        ])
        .unwrap();
        let outward = mitre_crossing(u.points(), u.miters(), 1.0, 10.0).expect("the notch closes");
        assert!((outward - 2.0).abs() < 1e-9, "crossed at {outward}");
        let sides = [DraftedSide { length: 20.0, tangent: 2.5 / 20.0 }];
        assert!(check_draft_room(fieldform_scene::NodeId::new(), &u.clone().into(), &sides, Corners::Sharp).is_err());
        check_draft_room(fieldform_scene::NodeId::new(), &u.into(), &sides, Corners::Round).expect("a round draft only unions");
    }

    /// **Refused just past the crossing and accepted just before it**, for either corner: the
    /// slot's walls may move 2 mm in, so over 20 mm a tangent of 0.0999 fits and 0.1001 does not.
    #[test]
    fn a_draft_is_refused_just_past_the_crossing_and_accepted_just_before_it() {
        for corners in Corners::ALL {
            let room = |tangent: f64| {
                check_draft_room(fieldform_scene::NodeId::new(), &slot().into(), &[DraftedSide { length: 20.0, tangent }], corners)
            };
            room(-0.0999).unwrap_or_else(|e| panic!("{corners:?} refused a draft that fits: {e}"));
            assert!(room(-0.1001).is_err(), "{corners:?} passed a draft that closes the slot");
            // Outward, a slot has nothing to cross.
            room(1.0).unwrap_or_else(|e| panic!("{corners:?} refused a widening draft: {e}"));
        }
    }

    /// **The refusal names the numbers**: how far the walls may move, how far along that is,
    /// and the steepest draft that would fit -- each one a fix.
    #[test]
    fn the_refusal_names_how_far_the_walls_may_move_and_the_draft_that_fits() {
        // 6° in over 30 mm moves the walls 3.15 mm; the slot has 2, reached 19.03 mm along,
        // and atan(2/30) is 3.81°.
        let tangent = -(6f64.to_radians().tan());
        let err = check_draft_room(
            fieldform_scene::NodeId::new(),
            &slot().into(),
            &[DraftedSide { length: 30.0, tangent }],
            Corners::Sharp,
        )
        .expect_err("past the crossing");
        let text = err.to_string();
        for number in ["6.0\u{b0} inward", "2.00 mm", "19.03 mm", "30.00 mm", "3.8\u{b0}"] {
            assert!(text.contains(number), "{number:?} is missing from: {text}");
        }
        assert!(text.contains("round the corners"), "a sharp draft is offered rounding: {text}");
        let round = check_draft_room(
            fieldform_scene::NodeId::new(),
            &slot().into(),
            &[DraftedSide { length: 30.0, tangent }],
            Corners::Round,
        )
        .expect_err("past the depth")
        .to_string();
        assert!(round.contains("closes the section to nothing"), "{round}");
        assert!(!round.contains("round the corners"), "a round draft is already round: {round}");
    }

    /// **Each side of a two-sided solid is held to its own draft**, and the one that crosses is
    /// the one named.
    #[test]
    fn each_side_is_held_to_its_own_draft() {
        let sides = [
            DraftedSide { length: 20.0, tangent: -0.05 },
            DraftedSide { length: 10.0, tangent: -0.25 },
        ];
        match check_draft_room(fieldform_scene::NodeId::new(), &slot().into(), &sides, Corners::Sharp) {
            Err(CompileError::DraftCrossesTheSection { length, .. }) => assert_eq!(length, 10.0),
            other => panic!("side two crosses at 2.5 mm, and this said {other:?}"),
        }
    }

    /// **Up to its crossing a sharp draft never overstates**, which is the reason the crossing
    /// is the limit rather than the depth: just inside it the walk is still a simple outline.
    #[test]
    fn a_sharp_draft_just_inside_its_crossing_never_overstates() {
        let tangent = -2.95 / 20.0;
        let solid = extrude(l_shape(), Corners::Sharp, tangent * 1.0, 20.0);
        let mut points = Vec::new();
        for i in 0..14 {
            for j in 0..14 {
                for k in 0..6 {
                    points.push(Point3::new(
                        -4.0 + i as f64 * 2.7,
                        -4.0 + j as f64 * 2.7,
                        -3.0 + k as f64 * 5.0,
                    ));
                }
            }
        }
        let worst = crate::overstatement(&solid, &points, &crate::probe_directions(), 0.01, 1.5);
        assert!(worst < 0.02, "overstated by {worst} mm");
    }
}
