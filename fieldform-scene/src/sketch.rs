// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A sketch: a curve that lies on a plane block and follows it.
//!
//! **A 2D sketch is a curve whose control points are held in the plane's own coordinates.** The
//! world positions the curve stores are derived from those and the plane's frame, so moving the
//! plane -- dragging it, turning it, sliding it over a surface, changing the body it stands on --
//! moves the sketch, and whatever is built on the sketch follows. A **3D sketch** is any curve
//! without this record. A sketch becomes one when a control point leaves its plane on purpose,
//! and never goes back on its own (the owner, 2026-10-31).
//!
//! **A side record and not a field on the curve**, for the reason a point on a surface is one:
//! the curve is an ordinary curve with its points stored, so everything that reads a curve reads
//! this one and a build that does not know this record opens the file and draws the sketch where
//! it last lay. It is also the one place a curve reads a block -- the dependency direction the
//! graph otherwise refuses -- which is why it is resolved in order at the start of a rebuild, as a
//! surface point is, and never by the compile.

use serde::{Deserialize, Serialize};

use crate::id::NodeId;
use nalgebra::{Point3, Vector3};

/// Where a curve lies on a plane block, and how it was last laid there.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sketch {
    /// The plane it lies on: a plane block of the active part's graph, by id.
    pub plane: NodeId,
    /// Each control point across the plane's own X and Y, in millimetres: the sketch's truth.
    pub local: Vec<[f64; 2]>,
    /// Live: re-laid from the plane after every rebuild. Fixed: left where it is, which is what a
    /// sketch becomes when a point leaves its plane, when the plane goes, or when the plane comes
    /// to stand on the sketch.
    pub live: bool,
    /// The plane's frame as the sketch was last laid on it, in world. What tells a plane that
    /// moved from a curve that was edited: with the frame unchanged a point off the laid position
    /// is the person's doing, and with it changed the plane's. Written by the refresh, never an
    /// input to the layout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub laid: Option<Laid>,
}

/// A plane's frame as a sketch last used it: its origin and its X and Y, unit length, in world.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Laid {
    pub origin: [f64; 3],
    pub x: [f64; 3],
    pub y: [f64; 3],
}

impl Laid {
    pub fn new(origin: Point3<f64>, x: Vector3<f64>, y: Vector3<f64>) -> Self {
        Laid { origin: origin.coords.into(), x: x.into(), y: y.into() }
    }

    /// The plane's normal: what its X and Y leave.
    pub fn normal(&self) -> Vector3<f64> {
        Vector3::from(self.x).cross(&Vector3::from(self.y))
    }

    /// A point of the plane, in world.
    pub fn lay(&self, local: [f64; 2]) -> Point3<f64> {
        Point3::from(self.origin) + Vector3::from(self.x) * local[0] + Vector3::from(self.y) * local[1]
    }

    /// A world point across the plane, and how far off the plane it is, signed along the normal.
    pub fn local_of(&self, world: Point3<f64>) -> ([f64; 2], f64) {
        let d = world - Point3::from(self.origin);
        ([d.dot(&Vector3::from(self.x)), d.dot(&Vector3::from(self.y))], d.dot(&self.normal()))
    }

    /// Whether two frames are the same frame, to the arithmetic.
    pub fn same_as(&self, other: &Laid) -> bool {
        let close = |a: [f64; 3], b: [f64; 3]| (Vector3::from(a) - Vector3::from(b)).norm() < 1e-9;
        close(self.origin, other.origin) && close(self.x, other.x) && close(self.y, other.y)
    }
}

impl Sketch {
    /// A sketch laid fresh: its points read off the curve against the frame it was drawn on.
    pub fn of(plane: NodeId, frame: &Laid, points: &[Point3<f64>]) -> Self {
        Sketch {
            plane,
            local: points.iter().map(|p| frame.local_of(*p).0).collect(),
            live: true,
            laid: Some(*frame),
        }
    }

    /// The curve this record belongs to is named by the map key; the plane is the one block it
    /// reads, so a loop through it can be found and a deletion can let go of it.
    pub fn source(&self) -> NodeId {
        self.plane
    }
}

/// How far off its plane a sketch's point may read and still be on it, in millimetres.
///
/// A thousandth: below anything a drawing shows, above what a plane standing on a surface point
/// is found to — that point is bisected to a thousandth of its body, so a point typed at the
/// round number the plane was asked for reads a few hundred-thousandths off it. Points laid by the
/// sketch and drags projected onto the plane land to rounding; anything further off was put there.
pub const ON_PLANE_MM: f64 = 1e-3;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::CurveId;

    fn tilted() -> Laid {
        let x = Vector3::new(1.0, 0.0, 0.0);
        let y = Vector3::new(0.0, 1.0, 1.0).normalize();
        Laid::new(Point3::new(10.0, 0.0, 5.0), x, y)
    }

    /// **Laying a point and reading it back are inverse**, and a point off the plane reads its
    /// distance along the normal, signed.
    #[test]
    fn a_point_laid_on_a_plane_reads_back_as_the_numbers_it_was_laid_from() {
        let frame = tilted();
        for local in [[0.0, 0.0], [3.0, -4.0], [-12.5, 7.25]] {
            let (back, off) = frame.local_of(frame.lay(local));
            assert!((back[0] - local[0]).abs() < 1e-12 && (back[1] - local[1]).abs() < 1e-12, "{local:?} came back {back:?}");
            assert!(off.abs() < 1e-12);
        }
        let (_, off) = frame.local_of(frame.lay([1.0, 1.0]) + frame.normal() * 2.5);
        assert!((off - 2.5).abs() < 1e-12, "{off}");
        assert!(frame.same_as(&frame) && !frame.same_as(&Laid::new(Point3::origin(), Vector3::x(), Vector3::y())));
    }

    /// **The record survives a save**, numbers as numbers.
    #[test]
    fn a_sketch_round_trips() {
        let frame = tilted();
        let points = [frame.lay([1.0, 2.0]), frame.lay([30.0, 2.0]), frame.lay([30.0, 20.0])];
        let sketch = Sketch::of(NodeId::new(), &frame, &points);
        for (read, meant) in sketch.local.iter().zip([[1.0, 2.0], [30.0, 2.0], [30.0, 20.0]]) {
            assert!((read[0] - meant[0]).abs() < 1e-9 && (read[1] - meant[1]).abs() < 1e-9, "{read:?} for {meant:?}");
        }
        assert!(sketch.live && sketch.laid.is_some_and(|l| l.same_as(&frame)));
        let json = serde_json::to_string(&sketch).unwrap();
        assert_eq!(serde_json::from_str::<Sketch>(&json).unwrap(), sketch, "{json}");
        let by_curve: std::collections::BTreeMap<CurveId, Sketch> = [(CurveId::new(), sketch)].into_iter().collect();
        let json = serde_json::to_string(&by_curve).unwrap();
        assert_eq!(serde_json::from_str::<std::collections::BTreeMap<CurveId, Sketch>>(&json).unwrap(), by_curve);
    }
}
