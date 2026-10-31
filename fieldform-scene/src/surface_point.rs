// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A point on a body's surface: a question with a remembered answer.
//!
//! **The question** is where the point was put and which way the surface faced there -- or, typed
//! from the console, a ray or a seed -- in millimetres, in a frame that belongs to the body or to a
//! plane the user chose, and never in fractions of a bounding box: a box belongs to the whole body,
//! not to the place on it, so growing the body elsewhere or turning it used to move every place
//! measured across one. **The answer** is the point's own stored position. Where the question has
//! several answers -- a line through several walls, a seed equally near two -- the solve takes the
//! one facing the same way nearest where the point already stands, so the point moves when its
//! numbers move or the surface under it moves, and otherwise stays.
//!
//! **A side record and not a block of its own.** The block is an ordinary `Node::Point`, whose `at`
//! the solve writes as three literals, so everything that stands on a point stands on this one, `at`
//! *is* the remembered answer with no second copy to fall out of step with it, and a build that does
//! not know this record still opens the file and shows the point where it last stood. This says how
//! that position is derived (`Document::surface_points`), the arrangement a surface axis has for its
//! line. It names a body by id, never one of its faces (the boundary rule in `graph/mod.rs`).

use serde::{Deserialize, Serialize};

use crate::graph::{Axis, ParamRef};
use crate::id::NodeId;
use crate::surface_axis::{axis_index, number_or_ref, seed_numbers};

/// Where a point stands on a body, and what the last solve found there.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SurfacePoint {
    /// The body it stands on: a block of the active part's graph, by id.
    pub body: NodeId,
    /// What the guide's numbers are measured in.
    pub frame: GuideFrame,
    /// The question.
    pub guide: Guide,
    /// Live: solved again after every rebuild. Fixed: left where it is, as a plane's Live and Fixed
    /// work, which is also what a point becomes when its body comes to depend on it.
    pub live: bool,
    /// What the last solve found besides the position, in world terms. Written by the solve with
    /// the point's `at`, read by whatever stands on the point, and never an input to the solve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solved: Option<Solved>,
}

/// The surface's own directions at the point, which an axis or a plane on it reads.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Solved {
    pub normal: [f64; 3],
    /// The roll about the normal: the frame's first across axis laid into the tangent plane.
    pub x: [f64; 3],
}

/// What a guide's millimetres are measured in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum GuideFrame {
    /// The body's own frame, which moves and turns with it and depends on nothing of its extent.
    Body,
    /// A plane block's frame: its origin, its normal as Z, its own X as X.
    Plane { node: NodeId },
}

/// The question a point asks of its body.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Guide {
    /// Where the point was put and the way the surface faced there, both in the frame: what a click
    /// or a drag leaves behind, and the only kind the viewport makes.
    ///
    /// The answer is the surface met by the line through `at` along `facing`, facing that way,
    /// nearest where the point stood. **Along the facing and not the nearest surface**, because a
    /// face that moves away along its own normal -- a block made taller, a boss made wider -- is
    /// then further from the spot than the face beside it, and the point would hop round the corner.
    /// **Facing that way**, because the inner face of a thin wall lies on the same line and is not
    /// the face the point was put on. Neither needs an axis chosen or a U and a V typed, which is
    /// what the ray below asked of a click.
    Spot {
        #[serde(with = "seed_numbers::required")]
        at: [ParamRef; 3],
        /// Unit length, in the frame. Written once by the gesture; the solve never rewrites it.
        facing: [f64; 3],
    },
    /// A line along the frame's `along` axis through `(u, v)` on the other two, which follow
    /// [`across`]. By default it comes from the plus side and travels toward the minus side, the way
    /// you look down at the top of something; `flipped` reverses it. Typed from the console, for a
    /// place given as numbers on a grid.
    Ray {
        along: Axis,
        flipped: bool,
        #[serde(with = "number_or_ref")]
        u: ParamRef,
        #[serde(with = "number_or_ref")]
        v: ParamRef,
        facing: Facing,
    },
    /// The surface point nearest a seed at three millimetre coordinates in the frame.
    Nearest {
        #[serde(with = "seed_numbers::required")]
        at: [ParamRef; 3],
    },
}

/// Which crossings of a ray count.
///
/// **What stops a point swapping walls.** Move a thin wall outward and its inner face can end up
/// nearer the old outer position than the new outer face is, but the inner face is the other facing
/// and does not count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Facing {
    /// The surface faces back along the ray, so the ray enters material there: what you see
    /// looking down it.
    #[default]
    Front,
    /// The surface faces away, so the ray leaves material there: the inside of a cup's far wall.
    Back,
}

impl Facing {
    /// Whether a crossing that enters or leaves material is one this facing counts.
    pub fn counts(self, entering: bool) -> bool {
        entering == (self == Facing::Front)
    }
}

/// The indices of the two axes across `along`, cyclic so a look down each is right-handed: a ray
/// along Z has U on X and V on Y. The rule a surface axis follows, so the two read alike.
pub fn across(along: Axis) -> (usize, usize) {
    let d = axis_index(along);
    ((d + 1) % 3, (d + 2) % 3)
}

impl SurfacePoint {
    /// The numbers this record carries, so a parameter it names counts as read.
    pub fn params(&self) -> impl Iterator<Item = &ParamRef> {
        let (pair, seed) = match &self.guide {
            Guide::Ray { u, v, .. } => (Some([u, v]), None),
            Guide::Nearest { at } | Guide::Spot { at, .. } => (None, Some(at)),
        };
        pair.into_iter().flatten().chain(seed.into_iter().flatten())
    }

    /// The place in the frame the question is asked from, for the kinds that have one: a spot or a
    /// seed. A ray is asked across a whole line.
    pub fn seed(&self) -> Option<&[ParamRef; 3]> {
        match &self.guide {
            Guide::Nearest { at } | Guide::Spot { at, .. } => Some(at),
            Guide::Ray { .. } => None,
        }
    }

    /// Every block the record reads besides the point it belongs to: what it depends on, so a loop
    /// through it can be found and a deletion can let go of it.
    pub fn sources(&self) -> impl Iterator<Item = NodeId> {
        let plane = match self.frame {
            GuideFrame::Body => None,
            GuideFrame::Plane { node } => Some(node),
        };
        std::iter::once(self.body).chain(plane)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ray(u: ParamRef) -> SurfacePoint {
        SurfacePoint {
            body: NodeId::new(),
            frame: GuideFrame::Body,
            guide: Guide::Ray { along: Axis::Z, flipped: false, u, v: ParamRef::Literal(-4.5), facing: Facing::Front },
            live: true,
            solved: None,
        }
    }

    /// **The record survives a save, each kind of frame and guide, and a literal stays the bare
    /// number it is**, so a person reading the file reads millimetres.
    #[test]
    fn a_surface_point_round_trips_and_writes_its_literals_as_numbers() {
        let plane = NodeId::new();
        let cases = [
            ray(ParamRef::Literal(12.5)),
            ray(ParamRef::Named(crate::ParamId::new())),
            SurfacePoint {
                guide: Guide::Spot { at: [ParamRef::Literal(5.0); 3], facing: [0.0, 0.0, 1.0] },
                ..ray(ParamRef::Literal(0.0))
            },
            SurfacePoint {
                frame: GuideFrame::Plane { node: plane },
                guide: Guide::Nearest { at: [ParamRef::Literal(1.0), ParamRef::Literal(2.0), ParamRef::Literal(3.0)] },
                live: false,
                solved: Some(Solved { normal: [0.0, 0.0, 1.0], x: [1.0, 0.0, 0.0] }),
                ..ray(ParamRef::Literal(0.0))
            },
        ];
        for point in cases {
            let json = serde_json::to_string(&point).unwrap();
            assert_eq!(serde_json::from_str::<SurfacePoint>(&json).unwrap(), point, "{json}");
        }
        let json = serde_json::to_value(ray(ParamRef::Literal(12.5))).unwrap();
        assert_eq!(json["guide"]["u"], serde_json::json!(12.5));
        assert!(json.get("solved").is_none(), "nothing solved, nothing written");
        let json = serde_json::to_value(SurfacePoint {
            guide: Guide::Spot { at: [ParamRef::Literal(5.0); 3], facing: [0.0, 0.0, 1.0] },
            ..ray(ParamRef::Literal(0.0))
        })
        .unwrap();
        assert_eq!(json["guide"]["kind"], serde_json::json!("spot"));
        assert_eq!(json["guide"]["at"], serde_json::json!([5.0, 5.0, 5.0]));
    }

    /// A front face is one the ray enters, a back face one it leaves, and the across axes turn
    /// cyclically with the ray.
    #[test]
    fn a_facing_counts_its_own_crossings_and_the_across_axes_are_cyclic() {
        assert!(Facing::Front.counts(true) && !Facing::Front.counts(false));
        assert!(Facing::Back.counts(false) && !Facing::Back.counts(true));
        assert_eq!(across(Axis::Z), (0, 1));
        assert_eq!(across(Axis::X), (1, 2));
        assert_eq!(across(Axis::Y), (2, 0));
    }

    /// Whatever a record's numbers name counts as read, whichever guide holds them.
    #[test]
    fn a_surface_point_lists_the_numbers_it_reads_and_the_blocks_it_stands_on() {
        let named = ParamRef::Named(crate::ParamId::new());
        let point = ray(named);
        assert_eq!(point.params().copied().collect::<Vec<_>>(), vec![named, ParamRef::Literal(-4.5)]);
        assert_eq!(point.sources().collect::<Vec<_>>(), vec![point.body]);

        let plane = NodeId::new();
        let seeded = SurfacePoint {
            frame: GuideFrame::Plane { node: plane },
            guide: Guide::Nearest { at: [named; 3] },
            ..point
        };
        assert_eq!(seeded.params().count(), 3);
        assert_eq!(seeded.sources().collect::<Vec<_>>(), vec![point.body, plane]);
        let spot = SurfacePoint { guide: Guide::Spot { at: [named; 3], facing: [0.0, 0.0, 1.0] }, ..point };
        assert_eq!(spot.params().count(), 3);
        assert!(spot.seed().is_some() && point.seed().is_none());
    }
}
