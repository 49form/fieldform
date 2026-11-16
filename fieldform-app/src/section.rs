// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The section cut as the panel edits it.

use crate::*;

/// A cutaway view, resolved: where a [`fieldform_scene::SectionCut`]'s plane is this frame.
///
/// **The record is the document's; this is a reading of it** (the owner, 2026-11-07: a section
/// is a block, saved and undone, and placed with a plane block's rows). It was view state held
/// here until then, and the paragraphs below are the arrangement the block kept: the row, the
/// switch, the claim on the handles. What changed is where the truth is -- `Document::sections`,
/// with an origin that is a `Place` and a facing that is a `Turn` -- and that this is derived from
/// it each time it is asked, through [`App::resolve_section`], rather than kept beside it.
///
/// **Each one has a row in the block list all the same, and the row is not a block** (the owner,
/// 2026-10-24: the section should stay on while every other tool is used and still be
/// adjustable, "like a block whose on/off is the toggle"; then, trying it: a section is *made* by
/// going to View → Section, there can be several, each switched as you see fit, and several may
/// be on at once since the field allows it). The row gives the view state the three things a
/// block's row gives a block — a switch, a place to click to take the handles, and a mark saying
/// the handles are its — without any of the four things a `Node` would bring: serialisation, a
/// place in the field, an operand slot, and an undo entry per drag. `App::sections` holds them in
/// the order they were made, `App::section_rows` draws them, and the ones that are on are what
/// the shader cuts with (`App::section_half_spaces`): what is past any of them is cut away.
///
/// Kept as an origin and a rotation rather than as a normal and an offset, even though the
/// shader wants the latter, because the manipulator moves a *frame*: a stored normal beside a
/// rotation would be the same orientation written down twice, which is the failure this
/// codebase keeps meeting. The normal is derived, never stored.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Section {
    /// The record this was read from.
    pub(crate) id: fieldform_scene::SectionCutId,
    /// Which one this is, in the order made: kept rather than counted from the row, so removing
    /// the first does not rename the second.
    pub(crate) number: u32,
    pub(crate) on: bool,
    pub(crate) origin: Point3<f64>,
    pub(crate) rotation: Rotation3<f64>,
    /// Which half is kept. Turning a plane over is not a rotation — it is the same plane
    /// facing the other way — so spending 180 degrees of the frame on it would fight the
    /// manipulator for one control. Same reasoning as `Node::Plane`.
    pub(crate) flipped: bool,
}

impl Section {
    /// The outward normal: the half-space *past* this is what gets cut away.
    pub(crate) fn normal(&self) -> Vector3<f64> {
        let facing = if self.flipped { -1.0 } else { 1.0 };
        self.rotation * Vector3::z() * facing
    }

    /// This section, with a manipulator motion applied about `pivot`.
    ///
    /// Scale is ignored rather than refused: a plane has no extent for a factor to act on, and
    /// the handles for it are not offered. Translation along the normal is what "move the
    /// section through the part" means, and rotation turns the frame the normal is read from.
    pub(crate) fn with(self, motion: gizmo::Motion, pivot: Point3<f64>) -> Self {
        match motion {
            gizmo::Motion::Translate(d) => Self { origin: self.origin + d, ..self },
            gizmo::Motion::Rotate { axis, degrees } => {
                let Some(unit) = nalgebra::Unit::try_new(axis, 1e-9) else { return self };
                let turn = Rotation3::from_axis_angle(&unit, degrees.to_radians());
                Self {
                    origin: turn * (self.origin - pivot.coords) + pivot.coords,
                    rotation: turn * self.rotation,
                    ..self
                }
            }
            gizmo::Motion::Scale { .. } => self,
        }
    }

    /// The plane as the shader wants it, or `None` when there is nothing to cut.
    pub(crate) fn half_space(&self) -> Option<(Vector3<f64>, f64)> {
        if !self.on {
            return None;
        }
        let n = self.normal();
        Some((n, n.dot(&self.origin.coords)))
    }
}

/// A section cut placed where a drag left its plane: an origin and a facing typed, so what it was
/// taken from, if anything, is let go of -- as a dragged plane block's typed place would be.
pub(crate) fn placed_section(before: fieldform_scene::SectionCut, moved: &Section) -> fieldform_scene::SectionCut {
    let (x, y, z) = moved.rotation.euler_angles();
    fieldform_scene::SectionCut {
        origin: fieldform_scene::Place::At([
            ParamRef::Literal(moved.origin.x),
            ParamRef::Literal(moved.origin.y),
            ParamRef::Literal(moved.origin.z),
        ]),
        turn: fieldform_scene::Turn::By([
            ParamRef::Literal(x.to_degrees()),
            ParamRef::Literal(y.to_degrees()),
            ParamRef::Literal(z.to_degrees()),
        ]),
        ..before
    }
}
