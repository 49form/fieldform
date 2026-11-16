// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Where a new block is put: a position, a turn and a size, before it becomes a node.

use crate::*;

/// A body's placement, as the manipulator thinks of it.
///
/// The same three fields [`Node::Transform`] carries, in a form that composes: reading them
/// out, applying a motion and writing them back is easier to get right than editing three
/// arrays of parameter references in place.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Placement {
    pub(crate) translation: Vector3<f64>,
    pub(crate) rotation: Rotation3<f64>,
    pub(crate) scale: f64,
}

impl Placement {
    pub(crate) fn identity() -> Self {
        Self {
            translation: Vector3::zeros(),
            rotation: Rotation3::identity(),
            scale: 1.0,
        }
    }

    /// Reads a placement node, resolving its parameters. Anything else reads as the identity,
    /// which is exactly right: an unplaced body sits where it was built.
    pub(crate) fn of(app: &App, node: Option<&Node>) -> Self {
        let Some(Node::Transform { translation, rotation, scale, .. }) = node else {
            return Self::identity();
        };
        let r = app.resolve_point(rotation);
        Self {
            translation: app.resolve_point(translation).coords,
            rotation: Rotation3::from_euler_angles(
                r.x.to_radians(),
                r.y.to_radians(),
                r.z.to_radians(),
            ),
            scale: app.resolve(*scale).max(1e-6),
        }
    }

    /// The three fields, as the literals a placement node stores.
    pub(crate) fn into_node(self, input: impl Into<fieldform_scene::Operand>) -> Node {
        let (rx, ry, rz) = self.rotation.euler_angles();
        Node::Transform {
            // Whatever the caller holds: a manipulator wrapping a body has an id, and one
            // re-wrapping a block that is already placed has that block's slot, which may be
            // empty. Neither is this function's business to open.
            input: input.into(),
            translation: [
                ParamRef::Literal(self.translation.x),
                ParamRef::Literal(self.translation.y),
                ParamRef::Literal(self.translation.z),
            ],
            rotation: [
                ParamRef::Literal(rx.to_degrees()),
                ParamRef::Literal(ry.to_degrees()),
                ParamRef::Literal(rz.to_degrees()),
            ],
            scale: ParamRef::Literal(self.scale),
            about: fieldform_scene::Place::origin(),
        }
    }

    /// This placement with a manipulator motion applied about `pivot`.
    ///
    /// The pivot is the body's bounding-box centre, not its own origin, so every motion has to
    /// move the translation as well as the field it names. Rotating without that would swing
    /// the body around the part's origin instead of turning it in place — the single most
    /// obvious way for a gizmo to feel broken.
    pub(crate) fn with(self, motion: gizmo::Motion, pivot: Point3<f64>) -> Self {
        match motion {
            gizmo::Motion::Translate(d) => Self {
                translation: self.translation + d,
                ..self
            },
            gizmo::Motion::Rotate { axis, degrees } => {
                let Some(unit) = nalgebra::Unit::try_new(axis, 1e-9) else { return self };
                let turn = Rotation3::from_axis_angle(&unit, degrees.to_radians());
                Self {
                    translation: turn * (self.translation - pivot.coords) + pivot.coords,
                    rotation: turn * self.rotation,
                    scale: self.scale,
                }
            }
            // A body only ever receives the uniform factor — the per-axis cubes are not
            // offered for one, because a non-uniform scale would stop it being a distance
            // field. See `Handles::per_axis_scale`.
            gizmo::Motion::Scale { factor, .. } => Self {
                translation: (self.translation - pivot.coords) * factor + pivot.coords,
                rotation: self.rotation,
                scale: self.scale * factor,
            },
        }
    }
}
