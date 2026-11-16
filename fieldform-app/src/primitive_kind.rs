// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The primitives a person can add, each with its label, its console word and the node it makes.

use crate::*;

/// The analytic solids Create offers, and the one place their default proportions live.
///
/// An enum rather than the boolean this used to be, which only ever had a sphere and a cube to
/// choose between. Keeping the default size here — instead of at each call site — is what
/// makes "every new body arrives at the same size" a property of the list rather than a habit
/// each entry has to remember.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PrimitiveKind {
    Sphere,
    Cube,
    SkewedBox,
    Cylinder,
    Cone,
    Torus,
    Capsule,
    Coil,
    Thread,
    Plane,
    Point,
}

impl PrimitiveKind {
    /// Every primitive, in the order they are offered.
    ///
    /// **Reference geometry first, then solids.** A point locates a plane and a plane locates a
    /// curve, so these come before the thing they are reached for; the solids after them run
    /// simplest idea first -- a ball, a box, then the ones taking two ends and a radius. It used
    /// to be solids first with the plane and the point trailing, on the argument that neither is
    /// a solid. True, and beside the point: what a menu orders is the order they are reached in.
    ///
    /// **The menu and the console read the command registry now** (`commands.rs`), which lists
    /// each primitive in this order; this is what the tests walk to hold the two together —
    /// `every_primitive_is_reachable_from_the_console` asks that every kind has an entry with its
    /// word and its operation, in this order, in Create › Primitives — the capsule, the coil and
    /// the thread carrying on in Create › Features, below Sweep.
    #[cfg(test)]
    pub(crate) const ALL: [PrimitiveKind; 11] = [
        PrimitiveKind::Point,
        PrimitiveKind::Plane,
        PrimitiveKind::Sphere,
        PrimitiveKind::Cube,
        PrimitiveKind::SkewedBox,
        PrimitiveKind::Cylinder,
        PrimitiveKind::Cone,
        PrimitiveKind::Torus,
        PrimitiveKind::Capsule,
        PrimitiveKind::Coil,
        PrimitiveKind::Thread,
    ];

    /// Where the reference geometry ends and the solids begin, which is where the menu puts a
    /// curve and a separator.
    #[cfg(test)]
    pub(crate) const SOLIDS_START: usize = 2;

    pub(crate) fn label(self) -> &'static str {
        match self {
            PrimitiveKind::Sphere => "Sphere",
            PrimitiveKind::Cube => "Cube",
            PrimitiveKind::SkewedBox => "Skewed box",
            PrimitiveKind::Cylinder => "Cylinder",
            PrimitiveKind::Cone => "Cone",
            PrimitiveKind::Torus => "Torus",
            PrimitiveKind::Capsule => "Capsule",
            PrimitiveKind::Coil => "Coil",
            PrimitiveKind::Thread => "Thread",
            PrimitiveKind::Plane => "Plane",
            PrimitiveKind::Point => "Point",
        }
    }

    /// The one word the console knows this by: its label lowercased, with the spaces closed up,
    /// because a console word is a single token and "skewed box" would read as a word and an
    /// argument.
    pub(crate) fn word(self) -> String {
        self.label().to_lowercase().replace(' ', "")
    }

    /// The operation name `dispatch` knows this by: the inverse of `dispatch`'s own arms, which
    /// the tests hold against the registry.
    #[cfg(test)]
    pub(crate) fn command(self) -> &'static str {
        match self {
            PrimitiveKind::Sphere => "create.sphere",
            PrimitiveKind::Cube => "create.cube",
            PrimitiveKind::SkewedBox => "create.skewbox",
            PrimitiveKind::Cylinder => "create.cylinder",
            PrimitiveKind::Cone => "create.cone",
            PrimitiveKind::Torus => "create.torus",
            PrimitiveKind::Capsule => "create.capsule",
            PrimitiveKind::Coil => "create.coil",
            PrimitiveKind::Thread => "create.thread",
            PrimitiveKind::Plane => "create.plane",
            PrimitiveKind::Point => "create.point",
        }
    }

    /// The node this makes, sized so every kind arrives inside the same `size` envelope.
    ///
    /// Same size on screen whichever one is asked for, which is why the sphere takes half of
    /// `size` as a radius and the torus splits it between its two: a Create menu where each
    /// entry appeared at a different scale would make the first thing you do after every one
    /// of them a resize.
    pub(crate) fn node(self, centre: [ParamRef; 3], size: f64) -> Node {
        let literal = ParamRef::Literal;
        // An end offset up or down Z from the centre, resolved to a number: a new body has
        // literal coordinates, and ends that were parameter references to a centre the shape
        // no longer has would be unreadable.
        let along_z = |c: [ParamRef; 3], dz: f64| {
            let at = |r: ParamRef| match r {
                ParamRef::Literal(v) => v,
                ParamRef::Named(_) | ParamRef::Field(_) => 0.0,
            };
            [literal(at(c[0])), literal(at(c[1])), literal(at(c[2]) + dz)]
        };
        match self {
            PrimitiveKind::Sphere => Node::Sphere {
                centre: fieldform_scene::Place::At(centre),
                radius: literal(size * 0.5),
            },
            PrimitiveKind::Cube => Node::Cuboid {
                centre: fieldform_scene::Place::At(centre),
                size: [literal(size); 3],
                turn: fieldform_scene::no_turn(),
                edge: fieldform_scene::Fillet::hard(),
            },
            // The two-point solids arrive standing on Z and centred like the rest, so no
            // Create entry lands somewhere different from its neighbours. Where they go after
            // that is a drag of either end.
            // Leaning on every axis at once, so what arrives is plainly not a cube and a
            // click on Create shows what the entry is for. The first edge stays on X so the
            // lean reads as a lean rather than as a turn, and each edge is a full edge, as a
            // cube's `size` is a full side. About the same volume as the cube beside it.
            PrimitiveKind::SkewedBox => Node::SkewedBox {
                centre: fieldform_scene::Place::At(centre),
                edges: [
                    [literal(size), literal(0.0), literal(0.0)],
                    [literal(size * 0.3), literal(size * 0.85), literal(0.0)],
                    [literal(size * 0.15), literal(size * 0.1), literal(size * 0.85)],
                ],
            },
            PrimitiveKind::Cylinder => Node::Cylinder {
                ends: Ends::between(
                    fieldform_scene::Place::At(along_z(centre, -size * 0.5)),
                    fieldform_scene::Place::At(along_z(centre, size * 0.5))),
                radius: literal(size * 0.5),
                edge: fieldform_scene::Fillet::hard(),
            },
            // A plain cone, apex up: the shape the word names. A top radius above zero
            // truncates it, which is why there is no separate entry for that.
            PrimitiveKind::Cone => Node::Cone {
                ends: Ends::between(
                    fieldform_scene::Place::At(along_z(centre, -size * 0.5)),
                    fieldform_scene::Place::At(along_z(centre, size * 0.5))),
                base_radius: literal(size * 0.5),
                top_radius: literal(0.0),
                edge: fieldform_scene::Fillet::hard(),
            },
            // Split so the outer radius is still half the envelope.
            PrimitiveKind::Torus => Node::Torus {
                centre: fieldform_scene::Place::At(centre),
                major_radius: literal(size * 0.35),
                minor_radius: literal(size * 0.15),
                turn: fieldform_scene::no_turn(),
            },
            // The points are cap centres, so the ends reach a radius past them: a quarter of
            // the envelope each way leaves the whole capsule `size` long.
            PrimitiveKind::Capsule => Node::Capsule {
                ends: Ends::between(
                    fieldform_scene::Place::At(along_z(centre, -size * 0.25)),
                    fieldform_scene::Place::At(along_z(centre, size * 0.25))),
                start_radius: literal(size * 0.25),
                end_radius: literal(size * 0.25),
            },
            // Standing on the centre and climbing Z, four turns filling the envelope's height
            // and most of its width. Turns and pitch, the pair a spring is usually given in.
            PrimitiveKind::Coil => Node::Coil {
                frame: fieldform_scene::Frame::At {
                    origin: along_z(centre, -size * 0.4),
                    axis: [literal(0.0), literal(0.0), literal(1.0)],
                },
                diameter: literal(size * 0.8),
                size: fieldform_scene::CoilSize::TurnsPitch {
                    turns: literal(4.0),
                    pitch: literal(size * 0.2),
                },
                section: fieldform_scene::CoilSection::Circle { wire: literal(size * 0.08) },
                hand: fieldform_scene::Hand::Right,
            },
            // A custom thread at the envelope's scale rather than a named one, which would arrive
            // a twelfth of the size of everything beside it: the major diameter four tenths of
            // it, at the coarse series' own ratio of pitch to diameter, and as long as it is.
            // Naming a size in the block puts the table's numbers in their place.
            PrimitiveKind::Thread => Node::Thread {
                frame: fieldform_scene::Frame::At {
                    origin: along_z(centre, -size * 0.5),
                    axis: [literal(0.0), literal(0.0), literal(1.0)],
                },
                major: literal(size * 0.4),
                pitch: literal(size * 0.4 * 0.15),
                length: literal(size),
                hand: fieldform_scene::Hand::Right,
                clearance: literal(0.0),
                designation: None,
            },
            // Both arrive at the centre rather than offset like the solids: there is no
            // envelope to fill, since one has no edges and the other no volume.
            PrimitiveKind::Plane => Node::Plane {
                origin: fieldform_scene::Place::At(centre),
                flipped: false,
                turn: fieldform_scene::no_turn(),
            },
            PrimitiveKind::Point => Node::Point { at: centre },
        }
    }
}
