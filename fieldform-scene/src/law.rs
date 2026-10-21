// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How a swept section changes along its path: taper, twist, the law that distributes them,
//! and the stretch of path the solid occupies. the sweep-v2-spec design note.
//!
//! One type rather than four fields on the block, so that a block which carries a section
//! along a path carries all of this the same way (D9) — a loft between two sections, when
//! there is one, will take the same thing. In the file the four are written flat on the node
//! (`taper_scale` or `taper_draft`, `twist`, `law`, `extent`), each only when it is not the
//! default, so a sweep that has not asked writes exactly the bytes it did — `graph/serde.rs`'s
//! `law_fields` and `read_law`.

use crate::graph::{ParamRef, RampShape};

/// How a section grows or shrinks along the path.
///
/// **Two, because they answer different engineering questions** (ruled 2026-09-30). Scale
/// keeps the section's shape and is what a guide rail will drive; a draft leans every wall at
/// one angle, which is what a moulded part wants, and is the extrude's own draft — a drafted
/// sweep along a straight path is the drafted extrude of the same section. Never both.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Taper {
    #[default]
    None,
    /// The section is this many times its drawn size where the path ends, following the law.
    Scale(ParamRef),
    /// Every wall leans out by this many degrees. A constant angle by definition, so the law
    /// does not apply to it.
    Draft(ParamRef),
}

impl Taper {
    pub(crate) fn params(&self) -> impl Iterator<Item = &ParamRef> {
        let typed = match self {
            Taper::None => None,
            Taper::Scale(v) | Taper::Draft(v) => Some(v),
        };
        typed.into_iter()
    }

    pub(crate) fn params_mut(&mut self) -> Vec<&mut ParamRef> {
        match self {
            Taper::None => Vec::new(),
            Taper::Scale(v) | Taper::Draft(v) => vec![v],
        }
    }

    /// What the block calls it.
    pub fn label(&self) -> &'static str {
        match self {
            Taper::None => "None",
            Taper::Scale(_) => "Scale",
            Taper::Draft(_) => "Draft",
        }
    }
}

/// What a section does on its way along the path.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepLaw {
    pub taper: Taper,
    /// Whether a draft taper rounds the section's corners or mitres them — the same type an
    /// extrude carries, because a drafted sweep along a straight path *is* the drafted
    /// extrude (D9). Nothing for a scale, which moves no wall along its own normal and so
    /// leaves every corner as it was drawn.
    pub corners: crate::Corners,
    /// How far the section turns about the path over the window, in degrees.
    pub twist: ParamRef,
    /// How the scale and the turn are distributed along the window: G0 is straight, G1 and G2
    /// ease in and out — the same three a ramp has, because they are the same idea.
    pub shape: RampShape,
    /// Where along the path the solid runs, as percentages of the path's length: start and
    /// end. `[0, 100]` is the whole path.
    pub extent: [ParamRef; 2],
}

impl Default for SweepLaw {
    fn default() -> Self {
        Self {
            taper: Taper::None,
            corners: crate::Corners::default(),
            twist: ParamRef::Literal(0.0),
            shape: RampShape::Linear,
            extent: [ParamRef::Literal(0.0), ParamRef::Literal(100.0)],
        }
    }
}

impl SweepLaw {
    /// Whether this is the law every sweep had before there was one, so that a block under it
    /// writes nothing and compiles to the sweep it always was.
    pub fn is_plain(&self) -> bool {
        *self == Self::default()
    }

    /// The numbers it carries, so they take part in parameter resolution like any other.
    pub(crate) fn params(&self) -> impl Iterator<Item = &ParamRef> {
        self.taper.params().chain(std::iter::once(&self.twist)).chain(&self.extent)
    }

    /// The same, to be edited.
    pub(crate) fn params_mut(&mut self) -> Vec<&mut ParamRef> {
        let mut out = self.taper.params_mut();
        out.push(&mut self.twist);
        out.extend(self.extent.iter_mut());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block that has not asked carries the plain law, which writes nothing; and every
    /// number the law holds is reachable by the route every other number is.
    #[test]
    fn the_plain_law_is_the_default_and_every_number_is_a_parameter() {
        let plain = SweepLaw::default();
        assert!(plain.is_plain());
        assert_eq!(plain.params().count(), 3, "twist, start, end");
        let mut law = SweepLaw {
            taper: Taper::Scale(ParamRef::Literal(0.5)),
            corners: crate::Corners::Sharp,
            twist: ParamRef::Literal(90.0),
            shape: RampShape::Smooth,
            extent: [ParamRef::Literal(10.0), ParamRef::Literal(90.0)],
        };
        assert!(!law.is_plain());
        assert_eq!(law.params().count(), 4, "scale, twist, start, end");
        *law.params_mut()[0] = ParamRef::Literal(2.0);
        assert_eq!(law.taper, Taper::Scale(ParamRef::Literal(2.0)));
        assert_eq!(Taper::Draft(ParamRef::Literal(5.0)).label(), "Draft");
    }
}
