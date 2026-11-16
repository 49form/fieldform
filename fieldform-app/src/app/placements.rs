// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Where a body sits once every transform above it is applied.

use crate::*;

impl App {
    /// Where the cage sits in the part, as against where it stores itself.
    ///
    /// The cage keeps its own coordinates — it is the direct layer, and they are the truth —
    /// while a [`Node::Transform`] anywhere above it moves the *field* those coordinates
    /// describe. This is the one place the two are reconciled, and everything that puts the
    /// cage on screen or reads a click off it goes through here, so that what is drawn, what
    /// is picked, and where the manipulator sits are the same points rather than three
    /// descriptions that can drift apart.
    ///
    /// The identity for a cage nothing has placed, which is the ordinary case.
    pub(crate) fn cage_placement(&self) -> nalgebra::Similarity3<f64> {
        let graph = self.part_graph();
        let Some(target) = graph
            .nodes()
            .find(|(_, n)| matches!(n, Node::Cage(c) if *c == self.active_cage))
            .map(|(id, _)| id)
        else {
            return nalgebra::Similarity3::identity();
        };
        self.placement_above(target)
    }

    /// Every placement standing above `target`, composed: where the block sits and which way it
    /// is turned, after everything above it.
    ///
    /// A node's own numbers say where it was built; the placements above it say where it is
    /// drawn. Anything that positions something in the viewport has to ask this or it draws
    /// the first and calls it the second — the disagreement the cage hit first, and then the
    /// point marker and the plane preview hit again the moment the block list let an *inner*
    /// node be selected.
    ///
    /// **One walk, and it is the kernel's.** This used to be a second implementation that read
    /// the unexpanded graph and composed each block's own turn as it went -- the same idea said
    /// twice, which D9 exists to stop and which cost three rounds of one bug being fixed in one
    /// copy and not the other. Asked for directly: "please make sure we are re-using code where
    /// possible -- it feels like some bugs are repeating."
    ///
    /// The two differed only in whether turns had been expanded, and that difference is itself
    /// one function: a turned block *is* that block inside a placement. So the graph goes
    /// through `with_turns_flattened` and the kernel's walk answers -- which also means the
    /// square drawn for a plane and the field built from it cannot come to different answers
    /// about where it is, because there is nothing left to disagree.
    pub(crate) fn placement_above(&self, target: NodeId) -> nalgebra::Similarity3<f64> {
        self.placement_in(&self.placed_graph(), target)
    }

    /// The graph with every turn rewritten as the placement it means, for
    /// [`Self::placement_above`]'s single block. The overlay's points and planes are placed by the
    /// one its readings keep ([`Self::overlay_readings`]), since making it clones the graph.
    pub(crate) fn placed_graph(&self) -> Graph {
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        fieldform_field::with_turns_flattened(&self.doc, self.part_graph(), &numbers)
    }

    /// [`Self::placement_above`], over a graph whose turns are already placements.
    pub(crate) fn placement_in(&self, placed: &Graph, target: NodeId) -> nalgebra::Similarity3<f64> {
        guard_stack("App::placement_above");
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        fieldform_field::placement_above(&self.doc, placed, target, None, &numbers)
    }
}
