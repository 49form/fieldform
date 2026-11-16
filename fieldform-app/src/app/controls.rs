// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The control mesh being edited: which one, where it sits, and how an edit to it is recorded.

use crate::*;

impl App {
    /// Drops whatever is picked on the control mesh, and the gesture that was using it.
    ///
    /// A selection names vertices and faces by index, and a refine renumbers both: the
    /// originals keep their indices but have moved under the vertex-point rule, and every face
    /// is new. A selection that survived would keep the manipulator on indices that now mean
    /// something else and move the wrong points without a word — the one thing in
    /// `cage-refine-spec.md` §7 that fails quietly. It is also why this runs on undo and redo:
    /// they put the other numbering back.
    pub(crate) fn forget_cage_selection(&mut self) {
        self.selection.clear();
        self.hover = Selection::None;
        self.move_state = None;
    }

    /// `refine`: one Catmull-Clark step on the cage the tools are editing.
    ///
    /// Through the registry like any document operation, so it is one entry on the undo stack
    /// and the console and the menu cannot differ — what this adds is the two things the
    /// registry cannot know. *Which* cage: the registry takes the only one and refuses to
    /// guess among several, but the app has an answer, the one on screen. And the selection,
    /// above.
    ///
    /// Refused while a surface is the thing being edited. `subdivide` is the closed cage's
    /// rule, a surface to follow is open, and refining "the cage" while a surface is
    /// highlighted would change something other than what is selected.
    pub(crate) fn refine_cage(&mut self, args: &Args) {
        if self.control != ControlMesh::Cage {
            self.warn(
                "Refine works on the cage, and a surface is selected — pick the cage's body first",
            );
            return;
        }
        let args = match args.get("cage") {
            Some(_) => args.clone(),
            None => args
                .clone()
                .with("cage", Value::Str(self.active_cage.to_string())),
        };
        let command = match self.registry.build("cage.refine", &args, &self.doc) {
            Ok(command) => command,
            Err(e) => {
                self.warn(e.to_string());
                return;
            }
        };
        if let Err(e) = self.stack.execute(&mut self.doc, command) {
            self.warn(e.to_string());
            return;
        }
        self.forget_cage_selection();
        self.after_document_change();
        let (points, faces) = (self.cage().vertex_count(), self.cage().face_count());
        self.say(format!(
            "Refined the cage: {points} control points, {faces} faces. The surface has not moved."
        ));
    }

    /// The cage the UI is editing, as its mesh — red or not, since framing, picking and the
    /// cage overlay want the mesh whatever its state. What needs a solid asks `doc.cage`.
    ///
    /// Panics only if the active id has been removed without updating it, which would be a
    /// bug in this file rather than bad input — every path that replaces the document sets
    /// `active_cage` in the same step.
    pub(crate) fn cage(&self) -> &QuadMesh {
        self.doc
            .cage_body(self.active_cage)
            .expect("active_cage must name a cage in the document")
            .mesh()
    }

    /// The mesh `mesh` names, if the document still holds it.
    pub(crate) fn mesh_of(&self, mesh: ControlMesh) -> Option<&QuadMesh> {
        match mesh {
            ControlMesh::Cage => self.doc.cage_body(self.active_cage).map(|c| c.mesh()),
            ControlMesh::Surface(id) => self.doc.quad_meshes.get(&id),
        }
    }

    pub(crate) fn mesh_of_mut(&mut self, mesh: ControlMesh) -> Option<&mut QuadMesh> {
        // A cage's extent is a body's, which the pick box is sized to, on every frame of a drag.
        self.document_moved();
        match mesh {
            ControlMesh::Cage => self.doc.cage_mesh_mut(self.active_cage),
            ControlMesh::Surface(id) => self.doc.quad_meshes.get_mut(&id),
        }
    }

    /// The mesh the cage tools are editing — see [`ControlMesh`].
    ///
    /// The cage when the chosen surface has gone, which lasts only until the start of the next
    /// frame: `refresh_surface_display` notices and hands the tools back to the cage properly.
    pub(crate) fn control(&self) -> &QuadMesh {
        match self.mesh_of(self.control) {
            Some(mesh) => mesh,
            None => self.cage(),
        }
    }

    /// Where `mesh` is drawn from its own coordinates. A cage can sit under a placement in the
    /// graph; a surface is not a node, and is drawn, picked and followed where its points are.
    pub(crate) fn mesh_placement(&self, mesh: ControlMesh) -> nalgebra::Similarity3<f64> {
        match mesh {
            ControlMesh::Cage => self.cage_placement(),
            ControlMesh::Surface(_) => nalgebra::Similarity3::identity(),
        }
    }

    pub(crate) fn control_placement(&self) -> nalgebra::Similarity3<f64> {
        self.mesh_placement(self.control)
    }

    /// Whether the control mesh is on screen at all — the cage by the rules
    /// `visible_cage_node` gives, a surface by its own eye.
    pub(crate) fn control_visible(&self) -> bool {
        match self.control {
            ControlMesh::Cage => self.visible_cage_node().is_some(),
            ControlMesh::Surface(id) => {
                self.doc.quad_meshes.contains_key(&id) && !self.hidden_surfaces.contains(&id)
            }
        }
    }

    /// Whether a click in the viewport picks on the control mesh: a mode that draws it, and it
    /// there to be drawn.
    pub(crate) fn control_editable(&self) -> bool {
        self.view.mode.cage_is_editable() && self.control_visible()
    }

    /// Hands the cage tools to another mesh.
    ///
    /// The selection goes with it — it names vertices by index, and index 3 of a surface is not
    /// index 3 of the cage — and the drawn control mesh is rebuilt from the new one. No compile:
    /// nothing about the field has changed, only which of its inputs is being pointed at.
    pub(crate) fn set_control(&mut self, mesh: ControlMesh) {
        if self.control == mesh {
            return;
        }
        self.control = mesh;
        self.selection.clear();
        self.hover = Selection::None;
        self.move_state = None;
        self.cage_mesh = CageMesh::from_cage(self.control());
        self.refresh_drawn_meshes();
    }

    /// Makes a surface the control mesh, from its row.
    pub(crate) fn edit_surface(&mut self, id: fieldform_scene::QuadMeshId) {
        let was = self.control;
        self.set_control(ControlMesh::Surface(id));
        // Whatever else was selected would otherwise keep the manipulator in Field mode, over
        // something that is not what was just clicked.
        self.selected_body = None;
        self.selected_curve = None;
        self.clear_point_selection();
        self.selected_block = None;
        self.section_selected = None;
        if was != self.control {
            let name = self.doc.surface_name(id).to_string();
            if self.view.mode.cage_is_editable() {
                self.say(format!(
                    "Editing {name}: click a point, edge or face and drag it, or use the handles"
                ));
            } else {
                self.say(format!(
                    "Editing {name}: its points are picked in Polygon or Overlay view (Tab)"
                ));
            }
        }
    }

    /// The command that records a finished move of `vertices` on `mesh`.
    pub(crate) fn positions_command(
        &self,
        mesh: ControlMesh,
        vertices: Vec<VertexId>,
        before: Vec<Point3<f64>>,
        after: Vec<Point3<f64>>,
    ) -> Box<dyn fieldform_ops::Command> {
        match mesh {
            ControlMesh::Cage => {
                Box::new(SetVertexPositions::new(self.active_cage, vertices, before, after))
            }
            ControlMesh::Surface(id) => {
                Box::new(SetSurfacePositions::new(id, vertices, before, after))
            }
        }
    }
}
