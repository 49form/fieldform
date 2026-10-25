// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a finished cage edit costs on the CPU, stage by stage.
use fieldform_field::{CageField, GraphField, LeafCache, levels_for_tolerance};
use fieldform_scene::{Document, MeshDetail, Node};

fn main() {
    for size in [100.0f64, 1000.0] {
        let cage = fieldform_cage::Cage::cube_of(size);
        let mut doc = Document::new();
        let id = doc.add_cage(cage.clone());
        let mut graph = fieldform_scene::Graph::empty();
        graph.add(Node::Cage(id));
        let diagonal = size * 3f64.sqrt();
        let tolerance = (diagonal * 1e-3).max(1e-4);
        let levels = levels_for_tolerance(diagonal, tolerance);
        let t = std::time::Instant::now();
        let field = CageField::from_cage(&cage, levels).expect("subdivides");
        let built = t.elapsed().as_secs_f64();
        let _ = &field;
        let mut cache = LeafCache::default();
        let t = std::time::Instant::now();
        let _ = GraphField::compile_cached(&doc, &graph, tolerance, &mut cache).expect("compiles");
        let cold = t.elapsed().as_secs_f64();
        // Moving a vertex: the leaf is a different leaf, so it is rebuilt.
        let v = doc.cage(id).unwrap().vertex_ids().next().unwrap();
        let was = doc.cage(id).unwrap().vertex_position(v);
        doc.cage_mesh_mut(id).unwrap().set_vertex_position(v, was + nalgebra::Vector3::new(1.0, 1.0, 1.0));
        let t = std::time::Instant::now();
        let _ = GraphField::compile_cached(&doc, &graph, tolerance, &mut cache).expect("compiles");
        let after_move = t.elapsed().as_secs_f64();
        println!(
            "{size:>6.0} mm cage: levels {levels}, from_cage {:.0} ms, first compile {:.0} ms, compile after a move {:.0} ms",
            built * 1e3, cold * 1e3, after_move * 1e3
        );
        // And what a Tessellate block's own compile costs, which the application does on the
        // interface thread when a block restarts after an edit.
        for detail in MeshDetail::PRESETS {
            let tol = detail.tolerance(size);
            let t = std::time::Instant::now();
            let _ = GraphField::compile_graph(&doc, &graph, (tol.min_cell_mm * 0.5).max(1e-4))
                .expect("compiles");
            println!(
                "        {:>9} block compile: levels {:>2}, {:>7.0} ms",
                detail.label(),
                levels_for_tolerance(diagonal, (tol.min_cell_mm * 0.5).max(1e-4)),
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}
