// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Whether a lattice cut by a box comes out of the mesher with its cut faces.
//!
//! **Written from a screenshot.** A gyroid infilled cube, tessellated and drawn by the Tessellate
//! block's preview, had no flat faces where the box cuts the walls — the walls ended in frayed
//! spikes — where the traced view of the same part shows clean flat caps. The preview draws
//! exactly the triangles the mesher made, so if the caps are missing there they are missing from
//! the STL too, and this asks the mesher directly.
//!
//! **It was the box, not the mesher.** Given 1 mm of room past the cube, the caps came out every
//! time; given the cube's own box, which is what the block passes, they were not meshed at all —
//! a face lying on the mesher's boundary has no cells beyond it, and dual contouring makes no
//! quad on a boundary edge. `tessellate` now grows the box by a finest cell itself. This runs the
//! block's path, the exact box:
//!
//! | detail | triangles | open edges | cap triangles | worst off | time |
//! |---|---|---|---|---|---|
//! | Coarse | 1 243 992 | 0 | 40 158 | 0.828 mm | 9.4 s |
//! | Normal | 4 364 796 | 0 | 115 518 | 0.464 mm | 32.2 s |
//! | Full | 15 240 396 | 0 | 812 772 | 0.230 mm | 166 s |
//!
//! 100 mm cube, 20 mm gyroid cells, 2 mm sheet, release build, 2026-09-18. Not rerun without the
//! margin — the owner's STL is that measurement, open on the cut faces in a separate viewer — but
//! `tessellate`'s `a_body_whose_faces_lie_on_its_bounds_comes_out_closed` was, and fails with the
//! margin set to zero.
//!
//! Three numbers per detail. **Open edges**: an edge with one triangle is a hole, and a closed
//! part should have none. **Cap triangles**: triangles lying on the box's six faces, which is
//! where the cut walls should be. **Worst off**: how far any vertex sits from the true surface,
//! since a spike is a vertex placed well away from it.
//!
//! Run with: `cargo run --release -p fieldform-field --example capped_lattice`

use fieldform_field::{Cancel, GraphField, Progress, ScalarField, tessellate};
use fieldform_scene::{
    BooleanOp, CellMap, Document, Graph, LatticeCell, LatticePhase, MeshDetail, Node, ParamRef,
};
use nalgebra::Point3;

const HALF: f64 = 50.0;

fn part() -> (Document, Graph) {
    let lit = ParamRef::Literal;
    let doc = Document::new();
    let mut graph = Graph::empty();
    let cube = graph.add(Node::Cuboid { edge: Default::default(),
        centre: fieldform_scene::Place::At([lit(0.0); 3]),
        size: [lit(HALF * 2.0); 3],
        turn: fieldform_scene::no_turn(),
    });
    let lattice = graph.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell: LatticeCell::Gyroid,
        map: CellMap::Rectangular { cell: lit(20.0) },
        phase: LatticePhase::sheet(lit(2.0)),
    });
    graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![cube.into(), lattice.into()] });
    (doc, graph)
}

fn main() {
    let (doc, graph) = part();
    println!(
        "  {:<10} {:>8} {:>10} {:>11} {:>9} {:>11} {:>7}",
        "detail", "improve", "triangles", "open edges", "caps", "worst off", "time"
    );
    for detail in [MeshDetail::Coarse, MeshDetail::Normal, MeshDetail::Full] {
        let tolerance = detail.tolerance(HALF * 2.0);
        let field = GraphField::compile_graph(&doc, &graph, (tolerance.min_cell_mm * 0.5).max(1e-4))
            .expect("compiles");
        // Exactly the cube, which is what the block hands over: `graph_extent` of an
        // intersection is the box it is cut by, and the room the mesher needs past it is
        // `tessellate`'s to add.
        let bounds = (Point3::new(-HALF, -HALF, -HALF), Point3::new(HALF, HALF, HALF));
        for improve in [false] {
            let started = std::time::Instant::now();
            let out = tessellate(
                &field,
                bounds,
                &tolerance,
                improve,
                false,
                &Cancel::new(),
                &Progress::new(),
            )
            .expect("not cancelled");
            let took = started.elapsed().as_secs_f64();
            let mesh = out.mesh;

            let mut edges: std::collections::HashMap<(u32, u32), u32> =
                std::collections::HashMap::new();
            for t in &mesh.triangles {
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
                }
            }
            let open = edges.values().filter(|&&n| n == 1).count();

            // On a face of the box: every corner within a hundredth of a millimetre of one of the
            // six planes, the same one.
            let caps = mesh
                .triangles
                .iter()
                .filter(|t| {
                    (0..3).any(|axis| {
                        [HALF, -HALF].iter().any(|&plane| {
                            t.iter().all(|&v| (mesh.positions[v as usize][axis] - plane).abs() < 0.01)
                        })
                    })
                })
                .count();

            let worst = mesh
                .positions
                .iter()
                .map(|p| field.distance(*p).abs())
                .fold(0.0f64, f64::max);

            println!(
                "  {:<10} {:>8} {:>10} {:>11} {:>9} {:>9.3}mm {:>6.1}s",
                detail.label(),
                improve,
                mesh.triangles.len(),
                open,
                caps,
                worst,
                took
            );
        }
    }
}
