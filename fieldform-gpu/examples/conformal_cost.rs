// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a conformal lattice costs, and what the surface's tessellation costs it.
//!
//! **Written because the first one hung the application.** Every sample of a conformal lattice
//! walks a hierarchy over the tessellated surface, where every other lattice here is closed-form
//! arithmetic — so this is the first field in the crate whose per-sample cost depends on how
//! finely something *else* was resolved. That dependence is what decides how the tessellation
//! depth should be chosen, which is the question `surface_levels` answers.
//!
//! The comparison is a rectangular gyroid in the same box at the same cell size: same rays, same
//! camera, same wall. Anything above that figure is what following a surface costs.
//!
//! Deliberately modest: 512 and 1024 pixels rather than the viewport's own 1440, and two frames
//! rather than five. A run at modelling tolerance wedged the GPU driver twice — processes that
//! `taskkill` reports as not running and that need a reboot — so this walks up from cheap rather
//! than starting at the top.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example conformal_cost`

use fieldform_field::SurfaceMap;
use fieldform_gpu::{generate, Camera, Renderer};
use fieldform_scene::{
    BooleanOp, CellMap, Document, Graph, LatticeCell, LatticePhase, Node, ParamRef, QuadMeshId,
};
use nalgebra::Point3;

const SPAN: f64 = 200.0;
const RISE: f64 = 20.0;
/// The panel the owner imported: six quads across, domed.
const FACES: usize = 6;
const CELLS_PER_FACE: u32 = 2;

fn panel() -> fieldform_cage::QuadMesh {
    let step = SPAN / FACES as f64;
    let mut positions = Vec::new();
    for j in 0..=FACES {
        for i in 0..=FACES {
            let (x, y) = (i as f64 * step - SPAN * 0.5, j as f64 * step - SPAN * 0.5);
            let (u, v) = (x / (SPAN * 0.5), y / (SPAN * 0.5));
            positions.push(Point3::new(x, y, RISE * (1.0 - u * u) * (1.0 - v * v)));
        }
    }
    let at = |i: usize, j: usize| (j * (FACES + 1) + i) as u32;
    let mut faces = Vec::new();
    for j in 0..FACES {
        for i in 0..FACES {
            faces.push(vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
        }
    }
    fieldform_cage::QuadMesh::from_faces(positions, &faces).expect("a domed panel")
}

/// The box and the lattice, with whichever map is asked for.
fn scene(surface: QuadMeshId, conformal: bool, shell: Option<f64>) -> (Document, Graph) {
    let mut doc = Document::new();
    doc.quad_meshes.insert(surface, panel());
    let map = if conformal {
        CellMap::Conformal {
            surface,
            cells_per_face: CELLS_PER_FACE,
            // One period through the surface, at the cell size the rectangular row uses.
            depth: ParamRef::Literal(17.0),
            thickness: shell.map(ParamRef::Literal),
        }
    } else {
        CellMap::Rectangular { cell: ParamRef::Literal(17.0) }
    };
    let mut graph = Graph::empty();
    let slab = graph.add(Node::Cuboid { edge: Default::default(),
        centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
        size: [
            ParamRef::Literal(SPAN),
            ParamRef::Literal(SPAN),
            ParamRef::Literal(40.0),
        ],
        turn: fieldform_scene::no_turn(),
    });
    let lattice = graph.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell: LatticeCell::Gyroid,
        map,
        phase: LatticePhase::sheet(ParamRef::Literal(2.0)),
    });
    graph.add(Node::Boolean {
        op: BooleanOp::Intersection,
        inputs: vec![slab.into(), lattice.into()],
    });
    (doc, graph)
}

/// One configuration, timed at two resolutions.
fn time(label: &str, conformal: bool, levels: Option<usize>, shell: Option<f64>) {
    let surface = QuadMeshId::new();
    let (doc, graph) = scene(surface, conformal, shell);
    let program = generate(&doc, &graph).expect("generates");
    // Built here at the depth being measured, rather than through `packed_surfaces`, which
    // derives one depth from the cell count -- the whole point of this table is to see what that
    // derivation is choosing between.
    let surfaces: Vec<fieldform_field::PackedSurface> = match levels {
        Some(levels) => vec![SurfaceMap::new(&panel(), levels)
            .expect("tessellates")
            .packed()],
        None => Vec::new(),
    };
    let triangles: usize = surfaces
        .iter()
        .map(|s| s.layout.triangle_count as usize)
        .sum();
    let renderer = match pollster::block_on(Renderer::new(&program, &[], &[], &surfaces, 64, 1.0)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("no GPU available ({e})");
            std::process::exit(0);
        }
    };
    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| match p {
            ParamRef::Literal(v) => *v as f32,
            _ => 0.0,
        })
        .collect();
    renderer.set_params(&values);

    let eye = Point3::new(SPAN * 1.2, SPAN * 0.9, SPAN * 0.8);
    let mut row = format!("  {label:<30} {triangles:>8}");
    for edge in [512u32, 1024] {
        let camera = Camera::looking_at(
            eye,
            Point3::origin(),
            nalgebra::Vector3::z(),
            (Point3::origin(), SPAN),
            45.0,
            1.0,
        );
        let _ = renderer.render(&camera, edge, edge, 512);
        let start = std::time::Instant::now();
        for _ in 0..2 {
            renderer.render(&camera, edge, edge, 512).expect("renders");
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / 2.0;
        row.push_str(&format!(" {ms:>9.1}"));
    }
    println!("{row}");
}

fn main() {
    println!("  levels   triangles   (a {FACES}x{FACES} panel)");
    for levels in 2..=6 {
        let map = SurfaceMap::new(&panel(), levels).expect("tessellates");
        println!("  {levels:>6}   {:>9}", map.triangle_count());
    }
    println!(
        "\n  surface_levels({CELLS_PER_FACE}) = {}",
        fieldform_field::surface_levels(CELLS_PER_FACE)
    );

    println!("\n  {:<30} {:>8} {:>9} {:>9}", "map", "tris", "512 px", "1024 px");
    time("rectangular, 17 mm cells", false, None, None);
    for levels in 3..=5 {
        time(&format!("conformal, levels {levels}"), true, Some(levels), None);
    }
    // The shell is not only what makes the block a part: it is what lets a ray far from the
    // surface be answered from the surface's own box without walking anything.
    for thickness in [40.0, 10.0] {
        time(
            &format!("conformal, levels 3, shell {thickness:.0} mm"),
            true,
            Some(3),
            Some(thickness),
        );
    }
}
