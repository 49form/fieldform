// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How often an operand mask actually fires, operand by operand.
//!
//! **The reading `mask_frame_cost` cannot take.** That harness times the frame both ways and
//! says masked is slower; it cannot say *why*, and the two reasons want opposite fixes. Either
//! the masks fire and the saving is smaller than the branch and the extra function — in which
//! case the feature is simply not worth it — or they never fire at all, in which case the
//! firing condition is wrong and the feature is fine. `MASK_WEIGHT` asks whether an operand is
//! dear to evaluate; it never asks whether any cell could skip it. This is what tells the two
//! apart.
//!
//! The skip grid records, per cell, which masked operands that cell has ruled out
//! (`march.wgsl`, `skip_build`). So: fill it the way the viewport does, read the plane back, and
//! count. A mask whose bit is set in no cell is pure cost.
//!
//! Two columns, because a mask firing where the march never goes buys nothing either. `all` is
//! every cell of the grid; `near` is the cells whose clearance does not carry the ray past
//! their own corner — the ones the march has to step inside rather than hop over.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example mask_fire_rate -- <file.fform>`

use fieldform_gpu::{generate_with_masks, Renderer};
use fieldform_scene::{Document, Graph, Node, ParamRef};
use nalgebra::{Point3, Vector3};

const SPINE_DEGREES: f64 = 1.0;
const TOLERANCE_MM: f64 = 0.5;
/// The two grids the viewport fills: coarse while anything moves, fine once it rests.
const GRIDS: [u32; 2] = [32, 64];

fn main() {
    let (doc, owned, path) = match std::env::args().nth(1) {
        Some(path) => {
            let (doc, _) = fieldform_io::open(std::path::Path::new(&path)).expect("opens");
            let graph = doc.parts_ordered().first().copied().expect("a part").graph.clone();
            (doc, graph, path)
        }
        None => {
            let (doc, graph) = lattice_in_a_cube();
            (doc, graph, "a lattice intersected with a cube (synthetic)".to_string())
        }
    };
    let graph = &owned;

    let extent = match fieldform_field::part_extent(&doc, graph) {
        fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
        _ => panic!("this document is unbounded"),
    };
    let radius = (extent.1 - extent.0).norm() * 0.5;
    let centre = nalgebra::center(&extent.0, &extent.1);

    // Always on here: the question is what the masks do when they are asked for, and the
    // application no longer asks.
    let program = generate_with_masks(&doc, graph, true).expect("generates");
    println!("{path}");
    if program.masked.is_empty() {
        println!("\nno operand is masked in this document — nothing to measure");
        return;
    }
    let curves = curves_for(&doc, graph, &program);
    let renderer = match pollster::block_on(Renderer::new(&program, &[], &curves, &[], 64, 1.0)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("no GPU available ({e})");
            return;
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

    let doc_ref = &doc;
    println!("{} operands masked, {} params\n", program.masked.len(), program.params.len());
    let reach = Vector3::repeat(radius);
    for cells in GRIDS {
        renderer.build_skip_grid(centre - reach, radius * 2.0, cells, true);
        let reading = renderer.read_skip_grid(cells).expect("reads back");
        // A cell the ray cannot hop over: its clearance does not reach its own far corner.
        let cell = radius * 2.0 / f64::from(cells);
        let corner = (cell * 1.7320508) as f32;
        let near: Vec<usize> =
            (0..reading.clearance.len()).filter(|i| reading.clearance[*i] < corner).collect();
        let total = reading.ruled_out.len();

        println!("grid {cells}³ — {total} cells, {} of them near the surface", near.len());
        println!("{:<6} {:<22} {:>7} {:>9} {:>9}", "bit", "operand", "rate", "all", "near");
        for (bit, id) in program.masked.iter().enumerate() {
            let kind = match graph.node(*id) {
                Some(n) => {
                    let s = format!("{n:?}");
                    s.split(['{', '(']).next().unwrap_or("?").trim().to_string()
                }
                None => "gone".to_string(),
            };
            let set = |i: &usize| reading.ruled_out[*i] & (1 << bit) != 0;
            let all = (0..total).filter(|i| set(i)).count();
            let hot = near.iter().filter(|i| set(i)).count();
            println!(
                "{bit:<6} {:<22} {:>7.2} {:>8.1}% {:>8.1}%",
                format!("{kind} {}", id.raw() % 10000),
                fieldform_field::graph_lipschitz(doc_ref, graph, *id).max(1.0),
                100.0 * all as f64 / total.max(1) as f64,
                100.0 * hot as f64 / near.len().max(1) as f64,
            );
        }
        let any = (0..total).filter(|i| reading.ruled_out[*i] != 0).count();
        println!(
            "any operand ruled out in {:.1}% of all cells, {:.1}% of near cells\n",
            100.0 * any as f64 / total.max(1) as f64,
            100.0 * near.iter().filter(|i| reading.ruled_out[**i] != 0).count() as f64
                / near.len().max(1) as f64,
        );
    }
    println!(
        "A bit set in no cell is an operand that was masked and never skipped: a WGSL function\n\
         and a branch, for nothing. A bit set in most near cells and the frame still slower is\n\
         the other answer — the masking works and does not pay."
    );
}

/// The shape `OperandMasks` was written for, as `mask_frame_cost` builds it.
fn lattice_in_a_cube() -> (Document, Graph) {
    let lit = ParamRef::Literal;
    let mut g = Graph::empty();
    let cube = g.add(Node::Cuboid { edge: Default::default(),
        centre: fieldform_scene::Place::At([lit(0.0); 3]),
        size: [lit(200.0), lit(200.0), lit(200.0)],
        turn: fieldform_scene::Turn::default(),
    });
    let cell = g.add(Node::Lattice {
        cell: fieldform_scene::LatticeCell::Gyroid,
        map: fieldform_scene::CellMap::Rectangular { cell: lit(12.0) },
        phase: fieldform_scene::LatticePhase::Sheet { thickness: lit(1.5), offset: lit(0.0) },
        frame: fieldform_scene::Frame::default(),
    });
    g.add(Node::Boolean {
        op: fieldform_scene::BooleanOp::Intersection,
        inputs: vec![cube.into(), cell.into()],
    });
    (Document::new(), g)
}

/// The spines, frames and profiles the shader asks for — the same reader `mask_frame_cost` uses.
fn curves_for(
    doc: &Document,
    graph: &Graph,
    program: &fieldform_gpu::ShaderProgram,
) -> Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> {
    let extent = match fieldform_field::part_extent(doc, graph) {
        fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
        _ => panic!("bounded"),
    };
    let floor = ((extent.1 - extent.0).norm() * 1e-5).max(1e-4);
    program
        .curves
        .iter()
        .map(|use_of| match *use_of {
            fieldform_gpu::CurveUse::Spine(id) => {
                (doc.curves[&id].tessellate_to_angle(SPINE_DEGREES, floor), None)
            }
            fieldform_gpu::CurveUse::Miters(_) => {
                unreachable!("no scene here mitres its draft")
            }
            fieldform_gpu::CurveUse::LoftSpine(_)
            | fieldform_gpu::CurveUse::LoftSection(..)
            | fieldform_gpu::CurveUse::LoftSlope(..) => {
                unreachable!("no scene here lofts")
            }
            fieldform_gpu::CurveUse::Frame(node) => {
                let Some(Node::Sweep { outline, spine, .. }) = graph.node(node) else {
                    unreachable!("a frame belongs to a sweep");
                };
                let points = doc.curves[spine].tessellate_to_angle(SPINE_DEGREES, floor);
                let frame = fieldform_field::sweep_frame(doc, outline, *spine, &points)
                    .expect("a spine with a direction");
                (frame, None)
            }
            fieldform_gpu::CurveUse::Profile(node) => {
                let (curve, ignore) = match graph.node(node) {
                    Some(Node::Extrude { outline, sweep, .. }) => (outline.curve(), sweep.source()),
                    Some(Node::Sweep { outline, spine, .. }) => (outline.curve(), Some(*spine)),
                    Some(Node::Revolve { outline, frame, .. }) => (outline.curve(), frame.curve_source()),
                    _ => unreachable!("a profile belongs to an extrude, a revolve or a sweep"),
                };
                let world = fieldform_field::profile_loop_turning(doc, curve, SPINE_DEGREES, ignore)
                    .expect("a closed loop")
                    .points;
                let (origin, normal) = fieldform_field::plane_of(&world).expect("planar");
                let rotation = fieldform_field::rotation_from_z(normal);
                let into = move |p: Point3<f64>| {
                    Point3::from(rotation.inverse_transform_vector(&(p.coords - origin.coords)))
                };
                let profile =
                    fieldform_field::Profile::in_plane(&world, into, TOLERANCE_MM).expect("planar");
                let points: Vec<Point3<f64>> =
                    profile.points().iter().map(|p| Point3::new(p.x, p.y, 0.0)).collect();
                let reads = fieldform_field::profile_reads_out_to(doc, graph, node);
                let (lo, hi) = profile.bounds();
                let bake = fieldform_field::bake_cells((hi - lo).max(), TOLERANCE_MM, reads)
                    .map(|cells| profile.baked(cells, reads))
                    .and_then(|p| p.bake().cloned());
                (points, bake)
            }
            fieldform_gpu::CurveUse::Point(_) => unreachable!("nothing here attracts"),
            fieldform_gpu::CurveUse::Wall(..) => unreachable!("nothing here is a thin extrude"),
        })
        .collect()
}
