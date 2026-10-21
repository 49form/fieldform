// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a sharper viewport costs, in milliseconds per frame.
//!
//! The traced view renders at most [`MAX_PIXELS`] on its longest edge and is then stretched over
//! the viewport, so on a large window the silhouette is a staircase — the shading is a smooth
//! gradient and hides the upscale, but a hard edge cannot. The cap is not arbitrary: the image
//! is read back to the CPU and handed to egui as a texture, so the cost is a render plus two
//! copies, and all three grow with the area.
//!
//! This measures that growth, on the cheapest body worth drawing and the dearest one, so the
//! number the viewport picks is chosen against evidence rather than against a feeling about
//! what 720 sounds like.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example traced_cost`

use fieldform_gpu::{generate, Camera, Renderer};
use fieldform_scene::{Fillet, 
    BooleanOp, CurveId, Document, Graph, LatticePhase, Node, ParamRef, Sweep,
};
use nalgebra::Point3;

const MM: f64 = 300.0;

fn main() {
    for (label, (doc, graph)) in [
        ("an extruded profile", extruded()),
        ("a cube infilled with a gyroid", gyroid()),
        ("a revolve", revolved(1)),
        ("a revolve patterned six times", revolved(6)),
        ("a revolve in a row of six", rowed(6)),
        ("a revolve patterned 100 times", revolved(100)),
        ("a revolve patterned 500 times", revolved(500)),
    ] {
        println!("\n=== {label} ===");
        let program = generate(&doc, &graph).expect("generates");
        let curves: Vec<Vec<Point3<f64>>> = program
            .curves
            .iter()
            // Through the same walk the application uses, so what is measured is what is
            // drawn -- tessellating here by hand measured a polyline nothing builds.
            .map(|use_of| match use_of {
                fieldform_gpu::CurveUse::Profile(_) => doc
                    .curves
                    .values()
                    .next()
                    .and_then(|c| fieldform_field::profile_loop(&doc, c.id, None).ok())
                    .map(|found| found.points)
                    .unwrap_or_default(),
                fieldform_gpu::CurveUse::Spine(id) => doc.curves[id].tessellate(1e-4),
                // No scene here attracts to a point, so there is nothing to supply.
                fieldform_gpu::CurveUse::Point(_) => Vec::new(),
                fieldform_gpu::CurveUse::Miters(_) => Vec::new(),
                fieldform_gpu::CurveUse::Frame(_) => Vec::new(),
                fieldform_gpu::CurveUse::LoftSpine(_)
                | fieldform_gpu::CurveUse::LoftSection(..)
                | fieldform_gpu::CurveUse::LoftSlope(..) => Vec::new(),
                fieldform_gpu::CurveUse::Wall(..) => unreachable!("no scene here is a thin extrude"),
            })
            .collect();
        println!(
            "  profile polylines: {:?} points",
            curves.iter().map(|c| c.len()).collect::<Vec<_>>()
        );
        println!(
            "  profile polylines: {:?} points",
            curves.iter().map(|c| c.len()).collect::<Vec<_>>()
        );
        // Baked as the viewport bakes, which is the thing being measured.
        let curves: Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> = program
            .curves
            .iter()
            .zip(curves.iter())
            .map(|(use_of, points)| match use_of {
                fieldform_gpu::CurveUse::Profile(_) => {
                    let flat: Vec<nalgebra::Point2<f64>> =
                        points.iter().map(|p| nalgebra::Point2::new(p.x, p.y)).collect();
                    let bake = fieldform_field::Profile::new(flat).ok().and_then(|profile| {
                        let (lo, hi) = profile.bounds();
                        fieldform_field::bake_cells((hi - lo).max(), 0.5, (0.0, 0.0))
                            .map(|cells| profile.baked(cells, (0.0, 0.0)))
                            .and_then(|p| p.bake().cloned())
                    });
                    (points.clone(), bake)
                }
                fieldform_gpu::CurveUse::Spine(_) => (points.clone(), None),
                fieldform_gpu::CurveUse::Point(_) => (points.clone(), None),
                fieldform_gpu::CurveUse::Miters(_) => (points.clone(), None),
                fieldform_gpu::CurveUse::Frame(_) => (points.clone(), None),
                fieldform_gpu::CurveUse::LoftSpine(_)
                | fieldform_gpu::CurveUse::LoftSection(..)
                | fieldform_gpu::CurveUse::LoftSlope(..) => (points.clone(), None),
                fieldform_gpu::CurveUse::Wall(..) => unreachable!("no scene here is a thin extrude"),
            })
            .collect();
        // The driver's compile, which a program with an outline pays twice over when a frame may
        // finish on it: the exact read is compiled beside the bake's.
        let compiling = std::time::Instant::now();
        let renderer = match pollster::block_on(Renderer::new(&program, &[], &curves, &[], 64, 1.0)) {
            Ok(r) => {
                println!("  compiled in {:.0} ms", compiling.elapsed().as_secs_f64() * 1e3);
                r
            }
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

        let centre = Point3::origin();
        let eye = Point3::new(MM * 1.6, MM * 1.2, MM * 0.9);
        println!("  longest edge   frame      relative");
        let mut base = 0.0f64;
        for edge in [720u32, 1440] {
            let mut camera = Camera::looking_at(
                eye,
                centre,
                nalgebra::Vector3::z(),
                (centre, MM),
                45.0,
                1.0,
            );
            // `FIELDFORM_EXACT_FINISH=1` finishes every ray on the outline, as the still tier does.
            camera.exact_finish = std::env::var_os("FIELDFORM_EXACT_FINISH").is_some();
            // One warm render, then five timed: the first allocates the buffers the rest reuse,
            // which is what the viewport does too after its first frame at a given size.
            let _ = renderer.render(&camera, edge, edge, 512);
            let start = std::time::Instant::now();
            for _ in 0..5 {
                renderer.render(&camera, edge, edge, 512).expect("renders");
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / 5.0;
            if base == 0.0 {
                base = ms;
            }
            println!("  {edge:>10}   {ms:>6.1} ms   {:.2}x", ms / base);
        }
    }
}

fn extruded() -> (Document, Graph) {
    let mut doc = Document::new();
    let id = CurveId::new();
    let h = 0.4 * MM;
    doc.curves.insert(
        id,
        fieldform_scene::Curve::polyline(
            id,
            "Profile".to_string(),
            vec![
                Point3::new(-h, -h, 0.0),
                Point3::new(h, -h, 0.0),
                Point3::new(h, h, 0.0),
                Point3::new(-h, h, 0.0),
            ],
            true,
        )
        .expect("closed"),
    );
    let mut graph = Graph::empty();
    graph.add(Node::Extrude { rims: Default::default(),
        outline: fieldform_scene::Outline::closed(id),
        sweep: Sweep::Normal,
        height: ParamRef::Literal(0.6 * MM),
                draft: ParamRef::Literal(0.0),
                start: ParamRef::Literal(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
    });
    (doc, graph)
}

/// A profile revolved, and optionally patterned about Z.
///
/// **The scene the cheap ones did not stand in for.** A revolve evaluates a profile polyline per
/// sample and a pattern multiplies that by its count, so unlike the analytic solids above this
/// one's cost is genuinely in the field rather than in the readback -- which is exactly the case
/// a measurement taken on a cube and a gyroid would miss.
fn revolved(copies: u32) -> (Document, Graph) {
    let mut doc = Document::new();
    let id = CurveId::new();
    let h = 0.25 * MM;
    // **A curved section, not a four-cornered one.** A polyline tessellates to its own corners
    // however finely it is asked for, so a square profile measures none of what the sampling
    // rate costs -- and the sampling rate is what a pattern multiplies. This is the shape the
    // slowdown was reported on: a curve drawn by hand and revolved.
    let mut points = Vec::new();
    for i in 0..24 {
        let t = i as f64 / 24.0 * std::f64::consts::TAU;
        points.push(Point3::new(
            0.6 * MM + h * t.cos(),
            0.0,
            h * t.sin() * 0.8,
        ));
    }
    doc.curves.insert(
        id,
        fieldform_scene::Curve::new(
            id,
            "Section".to_string(),
            3,
            points,
            vec![1.0; 24],
            {
                let n = 24 + 3 + 1;
                (0..n).map(|i| i as f64).collect()
            },
            true,
        )
        .expect("a closed cubic section"),
    );
    let mut graph = Graph::empty();
    let solid = graph.add(Node::Revolve { rims: Default::default(),
        outline: fieldform_scene::Outline::closed(id),
        frame: fieldform_scene::Frame::default(),
        angle: fieldform_scene::ParamRef::Literal(360.0),
        sides: fieldform_scene::AngleSides::One,
    });
    if copies > 1 {
        graph.add(Node::Pattern {
            options: Default::default(),
            input: solid.into(),
            count: ParamRef::Literal(f64::from(copies)),
            kind: fieldform_scene::PatternKind::Radial {
                frame: fieldform_scene::Frame::default(),
            },
        
            join: true,
            fillet: Fillet::hard(),
        });
    }
    (doc, graph)
}


/// The same body in a row rather than a ring, so the two folds can be compared directly.
///
/// A linear fold is a dot product and a subtraction; a radial one has to get into the axis's
/// frame, which means building a rotation from three Euler angles. The gap between these two
/// numbers is what that costs.
fn rowed(copies: u32) -> (Document, Graph) {
    let (doc, mut graph) = revolved(1);
    let solid = graph.roots()[0];
    graph.add(Node::Pattern {
        options: Default::default(),
        input: solid.into(),
        count: ParamRef::Literal(f64::from(copies)),
        kind: fieldform_scene::PatternKind::Linear {
            step: [
                ParamRef::Literal(MM * 0.8),
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
            ],
        },
    
        join: true,
        fillet: Fillet::hard(),
    });
    (doc, graph)
}

fn gyroid() -> (Document, Graph) {
    let doc = Document::new();
    let mut graph = Graph::empty();
    let cube = graph.add(Node::Cuboid { edge: Default::default(),
        centre: fieldform_scene::Place::At([ParamRef::Literal(0.0); 3]),
        size: [ParamRef::Literal(MM * 1.2); 3],
        turn: fieldform_scene::no_turn(),
    });
    let lattice = graph.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell: fieldform_scene::LatticeCell::Gyroid,
        map: fieldform_scene::CellMap::Rectangular {
            cell: ParamRef::Literal(MM * 0.15),
        },
        phase: LatticePhase::sheet(ParamRef::Literal(MM * 0.02)),
    });
    graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![cube.into(), lattice.into()] });
    (doc, graph)
}
