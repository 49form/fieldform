// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a loft costs the viewport, in milliseconds to compile and per frame, G0 against G1,
//! against the sweep of its first section along the same path.
//!
//! Reported 2026-10-03 as "lofting G1 kills the viewport, it's all very slow". Two things could
//! be the cost and they scale differently: the tracer's step budget grows with the loft's bound
//! (`loft_stretch`, larger at G1), and every reading at G1 walks up to four sections where G0
//! walks two -- and the normal takes seven readings, so the driver inlines that many walks into
//! one function, which is the pattern that crashed it on the mitred wall sweep. The compile
//! time says which: a budget costs frames, an inlining costs the compile and the frames both.
//!
//! Run with: `cargo run -p fieldform-gpu --example loft_cost`

use fieldform_gpu::{generate, Camera, Renderer};
use fieldform_scene::{Curve, CurveId, Document, Graph, LoftSmooth, Node, ParamRef};
use nalgebra::{Point3, Vector3};

/// The viewport's own numbers, from `traced/mod.rs`.
const STEP_BUDGET: f64 = 512.0;
const STEP_HEADROOM: f64 = 2.0;
const MAX_STEP_BUDGET: u32 = 16_384;
const S: f64 = 1000.0;

fn main() {
    let scenes: Vec<(&str, (Document, Graph))> = vec![
        ("the square swept along the path", swept()),
        ("four sections lofted at G0", lofted(LoftSmooth::G0, 4)),
        ("four sections lofted at G1", lofted(LoftSmooth::G1, 4)),
        ("two sections lofted at G0", lofted(LoftSmooth::G0, 2)),
        ("two sections lofted at G1", lofted(LoftSmooth::G1, 2)),
        // Identical squares, so the bound is one and only the walks differ between G0 and G1.
        ("two equal squares lofted at G0", squares(LoftSmooth::G0, 2)),
        ("two equal squares lofted at G1", squares(LoftSmooth::G1, 2)),
        ("four equal squares lofted at G1", squares(LoftSmooth::G1, 4)),
    ];
    println!(
        "{:<36} {:>7} {:>8} {:>6} {:>7} {:>9} {:>10} {:>9} {:>8}",
        "scene", "points", "stretch", "mult", "budget", "gen ms", "compile ms", "ms/frame", "us/kstep"
    );
    for (label, (doc, graph)) in scenes {
        let generating = std::time::Instant::now();
        let program = generate(&doc, &graph).expect("generates");
        let gen_ms = generating.elapsed().as_secs_f64() * 1000.0;
        let extent = match fieldform_field::part_extent(&doc, &graph) {
            fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
            _ => panic!("every scene here is bounded"),
        };
        let size = (extent.1 - extent.0).norm();

        let mut points = 0usize;
        let mut stretch = 1.0f64;
        let curves: Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> = program
            .curves
            .iter()
            .map(|use_of| match *use_of {
                fieldform_gpu::CurveUse::Spine(id) => {
                    (doc.curves[&id].tessellate(fieldform_field::DEFAULT_TOLERANCE_MM), None)
                }
                fieldform_gpu::CurveUse::Frame(node) => match graph.node(node) {
                    Some(Node::Sweep { outline, spine, .. }) => {
                        let path = doc.curves[spine].tessellate(fieldform_field::DEFAULT_TOLERANCE_MM);
                        stretch = fieldform_field::sweep_stretch(&doc, outline, *spine, &Default::default());
                        (
                            fieldform_field::sweep_frame(&doc, outline, *spine, &path)
                                .expect("a spine with a direction"),
                            None,
                        )
                    }
                    Some(Node::Loft { .. }) => (
                        loft_of(&doc, &graph, node).frame_normals().iter().map(|n| Point3::from(*n)).collect(),
                        None,
                    ),
                    _ => unreachable!("a frame belongs to a sweep or a loft"),
                },
                fieldform_gpu::CurveUse::Profile(node) => {
                    let Some(Node::Sweep { outline, spine, .. }) = graph.node(node) else {
                        unreachable!("the only profile here is the sweep's");
                    };
                    let laid = fieldform_field::sweep_outline_section(
                        &doc,
                        outline,
                        *spine,
                        fieldform_field::PROFILE_TURN_DEGREES,
                    )
                    .expect("a section");
                    let rotation = fieldform_field::rotation_from_z(laid.normal);
                    let origin = laid.origin;
                    let into = move |p: Point3<f64>| {
                        Point3::from(rotation.inverse_transform_vector(&(p.coords - origin.coords)))
                    };
                    let flat = fieldform_field::Profile::in_plane(&laid.points, into, 1e-3).expect("planar");
                    points += flat.points().len();
                    (flat.points().iter().map(|p| Point3::new(p.x, p.y, 0.0)).collect(), None)
                }
                fieldform_gpu::CurveUse::LoftSpine(node) => {
                    let built = loft_of(&doc, &graph, node);
                    stretch = built.stretch();
                    (built.spine.points().to_vec(), None)
                }
                fieldform_gpu::CurveUse::LoftSlope(node, k) => (
                    loft_of(&doc, &graph, node).section_slopes(k).iter().map(|m| Point3::new(m.x, m.y, 0.0)).collect(),
                    None,
                ),
                fieldform_gpu::CurveUse::LoftSection(node, k) => {
                    let section = loft_of(&doc, &graph, node).section_points(k).to_vec();
                    points += section.len();
                    (section.iter().map(|q| Point3::new(q.x, q.y, 0.0)).collect(), None)
                }
                ref other => unreachable!("nothing here binds {other:?}"),
            })
            .collect();

        let compiling = std::time::Instant::now();
        let renderer = match pollster::block_on(Renderer::new(&program, &[], &curves, &[], 64, 1.0)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("no GPU available ({e})");
                return;
            }
        };
        let compile_ms = compiling.elapsed().as_secs_f64() * 1000.0;
        let values: Vec<f32> = program
            .params
            .iter()
            .map(|p| match p {
                ParamRef::Literal(v) => *v as f32,
                _ => 0.0,
            })
            .collect();
        renderer.set_params(&values);

        let multiplier = fieldform_field::step_multiplier(&doc, &graph);
        let budget = ((STEP_BUDGET * multiplier * STEP_HEADROOM).ceil() as u32)
            .clamp(STEP_BUDGET as u32, MAX_STEP_BUDGET);
        let centre = nalgebra::center(&extent.0, &extent.1);
        let radius = size * 0.5;
        let eye = centre + Vector3::new(1.6, 1.2, 0.9) * radius;
        let mut camera = Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), 45.0, 1.0);
        camera.value_scale = multiplier;

        let edge = 720u32;
        // The first frame carries whatever the driver left for first use; it is not counted.
        // **The least of three runs of twenty**, because the first table was taken beside a
        // live session of the application and read three times what a quiet machine reads --
        // the GPU is shared, and the least run is the one least shared.
        let _ = renderer.render(&camera, edge, edge, budget);
        let mut ms = f64::INFINITY;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            for _ in 0..20 {
                renderer.render(&camera, edge, edge, budget).expect("renders");
            }
            ms = ms.min(start.elapsed().as_secs_f64() * 1000.0 / 20.0);
        }
        let per_step = ms / budget as f64 * 1000.0;
        println!(
            "{label:<36} {points:>7} {stretch:>8.2} {multiplier:>6.2} {budget:>7} {gen_ms:>9.1} {compile_ms:>10.0} {ms:>9.1} {per_step:>8.1}"
        );
    }
}

fn loft_of(doc: &Document, graph: &Graph, node: fieldform_scene::NodeId) -> std::sync::Arc<fieldform_field::Lofted> {
    let Some(Node::Loft { rims: _, sections, spine, smooth }) = graph.node(node) else {
        unreachable!("a loft binding belongs to a loft");
    };
    fieldform_field::loft_built(doc, sections, *spine, *smooth).expect("builds")
}

/// The harness's path: a single-span cubic bending in two planes.
fn path(doc: &mut Document) -> CurveId {
    let id = CurveId::new();
    doc.curves.insert(
        id,
        Curve::new(
            id,
            "Path",
            3,
            vec![
                Point3::new(0.0, 0.0, -0.30 * S),
                Point3::new(-0.10 * S, 0.02 * S, 0.0),
                Point3::new(0.06 * S, 0.14 * S, 0.12 * S),
                Point3::new(0.30 * S, 0.10 * S, 0.34 * S),
            ],
            vec![1.0; 4],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            false,
        )
        .expect("a single-span cubic"),
    );
    id
}

/// The harness's four sections, drawn flat about the path at a quarter of its parameter each:
/// a square, a rational circle, a rounded rectangle and a rational ellipse.
fn sections(doc: &mut Document, bend: &Curve, count: usize) -> Vec<CurveId> {
    let flat = |centre: Point3<f64>, shape: Vec<Point3<f64>>| -> Vec<Point3<f64>> {
        shape.into_iter().map(|p| centre + p.coords).collect()
    };
    let w = std::f64::consts::FRAC_1_SQRT_2;
    let oval = |a: f64, b: f64| -> Vec<Point3<f64>> {
        vec![
            Point3::new(a, 0.0, 0.0),
            Point3::new(a, b, 0.0),
            Point3::new(0.0, b, 0.0),
            Point3::new(-a, b, 0.0),
            Point3::new(-a, 0.0, 0.0),
            Point3::new(-a, -b, 0.0),
            Point3::new(0.0, -b, 0.0),
            Point3::new(a, -b, 0.0),
            Point3::new(a, 0.0, 0.0),
        ]
    };
    let oval_weights = vec![1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0];
    let oval_knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0];
    let rounded = |half_x: f64, half_y: f64, r: f64| -> Vec<Point3<f64>> {
        let mut out = Vec::new();
        for (sx, sy, from) in [(1.0, 1.0, 0.0), (-1.0, 1.0, 90.0), (-1.0, -1.0, 180.0), (1.0, -1.0, 270.0)] {
            let (cx, cy) = (sx * (half_x - r), sy * (half_y - r));
            for i in 0..=6 {
                let a = (from + 90.0 * i as f64 / 6.0f64).to_radians();
                out.push(Point3::new(cx + r * a.cos(), cy + r * a.sin(), 0.0));
            }
        }
        out
    };
    let h = 0.10 * S;
    let square = CurveId::new();
    doc.curves.insert(
        square,
        Curve::polyline(
            square,
            "Square",
            flat(
                bend.evaluate(0.0),
                vec![
                    Point3::new(-h, -h, 0.0),
                    Point3::new(h, -h, 0.0),
                    Point3::new(h, h, 0.0),
                    Point3::new(-h, h, 0.0),
                ],
            ),
            true,
        )
        .expect("closed"),
    );
    let ellipse = CurveId::new();
    doc.curves.insert(
        ellipse,
        Curve::new(
            ellipse,
            "Ellipse",
            2,
            flat(bend.evaluate(1.0), oval(0.11 * S, 0.06 * S)),
            oval_weights.clone(),
            oval_knots.clone(),
            true,
        )
        .expect("a rational quadratic ellipse"),
    );
    if count == 2 {
        return vec![square, ellipse];
    }
    let circle = CurveId::new();
    doc.curves.insert(
        circle,
        Curve::new(
            circle,
            "Circle",
            2,
            flat(bend.evaluate(1.0 / 3.0), oval(0.09 * S, 0.09 * S)),
            oval_weights,
            oval_knots,
            true,
        )
        .expect("a rational quadratic circle"),
    );
    let rect = CurveId::new();
    doc.curves.insert(
        rect,
        Curve::polyline(
            rect,
            "Rounded rectangle",
            flat(bend.evaluate(2.0 / 3.0), rounded(0.12 * S, 0.07 * S, 0.03 * S)),
            true,
        )
        .expect("closed"),
    );
    vec![square, circle, rect, ellipse]
}

fn lofted(smooth: LoftSmooth, count: usize) -> (Document, Graph) {
    let mut doc = Document::new();
    let path = path(&mut doc);
    let bend = doc.curves[&path].clone();
    let ids = sections(&mut doc, &bend, count);
    let mut graph = Graph::empty();
    graph.add(Node::Loft { rims: Default::default(),
        sections: ids.into_iter().map(fieldform_scene::Outline::closed).collect(),
        spine: Some(path),
        smooth,
    });
    (doc, graph)
}

fn swept() -> (Document, Graph) {
    let mut doc = Document::new();
    let path = path(&mut doc);
    let bend = doc.curves[&path].clone();
    let ids = sections(&mut doc, &bend, 2);
    let mut graph = Graph::empty();
    graph.add(Node::Sweep { rims: Default::default(),
        outline: fieldform_scene::Outline::closed(ids[0]),
        spine: path,
        section: Default::default(),
        law: Default::default(),
    });
    (doc, graph)
}

/// `count` copies of the square along the path, so the sections are equal and the bound is one.
fn squares(smooth: LoftSmooth, count: usize) -> (Document, Graph) {
    let mut doc = Document::new();
    let path = path(&mut doc);
    let bend = doc.curves[&path].clone();
    let h = 0.10 * S;
    let mut ids = Vec::new();
    for k in 0..count {
        let centre = bend.evaluate(k as f64 / (count - 1) as f64);
        let id = CurveId::new();
        doc.curves.insert(
            id,
            Curve::polyline(
                id,
                "Square",
                vec![
                    centre + Vector3::new(-h, -h, 0.0),
                    centre + Vector3::new(h, -h, 0.0),
                    centre + Vector3::new(h, h, 0.0),
                    centre + Vector3::new(-h, h, 0.0),
                ],
                true,
            )
            .expect("closed"),
        );
        ids.push(id);
    }
    let mut graph = Graph::empty();
    graph.add(Node::Loft { rims: Default::default(),
        sections: ids.into_iter().map(fieldform_scene::Outline::closed).collect(),
        spine: Some(path),
        smooth,
    });
    (doc, graph)
}
