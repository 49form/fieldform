// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the driver charges for the owner's wing, and how much of it is the normal pass.
//!
//! **Written from a freeze the owner reported three times.** A swept wing, mirrored, blended to a
//! fuselage: ticking "include in this block" on the mirror stopped the application for about five
//! seconds; changing the boolean to a union and then to G1 did it again, longer each time. The
//! application's own slow-frame line named `create_compute_pipeline` every time, so the cost is
//! the driver's, and it is paid on whatever thread asks.
//!
//! A survey measured that cost against the number of spine walks inlined into one function —
//! 8 walks 2.0 s, 16 walks 6.2 s, 32 walks 53.5 s, 64 walks 267 s — which is faster than cubic
//! and is the shape of a compiler that is superlinear in a function's size. The value pass inlines
//! one walk per sweep. The normal pass used to inline seven more.
//!
//! So this counts the walks in the generated source and times the driver on each scene, with the
//! wing alone, mirrored, and mirrored *and* blended — which is the sequence the owner walked.
//!
//! **Measured on an RTX A500 Laptop GPU, 2026-09-20**, each figure the first time that source was
//! ever compiled. The driver keeps a disk cache, so a second run of an unchanged shader reports
//! about 15 ms and means nothing; the way to take a cold reading is to change the shader, which
//! is also why `FIELDFORM_NO_MIRROR_FOLD=1` exists — it makes the other composition, so both can
//! be read cold in one sitting.
//!
//! **These are single cold samples and they vary.** One later reading of the mirrored row came
//! back at 14525 ms where the table says 3239, with nothing changed but the prelude gaining a
//! function that row does not call. Treat a difference under about 2x as noise; the 8419 -> 3756
//! below is well clear of it, and the 754 -> 637 further down is not.
//!
//! | scene | walks | readings | as first written | one walk per reading | one walk for all seven |
//! |---|---|---|---|---|---|
//! | the wing alone | 8 → 2 | 8 | 1697 ms | 1098 ms | **1104 ms** |
//! | mirrored | 16 → 4 | 16 | 8085 ms | 3608 ms | **3239 ms** |
//! | mirrored and blended, G1 | 16 → 4 | 16 | 8419 ms | 4461 ms | **3756 ms** |
//!
//! Two things the table settles, both of which were open questions:
//!
//! - **The walk is what costs, and the profile reading beside it is not.** The last row of the
//!   run — the same section *extruded* rather than swept, so sixteen profile readings and no
//!   walk at all — compiles in 883 ms. A walk is therefore worth about 720 ms here and a profile
//!   reading about 55. Giving an extrude or a revolve an analytic normal, which was proposed
//!   alongside the sweep, would buy a tenth of what the sweep bought. It is not the lever.
//! - **What is left is the mirror, and it did not pay here.** A joined mirror is
//!   `min(f(p), f(mirror(p)))`, so it writes its whole subtree twice — which is why the mirrored
//!   rows carry four walks rather than two. When the body lies wholly on one side of its own
//!   plane one reading does, exactly rather than approximately, and `mirror_one_side` now emits
//!   that. The `mirror` column says which each row got.
//!
//!   The swept rows still say `twice`, and the reason is the extent rather than the mirror: a
//!   sweep's bound is a ball of the profile's reach at every spine sample
//!   (`Node::Sweep`'s arm of `graph_extent`), so a wing whose section is 400 mm across and whose
//!   root is 200 mm out has a bound that reaches **0.0006 mm past** the centreline. The body does
//!   not cross; its bound does, and a bound is all this can safely read. Measured on the one row
//!   that does fold, the extruded one: 754 ms written twice against 637 ms written once.
//!
//!   So the remaining lever is a tighter extent for a sweep — the oriented section at each spine
//!   sample rather than a ball around it — which would also tighten every box guard and march
//!   bound built on it.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example wing_compile_cost`

use fieldform_scene::{
    BooleanOp, Continuity, Curve, CurveId, Document, Ends, Fillet, Frame, Graph, Node, ParamRef,
    Place,
};
use nalgebra::Point3;

fn main() {
    let scenes: Vec<(&str, (Document, Graph))> = vec![
        ("the wing alone", wing(Stage::Alone)),
        ("the wing mirrored", wing(Stage::Mirrored)),
        ("mirrored and blended to the fuselage, G1", wing(Stage::Blended)),
        // The same shape without a spine to walk, so what an extrude's own six differences
        // cost can be read off against the sweep's beside it.
        ("the section extruded, mirrored, blended", wing(Stage::Extruded)),
    ];

    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("an adapter");
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: fieldform_gpu::required_limits(&adapter),
                ..Default::default()
            })
            .await
            .expect("a device")
    });
    println!("{}\n", device.adapter_info().name);
    println!(
        "{:<42} {:>6} {:>7} {:>9} {:>7} {:>10}",
        "scene", "lines", "walks", "readings", "mirror", "driver ms"
    );

    for (label, (doc, graph)) in scenes {
        let program = fieldform_gpu::generate(&doc, &graph).expect("generates");
        let lines = program.source.lines().count();
        // What the driver has to unroll, counted where it is asked for rather than where it is
        // written. The generated half of the source is everything after the prelude, so a call
        // site there is a copy the driver inlines, and each of these functions walks the spine
        // exactly once however many readings it takes.
        let cut = program
            .source
            .find("// ---- generated from the composition graph ----")
            .expect("every program carries the marker");
        let generated = &program.source[cut..];
        let walks = generated.matches("sweep_distance(").count()
            + generated.matches("sweep_value_normal(").count()
            + generated.matches("pipe_distance(").count()
            + generated.matches("pipe_normal(").count();
        // The profile readings the driver unrolls beside them: a sweep takes seven for a value
        // and a normal, an extrude or a revolve six for its central difference and one more for
        // its value.
        let readings = generated.matches("sweep_distance(").count()
            + 7 * generated.matches("sweep_value_normal(").count()
            + generated.matches("extrude_distance(").count()
            + 6 * generated.matches("extrude_normal(").count()
            + generated.matches("revolve_distance(").count()
            + 6 * generated.matches("revolve_normal(").count();

        // Whether each joined mirror writes its body once or twice. It writes it once when the
        // body never crosses its own plane, which is decided from the extent -- and a sweep's
        // extent is a ball of the profile's reach at every spine sample, so a swept body's bound
        // reaches back past its own root and a swept wing does not qualify.
        let folds = graph
            .nodes()
            .filter(|(_, n)| matches!(n, Node::Mirror { .. }))
            .map(|(id, _)| match fieldform_field::mirror_one_side(&doc, &graph, id) {
                Some(_) => "once",
                None => "twice",
            })
            .collect::<Vec<_>>()
            .join(",");
        let folds = if folds.is_empty() { "-".to_string() } else { folds };
        let curves = curves_for(&doc, &graph, &program);
        let t = std::time::Instant::now();
        let built = fieldform_gpu::Renderer::with_device(
            device.clone(),
            queue.clone(),
            &program,
            &[],
            &curves,
            &[],
        );
        let ms = t.elapsed().as_secs_f64() * 1e3;
        match built {
            Ok(_) => println!("{label:<42} {lines:>6} {walks:>7} {readings:>9} {folds:>7} {ms:>10.0}"),
            Err(e) => println!("{label:<42} {lines:>6} {walks:>7} {readings:>9} {folds:>7}   refused: {e}"),
        }
    }
}

/// The spines, frames and profiles the shader asks for, read the way the application reads them.
///
/// The same walk `sweep_cost` does: a spine tessellated to the viewport's degree of turn, a frame
/// built on it, and a profile flattened into its own plane and baked to the display tolerance.
fn curves_for(
    doc: &Document,
    graph: &Graph,
    program: &fieldform_gpu::ShaderProgram,
) -> Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> {
    const DEGREES: f64 = 1.0;
    const TOLERANCE_MM: f64 = 0.5;
    let extent = match fieldform_field::part_extent(doc, graph) {
        fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
        _ => panic!("every scene here is bounded"),
    };
    let floor = ((extent.1 - extent.0).norm() * 1e-5).max(1e-4);
    program
        .curves
        .iter()
        .map(|use_of| match *use_of {
            fieldform_gpu::CurveUse::Spine(id) => {
                (doc.curves[&id].tessellate_to_angle(DEGREES, floor), None)
            }
            // No scene here mitres its draft, so there are no corners to displace.
            fieldform_gpu::CurveUse::Miters(_) => {
                unreachable!("this scene's draft rounds its corners")
            }
            fieldform_gpu::CurveUse::LoftSpine(_)
            | fieldform_gpu::CurveUse::LoftSection(..)
            | fieldform_gpu::CurveUse::LoftSlope(..) => {
                unreachable!("this scene lofts nothing")
            }
            fieldform_gpu::CurveUse::Frame(node) => {
                let Some(Node::Sweep { outline, spine, .. }) = graph.node(node) else {
                    unreachable!("a frame belongs to a sweep");
                };
                let points = doc.curves[spine].tessellate_to_angle(DEGREES, floor);
                let frame = fieldform_field::sweep_frame(doc, outline, *spine, &points)
                    .expect("a spine with a direction");
                (frame, None)
            }
            fieldform_gpu::CurveUse::Profile(node) => {
                let (curve, ignore) = match graph.node(node) {
                    Some(Node::Extrude { outline, sweep, .. }) => (outline.curve(), sweep.source()),
                    Some(Node::Sweep { outline, spine, .. }) => (outline.curve(), Some(*spine)),
                    _ => unreachable!("a profile belongs to an extrude or a sweep"),
                };
                let world = fieldform_field::profile_loop_turning(doc, curve, DEGREES, ignore)
                    .expect("a closed loop")
                    .points;
                let (origin, normal) = fieldform_field::plane_of(&world).expect("planar");
                let rotation = fieldform_field::rotation_from_z(normal);
                let into = move |p: Point3<f64>| {
                    Point3::from(rotation.inverse_transform_vector(&(p.coords - origin.coords)))
                };
                let profile = fieldform_field::Profile::in_plane(&world, into, TOLERANCE_MM)
                    .expect("planar");
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

#[derive(Clone, Copy)]
enum Stage {
    Alone,
    Mirrored,
    Blended,
    Extruded,
}

/// An aerofoil section, swept along a slightly swept and tapered path, then mirrored about the
/// centreline and blended into a cylindrical fuselage.
fn wing(stage: Stage) -> (Document, Graph) {
    let mut doc = Document::new();

    // A section in the XZ plane at y = 200, so the sweep runs outboard along +Y.
    let section = CurveId::new();
    let mut points = Vec::new();
    let chord = 400.0;
    for i in 0..24 {
        let u = i as f64 / 24.0;
        let x = chord * (1.0 - (u * std::f64::consts::TAU).cos()) * 0.5;
        let t = 5.0 * chord * 0.12 * (0.2969 * (x / chord).sqrt() - 0.1260 * (x / chord)
            - 0.3516 * (x / chord).powi(2)
            + 0.2843 * (x / chord).powi(3)
            - 0.1015 * (x / chord).powi(4));
        let z = if (u * std::f64::consts::TAU).sin() >= 0.0 { t } else { -t };
        points.push(Point3::new(x, 200.0, z));
    }
    doc.curves.insert(
        section,
        Curve::polyline(section, "Section", points, true).expect("a closed section"),
    );

    let spine = CurveId::new();
    doc.curves.insert(
        spine,
        Curve::new(
            spine,
            "Span",
            3,
            vec![
                Point3::new(0.0, 200.0, 0.0),
                Point3::new(40.0, 700.0, 20.0),
                Point3::new(90.0, 1300.0, 60.0),
                Point3::new(140.0, 1900.0, 120.0),
            ],
            vec![1.0; 4],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            false,
        )
        .expect("a single-span cubic"),
    );

    let mut graph = Graph::empty();
    let body = match stage {
        Stage::Extruded => graph.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(section),
            sweep: fieldform_scene::Sweep::Normal,
            height: ParamRef::Literal(1700.0),
            draft: ParamRef::Literal(0.0),
            start: ParamRef::Literal(0.0),
            sides: fieldform_scene::Sides::One,
            corners: Default::default(),
        }),
        _ => graph.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(section), spine, section: Default::default(), law: Default::default() }),
    };
    if matches!(stage, Stage::Alone) {
        return (doc, graph);
    }

    let mirrored = graph.add(Node::Mirror {
        input: body.into(),
        frame: Frame::At {
            origin: [ParamRef::Literal(0.0); 3],
            axis: [ParamRef::Literal(0.0), ParamRef::Literal(1.0), ParamRef::Literal(0.0)],
        },
        join: true,
    });
    if matches!(stage, Stage::Mirrored) {
        return (doc, graph);
    }

    let fuselage = graph.add(Node::Cylinder { edge: Default::default(),
        ends: Ends::At {
            from: Place::At([
                ParamRef::Literal(-600.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
            ]),
            to: Place::At([
                ParamRef::Literal(1400.0),
                ParamRef::Literal(0.0),
                ParamRef::Literal(0.0),
            ]),
        },
        radius: ParamRef::Literal(180.0),
    });
    graph.add(Node::Blend {
        op: BooleanOp::Union,
        a: mirrored.into(),
        b: fuselage.into(),
        fillet: Fillet { radius: ParamRef::Literal(40.0), continuity: Continuity::G1, second: None },
    });
    (doc, graph)
}
