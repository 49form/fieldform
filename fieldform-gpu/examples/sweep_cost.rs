// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a sweep costs the viewport, in milliseconds per frame, against the extrude it replaces.
//!
//! Reported after sweeping a large glyph made the viewport crawl. Two things could be the cost
//! and they scale differently: the tracer's step budget grows with the sweep's stretch
//! (`sweep_stretch`), and every step walks the whole spine, which the viewport samples at one
//! degree of turn (`spine_degrees`). So the same profile is drawn extruded, swept along a
//! straight line (a stretch of exactly one, so only the walk differs), and swept round two
//! bends; a pipe along the tighter bend is the walk without the profile or the stretch.
//!
//! **And again with the spine walked as a draft walks it** (`DRAFT_SPINE_DEGREES`, loop spec
//! P9): two degrees of turn on every rung where a settled rebuild walks the rung's own, one on
//! Balanced. The profile is the same in both tables, so only the spine differs. The straight
//! line is the control, one segment at any angle, and it does not read alike: 6.3 ms in the
//! first table and 5.9 in the second, and the same with both rows at two degrees, so about
//! 0.4 ms of any gap between the tables is where a row sits rather than how its spine was
//! walked. `walk ms` is what a rebuild spends on the spine on this thread -- its walk, and for
//! a sweep the frame over it, which `traced/rebuild.rs` works out on every rebuild.
//!
//! Medians of seven runs, 2026-10-12, dev profile, the first table first and then the second
//! (the order reversed read within 0.4 ms of these):
//!
//! | scene | segments, 1° → 2° | ms/frame | walk ms |
//! |---|---|---|---|
//! | straight line (control) | 1 → 1 | 6.3 → 5.9 | 0.47 → 0.47 |
//! | gentle bend | 96 → 52 | 15.6 → **12.4** | 0.75 → 0.68 |
//! | tighter bend | 150 → 86 | 84.8 → **73.2** | 0.95 → 0.81 |
//! | pipe round the tighter bend | 173 → 86 | 11.6 → **9.5** | 0.32 → 0.17 |
//!
//! **And with the section read in its tube** (loop spec P2, `sweep-spec.md` §4.1): a square
//! section under the plain law reads its walls truly, so the stretch no longer shortens the
//! steps that find them; it still sets the budget, which a sweep's caps read from inside need.
//! `tube` is the tube the shader is told: `inf` for a straight spine, `-` where there is none.
//! Medians of seven runs, 2026-10-13, dev profile, interleaved with the commit before's binary,
//! which went first in every other round:
//!
//! | scene | stretch | tube | 1°, before → after | 2°, before → after |
//! |---|---|---|---|---|
//! | the glyph extruded (control) | — | — | 6.5 → 6.4 | — |
//! | straight line | 1.00 | inf | 6.2 → 6.3 | 5.9 → 6.0 |
//! | gentle bend | 1.40 | 1559 | 15.7 → **13.8** | 12.3 → **10.7** |
//! | tighter bend | 14.37 | 481 | 86.0 → **18.6** | 73.0 → **15.6** |
//! | pipe round the tighter bend | — | — | 11.5 → 12.7 | 9.4 → 10.8 |
//!
//! The pipe draws no sweep, and the gap in its row is where it sits: run on its own it reads
//! 11.7 → 11.4 ms, and the tenth-of-a-degree pipe 27.8 → 27.9. In the full run it follows the
//! tight glyph, which kept the card four times busier just before it in the earlier binary.
//!
//! **And with the reading in the tube allowing for the polyline's frame** (`sweep-spec.md` §4.1,
//! 4a–4c), which it must round a coarse cut. Medians of seven runs, 2026-10-13, interleaved with
//! the binary before that correction: the tight glyph 18.5 → 19.4 ms at 1° and 15.7 → 15.3 at
//! 2°, the gentle bend 13.7 → 13.9 and 10.7 → 11.6, the straight line 6.2 → 6.3 and 6.0 → 6.0,
//! and the pipe, which draws no sweep, 12.9 → 13.3 and 10.9 → 10.9.
//!
//! Run with: `cargo run -p fieldform-gpu --example sweep_cost`

use fieldform_gpu::{generate, Camera, Renderer};
use fieldform_scene::{Curve, CurveId, Document, Graph, Node, ParamRef};
use nalgebra::{Point3, Vector3};

/// The viewport's own numbers, from `traced/mod.rs`.
const STEP_BUDGET: f64 = 512.0;
const STEP_HEADROOM: f64 = 2.0;
const MAX_STEP_BUDGET: u32 = 16_384;
const SPINE_DEGREES: f64 = 1.0;
const DRAFT_SPINE_DEGREES: f64 = 2.0;
const TOLERANCE_MM: f64 = 0.5;

fn main() {
    // The last row samples the spine ten times finer than the viewport ever asks, so the
    // spine index (`segment_tree`) is measured where a walk of every segment should hurt;
    // `FIELDFORM_NO_SPINE_INDEX=1` runs the same rows with the walk.
    let mut scenes: Vec<(&str, (Document, Graph), f64)> = vec![
        ("the glyph extruded", extruded(), SPINE_DEGREES),
        ("the glyph swept along a straight line", swept(Path::Straight), SPINE_DEGREES),
        ("the glyph swept round a gentle bend", swept(Path::Gentle), SPINE_DEGREES),
        ("the glyph swept round a tighter bend", swept(Path::Tight), SPINE_DEGREES),
        ("a pipe round the tighter bend", piped(), SPINE_DEGREES),
        ("the same pipe, spine at a tenth of a degree", piped(), SPINE_DEGREES / 10.0),
    ];
    let drafts = scenes.len();
    scenes.extend([
        ("the glyph swept along a straight line", swept(Path::Straight), DRAFT_SPINE_DEGREES),
        ("the glyph swept round a gentle bend", swept(Path::Gentle), DRAFT_SPINE_DEGREES),
        ("the glyph swept round a tighter bend", swept(Path::Tight), DRAFT_SPINE_DEGREES),
        ("a pipe round the tighter bend", piped(), DRAFT_SPINE_DEGREES),
    ]);
    let header = || {
        println!(
            "{:<42} {:>6} {:>7} {:>8} {:>6} {:>6} {:>7} {:>9} {:>8}",
            "scene", "spine", "profile", "stretch", "tube", "mult", "budget", "ms/frame", "walk ms"
        )
    };
    header();
    for (k, (label, (doc, graph), degrees)) in scenes.into_iter().enumerate() {
        if k == drafts {
            println!("\nThe spine walked as a draft walks it, at {DRAFT_SPINE_DEGREES} degrees:");
            header();
        }
        let program = generate(&doc, &graph).expect("generates");
        let extent = match fieldform_field::part_extent(&doc, &graph) {
            fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
            _ => panic!("every scene here is bounded"),
        };
        let size = (extent.1 - extent.0).norm();
        let floor = (size * 1e-5).max(1e-4);
        // The finer row needs a finer floor too, or the tessellation stops at the floor long
        // before it reaches the angle asked for.
        let floor = if degrees < SPINE_DEGREES { floor * 1e-4 } else { floor };
        let spine_points = |id: &CurveId| doc.curves[id].tessellate_to_angle(degrees, floor);

        let mut spine_segments = 0usize;
        let mut profile_points = 0usize;
        let mut stretch = 1.0f64;
        // Where the section reads as it is (`fieldform_field::sweep_tube`), and none where the
        // whole reading is divided by the stretch.
        let mut tube: Option<f64> = None;
        let curves: Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> = program
            .curves
            .iter()
            .map(|use_of| match *use_of {
                fieldform_gpu::CurveUse::Spine(id) => {
                    let points = spine_points(&id);
                    spine_segments = points.len().saturating_sub(1);
                    (points, None)
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
                    let Some(Node::Sweep { rims: _, outline, spine, section, law }) = graph.node(node) else {
                        unreachable!("a frame belongs to a sweep");
                    };
                    stretch = fieldform_field::sweep_stretch(&doc, outline, *spine, law);
                    tube = fieldform_field::sweep_tube(&doc, outline, *spine, section, law)
                        .map(|(_, tube)| tube);
                    let points = spine_points(spine);
                    let frame = fieldform_field::sweep_frame(&doc, outline, *spine, &points)
                        .expect("a spine with a direction");
                    (frame, None)
                }
                // The application's own reading of a profile (`profile_points` in
                // `traced/rebuild.rs`): the loop walked from the block's curve, flattened into its
                // fitted plane, and baked to the display tolerance.
                fieldform_gpu::CurveUse::Profile(node) => {
                    let (curve, ignore) = match graph.node(node) {
                        Some(Node::Extrude { outline, sweep, .. }) => (outline.curve(), sweep.source()),
                        Some(Node::Sweep { outline, spine, .. }) => (outline.curve(), Some(*spine)),
                        _ => unreachable!("a profile belongs to an extrude or a sweep"),
                    };
                    let world =
                        fieldform_field::profile_loop_turning(&doc, curve, SPINE_DEGREES, ignore)
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
                    profile_points = points.len();
                    let reads = fieldform_field::profile_reads_out_to(&doc, &graph, node);
                    let (lo, hi) = profile.bounds();
                    let bake = fieldform_field::bake_cells((hi - lo).max(), TOLERANCE_MM, reads)
                        .map(|cells| profile.baked(cells, reads))
                        .and_then(|p| p.bake().cloned());
                    (points, bake)
                }
                fieldform_gpu::CurveUse::Point(_) => unreachable!("nothing here attracts"),
                fieldform_gpu::CurveUse::Wall(..) => unreachable!("nothing here is a thin extrude"),
            })
            .collect();

        let renderer = match pollster::block_on(Renderer::new(&program, &[], &curves, &[], 64, 1.0))
        {
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

        // The budget the viewport would give this graph, and the value scale it traces with.
        let multiplier = fieldform_field::step_multiplier(&doc, &graph);
        let budget = ((STEP_BUDGET * multiplier * STEP_HEADROOM).ceil() as u32)
            .clamp(STEP_BUDGET as u32, MAX_STEP_BUDGET);
        let centre = nalgebra::center(&extent.0, &extent.1);
        let radius = size * 0.5;
        let eye = centre + Vector3::new(1.6, 1.2, 0.9) * radius;
        let mut camera =
            Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), 45.0, 1.0);
        camera.value_scale = fieldform_field::value_scale_with(&doc, &graph, &[]);

        let edge = 720u32;
        let _ = renderer.render(&camera, edge, edge, budget);
        let start = std::time::Instant::now();
        for _ in 0..5 {
            renderer.render(&camera, edge, edge, budget).expect("renders");
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / 5.0;

        // Each spine walked once, as the application walks it however many bindings read it,
        // and a sweep's frame worked out over that walk; the median of nine.
        let sweep_of = |node| match graph.node(node) {
            Some(Node::Sweep { outline, spine, .. }) => (outline, *spine),
            _ => unreachable!("a frame belongs to a sweep"),
        };
        let mut walks: Vec<f64> = (0..9)
            .map(|_| {
                let start = std::time::Instant::now();
                let mut walked = std::collections::HashMap::new();
                for use_of in &program.curves {
                    let id = match *use_of {
                        fieldform_gpu::CurveUse::Spine(id) => id,
                        fieldform_gpu::CurveUse::Frame(node) => sweep_of(node).1,
                        _ => continue,
                    };
                    walked.entry(id).or_insert_with(|| spine_points(&id));
                }
                for use_of in &program.curves {
                    if let fieldform_gpu::CurveUse::Frame(node) = *use_of {
                        let (outline, spine) = sweep_of(node);
                        std::hint::black_box(fieldform_field::sweep_frame(
                            &doc,
                            outline,
                            spine,
                            &walked[&spine],
                        ));
                    }
                }
                std::hint::black_box(walked);
                start.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        walks.sort_by(f64::total_cmp);
        let walk_ms = walks[walks.len() / 2];
        let tube = match tube {
            Some(t) if t > 1e6 => "inf".to_string(),
            Some(t) => format!("{t:.0}"),
            None => "-".to_string(),
        };
        println!(
            "{label:<42} {spine_segments:>6} {profile_points:>7} {stretch:>8.2} {tube:>6} {multiplier:>6.2} {budget:>7} {ms:>9.1} {walk_ms:>8.2}"
        );
    }
}

/// The glyph a user drew: a P-shaped outline of seven lines and two arcs, in the ground plane,
/// about 200 by 500 mm. Its corners reach some 270 mm from its centre, which is what makes a
/// bend expensive for it.
fn glyph(doc: &mut Document) -> CurveId {
    let line = |doc: &mut Document, a: (f64, f64), b: (f64, f64)| {
        let id = CurveId::new();
        let curve = Curve::polyline(
            id,
            "Line",
            vec![Point3::new(a.0, a.1, 0.0), Point3::new(b.0, b.1, 0.0)],
            false,
        )
        .expect("a line");
        doc.curves.insert(id, curve);
        id
    };
    let arc = |doc: &mut Document, a: (f64, f64), c: (f64, f64), b: (f64, f64)| {
        let id = CurveId::new();
        let curve = Curve::new(
            id,
            "Curve",
            2,
            vec![
                Point3::new(a.0, a.1, 0.0),
                Point3::new(c.0, c.1, 0.0),
                Point3::new(b.0, b.1, 0.0),
            ],
            vec![1.0; 3],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            false,
        )
        .expect("a quadratic");
        doc.curves.insert(id, curve);
        id
    };
    let seed = line(doc, (0.0, -100.0), (0.0, 200.0));
    line(doc, (0.0, -100.0), (100.0, -100.0));
    line(doc, (100.0, -100.0), (100.0, 100.0));
    line(doc, (100.0, 100.0), (200.0, 100.0));
    line(doc, (200.0, 100.0), (200.0, 200.0));
    line(doc, (200.0, 200.0), (100.0, 200.0));
    arc(doc, (100.0, 200.0), (100.0, 300.0), (200.0, 300.0));
    line(doc, (200.0, 300.0), (200.0, 400.0));
    arc(doc, (200.0, 400.0), (0.0, 400.0), (0.0, 200.0));
    seed
}

#[derive(Clone, Copy)]
enum Path {
    Straight,
    Gentle,
    Tight,
}

/// A path starting at the origin along +Z, so the section is the glyph as drawn.
fn path(doc: &mut Document, which: Path) -> CurveId {
    let id = CurveId::new();
    let curve = match which {
        Path::Straight => Curve::polyline(
            id,
            "Path",
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 1200.0)],
            false,
        )
        .expect("a line"),
        Path::Gentle => cubic(
            id,
            [(0.0, 0.0, 0.0), (0.0, 0.0, 500.0), (200.0, 0.0, 1000.0), (200.0, 300.0, 1500.0)],
        ),
        Path::Tight => cubic(
            id,
            [(0.0, 0.0, 0.0), (0.0, 0.0, 400.0), (500.0, 0.0, 700.0), (500.0, 500.0, 1000.0)],
        ),
    };
    doc.curves.insert(id, curve);
    id
}

fn cubic(id: CurveId, p: [(f64, f64, f64); 4]) -> Curve {
    Curve::new(
        id,
        "Path",
        3,
        p.iter().map(|&(x, y, z)| Point3::new(x, y, z)).collect(),
        vec![1.0; 4],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        false,
    )
    .expect("a single-span cubic")
}

fn extruded() -> (Document, Graph) {
    let mut doc = Document::new();
    let profile = glyph(&mut doc);
    let mut graph = Graph::empty();
    graph.add(Node::Extrude { rims: Default::default(),
        outline: fieldform_scene::Outline::closed(profile),
        sweep: fieldform_scene::Sweep::Normal,
        height: ParamRef::Literal(140.0),
        draft: ParamRef::Literal(0.0),
        start: ParamRef::Literal(0.0),
        sides: fieldform_scene::Sides::One,
        corners: Default::default(),
    });
    (doc, graph)
}

fn swept(which: Path) -> (Document, Graph) {
    let mut doc = Document::new();
    let profile = glyph(&mut doc);
    let spine = path(&mut doc, which);
    let mut graph = Graph::empty();
    graph.add(Node::Sweep { rims: Default::default(), outline: fieldform_scene::Outline::closed(profile), spine, section: Default::default(), law: Default::default() });
    (doc, graph)
}

fn piped() -> (Document, Graph) {
    let mut doc = Document::new();
    let spine = path(&mut doc, Path::Tight);
    let mut graph = Graph::empty();
    graph.add(Node::Pipe { curve: spine, radius: ParamRef::Literal(40.0) });
    (doc, graph)
}
