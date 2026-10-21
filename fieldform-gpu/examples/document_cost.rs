// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a saved document costs the viewport: the shader's compile, a draft frame and a still-sized
//! frame, with every loft in it described -- its sections' resampled counts, stations, tilts,
//! stretch and bend -- so a report of "it is very slow" can be measured on the very scene it was
//! made about rather than on a scene built to resemble it.
//!
//! Run with: `cargo run -p fieldform-gpu --example document_cost -- path/to/file.fform`
//!
//! Loads the document as the application does (`fieldform_io::open`), takes each part's graph as
//! it stands, and supplies the shader's bindings the way `traced/rebuild.rs` does: spines and
//! frames, a loft's own, and since 2026-10-13 the profiles of extrudes, sweeps and revolves and
//! the walls and corners of thin ones, without which a sweep of a closed profile -- the usual
//! kind -- could not be measured here at all. Numbers are the least of three runs of twenty
//! frames.
//!
//! **The compile is timed cold, and can be timed with a piece cut out.** Every run salts the
//! march's one field reading so the driver cannot answer from its cache (`FIELDFORM_SALT=<n>`
//! to choose the salt); `FIELDFORM_VARIANT=<name>` cuts a named piece out of the source before
//! the build, so its share of the compile is the difference against a plain run -- the list is
//! at the `match` below; `FIELDFORM_DUMP_WGSL=<path>` writes the source as compiled. The still's
//! own `occlude` entry is compiled and timed as well, since `Renderer::new` does not make it.
//! This is how four partial revolves' 106 s was laid at their normal's door
//! (`round-chamfer-spec.md` §4.4).

use fieldform_gpu::{generate, Camera, Renderer};
use fieldform_scene::{Node, ParamRef};
use nalgebra::{Point3, Vector3};

/// The viewport's own numbers, from `traced/mod.rs`.
const STEP_BUDGET: f64 = 512.0;
const STEP_HEADROOM: f64 = 2.0;
const MAX_STEP_BUDGET: u32 = 16_384;
/// What a profile is flattened and baked to, as `sweep_cost` reads one.
const PROFILE_TOLERANCE_MM: f64 = 0.5;

fn main() {
    let path = std::env::args().nth(1).expect("a .fform path");
    let (doc, _archive) = fieldform_io::open(std::path::Path::new(&path)).expect("the document opens");
    for part in doc.parts.values() {
        // **Read through the rewrite the shader reads first** (backlog item 113): a joined
        // mirror of a section touching its plane is one extrude there, and its section is a
        // curve only the rewritten document holds -- as `TracedView::rebuild` resolves it.
        let merged = fieldform_field::with_shared_faces_merged(&doc, &part.graph);
        let (doc, graph) = match &merged {
            Some((merged_doc, merged_graph)) => (merged_doc, merged_graph),
            None => (&doc, &part.graph),
        };
        println!("=== part {:?}: {} nodes ===", part.name, graph.nodes().count());
        // What each curve costs to walk at the profile's turn, which every reader of a loft or
        // a sweep pays per section curve per build.
        let mut walk_total = 0.0;
        for curve in doc.curves.values() {
            let floor = (curve.length(0.5) * 1e-5).max(1e-4);
            let timing = std::time::Instant::now();
            let points = curve.tessellate_to_angle(fieldform_field::PROFILE_TURN_DEGREES, floor);
            let ms = timing.elapsed().as_secs_f64() * 1000.0;
            walk_total += ms;
            println!("  curve {:?}: degree {}, {} control points -> {} points in {ms:.2} ms", curve.name, curve.degree(), curve.control_points().len(), points.len());
        }
        println!("  all curves walked: {walk_total:.1} ms");
        // And what each loft section's loop walk costs, which samples more than its own curves.
        for (_, node) in graph.nodes() {
            let Node::Loft { sections, spine, .. } = node else { continue };
            for (k, section) in sections.iter().enumerate() {
                let timing = std::time::Instant::now();
                let found = fieldform_field::profile_loop_turning(
                    &doc,
                    section.curve(),
                    fieldform_field::PROFILE_TURN_DEGREES,
                    *spine,
                );
                let ms = timing.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "  section {k}: loop of {} curves, {} points, walked in {ms:.1} ms",
                    found.as_ref().map(|f| f.curves.len()).unwrap_or(0),
                    found.as_ref().map(|f| f.points.len()).unwrap_or(0)
                );
            }
        }

        // Every loft, described as the panel describes it.
        for (id, node) in graph.nodes() {
            let Node::Loft { rims: _, sections, spine, smooth } = node else { continue };
            // The build in its two halves: walking the sections' curves into polylines and
            // deriving the centerline (`loft_parts`), then the correspondence, the frame and
            // the bound (`build_loft`) -- the second measured on a build the memo has not seen.
            let walking = std::time::Instant::now();
            let parts = fieldform_field::loft_parts(
                &doc,
                sections,
                *spine,
                fieldform_field::PROFILE_TURN_DEGREES,
                fieldform_field::DEFAULT_TOLERANCE_MM,
            );
            let walk_ms = walking.elapsed().as_secs_f64() * 1000.0;
            let building = std::time::Instant::now();
            let built = fieldform_field::loft_built(&doc, sections, *spine, *smooth);
            let build_ms = building.elapsed().as_secs_f64() * 1000.0;
            let again = std::time::Instant::now();
            let _ = fieldform_field::loft_built(&doc, sections, *spine, *smooth);
            let again_ms = again.elapsed().as_secs_f64() * 1000.0;
            println!(
                "  walking the sections and deriving the centerline {walk_ms:.1} ms; the whole build                  {build_ms:.1} ms (the walk again, then the correspondence, frame and bound); asked a                  second time, memo in hand, {again_ms:.1} ms{}",
                parts.map(|_| String::new()).unwrap_or_else(|e| format!(" -- parts failed: {e}"))
            );
            match built {
                Some(l) => {
                    let counts: Vec<usize> = (0..l.section_count()).map(|k| l.section_points(k).len()).collect();
                    let drawn: Vec<usize> = sections
                        .iter()
                        .map(|o| {
                            fieldform_field::profile_loop_turning(
                                &doc,
                                o.curve(),
                                fieldform_field::PROFILE_TURN_DEGREES,
                                *spine,
                            )
                            .map(|found| found.points.len())
                            .unwrap_or(0)
                        })
                        .collect();
                    println!(
                        "loft {id:?}: {:?}, centerline {} ({} samples, {:.1} mm), sections drawn as {:?} points, \
                         resampled to {:?}; stations {:?}; tilts {:?}; stretch {:.3}, bend {:.3} of the fold, \
                         overshoot {:.2} mm; built in {build_ms:.1} ms",
                        smooth,
                        if spine.is_some() { "named" } else { "derived" },
                        l.spine.points().len(),
                        l.spine.length(),
                        drawn,
                        counts,
                        l.stations().iter().map(|a| (a * 10.0).round() / 10.0).collect::<Vec<_>>(),
                        l.tilts().iter().map(|t| (t * 10.0).round() / 10.0).collect::<Vec<_>>(),
                        l.stretch(),
                        l.bend_share(),
                        l.overshoot(),
                    );
                }
                None => println!("loft {id:?} cannot be built"),
            }
        }

        let generating = std::time::Instant::now();
        let mut program = match generate(&doc, graph) {
            Ok(p) => p,
            Err(e) => {
                println!("the part does not generate: {e}");
                continue;
            }
        };
        let gen_ms = generating.elapsed().as_secs_f64() * 1000.0;
        let compiling_cpu = std::time::Instant::now();
        let cpu = fieldform_field::GraphField::compile(&doc, part, 0.25);
        let cpu_ms = compiling_cpu.elapsed().as_secs_f64() * 1000.0;
        if let Err(e) = &cpu {
            println!("the part does not compile on the CPU: {e}");
        }
        let extent = match fieldform_field::part_extent(&doc, graph) {
            fieldform_field::Extent::Bounded { lo, hi } => (lo, hi),
            _ => {
                println!("the part is unbounded; nothing to frame");
                continue;
            }
        };
        let size = (extent.1 - extent.0).norm();

        let loft_of = |node: fieldform_scene::NodeId| {
            let Some(Node::Loft { rims: _, sections, spine, smooth }) = graph.node(node) else {
                unreachable!("a loft binding belongs to a loft");
            };
            fieldform_field::loft_built(&doc, sections, *spine, *smooth).expect("builds")
        };
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
                        (
                            fieldform_field::sweep_frame(&doc, outline, *spine, &path)
                                .expect("a spine with a direction"),
                            None,
                        )
                    }
                    Some(Node::Loft { .. }) => (
                        loft_of(node).frame_normals().iter().map(|n| Point3::from(*n)).collect(),
                        None,
                    ),
                    _ => unreachable!("a frame belongs to a sweep or a loft"),
                },
                fieldform_gpu::CurveUse::LoftSpine(node) => (loft_of(node).spine.points().to_vec(), None),
                fieldform_gpu::CurveUse::LoftSlope(node, k) => (
                    loft_of(node).section_slopes(k).iter().map(|m| Point3::new(m.x, m.y, 0.0)).collect(),
                    None,
                ),
                fieldform_gpu::CurveUse::LoftSection(node, k) => (
                    loft_of(node).section_points(k).iter().map(|q| Point3::new(q.x, q.y, 0.0)).collect(),
                    None,
                ),
                // A closed section, read as the application reads it (`profile_points` in
                // `traced/rebuild.rs`): the loop walked from the block's curve, flattened about
                // the point a sweep's path carries it by or into an extrude's fitted plane, and
                // baked to the display tolerance. Without it a sweep of a closed profile, the
                // usual kind, could not be measured here at all.
                fieldform_gpu::CurveUse::Profile(node) => {
                    let (curve, ignore) = match graph.node(node) {
                        Some(Node::Extrude { outline, sweep, .. }) => (outline.curve(), sweep.source()),
                        Some(Node::Sweep { outline, spine, .. }) => (outline.curve(), Some(*spine)),
                        Some(Node::Revolve { outline, frame, .. }) => (outline.curve(), frame.curve_source()),
                        _ => unimplemented!("this example reads the profiles of extrudes, sweeps and revolves"),
                    };
                    let turn = fieldform_field::PROFILE_TURN_DEGREES;
                    let world = fieldform_field::profile_loop_turning(&doc, curve, turn, ignore)
                        .expect("a closed loop")
                        .points;
                    let numbers = |r: &ParamRef| r.resolve(&|id| doc.parameter_value(id));
                    let (origin, rotation) = match graph.node(node) {
                        Some(Node::Sweep { outline, spine, .. }) => {
                            let laid = fieldform_field::sweep_outline_section(&doc, outline, *spine, turn)
                                .expect("the section lays out");
                            (laid.origin.coords, fieldform_field::rotation_from_z(laid.normal))
                        }
                        Some(Node::Revolve { frame, .. }) => {
                            fieldform_field::resolve_frame(&doc, graph, frame, &numbers).expect("an axis")
                        }
                        _ => {
                            let (origin, normal) = fieldform_field::plane_of(&world).expect("planar");
                            (origin.coords, fieldform_field::rotation_from_z(normal))
                        }
                    };
                    let into = move |p: Point3<f64>| {
                        Point3::from(rotation.inverse_transform_vector(&(p.coords - origin)))
                    };
                    let profile = match graph.node(node) {
                        Some(Node::Revolve { .. }) => fieldform_field::Profile::about_axis(&world, into),
                        _ => fieldform_field::Profile::in_plane(&world, into, PROFILE_TOLERANCE_MM),
                    }
                    .expect("planar");
                    let points: Vec<Point3<f64>> =
                        profile.points().iter().map(|p| Point3::new(p.x, p.y, 0.0)).collect();
                    let reads = fieldform_field::profile_reads_out_to(&doc, graph, node);
                    let (lo, hi) = profile.bounds();
                    let bake = fieldform_field::bake_cells((hi - lo).max(), PROFILE_TOLERANCE_MM, reads)
                        .map(|cells| profile.baked(cells, reads))
                        .and_then(|p| p.bake().cloned());
                    (points, bake)
                }
                // A wall's three polylines and a mitred draft's corners, from the kernel's own
                // readings, as the application supplies them.
                fieldform_gpu::CurveUse::Wall(node, part) => {
                    let built = fieldform_field::outline_wall_baked(
                        &doc,
                        graph,
                        node,
                        fieldform_field::PROFILE_TURN_DEGREES,
                        PROFILE_TOLERANCE_MM,
                        0.0,
                    )
                    .expect("the wall builds");
                    let grid = match part {
                        fieldform_field::WallPart::Chain => built.bake().cloned(),
                        _ => None,
                    };
                    (built.packed(part), grid)
                }
                fieldform_gpu::CurveUse::Miters(node) => (
                    fieldform_field::outline_miters(&doc, graph, node, fieldform_field::PROFILE_TURN_DEGREES)
                        .expect("the corners read"),
                    None,
                ),
                ref other => unimplemented!("this example supplies lofts, sweeps and profiles, not {other:?}"),
            })
            .collect();

        // **Where a shader's compile time goes, by cutting a piece out and timing what is left.**
        // Four partial revolves took the driver 106 s where one took 4 s (spec
        // `round-chamfer-spec.md` §4.4, 2026-10-18), and no profiler reads a driver's optimizer;
        // `FIELDFORM_VARIANT` names a piece of the source to cut before the build, so its share is
        // the difference: `no_outline` reads the bake where the march would read the outline,
        // `no_wedge_normal` returns a constant for a wedge's differenced normal, `no_normal` for
        // every normal, `no_wedge_outside` reads the wedge's own section where the hard caps would
        // land, `no_rim` makes every rim read the hard corner, `no_shadow` and `no_occlusion` empty
        // the still's two loops. `FIELDFORM_DUMP_WGSL=<path>` writes the source as compiled.
        //
        // **Every compile here is cold, and that is arranged.** The driver keeps a cache keyed on
        // an entry point's compiled bytes, and answers from it in under a second where the cold
        // compile took 106 s -- which read as two variants costing nothing when what they had
        // changed was the still's loops, not the march. So the march's one field reading is
        // salted with a number that is new each run (`FIELDFORM_SALT` to choose it): a real
        // addition the optimizer cannot fold, of a size no reading can notice, and different bytes
        // every time. The salt is the first cut, so a variant's text is matched after it.
        let variant = std::env::var("FIELDFORM_VARIANT").unwrap_or_default();
        let salt = std::env::var("FIELDFORM_SALT").ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos() % 1_000_000)
                .unwrap_or(1)
                .max(1)
        });
        {
            let salted = format!("    let d = field(p) + {:.1}e-30;", salt as f64);
            let n = program.source.matches("    let d = field(p);").count();
            assert_eq!(n, 1, "the march reads the field once in `field_here`");
            program.source = program.source.replace("    let d = field(p);", &salted);
            println!("  salt {salt}");
            let mut cut = |from: &str, to: &str| {
                let n = program.source.matches(from).count();
                program.source = program.source.replace(from, to);
                println!("  variant {variant}: {n} x {from:?}");
            };
            let stub = |signature: &str, body: &str| -> (String, String) {
                let (name, rest) = signature.split_once('(').expect("a signature");
                (signature.to_string(), format!("{signature} {body} }}\n{name}_unused({rest}"))
            };
            match variant.as_str() {
                "no_outline" => cut("if (exact_reads) {", "if (false) {"),
                "no_wedge_normal" => {
                    let (from, to) = stub(
                        "fn revolve_wedge_rim_normal(index: u32, p: vec3<f32>, begins: f32, sweep: f32, rim: vec4<f32>, euler: vec3<f32>) -> vec3<f32> {",
                        "return vec3<f32>(0.0, 0.0, 1.0);",
                    );
                    cut(&from, &to);
                }
                "no_normal" => {
                    let (from, to) = stub("fn field_normal(p: vec3<f32>) -> vec3<f32> {", "return vec3<f32>(0.0, 0.0, 1.0);");
                    cut(&from, &to);
                }
                "no_wedge_outside" => {
                    cut("cap_distance(profile_distance(index, vec2<f32>(r * cos(psi_a), p.z)), r * sin(psi_a))", "cap_distance(x, r * sin(psi_a))");
                    cut("cap_distance(profile_distance(index, vec2<f32>(r * cos(psi_b), p.z)), r * sin(psi_b))", "cap_distance(x, r * sin(psi_b))");
                }
                "no_rim" => cut("fn rim_join(x: f32, y: f32, rim: vec4<f32>) -> f32 {", "fn rim_join(x: f32, y: f32, rim: vec4<f32>) -> f32 {\n    return max(x, y);"),
                // The wedge normal's six-count made one the driver cannot know, to ask whether
                // it unrolled the literal: a difference here is the unrolling's cost.
                "opaque_normal_bound" => {
                    let body = program.source.find("n[axis] += weight * revolve_wedge_rim_distance").expect("the wedge normal's loop");
                    let bound = program.source[..body].rfind("k < 6u;").expect("its literal bound");
                    program.source.replace_range(bound..bound + 7, "k < 6u + u32(sweep > 1e30);");
                    println!("  variant {variant}: the wedge normal's bound made opaque");
                }
                "no_shadow" => cut("for (var i = 0u; i < 48u; i = i + 1u) {", "for (var i = 0u; i < 0u; i = i + 1u) {"),
                "no_occlusion" => cut("for (var i = 1u; i <= 5u; i = i + 1u) {", "for (var i = 1u; i <= 0u; i = i + 1u) {"),
                "" => {}
                other => panic!("no variant named {other}"),
            }
        }
        if let Some(path) = std::env::var_os("FIELDFORM_DUMP_WGSL") {
            std::fs::write(&path, &program.source).expect("the dump path is writable");
            println!("  wrote {}", path.to_string_lossy());
        }
        let compiling = std::time::Instant::now();
        let renderer = match pollster::block_on(Renderer::new(&program, &[], &curves, &[], 64, size * 0.15)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("no GPU available ({e})");
                return;
            }
        };
        let compile_ms = compiling.elapsed().as_secs_f64() * 1000.0;
        // The still's own entry, which the application compiles only when a still first asks
        // for it, so `Renderer::new` above has not timed it.
        let (module_ms, occlude_ms) = renderer.time_occlude_pipeline(&|s| s.to_string());
        println!(
            "  occlude pipeline: module {module_ms:.0} ms, pipeline {occlude_ms:.0} ms{}",
            if variant.is_empty() { String::new() } else { format!(" ({variant})") }
        );
        let values: Vec<f32> = program
            .params
            .iter()
            .map(|p| match p {
                ParamRef::Literal(v) => *v as f32,
                ParamRef::Named(id) => doc.parameter_value(*id).unwrap_or(0.0) as f32,
                ParamRef::Field(_) => 0.0,
            })
            .collect();
        renderer.set_params(&values);

        let multiplier = fieldform_field::step_multiplier(&doc, graph);
        let budget = ((STEP_BUDGET * multiplier * STEP_HEADROOM).ceil() as u32)
            .clamp(STEP_BUDGET as u32, MAX_STEP_BUDGET);
        let centre = nalgebra::center(&extent.0, &extent.1);
        let radius = size * 0.5;
        println!(
            "shader: {} curves, {} params, {} lines; generate {gen_ms:.1} ms, CPU compile {cpu_ms:.1} ms, \
             pipeline {compile_ms:.0} ms; step multiplier {multiplier:.2}, budget {budget}",
            program.curves.len(),
            program.params.len(),
            program.source.lines().count(),
        );
        // **Every rung of the draft ladder, because the question P10 asks is about the floor.**
        // The viewport shrinks a draft until it fits `DRAFT_BUDGET_MS` (50 ms, a fifth of a
        // second at four frames); the ladder's last rung is 240 px, and a document whose draft
        // misses the budget *there* is one the interface cannot outrun -- every frame queues a
        // march the swapchain's present waits for, which is the "sluggish" of the loop spec's
        // §1 row 4. So the row to read in this table is the 240 one.
        for (label, w, h) in [
            ("draft 720", 720u32, 720u32),
            ("draft 480", 480, 480),
            ("draft 360", 360, 360),
            ("draft 240 (ladder floor)", 240, 240),
            ("still 1600x900", 1600, 900),
        ] {
            let eye = centre + Vector3::new(1.6, 1.2, 0.9) * radius;
            let mut camera =
                Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), 45.0, w as f64 / h as f64);
            camera.value_scale = fieldform_field::value_scale_with(&doc, graph, &[]);
            let _ = renderer.render(&camera, w, h, budget);
            let mut ms = f64::INFINITY;
            for _ in 0..3 {
                let start = std::time::Instant::now();
                for _ in 0..20 {
                    renderer.render(&camera, w, h, budget).expect("renders");
                }
                ms = ms.min(start.elapsed().as_secs_f64() * 1000.0 / 20.0);
            }
            // Against the two budgets that matter: a draft's own 50 ms, and the 16 ms an
            // interface frame has if it is to keep 60 a second while a march is queued ahead of
            // its present.
            let against = if ms > 50.0 {
                " -- over the 50 ms draft budget"
            } else if ms > 16.0 {
                " -- inside the draft budget, over a 16 ms interface frame"
            } else {
                ""
            };
            println!(
                "{label}: {ms:.1} ms/frame ({:.1} us per 1000 steps){against}",
                ms / budget as f64 * 1000.0
            );
        }
    }
}
