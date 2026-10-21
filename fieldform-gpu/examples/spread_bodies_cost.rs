// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a scene of bodies apart costs to trace, with and without the two exact skips.
//!
//! Every ray used to evaluate every body at every step, and started at the eye. On an assembly
//! — or a pattern, or any scene of parts apart — most of those evaluations were of bodies that
//! could not be the nearest, and the first several steps of every ray crossed empty space in
//! front of the model. Step 4 of `performance-plan-of-action.md` removes both exactly: a body
//! whose box lies no nearer than the running minimum is not evaluated, and the primary ray
//! starts where it enters the scene's sphere. This measures each on its own and both together,
//! on 2, 10, 100 and 500 spheres three radii apart, at the drag size and at 1080, and holds each
//! arm's frame pixel by pixel to one drawn without its skip: the guard bit for bit, the ray start
//! everywhere but on a ray grazing a sphere or within rounding of the same surface (`hold_to`
//! says why those are let through).
//!
//! `many_bodies.rs` measures what a body count costs to *compile*; this is what it costs to
//! trace, which is the number a frame is made of.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example spread_bodies_cost`

use fieldform_gpu::cpu_march::pixel_ray;
use fieldform_gpu::{
    generate_for_measurement, BakeSettings, Camera, EnvironmentMaps, Hit, Look, Renderer,
    ShadingResources, Studio,
};
use fieldform_scene::{Document, Graph, Material, Node, ParamRef, Place};
use nalgebra::{Point3, Vector3};

const RADIUS_MM: f64 = 10.0;
/// Three radii between centres, so the bodies are apart rather than touching.
const PITCH_MM: f64 = RADIUS_MM * 3.0;

/// `count` bodies on a cubic grid, and the box they all lie in. A body is a union of
/// `per_body` overlapping spheres — one is a single primitive, which costs about what the box
/// test does; eight is about what a short pipe does — so the guard is measured on both sides
/// of the weight the generator guards from (`GUARD_WEIGHT` in codegen.rs).
fn spheres(count: usize, per_body: usize) -> (Graph, (Point3<f64>, Point3<f64>)) {
    let side = (count as f64).cbrt().ceil() as usize;
    let mut graph = Graph::empty();
    let mut lo = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut hi = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for i in 0..count {
        let (x, y, z) = (i % side, (i / side) % side, i / (side * side));
        let c = Point3::new(x as f64 * PITCH_MM, y as f64 * PITCH_MM, z as f64 * PITCH_MM);
        let sphere = |graph: &mut Graph, at: Point3<f64>| {
            graph.add(Node::Sphere {
                centre: Place::At([
                    ParamRef::Literal(at.x),
                    ParamRef::Literal(at.y),
                    ParamRef::Literal(at.z),
                ]),
                radius: ParamRef::Literal(RADIUS_MM),
            })
        };
        if per_body > 1 {
            let parts: Vec<_> = (0..per_body)
                .map(|k| {
                    // Nudged inside the same ball, so the body's box is the one sphere's.
                    let a = k as f64 / per_body as f64 * std::f64::consts::TAU;
                    let nudge = RADIUS_MM * 0.001;
                    sphere(&mut graph, c + nalgebra::Vector3::new(a.cos(), a.sin(), 0.0) * nudge)
                })
                .collect();
            graph.add(Node::Boolean { op: fieldform_scene::BooleanOp::Union, inputs: parts.into_iter().map(Into::into).collect() });
        } else {
            sphere(&mut graph, c);
        }
        lo = Point3::new(
            lo.x.min(c.x - RADIUS_MM),
            lo.y.min(c.y - RADIUS_MM),
            lo.z.min(c.z - RADIUS_MM),
        );
        hi = Point3::new(
            hi.x.max(c.x + RADIUS_MM),
            hi.y.max(c.y + RADIUS_MM),
            hi.z.max(c.z + RADIUS_MM),
        );
    }
    (graph, (lo, hi))
}

/// How one arm's frame differs from the frame it is held to.
#[derive(Default)]
struct Against {
    /// Pixels that hit in one frame and missed in the other, or hit more than a millimetre
    /// apart, whose nearer hit lies in the tangency band.
    grazing: Vec<usize>,
    /// The largest `|n·dir|` among those, as a fraction of the band, and where.
    worst_grazing: Option<(f64, String)>,
    /// The largest gap between two hits on one surface, `|Δdepth|·|n·dir|` in epsilons, and where.
    worst_same_surface: Option<(f64, String)>,
    /// Every other difference: the pixel's index, and a line saying what differs there.
    faults: Vec<(usize, String)>,
}

/// Bit for bit, except that +0 and -0 are one number: a component that rounds to nought may
/// carry either sign without the surface being any different.
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a == 0.0 && b == 0.0)
}

/// Holds `arm`'s frame to `reference`'s pixel by pixel; `exact` arms bit for bit.
///
/// **Both skips are exact in arithmetic, and only one of them is exact in `f32`.** The box guard
/// leaves out a body that cannot be the nearest, so the fold returns the same number; this example
/// finds it bit-identical -- hit, depth, body and normal -- in every pixel of every frame, from
/// the eye against the plain march and from the sphere against the ray start alone, and holds it
/// to that. The normal is held as well as the depth because it comes from a second fold: the
/// generator emits a guarded copy of the union for `field_normal`, and a copy that ruled out a
/// different set from the value pass's would put another body's normal at the right depth, which
/// no comparison of depths sees.
///
/// The ray start changes nothing about the field, only where the samples fall along the ray: it
/// begins where the ray enters the scene's sphere, which on every pixel here lies at least 5.67 mm
/// before the first body. A ray that meets a surface squarely cannot tell the difference. One
/// that only grazes a sphere can, because the confirmation reads its rate of approach off two
/// readings (`confirm_distance`), and the ulp of a 10 mm distance in `f32` is about 1e-6 mm: near
/// tangency the rate stalls on rounding, and the march admits a graze up to about one epsilon
/// *outside* the sphere -- from one start and not the other.
///
/// The pixel that showed it, (943, 501) at two bodies and 1920×1080, grazes the sphere at the
/// origin 0.84 epsilon outside it. The plain march hits; the ray start misses; and the same march
/// in `f64` misses from both starts, as it does on every flip and every depth jump found. The hit
/// is the rounding, not the skip losing a surface.
///
/// **So a difference passes only where it can be that.** A hit against a miss, or two hits more
/// than a millimetre apart, must have its nearer hit inside the tangency band of a sphere of
/// radius R, `|n·dir| < 2·√(ε/R)`. A ray that tunnelled through a front face met that face
/// square-on and cannot hide in the band. Two hits within a millimetre are one surface, on one
/// body, and must agree to 3ε along its normal. Nothing passes at the same stop: a ray stopped at
/// the same depth on the same body in both frames is one point asked of one fold, so a normal that
/// differs there is a fault in any arm. Measured over this example's whole sweep, and printed
/// under its table: every grazing difference sits at most 0.701 of the way to the band's edge,
/// and no same-surface gap exceeds 2.024 ε -- both worst at 500 bodies and 1080.
///
/// **A hit count, which this used to compare, cannot say any of that.** Counts cancel: 500 bodies
/// at 720 matched with six pixels flipped, three each way. And up to forty pixels a frame hit in
/// both arms 17.4 to 207.5 mm apart -- one grazed a front sphere, the other passed it to the
/// sphere behind -- which no count sees at all.
///
/// **And the exemption was made to cross its own branch.** With the shader's ray start pushed
/// 10 mm past the sphere's entry, which puts it inside the nearer body on part of the frame, the
/// first row fails on 15164 pixels, the printed ones at `|n·dir|` above 0.5 and 10 to 320 ε along
/// the normal. Pushed 25 mm, which carries some rays past a body altogether, it fails on 20592;
/// the printed ones are hits lost near a silhouette at `|n·dir|` from 0.024, against a band of
/// 0.0107. Both counts are pixels, each counted once however many arms fail on it. The band lets
/// through what rounding does, and not what a skip that loses a surface does.
fn hold_to(
    (held_to, reference): (&str, &[Hit]),
    (name, arm): (&str, &[Hit]),
    camera: &Camera,
    frame: (u32, u32),
    exact: bool,
) -> Against {
    let mut out = Against::default();
    let band = 2.0 * (camera.epsilon / RADIUS_MM).sqrt();
    let describe = |h: &Hit| {
        if h.is_hit() {
            format!("{:.4} mm on body {}", h.depth, h.body)
        } else {
            "miss".to_string()
        }
    };
    for (i, (r, a)) in reference.iter().zip(arm).enumerate() {
        let same_stop = same(r.depth, a.depth) && r.body == a.body;
        let same_normal = (0..3).all(|k| same(r.normal[k], a.normal[k]));
        if same_stop && same_normal {
            continue;
        }
        let (x, y) = (i as u32 % frame.0, i as u32 / frame.0);
        let (_, dir) = pixel_ray(camera, frame, x, y);
        let dir = Vector3::new(dir.x as f64, dir.y as f64, dir.z as f64);
        // A normal that will not normalize counts as square-on, so it fails rather than hides.
        let facing = |h: &Hit| h.normal.try_normalize(1e-12).map_or(1.0, |m| m.dot(&dir).abs());
        let near = match (r.is_hit(), a.is_hit()) {
            (true, true) => Some(if r.depth <= a.depth { r } else { a }),
            (true, false) => Some(r),
            (false, true) => Some(a),
            (false, false) => None,
        };
        let mut line = format!(
            "({x}, {y}): {held_to} {}, {name} {}, nearer |n·dir| {}, band {band:.5}",
            describe(r),
            describe(a),
            near.map_or("-".to_string(), |h| format!("{:.5}", facing(h))),
        );
        // The normals only where they are what differs: anywhere else the stops differ too, and
        // two stops carry two normals however right both are.
        if same_stop {
            let n = |h: &Hit| format!("({:.7}, {:.7}, {:.7})", h.normal.x, h.normal.y, h.normal.z);
            line.push_str(&format!(", normal {} against {}", n(r), n(a)));
        }
        // A ray stopped at the same depth on the same body stopped at the same point, and every
        // arm asks the same fold for the normal there, guarded or not -- so a normal that differs
        // at the same stop is a fault in any arm, not a grazing ray.
        let Some(near) = near.filter(|_| !exact && !same_stop) else {
            out.faults.push((i, line));
            continue;
        };
        let gap = (r.depth - a.depth).abs();
        if r.is_hit() && a.is_hit() && gap <= 1.0 {
            let along = gap * facing(near) / camera.epsilon;
            if out.worst_same_surface.as_ref().is_none_or(|(w, _)| along > *w) {
                out.worst_same_surface = Some((along, format!("({x}, {y}) in {name}")));
            }
            // Written to fail on a NaN: `along > 3.0` would wave one through.
            if !(along <= 3.0) || r.body != a.body {
                out.faults.push((i, format!("{line}, {along:.2} ε along the normal")));
            }
        } else {
            let ratio = facing(near) / band;
            if out.worst_grazing.as_ref().is_none_or(|(w, _)| ratio > *w) {
                out.worst_grazing = Some((ratio, format!("({x}, {y}) in {name}")));
            }
            if ratio < 1.0 {
                out.grazing.push(i);
            } else {
                out.faults.push((i, line));
            }
        }
    }
    out
}

fn main() {
    let doc = Document::new();
    let maps = EnvironmentMaps::bake(&Studio::default(), BakeSettings::default());
    let mut shared: Option<(Renderer, ShadingResources)> = None;

    // Each arm after the first is held to an earlier one, and held bit for bit wherever the two
    // start their rays alike -- see `hold_to`. Both skips are held to the ray start alone rather
    // than to the plain march, which is what holds the guard exactly under either start: against
    // the plain march, both would differ from it only as the ray start does, and be let through
    // on the same grazing rays.
    const ARMS: [(&str, bool, bool, Option<usize>); 4] = [
        ("neither", false, false, None),
        ("box guard", true, false, Some(0)),
        ("ray start", false, true, Some(0)),
        ("both", true, true, Some(2)),
    ];
    println!("| bodies | frame | neither | box guard | ray start | both | hits, arms held |");
    println!("|---|---|---|---|---|---|---|");
    let mut worst_grazing: Option<(f64, String)> = None;
    let mut worst_same_surface: Option<(f64, String)> = None;
    // No scene here may take a submission near two seconds unguarded: a hundred dear bodies at
    // 1080 did, and the driver was reset -- the loop spec's S2 on the laptop, from an example
    // that traces a frame whole. The application bands; this keeps to what is safe whole.
    for (count, per_body, both_sizes) in [
        (2usize, 1usize, true),
        (10, 1, true),
        (100, 1, true),
        (500, 1, true),
        (10, 4, true),
        (100, 4, false),
        (10, 8, true),
        (100, 8, false),
    ] {
        let (graph, bounds) = spheres(count, per_body);
        let count =
            if per_body > 1 { format!("{count} × {per_body} spheres") } else { count.to_string() };
        let sizes: &[(u32, u32)] =
            if both_sizes { &[(720, 405), (1920, 1080)] } else { &[(720, 405)] };
        let programs = [
            // The cull on in both arms, so what moves between them is the guard alone.
            generate_for_measurement(&doc, &graph, false, true).expect("generates"),
            generate_for_measurement(&doc, &graph, true, true).expect("generates"),
        ];
        for &(width, height) in sizes {
            let mut row = format!("| {count} | {width}×{height} |");
            let mut frames: Vec<Vec<Hit>> = Vec::new();
            let mut grazing: Vec<usize> = Vec::new();
            let mut faults: Vec<(usize, String)> = Vec::new();
            for (name, guarded, start_at_scene, held_to) in ARMS {
                let program = &programs[usize::from(guarded)];
                let renderer = match &shared {
                    None => match pollster::block_on(Renderer::new(program, &[], &[], &[], 32, 1.0))
                    {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!("no GPU available ({e}); nothing else in the workspace needs one.");
                            return;
                        }
                    },
                    Some((first, _)) => Renderer::with_device(
                        first.device().clone(),
                        first.queue().clone(),
                        program,
                        &[],
                        &[],
                        &[],
                    )
                    .expect("the renderer builds on the shared device"),
                };
                let values: Vec<f32> = program
                    .params
                    .iter()
                    .map(|p| match p {
                        ParamRef::Literal(v) => *v as f32,
                        _ => unreachable!("literals only"),
                    })
                    .collect();
                renderer.set_params(&values);
                if shared.is_none() {
                    let shading = ShadingResources::new(renderer.device(), renderer.queue(), &maps)
                        .expect("the lighting pipeline builds");
                    shared = Some((
                        Renderer::with_device(
                            renderer.device().clone(),
                            renderer.queue().clone(),
                            program,
                            &[],
                            &[],
                            &[],
                        )
                        .expect("the renderer builds"),
                        shading,
                    ));
                }
                let shading = &mut shared.as_mut().expect("set above").1;

                let mut camera = Camera::framing(bounds, width as f64 / height as f64);
                camera.start_at_scene = start_at_scene;
                let look = Look::studio(Material::default(), camera.forward, camera.right, camera.up);
                let mut samples = Vec::new();
                for i in 0..6 {
                    let started = std::time::Instant::now();
                    renderer
                        .present(shading, &camera, width, height, 512, &look, 0)
                        .expect("the frame presents");
                    renderer
                        .device()
                        .poll(wgpu::PollType::wait_indefinitely())
                        .expect("the GPU finishes");
                    if i > 0 {
                        samples.push(started.elapsed().as_secs_f64() * 1e3);
                    }
                    let _ = shading.costs();
                }
                samples.sort_by(f64::total_cmp);
                row.push_str(&format!(" {:.1} ms |", samples[samples.len() / 2]));
                let frame =
                    renderer.render(&camera, width, height, 512).expect("the frame renders");
                let Some(held_to) = held_to else {
                    frames.push(frame);
                    continue;
                };
                let (reference, _, reference_starts_at_scene, _) = ARMS[held_to];
                let against = hold_to(
                    (reference, &frames[held_to]),
                    (name, &frame),
                    &camera,
                    (width, height),
                    start_at_scene == reference_starts_at_scene,
                );
                frames.push(frame);
                let at = |(w, pixel): (f64, String)| {
                    (w, format!("{pixel}, {count} at {width}×{height}"))
                };
                for (worst, this) in [
                    (&mut worst_grazing, against.worst_grazing.map(at)),
                    (&mut worst_same_surface, against.worst_same_surface.map(at)),
                ] {
                    if let Some(this) = this {
                        if worst.as_ref().is_none_or(|(w, _)| this.0 > *w) {
                            *worst = Some(this);
                        }
                    }
                }
                grazing.extend(against.grazing);
                faults.extend(against.faults);
            }
            // Pixels, not lines: a pixel two arms both fail on is one pixel of the frame.
            let mut failing: Vec<usize> = faults.iter().map(|(i, _)| *i).collect();
            for pixels in [&mut failing, &mut grazing] {
                pixels.sort_unstable();
                pixels.dedup();
            }
            let hits = frames[0].iter().filter(|h| h.is_hit()).count();
            if faults.is_empty() {
                row.push_str(&format!(" {hits} ✓ ({} grazing) |", grazing.len()));
            } else {
                row.push_str(&format!(" {hits} FAILS ({} pixels) |", failing.len()));
            }
            println!("{row}");
            for (_, fault) in faults.iter().take(20) {
                println!("    {fault}");
            }
            assert!(
                faults.is_empty(),
                "{} pixels differ from the march they are held to by more than a grazing ray or \
                 rounding can",
                failing.len()
            );
        }
    }
    let show = |worst: &Option<(f64, String)>| match worst {
        Some((w, at)) => format!("{w:.3}, at {at}"),
        None => "no difference anywhere".to_string(),
    };
    println!();
    println!(
        "Worst grazing difference, |n·dir| over the band 2·√(ε/R): {}",
        show(&worst_grazing)
    );
    println!("Worst same-surface gap, |Δdepth|·|n·dir| in ε: {}", show(&worst_same_surface));
}
