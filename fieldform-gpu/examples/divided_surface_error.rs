// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How far from its true surface each body is drawn, **in millimetres along the ray** — the
//! number `gpu_matches_cpu` cannot give for a divided body.
//!
//! `gpu_matches_cpu` scores a hit by what the CPU field *reads* there. For a body whose value is
//! divided to stay a lower bound, that reading is the distance over the divisor, so a hit drawn
//! 0.1 mm proud of a sweep divided by 83 reads 0.0012 and passes. Here the true surface is found
//! by its sign, which a divisor cannot change: from each GPU hit the CPU field is walked along the
//! ray to where it changes sign, and the distance walked is the error.
//!
//! Reported by the owner as a step between two sweeps leaving one profile in opposite directions
//! (`swweeeeeping.fform`): the shape is seamless, and the forward sweep, tapering to 0.01, is
//! divided by 82.9 where the backward one is divided by 1.18.
//!
//! `cargo run --release -p fieldform-gpu --example divided_surface_error -- <doc.fform> <node>...`
//! with the bodies to draw named by id; everything else in the part is left out. `SEAM_X`,
//! `SEAM_Y` and `SEAM_Z` aim the views (the owner's file: `SEAM_Y=-25.408 SEAM_Z=6`), and `TRACE=1`
//! follows a sample of the worst rays through `cpu_march`, the march's replica, step by step.
//! `BAKE=1` reads the profiles from a grid as the viewport does at `CELLS`, `EXACT=1` finishes the
//! rays on the outline as a still does, `WHOLE=1` adds a view of the whole part at `W` by `H`
//! pixels, `STRAYS=1` lists the hits the exact field reads as off any surface, and `NOWALK=1`
//! skips the walk, which is minutes of CPU on a large frame. `SHADE=<stem>` writes each view,
//! strays red and normals facing away blue.
//!
//! The strays are how the owner's specks were found (item 115): with the finish on, 261 hits in
//! one view sat on the bake one margin proud along the top knife, every one taken by the march's
//! "gone through what it refused" path, which did not flip to the outline. It does now, and the
//! count is nought in every view.
//!
//! Measured on the owner's file, 99th percentile of the error, forward sweep / backward sweep:
//!
//! | view | before `settle` | after |
//! |---|---|---|
//! | side, level | 0.2031 / 0.0029 mm | 0.0004 / 0.0028 mm |
//! | side, from below | 0.5127 / 0.0053 | 0.0013 / 0.0028 |
//! | from above | 0.2071 / 0.0029 | 0.0011 / 0.0028 |
//! | along the side, grazing | 2.3284 / 0.0950 | 1.0144 / 0.0004 |
//!
//! Along the ray, so the grazing row is a sliver square to the surface. The backward sweep's
//! 0.0028 is two thresholds, where a candidate that near stands as an exact body's does; so the
//! step at the seam went from 0.2 to 0.5 mm down to about 0.003. No hit lands more than 0.0005 mm
//! inside.

use fieldform_field::{GraphField, ScalarField};
use fieldform_gpu::{Camera, CurveUse, Renderer};
use fieldform_scene::{Node, NodeId};
use nalgebra::{Point3, Vector3};

/// What the spines are flattened to and the CPU field compiled at, so both read one path.
const TOLERANCE_MM: f64 = 0.01;
const WIDTH_DEFAULT: u32 = 480;
const HEIGHT_DEFAULT: u32 = 320;

/// A hit whose exact reading is further than this from nought is a stray: drawn where no
/// surface is, which `along_ray_error` cannot see when the surface is more than its reach away.
const STRAY_MM: f64 = 0.05;

/// The signed distance along `dir` from `at` to where the CPU field changes sign: positive when
/// the hit stopped short of the surface, negative when it went past it.
fn along_ray_error(cpu: &GraphField, at: Point3<f64>, dir: Vector3<f64>) -> Option<f64> {
    const STEP: f64 = 0.02;
    const REACH: usize = 250;
    let outside = cpu.distance(at) > 0.0;
    let way = if outside { 1.0 } else { -1.0 };
    let mut prev = 0.0;
    for i in 1..=REACH {
        let s = way * STEP * i as f64;
        if (cpu.distance(at + dir * s) > 0.0) != outside {
            let (mut lo, mut hi) = (prev, s);
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if (cpu.distance(at + dir * mid) > 0.0) == outside { lo = mid } else { hi = mid }
            }
            return Some(0.5 * (lo + hi));
        }
        prev = s;
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let width: u32 = std::env::var("W").map_or(WIDTH_DEFAULT, |v| v.parse().unwrap());
    let height: u32 = std::env::var("H").map_or(HEIGHT_DEFAULT, |v| v.parse().unwrap());
    let (doc, _archive) = fieldform_io::open(std::path::Path::new(&args[1])).expect("the document opens");
    let keep: Vec<NodeId> =
        args[2..].iter().map(|s| NodeId::from_raw(u64::from_str_radix(s, 16).expect("a node id in hex"))).collect();
    let (_, part) = doc.parts.iter().next().expect("a part");
    let mut graph = part.graph.clone();
    loop {
        let gone: Vec<NodeId> = graph
            .nodes()
            .filter(|(id, n)| n.is_body() && !keep.contains(id) && !graph.nodes().any(|(_, m)| m.inputs().contains(id)))
            .map(|(id, _)| id)
            .collect();
        if gone.is_empty() {
            break;
        }
        for id in gone {
            graph.remove(id);
        }
    }
    let cpu = GraphField::compile_graph(&doc, &graph, TOLERANCE_MM).expect("compiles");
    let program = fieldform_gpu::generate_with(&doc, &graph, cpu.morph_divisors()).expect("generates");
    let curves: Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> = program
        .curves
        .iter()
        .map(|use_of| match use_of {
            // A sweep's section flattened about the point its path carries it by, as the compile
            // reads it -- `gpu_matches_cpu`'s arm, unbaked, so the shader walks the polyline.
            CurveUse::Profile(node) => {
                let Some(Node::Sweep { outline, spine, .. }) = graph.node(*node) else {
                    panic!("only sweeps are bound here")
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
                let points = flat.points().iter().map(|p| Point3::new(p.x, p.y, 0.0)).collect();
                // `BAKE=1` answers the profile from a grid as the viewport does at `CELLS` (the
                // quality's resolution: 64 Fast, 128 Balanced, 192 Full), with the viewport's own
                // tolerance and band -- so what is measured includes the bake's margin, which the
                // polyline walk above has none of.
                if std::env::var_os("BAKE").is_none() {
                    return (points, None);
                }
                let cells_across: f64 = std::env::var("CELLS").map_or(192.0, |v| v.parse().unwrap());
                let (lo, hi) = fieldform_field::part_extent(&doc, &graph).as_box().expect("bounded");
                let extent = hi - lo;
                let tolerance = (extent.x.max(extent.y).max(extent.z) / cells_across * 0.5).max(1e-4);
                let reads = fieldform_field::reads_widened(
                    fieldform_field::profile_reads_out_to(&doc, &graph, *node),
                    fieldform_gpu::Occlusion::reach(extent.norm() * 0.5),
                );
                let (plo, phi) = flat.bounds();
                let baked = fieldform_field::bake_cells((phi - plo).max(), tolerance, reads)
                    .map(|cells| flat.baked(cells, reads))
                    .and_then(|p| p.bake().cloned());
                match &baked {
                    Some(b) => println!(
                        "profile of {node:?} baked at a tolerance of {tolerance:.3} mm: margin {:.3} mm",
                        b.margin()
                    ),
                    None => println!("profile of {node:?} would not bake at {tolerance:.3} mm"),
                }
                (points, baked)
            }
            CurveUse::Spine(id) => (doc.curves[id].tessellate(TOLERANCE_MM), None),
            CurveUse::Frame(node) => {
                let Some(Node::Sweep { outline, spine, .. }) = graph.node(*node) else {
                    panic!("only sweeps' frames are bound here")
                };
                let path = doc.curves[spine].tessellate(TOLERANCE_MM);
                (fieldform_field::sweep_frame(&doc, outline, *spine, &path).expect("a frame"), None)
            }
            other => panic!("{other:?} is not bound by this probe"),
        })
        .collect();
    let renderer =
        pollster::block_on(Renderer::new(&program, &[], &curves, &[], 32, 1.0)).expect("a GPU");
    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
        .collect();
    renderer.set_params(&values);

    let (lo, hi) = fieldform_field::part_extent(&doc, &graph).as_box().expect("bounded");
    let centre = nalgebra::center(&lo, &hi);
    let radius = (hi - lo).norm() * 0.5;
    // The views a person zooms in on a seam with: the target is the seam's own plane, where the
    // owner's screenshot looked, and the scene's radius is the whole part's, as the viewport's is.
    let seam = Point3::new(
        std::env::var("SEAM_X").map_or(0.0, |v| v.parse().unwrap()),
        std::env::var("SEAM_Y").map_or(centre.y, |v| v.parse().unwrap()),
        std::env::var("SEAM_Z").map_or(centre.z, |v| v.parse().unwrap()),
    );
    println!("scene radius {radius:.2} mm, so the hit threshold is {:.5} mm", radius * 1e-5);
    println!("| view | body | hits | worst short (mm) | worst past (mm) | 99th pct of |error| |");
    println!("|---|---|---|---|---|---|");
    for (name, from) in [
        ("side, level", Vector3::new(1.0, 0.25, 0.1)),
        ("side, from below", Vector3::new(1.0, 0.3, -0.6)),
        ("from above", Vector3::new(0.3, 0.3, 1.0)),
        ("along the side", Vector3::new(1.0, 1.6, -0.2)),
        ("whole part", Vector3::new(0.55, -0.7, 0.45)),
    ] {
        // The whole part is framed to fit, as the viewport frames it; the seam views zoom in.
        let whole = name == "whole part";
        if whole && std::env::var_os("WHOLE").is_none() {
            continue;
        }
        let fov: f64 = if whole { 30.0 } else { 12.0 };
        let target = if whole { centre } else { seam };
        let distance = if whole { radius / (fov.to_radians() * 0.5).sin() * 1.1 } else { 120.0 };
        let eye = target + from.normalize() * distance;
        let mut camera = Camera::looking_at(eye, target, Vector3::z(), (centre, radius), fov, f64::from(width) / f64::from(height));
        // `EXACT=1` finishes every ray on the outline, as the still tier does.
        camera.exact_finish = std::env::var_os("EXACT").is_some();
        let hits = renderer.render(&camera, width, height, 16384).expect("renders");
        // `SHADE=<stem>` writes each view lit by its normals, at this zoom -- about the owner's --
        // so a seam or an edge can be looked at as well as measured.
        // The ray of pixel `i` and where its hit stands.
        let ray_of = |i: usize| {
            let (px, py) = ((i as u32 % width) as f64 + 0.5, (i as u32 / width) as f64 + 0.5);
            let u = (px / f64::from(width) * 2.0 - 1.0) * camera.half_height * camera.aspect;
            let v = (1.0 - py / f64::from(height) * 2.0) * camera.half_height;
            (camera.forward + camera.right * u + camera.up * v).normalize()
        };
        // A stray is a hit the exact field reads as well off any surface: drawn in mid-air. Every
        // hit is checked, because `along_ray_error` drops the ones its walk never reaches.
        let mut strays: Vec<(usize, f64)> = Vec::new();
        let mut facing_away = 0usize;
        let mut away: Vec<(usize, f64)> = Vec::new();
        for (i, hit) in hits.iter().enumerate() {
            if !hit.is_hit() {
                continue;
            }
            let dir = ray_of(i);
            let at = camera.eye + dir * hit.depth;
            let reading = cpu.distance(at);
            if !reading.is_finite() || reading.abs() > STRAY_MM || !hit.normal.x.is_finite() {
                strays.push((i, reading));
            }
            if hit.normal.dot(&dir) > 0.0 {
                facing_away += 1;
                away.push((i, reading));
            }
        }
        {
            let worst = strays.iter().map(|(_, r)| r.abs()).fold(0.0f64, f64::max);
            let hits_n = hits.iter().filter(|h| h.is_hit()).count();
            println!(
                "{name}: {hits_n} hits, {} strays (exact reading over {STRAY_MM} mm; worst {worst:.3} mm), {facing_away} normals facing away",
                strays.len()
            );
            let listed = if std::env::var_os("STRAYS").is_some() { 12 } else { 0 };
            for (i, reading) in strays.iter().take(listed).chain(away.iter().take(listed / 2)) {
                let hit = &hits[*i];
                let dir = ray_of(*i);
                let at = camera.eye + dir * hit.depth;
                println!(
                    "  stray at px ({}, {}): depth {:.4}, reads {reading:.4} mm, normal ({:.3}, {:.3}, {:.3}) at ({:.3}, {:.3}, {:.3})",
                    *i as u32 % width, *i as u32 / width, hit.depth, hit.normal.x, hit.normal.y, hit.normal.z, at.x, at.y, at.z
                );
            }
        }
        if let Ok(stem) = std::env::var("SHADE") {
            let light = (camera.forward * -1.0 + camera.up * 0.6 + camera.right * 0.35).normalize();
            let mut rgb = Vec::with_capacity(hits.len() * 3);
            for (i, hit) in hits.iter().enumerate() {
                if strays.iter().any(|(j, _)| *j == i) {
                    rgb.extend_from_slice(&[255, 0, 0]);
                } else if hit.is_hit() && hit.normal.dot(&ray_of(i)) > 0.0 {
                    rgb.extend_from_slice(&[0, 90, 255]);
                } else if hit.is_hit() {
                    let shade = 0.15 + 0.85 * hit.normal.dot(&light).max(0.0);
                    rgb.extend_from_slice(&[(shade * 220.0) as u8, (shade * 150.0) as u8, (shade * 110.0) as u8]);
                } else {
                    rgb.extend_from_slice(&[24, 26, 30]);
                }
            }
            let file = format!("{stem}-{}.png", name.replace(", ", "-").replace(' ', "-"));
            image::save_buffer(&file, &rgb, width, height, image::ColorType::Rgb8).expect("writable");
            println!("wrote {file}");
        }
        let mut by_body: std::collections::BTreeMap<u32, Vec<f64>> = Default::default();
        // `NOWALK=1` skips the walk, which on a frame this size is minutes of CPU for a column
        // the strays above already answer.
        let every = if std::env::var_os("NOWALK").is_some() { usize::MAX } else if hits.len() > 400_000 { 4 } else { 1 };
        for (i, hit) in hits.iter().enumerate() {
            if !hit.is_hit() || every == usize::MAX || i % every != 0 {
                continue;
            }
            let dir = ray_of(i);
            let at = camera.eye + dir * hit.depth;
            if let Some(e) = along_ray_error(&cpu, at, dir) {
                // What the march saw at the worst of them: the reading where it stopped, the
                // reading a full look further on, and how far the surface really was -- whose
                // ratio to the reading is how short the reading is there, divisor and all.
                // The same ray through the CPU replica of the march, step by step, beside where
                // the GPU stopped -- so a bad hit can be followed into the loop that made it.
                if std::env::var_os("TRACE").is_some() && e > 0.03 && i % 211 == 0 {
                    let f32_field = |p: fieldform_gpu::cpu_march::V3| {
                        cpu.distance(Point3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z))) as f32
                    };
                    let arm = fieldform_gpu::cpu_march::Arm {
                        field: &f32_field,
                        grid: None,
                        relax: 1.4,
                        hop: fieldform_gpu::cpu_march::Hop::Vouched,
                        settles: program.settles,
                        exact: None,
                    };
                    let o32 = fieldform_gpu::cpu_march::V3::new(camera.eye.x as f32, camera.eye.y as f32, camera.eye.z as f32);
                    let d32 = fieldform_gpu::cpu_march::V3::new(dir.x as f32, dir.y as f32, dir.z as f32);
                    println!("  TRACE body {} GPU stopped {e:.4} mm short at t = {:.4}", hit.body, hit.depth);
                    let out = fieldform_gpu::cpu_march::march(&arm, o32, d32, 0.0, 1000.0, (radius * 1e-5) as f32, 16384, true);
                    if let Some(t) = out.hit {
                        let at = camera.eye + dir * f64::from(t);
                        println!("  TRACE replica stopped at t = {t:.4}: {:?} mm short", along_ray_error(&cpu, at, dir));
                    }
                }
                by_body.entry(hit.body).or_default().push(e);
            }
        }
        for (body, mut errors) in by_body {
            let short = errors.iter().cloned().fold(0.0f64, f64::max);
            let past = errors.iter().cloned().fold(0.0f64, f64::min);
            errors.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
            let p99 = errors[(errors.len() * 99 / 100).min(errors.len() - 1)].abs();
            println!("| {name} | {body} | {} | {short:.4} | {past:.4} | {p99:.4} |", errors.len());
        }
    }
}
