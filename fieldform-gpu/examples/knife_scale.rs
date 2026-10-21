// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Whether a sharp edge costs or shows the same at every size — the scale check behind backlog
//! item 115's fix.
//!
//! A lens with a 39-degree knife at each end is extruded at 100 mm, 1 m and 50 m across, and its
//! profile is baked exactly as the viewport bakes at a quality (`CELLS`: 64 Fast, 192 Full, and
//! what Enchilada's finer bake asks for). For each: how many cells the bake took and how long,
//! its margin as a fraction of the part, the error the march draws at the knife in millimetres
//! *and in pixels* of a frame that fits the part, and the frame's cost.
//!
//! The viewport's tolerance is a fraction of the scene's extent, so every number here but the
//! millimetres should come out the same at every size: a knife on a 50 m hull is neither slower
//! nor fatter on screen than the same knife on a 100 mm bracket. This is what says so.
//!
//! `cargo run --release -p fieldform-gpu --example knife_scale [-- <cells>...]`

use fieldform_field::{GraphField, ScalarField};
use fieldform_gpu::{Camera, CurveUse, Renderer};
use fieldform_scene::{Curve, CurveId, Document, Graph, Node, Outline, ParamRef, Sides, Sweep};
use nalgebra::{Point3, Vector3};

const WIDTH: u32 = 720;
const HEIGHT: u32 = 480;
const FRAMES: u32 = 10;

/// The lens of `swweeeeeping.fform`'s section, `across` millimetres wide, standing in z = 0.
fn lens(doc: &mut Document, across: f64) -> CurveId {
    let s = across / 40.0;
    let id = CurveId::new();
    let points = [(-20.0, 0.0), (0.0, -4.35), (20.0, 0.0), (10.0, 5.0), (0.0, 12.3), (-10.0, 5.0)]
        .iter()
        .map(|(x, y)| Point3::new(x * s, y * s, 0.0))
        .collect();
    doc.curves.insert(id, Curve::polyline(id, "lens", points, true).expect("closed"));
    id
}

/// The signed distance along `dir` from `at` to where the exact field changes sign.
fn along_ray_error(cpu: &GraphField, at: Point3<f64>, dir: Vector3<f64>, step: f64) -> Option<f64> {
    let outside = cpu.distance(at) > 0.0;
    let way = if outside { 1.0 } else { -1.0 };
    let mut prev = 0.0;
    for i in 1..=250 {
        let s = way * step * i as f64;
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
    let cells_list: Vec<f64> = {
        let given: Vec<f64> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
        if given.is_empty() { vec![64.0, 192.0, 768.0] } else { given }
    };
    println!(
        "| across | cells asked | bake cells | bake ms | margin | margin / across | knife error p99 | in px | frame ms |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for across in [100.0, 1000.0, 50_000.0] {
        let mut doc = Document::new();
        let curve = lens(&mut doc, across);
        let mut graph = Graph::empty();
        let node = graph.add(Node::Extrude { rims: Default::default(),
            outline: Outline::closed(curve),
            sweep: Sweep::Normal,
            height: ParamRef::Literal(across * 0.4),
            draft: ParamRef::Literal(0.0),
            start: ParamRef::Literal(0.0),
            sides: Sides::One,
            corners: Default::default(),
        });
        // Exact, for the truth: a tolerance this fine never bakes.
        let cpu = GraphField::compile_graph(&doc, &graph, across * 1e-6).expect("compiles");
        let program = fieldform_gpu::generate_with(&doc, &graph, &[]).expect("generates");
        let (lo, hi) = fieldform_field::part_extent(&doc, &graph).as_box().expect("bounded");
        let extent = hi - lo;
        let centre = nalgebra::center(&lo, &hi);
        let radius = extent.norm() * 0.5;
        for &cells_across in &cells_list {
            // The viewport's rule: half the cage bake's cell, and the display's reach added to the band.
            let tolerance = (extent.x.max(extent.y).max(extent.z) / cells_across * 0.5).max(1e-4);
            let reads = fieldform_field::reads_widened(
                fieldform_field::profile_reads_out_to(&doc, &graph, node),
                fieldform_gpu::Occlusion::reach(radius),
            );
            let mut bake_cells = 0usize;
            let mut bake_ms = 0.0;
            let mut margin = 0.0;
            let curves: Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> = program
                .curves
                .iter()
                .map(|use_of| match use_of {
                    CurveUse::Profile(_) => {
                        let world = doc.curves[&curve].tessellate(0.05);
                        let (origin, normal) = fieldform_field::plane_of(&world).expect("planar");
                        let rotation = fieldform_field::rotation_from_z(normal);
                        let into = move |p: Point3<f64>| {
                            Point3::from(rotation.inverse_transform_vector(&(p.coords - origin.coords)))
                        };
                        let flat = fieldform_field::Profile::in_plane(&world, into, 1e-3).expect("planar");
                        let points = flat.points().iter().map(|p| Point3::new(p.x, p.y, 0.0)).collect();
                        let (plo, phi) = flat.bounds();
                        let cells = fieldform_field::bake_cells((phi - plo).max(), tolerance, reads);
                        let bake = cells.and_then(|cells| {
                            bake_cells = cells;
                            let began = std::time::Instant::now();
                            let baked = flat.clone().baked(cells, reads);
                            bake_ms = began.elapsed().as_secs_f64() * 1e3;
                            baked.bake().cloned()
                        });
                        if let Some(b) = &bake {
                            margin = b.margin();
                        }
                        (points, bake)
                    }
                    other => panic!("{other:?} is not bound by this example"),
                })
                .collect();
            let renderer = pollster::block_on(Renderer::new(&program, &[], &curves, &[], 32, 1.0))
                .expect("a GPU");
            let values: Vec<f32> = program
                .params
                .iter()
                .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
                .collect();
            renderer.set_params(&values);
            // A frame that fits the part, from where both knives show.
            let fov = 30.0f64;
            let distance = radius / (fov.to_radians() * 0.5).sin() * 1.1;
            let eye = centre + Vector3::new(0.55, -0.7, 0.45).normalize() * distance;
            let mut camera = Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), fov, f64::from(WIDTH) / f64::from(HEIGHT));
            // `EXACT=1` finishes every ray on the outline, as the still tier does.
            camera.exact_finish = std::env::var_os("EXACT").is_some();
            let mm_per_px = 2.0 * distance * (fov.to_radians() * 0.5).tan() / f64::from(HEIGHT);
            let _ = renderer.render(&camera, WIDTH, HEIGHT, 4096).expect("renders");
            let began = std::time::Instant::now();
            for _ in 0..FRAMES {
                renderer.render(&camera, WIDTH, HEIGHT, 4096).expect("renders");
            }
            let frame_ms = began.elapsed().as_secs_f64() * 1e3 / f64::from(FRAMES);
            let hits = renderer.render(&camera, WIDTH, HEIGHT, 4096).expect("renders");
            // The error along the ray at every hit, scale-free: stepped in thousandths of the part.
            let mut errors: Vec<f64> = Vec::new();
            for (i, hit) in hits.iter().enumerate() {
                if !hit.is_hit() || i % 7 != 0 {
                    continue;
                }
                let (px, py) = ((i as u32 % WIDTH) as f64 + 0.5, (i as u32 / WIDTH) as f64 + 0.5);
                let u = (px / f64::from(WIDTH) * 2.0 - 1.0) * camera.half_height * camera.aspect;
                let v = (1.0 - py / f64::from(HEIGHT) * 2.0) * camera.half_height;
                let dir = (camera.forward + camera.right * u + camera.up * v).normalize();
                let at = camera.eye + dir * hit.depth;
                if let Some(e) = along_ray_error(&cpu, at, dir, across * 2e-4) {
                    errors.push(e.abs());
                }
            }
            errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p99 = errors.get(errors.len() * 99 / 100).copied().unwrap_or(0.0);
            println!(
                "| {across:>7.0} mm | {cells_across:>4.0} | {bake_cells:>5} | {bake_ms:>7.1} | {margin:>8.4} mm | {:>9.6} | {p99:>8.4} mm | {:>5.2} | {frame_ms:>6.2} |",
                margin / across,
                p99 / mm_per_px,
            );
        }
    }
}
