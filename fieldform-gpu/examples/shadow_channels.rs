// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A still's occlusion and shadow, each written out as a picture of its own, from a document.
//!
//! The lighting pass multiplies three things a person sees as one: the material under the
//! lights, how open the surface is (`occlusion` in `march.wgsl`), and how much of the key light
//! reaches it (`shadow`). When a shaded still looks wrong near a join, this says which of the
//! three it is: `<stem>-shaded.png` is the still, `<stem>-open.png` the occlusion alone and
//! `<stem>-shadow.png` the shadow alone, grey from black (closed, or shadowed) to white.
//!
//! `cargo run --release -p fieldform-gpu --example shadow_channels -- <doc.fform> <stem> [<node>...]`
//! draws the part's bodies (or only the nodes named, in hex). `VIEW=x,y,z` is the direction from
//! the part's centre toward the eye, `ZOOM` how many times closer than a frame that fits the part,
//! and `BAKE=1` reads every sweep's profile from a grid as the viewport does at `CELLS` across
//! (192 Full, 384 Enchilada's finer bake); without it the profiles are walked exactly, which is
//! the control for anything the bake's margins might be doing to a reading taken off the surface.
//! `EXACT=1` finishes the rays on the outline, as a still does.

use fieldform_field::ScalarField;
use fieldform_gpu::{Camera, CurveUse, EnvironmentChoice, Look, Occlusion, Renderer, ShadingResources};
use fieldform_scene::{Material, Node, NodeId};
use nalgebra::{Point3, Vector3};

const TOLERANCE_MM: f64 = 0.01;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (doc, _archive) = fieldform_io::open(std::path::Path::new(&args[1])).expect("the document opens");
    let stem = args[2].clone();
    let keep: Vec<NodeId> =
        args[3..].iter().map(|s| NodeId::from_raw(u64::from_str_radix(s, 16).expect("a node id in hex"))).collect();
    let (_, part) = doc.parts.iter().next().expect("a part");
    let mut graph = part.graph.clone();
    // Framed on the whole part whatever is drawn, so a body alone is looked at from where the
    // part is, and the occlusion reaches as far as it does in the viewport.
    let (lo, hi) = fieldform_field::part_extent(&doc, &graph).as_box().expect("bounded");
    // `NOBLEND=1` makes every blend a plain union, the control for what the blend's reading does.
    if std::env::var_os("NOBLEND").is_some() {
        let blends: Vec<NodeId> = graph.nodes().filter(|(_, n)| matches!(n, Node::Blend { .. })).map(|(id, _)| id).collect();
        for id in blends {
            if let Some(node) = graph.node_mut(id) {
                if let Node::Blend { op, a, b, .. } = node.clone() {
                    *node = Node::Boolean { op, inputs: vec![a, b] };
                }
            }
        }
    }
    if !keep.is_empty() {
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
    }
    let cpu = fieldform_field::GraphField::compile_graph(&doc, &graph, TOLERANCE_MM).expect("compiles");
    let program = fieldform_gpu::generate_with(&doc, &graph, cpu.morph_divisors()).expect("generates");
    let extent = hi - lo;
    let centre = nalgebra::center(&lo, &hi);
    let radius = extent.norm() * 0.5;
    let bake = std::env::var_os("BAKE").is_some();
    let cells_across: f64 = std::env::var("CELLS").map_or(192.0, |v| v.parse().unwrap());
    let curves: Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> = program
        .curves
        .iter()
        .map(|use_of| match use_of {
            CurveUse::Profile(node) => {
                let Some(Node::Sweep { outline, spine, .. }) = graph.node(*node) else {
                    panic!("only sweeps are bound here")
                };
                let laid = fieldform_field::sweep_outline_section(&doc, outline, *spine, fieldform_field::PROFILE_TURN_DEGREES)
                    .expect("a section");
                let rotation = fieldform_field::rotation_from_z(laid.normal);
                let origin = laid.origin;
                let into = move |p: Point3<f64>| Point3::from(rotation.inverse_transform_vector(&(p.coords - origin.coords)));
                let flat = fieldform_field::Profile::in_plane(&laid.points, into, 1e-3).expect("planar");
                let points = flat.points().iter().map(|p| Point3::new(p.x, p.y, 0.0)).collect();
                if !bake {
                    return (points, None);
                }
                // As the viewport bakes for a still: its tolerance, its band widened by the
                // occlusion's reach, so the samples off the surface read the fine level.
                let tolerance = (extent.x.max(extent.y).max(extent.z) / cells_across * 0.5).max(1e-4);
                let reads = fieldform_field::reads_widened(
                    fieldform_field::profile_reads_out_to(&doc, &graph, *node),
                    Occlusion::reach(radius),
                );
                let (plo, phi) = flat.bounds();
                let baked = fieldform_field::bake_cells((phi - plo).max(), tolerance, reads)
                    .map(|cells| flat.baked(cells, reads))
                    .and_then(|p| p.bake().cloned());
                if let Some(b) = &baked {
                    println!("profile of {node:?} baked at a tolerance of {tolerance:.3} mm: margin {:.3} mm", b.margin());
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
            other => panic!("{other:?} is not bound by this example"),
        })
        .collect();
    let renderer = pollster::block_on(Renderer::new(&program, &[], &curves, &[], 32, 1.0)).expect("a GPU");
    let values: Vec<f32> =
        program.params.iter().map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32).collect();
    renderer.set_params(&values);

    let (width, height) = (1440u32, 976u32);
    let from: Vector3<f64> = std::env::var("VIEW")
        .ok()
        .and_then(|v| {
            let n: Vec<f64> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            (n.len() == 3).then(|| Vector3::new(n[0], n[1], n[2]))
        })
        .unwrap_or_else(|| Vector3::new(1.0, -0.3, 0.6));
    let zoom: f64 = std::env::var("ZOOM").map_or(1.0, |v| v.parse().unwrap());
    let fov = 30.0f64;
    let distance = radius / (fov.to_radians() * 0.5).sin() * 1.1 / zoom;
    let eye = centre + from.normalize() * distance;
    let mut camera =
        Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), fov, f64::from(width) / f64::from(height));
    camera.exact_finish = std::env::var_os("EXACT").is_some();
    let look = Look::for_environment(EnvironmentChoice::Studio, Material::default(), camera.forward, camera.right, camera.up);
    let mut occlusion = Occlusion::for_scene(look.lights[0].direction, radius);
    // `REACH=<mm>` overrides how far out the occlusion looks, for telling a reach that is too
    // long from a field that reads short.
    if let Some(reach) = std::env::var("REACH").ok().and_then(|v| v.parse::<f64>().ok()) {
        occlusion.radius = reach;
    }
    camera.occlusion = Some(occlusion);
    // The divisor put back for the occlusion's readings, as the viewport does (`Camera::value_scale`):
    // without it a divided sweep reads short and occludes itself to black.
    camera.value_scale = fieldform_field::value_scale_with(&doc, &graph, cpu.morph_divisors());
    println!("value scale {:.3}, occlusion reach {:.2} mm, shadow reach {:.1} mm", camera.value_scale, occlusion.radius, occlusion.shadow_reach);
    camera.jitter = [0.0, 0.0];
    let maps = EnvironmentChoice::Studio.bake();
    let mut shading = ShadingResources::new(renderer.device(), renderer.queue(), &maps).expect("the lighting pipeline builds");
    renderer.present(&mut shading, &camera, width, height, 4096, &look, 0).expect("the frame presents");
    let shaded = renderer.read_display(&shading).expect("the frame reads back");
    let extras = renderer.read_extras(&shading).expect("the extras read back");
    let hits = renderer.render(&camera, width, height, 4096).expect("the hits read back");

    image::save_buffer(format!("{stem}-shaded.png"), &shaded.rgba, width, height, image::ColorType::Rgba8).expect("writable");
    let mut open = Vec::with_capacity(extras.len());
    let mut lit = Vec::with_capacity(extras.len());
    let (mut measured, mut hit_n) = (0usize, 0usize);
    for (hit, [o, l]) in hits.iter().zip(&extras) {
        if !hit.is_hit() {
            open.push(30u8);
            lit.push(30u8);
            continue;
        }
        hit_n += 1;
        if *o < 0.0 {
            // Left to the lighting pass to fill in: marked blue-grey rather than guessed.
            open.push(90);
            lit.push(90);
            continue;
        }
        measured += 1;
        open.push((o.clamp(0.0, 1.0) * 255.0) as u8);
        lit.push((l.clamp(0.0, 1.0) * 255.0) as u8);
    }
    // The occlusion in numbers: an open flat face reads 1, and a face that reads nought all over
    // is not a picture of a crevice but of a field reading short of its surface.
    let readings: Vec<f32> = hits.iter().zip(&extras).filter(|(h, [o, _])| h.is_hit() && *o >= 0.0).map(|(_, [o, _])| *o).collect();
    let mean = readings.iter().sum::<f32>() / readings.len().max(1) as f32;
    let closed = readings.iter().filter(|o| **o < 0.05).count();
    let lit_mean = hits.iter().zip(&extras).filter(|(h, [o, _])| h.is_hit() && *o >= 0.0).map(|(_, [_, l])| *l).sum::<f32>() / readings.len().max(1) as f32;
    println!(
        "occlusion: mean open {mean:.3}, {closed} of {} measured pixels read closed (under 0.05); shadow: mean lit {lit_mean:.3}",
        readings.len()
    );
    // The normals against the CPU field's gradient, and the occlusion replicated on the CPU from
    // the exact field with each normal: which of the two the GPU's reading is short of.
    let ray_of = |i: usize| {
        let (px, py) = ((i as u32 % width) as f64 + 0.5, (i as u32 / width) as f64 + 0.5);
        let u = (px / f64::from(width) * 2.0 - 1.0) * camera.half_height * camera.aspect;
        let v = (1.0 - py / f64::from(height) * 2.0) * camera.half_height;
        (camera.forward + camera.right * u + camera.up * v).normalize()
    };
    let gradient = |at: Point3<f64>| {
        let e = 1e-3;
        let g = Vector3::new(
            cpu.distance(at + Vector3::x() * e) - cpu.distance(at - Vector3::x() * e),
            cpu.distance(at + Vector3::y() * e) - cpu.distance(at - Vector3::y() * e),
            cpu.distance(at + Vector3::z() * e) - cpu.distance(at - Vector3::z() * e),
        );
        g.try_normalize(1e-12).unwrap_or_else(Vector3::z)
    };
    let cpu_open = |at: Point3<f64>, n: Vector3<f64>| {
        let r = occlusion.radius;
        let (mut occluded, mut falloff) = (0.0, 1.0);
        for i in 1..=5 {
            let h = r * i as f64 / 5.0;
            let d = cpu.distance(at + n * h) * camera.value_scale;
            occluded += (h - d) * falloff;
            falloff *= 0.7;
        }
        (1.0 - 1.5 * occluded / r).clamp(0.0, 1.0)
    };
    let (mut n_sum, mut n_bad, mut n_count, mut open_gpu_n, mut open_cpu_n) = (0.0f64, 0usize, 0usize, 0.0f64, 0.0f64);
    let mut worst: Option<(usize, f64)> = None;
    for (i, (hit, [o, _])) in hits.iter().zip(&extras).enumerate() {
        if !hit.is_hit() || *o < 0.0 || i % 97 != 0 {
            continue;
        }
        let at = camera.eye + ray_of(i) * hit.depth;
        let g = gradient(at);
        let angle = hit.normal.normalize().dot(&g).clamp(-1.0, 1.0).acos().to_degrees();
        n_sum += angle;
        n_bad += usize::from(angle > 10.0);
        n_count += 1;
        if worst.is_none_or(|(_, a)| angle > a) {
            worst = Some((i, angle));
        }
        open_gpu_n += cpu_open(at, hit.normal.normalize());
        open_cpu_n += cpu_open(at, g);
    }
    let n = n_count.max(1) as f64;
    println!(
        "normals, one hit in 97: mean {:.2} degrees off the CPU gradient, {n_bad} of {n_count} over 10 degrees; occlusion replicated on the CPU field: mean open {:.3} with the GPU's normals, {:.3} with the gradient",
        n_sum / n, open_gpu_n / n, open_cpu_n / n
    );
    if let Some((i, angle)) = worst {
        let hit = &hits[i];
        let at = camera.eye + ray_of(i) * hit.depth;
        let g = gradient(at);
        println!(
            "  worst at px ({}, {}): GPU normal ({:.3}, {:.3}, {:.3}), gradient ({:.3}, {:.3}, {:.3}), {angle:.1} degrees, at ({:.2}, {:.2}, {:.2}) reading {:.4}",
            i as u32 % width, i as u32 / width, hit.normal.x, hit.normal.y, hit.normal.z, g.x, g.y, g.z, at.x, at.y, at.z, cpu.distance(at)
        );
    }
    // The field along the normal from one hit, against the distance a true field would read
    // (the offset itself, until something else is nearer): how short, and from where.
    if let Some((i, _)) = worst {
        let hit = &hits[i];
        let at = camera.eye + ray_of(i) * hit.depth;
        let n = hit.normal.normalize();
        let along: Vec<String> = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0]
            .iter()
            .map(|h| format!("{h}: {:.3}", cpu.distance(at + n * *h) * camera.value_scale))
            .collect();
        println!("  along the normal from there, mm offset: reading times the scale -- {}", along.join(", "));
    }
    // And from six hits spread across the picture, with the GPU's own reading of openness.
    let measured_hits: Vec<usize> = hits.iter().zip(&extras).enumerate().filter(|(_, (h, [o, _]))| h.is_hit() && *o >= 0.0).map(|(i, _)| i).collect();
    for k in 0..6 {
        let Some(&i) = measured_hits.get(k * measured_hits.len() / 6 + measured_hits.len() / 12) else { break };
        let hit = &hits[i];
        let at = camera.eye + ray_of(i) * hit.depth;
        let n = hit.normal.normalize();
        let along: Vec<String> =
            [0.5, 2.0, 5.0, 25.0].iter().map(|h| format!("{h}: {:.2}", cpu.distance(at + n * *h) * camera.value_scale)).collect();
        println!(
            "  px ({}, {}) at ({:.1}, {:.1}, {:.1}) normal ({:.2}, {:.2}, {:.2}) GPU open {:.3}; along the normal -- {}",
            i as u32 % width, i as u32 / width, at.x, at.y, at.z, n.x, n.y, n.z, extras[i][0], along.join(", ")
        );
    }
    image::save_buffer(format!("{stem}-open.png"), &open, width, height, image::ColorType::L8).expect("writable");
    image::save_buffer(format!("{stem}-shadow.png"), &lit, width, height, image::ColorType::L8).expect("writable");
    println!(
        "{hit_n} pixels hit, {measured} measured for occlusion and shadow; wrote {stem}-shaded.png, -open.png and -shadow.png"
    );
}
