// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a document's shader is actually made of, and which block put it there.
//!
//! **Because the driver's compile time is superlinear in one function's size** — measured at
//! 8 walks 2.0 s, 16 walks 6.2 s, 32 walks 53.5 s by `wing_compile_cost` — the way to a faster
//! first render is to emit less, and the way to emit less is to know what is being emitted.
//!
//! A node can appear in the generated source more than once. Every block that *changes the point*
//! — a placement, a mirror, a twist, an attractor, a pattern — makes its subtree be read at a new
//! point, and a subtree read at two points is written twice. That is the multiplier worth seeing,
//! because it compounds down a chain: a mirror over a twist over an extrude writes the extrude
//! once per point either of them declares.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example shader_composition -- <file.fform>`

fn main() {
    // **The floor first.** The prelude declares every field kind this build knows, and the
    // question a document's own figure cannot answer is how much of the driver's time is the
    // document at all. So: a single sphere, whose generated half is a handful of lines, through
    // the same device.
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
    println!("{}", device.adapter_info().name);
    {
        let mut g = fieldform_scene::Graph::empty();
        g.add(fieldform_scene::Node::Sphere {
            centre: fieldform_scene::Place::At([fieldform_scene::ParamRef::Literal(0.0); 3]),
            radius: fieldform_scene::ParamRef::Literal(10.0),
        });
        let doc = fieldform_scene::Document::new();
        let program = fieldform_gpu::generate(&doc, &g).expect("a sphere generates");
        let cut = program
            .source
            .find("// ---- generated from the composition graph ----")
            .expect("marked");
        let t = std::time::Instant::now();
        let _ = fieldform_gpu::Renderer::with_device(
            device.clone(),
            queue.clone(),
            &program,
            &[],
            &[],
            &[],
        );
        println!(
            "floor: one sphere, {} generated lines, {:.1} KB whole -> driver {:.0} ms\n",
            program.source[cut..].lines().count(),
            program.source.len() as f64 / 1024.0,
            t.elapsed().as_secs_f64() * 1e3,
        );
    }

    let path = std::env::args().nth(1).expect("a .fform file");
    let (doc, _) = fieldform_io::open(std::path::Path::new(&path)).expect("opens");

    for part in doc.parts_ordered() {
        let graph = &part.graph;
        let program = match fieldform_gpu::generate(&doc, graph) {
            Ok(p) => p,
            Err(e) => {
                println!("{:?}: refused: {e}", part.name);
                continue;
            }
        };
        let cut = program
            .source
            .find("// ---- generated from the composition graph ----")
            .expect("every program carries the marker");
        let generated = &program.source[cut..];

        println!(
            "\n=== {:?}: {} nodes, {} bodies, {} lines of generated WGSL ({:.1} KB whole) ===",
            part.name,
            graph.nodes().count(),
            graph.roots().len(),
            generated.lines().count(),
            program.source.len() as f64 / 1024.0,
        );

        // How many times each block was written out. `p{n}` names a point context, and a
        // subtree under two contexts is emitted under both.
        println!("{:<22} {:>7} {:>9}", "block", "written", "lines");
        let mut rows: Vec<(String, usize, usize)> = Vec::new();
        for (id, node) in graph.nodes() {
            let kind = format!("{node:?}");
            let kind = kind.split(['{', '(']).next().unwrap_or("?").trim().to_string();
            let marker = emitted_as(node);
            let Some(marker) = marker else { continue };
            let written = generated.matches(&marker).count();
            // A rough share: how many generated lines mention this block's own call.
            let lines = generated.lines().filter(|l| l.contains(&marker)).count();
            rows.push((format!("{kind} {}", id.raw() % 10000), written, lines));
        }
        rows.sort_by_key(|(_, written, _)| std::cmp::Reverse(*written));
        for (name, written, lines) in rows {
            println!("{name:<22} {written:>7} {lines:>9}");
        }

        // And the constructs the driver charges most for, whoever emitted them.
        println!("\n{:<28} {:>7}", "expensive construct", "sites");
        for marker in [
            "nearest_segment(",
            "sweep_distance(",
            "sweep_value_normal(",
            "profile_distance(",
            "extrude_distance(",
            "extrude_normal(",
            "revolve_distance(",
            "revolve_normal(",
            "sample_baked(",
            "lattice_distance(",
            "conformal_lattice_distance(",
            "to_local(",
            "mirror_point(",
            "mirror_fold(",
            "twist_point(",
        ] {
            let n = generated.matches(marker).count();
            if n > 0 {
                println!("{marker:<28} {n:>7}");
            }
        }

        println!(
            "\n{} point contexts declared — each one is a place the subtree below it is \
             written out again",
            generated.matches("    let p").count()
        );

        // The generated half, for reading. The driver's time is spent on one function at a
        // time, so which function is large is the question a count of call sites cannot answer.
        println!("\n{:<26} {:>7}", "generated function", "lines");
        let mut current = String::new();
        let mut lines = 0usize;
        let mut sizes: Vec<(String, usize)> = Vec::new();
        for line in generated.lines() {
            if let Some(rest) = line.strip_prefix("fn ") {
                if !current.is_empty() {
                    sizes.push((current.clone(), lines));
                }
                current = rest.split('(').next().unwrap_or("?").to_string();
                lines = 0;
            }
            lines += 1;
        }
        if !current.is_empty() {
            sizes.push((current, lines));
        }
        sizes.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        for (name, n) in sizes.iter().take(12) {
            println!("{name:<26} {n:>7}");
        }

        if let Ok(out) = std::env::var("FIELDFORM_DUMP_WGSL") {
            std::fs::write(&out, &program.source).expect("writes");
            println!("\nwrote the whole shader to {out}");
        }
    }
}

/// The call a block emits, when it emits one this can count.
fn emitted_as(node: &fieldform_scene::Node) -> Option<String> {
    use fieldform_scene::Node;
    Some(
        match node {
            Node::Extrude { .. } => "extrude_distance(",
            Node::Revolve { .. } => "revolve_distance(",
            Node::Sweep { .. } => "sweep_distance(",
            Node::Pipe { .. } => "pipe_distance(",
            Node::Sphere { .. } => "sphere_distance(",
            Node::Cuboid { .. } => "cuboid_distance(",
            Node::Cylinder { .. } => "cylinder_distance(",
            Node::Cone { .. } => "cone_distance(",
            Node::Torus { .. } => "torus_distance(",
            Node::Capsule { .. } => "capsule_distance(",
            Node::Plane { .. } => "plane_distance(",
            Node::Lattice { .. } => "lattice_distance(",
            Node::Mirror { .. } => "mirror_",
            Node::Twist { .. } => "twist_point(",
            Node::Attract { .. } => "attract_point(",
            _ => return None,
        }
        .to_string(),
    )
}
