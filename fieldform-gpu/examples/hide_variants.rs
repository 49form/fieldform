// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What each combination of hidden bodies costs the driver, for a real document.
//!
//! **Because hiding a body is a different shader, not a different uniform.** The viewport draws
//! a graph with the hidden bodies taken out of it (`fieldform_scene::drawn_graph`), so every
//! combination of what is hidden is its own source, and a source the driver has not seen before
//! is a cold compile. `hide_cost.rs` said this in 2026-09 and measured the stages around it; what
//! it did not measure is how the *driver's* share moves as bodies come and go, which is the part
//! that now dominates.
//!
//! Reported by the owner as a document that "takes very long to be rendered", then: "i save the
//! file with all blocks visible and re-opened it - loads super fast". Both are the same fact seen
//! from two sides — the all-visible variant had already been compiled and was being served from
//! the driver's disk cache, and the variants with things hidden had not.
//!
//! The question this has to answer before anything is built: does hiding make the shader
//! **smaller**? If it does, hiding is cheap in principle and only the first visit to each
//! combination is dear; if it does not, keeping every body in the shader and masking at run time
//! would be paying the largest shader all the time.
//!
//! Every figure is the first time that source was compiled *on this machine ever*. A combination
//! visited before comes back in about 15 ms from the driver's own cache and means nothing, which
//! is why the table says which is which.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example hide_variants -- <file.fform>`

use fieldform_scene::NodeId;

fn main() {
    let path = std::env::args().nth(1).expect("a .fform file");
    let (doc, _) = fieldform_io::open(std::path::Path::new(&path)).expect("opens");
    let part = doc.parts_ordered().first().copied().expect("a part");
    let graph = &part.graph;
    let bodies = graph.roots();

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
    println!("{} bodies in {:?}\n", bodies.len(), part.name);

    // Every subset of the bodies, smallest hidden set first, so the all-visible case leads.
    let mut variants: Vec<Vec<NodeId>> = Vec::new();
    for mask in 0u32..(1 << bodies.len().min(5)) {
        let hidden: Vec<NodeId> = bodies
            .iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, id)| *id)
            .collect();
        variants.push(hidden);
    }
    variants.sort_by_key(|h| h.len());

    println!(
        "{:<28} {:>7} {:>8} {:>7} {:>11} {:>9}",
        "hidden", "bodies", "lines", "params", "driver ms", "again"
    );
    for hidden in variants {
        let mut drawn = graph.clone();
        // Hiding takes a body away **with everything only it used**, as the viewport does.
        // Removing the node alone would leave its operands unconsumed, which makes them bodies —
        // so hiding a shelled cube would draw the solid cube it was made from.
        for id in &hidden {
            drawn.remove_body_and_its_own(*id);
        }
        let label = if hidden.is_empty() {
            "(nothing — all visible)".to_string()
        } else {
            hidden.iter().map(|n| format!("{}", n.raw() % 10000)).collect::<Vec<_>>().join(",")
        };
        let program = match fieldform_gpu::generate(&doc, &drawn) {
            Ok(p) => p,
            Err(e) => {
                println!("{label:<28} refused: {e}");
                continue;
            }
        };
        let cut = program
            .source
            .find("// ---- generated from the composition graph ----")
            .expect("marked");
        let build = || {
            let t = std::time::Instant::now();
            let _ = fieldform_gpu::Renderer::with_device(
                device.clone(),
                queue.clone(),
                &program,
                &[],
                &[],
                &[],
            );
            t.elapsed().as_secs_f64() * 1e3
        };
        let first = build();
        let second = build();
        println!(
            "{label:<28} {:>7} {:>8} {:>7} {first:>11.0} {second:>9.0}",
            drawn.roots().len(),
            program.source[cut..].lines().count(),
            program.params.len(),
        );
    }
    println!(
        "\n`driver ms` is the first compile of that source and `again` the second. Where the two \
         are equal\nthe first was already in the driver's disk cache and says nothing about what \
         it costs to reach."
    );
}
