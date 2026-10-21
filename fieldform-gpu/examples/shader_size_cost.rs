// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where the generated shader stops being something the driver will take — plan step 13.
//!
//! **Because nothing warned before a driver refused one.** MPR's hard ceiling was a per-thread
//! register and tape budget (§4.6 of the IP research note kept outside the repository); here
//! it is the driver's compile of a
//! generated shader, and `union-compile-8fa06b7.md` measured that cost at a power of about 2 to 2.5
//! in the size of one entry point after inlining. Sixteen unioned pipes reached **123 seconds**, so
//! the wall is real and was not found on a customer's graph.
//!
//! This walks the size up until a row **fails or exceeds two seconds**, which
//! `tile-pruning-spec.md` §4 defines as the documented limit. It walks upward and stops, rather
//! than measuring every size, because the rows past the limit are minutes each and say nothing the
//! limit does not.
//!
//! Three arms:
//!
//! - **bodies** — `n` spheres unioned. The cheapest body there is, so this isolates the count from
//!   what a body contains.
//! - **nested ops** — `n` offsets stacked over one sphere. One graph, one body, a deep expression.
//! - **refused** — a shader that is valid WGSL and asks for more workgroup storage than the device
//!   allows. **The control**: it must come back as an error rather than a panic. A run where
//!   nothing fails otherwise proves only that the limit is far away; this proves the error scope in
//!   `Renderer::build` is wired at all, which is the half of step 13 that stops a crash.
//!
//! **Measured on an RTX A500 Laptop GPU, Vulkan, 2026-09-26.** Cold pipeline in milliseconds:
//!
//! | n | bodies: KB / compile | nested ops: KB / compile |
//! |---|---|---|
//! | 10 | 162 / 124 | 159 / 136 |
//! | 25 | 171 / 249 | 161 / 148 |
//! | 50 | 185 / 582 | 165 / 198 |
//! | 100 | 214 / **1 826** | 174 / 611 |
//! | 200 | 273 / **8 148** | 192 / 1 363 |
//! | 400 | — | 227 / **2 056** |
//!
//! **So the documented limit is 200 bodies, or 400 stacked ops**, taking the spec's two seconds as
//! the line. A hundred bodies is already 1.8 s, so a hundred is the practical ceiling for a
//! rebuild that happens while somebody waits — and that is *cheap* bodies, one sphere each. Sixteen
//! unioned pipes were 123 s before the union loop, which is the same wall reached with 16 operands
//! rather than 200, because what costs is the inlined size and a pipe carries a spine walk.
//!
//! **Read the arms apart.** A body is a root nothing consumes, so the `bodies` arm grows the top
//! level fold; the `nested ops` arm is one body whose expression gets longer. They cost differently:
//! 200 bodies is 8.1 s where 200 stacked ops is 1.4 s, so a fold over many roots is dearer per node
//! than a chain inside one. That is what step 11's per-tile body mask is aimed at.
//!
//! **The `points` column stays at 1 in both arms**, correctly — neither builds a point changer. A
//! graph with a mirror over a twist reads deeper, and that multiplier is the thing a node count
//! cannot show; `a_programs_size_report_counts_bodies_nodes_and_the_point_chain` holds it.
//!
//! **A re-run measures the driver's cache, not the compile**, at about twenty milliseconds a row.
//! Walk the ladder once per machine; if a row has to be re-read cold, change its count.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example shader_size_cost`

use fieldform_scene::{Document, Graph, Node, ParamRef, Place};

/// What `tile-pruning-spec.md` §4 calls the limit: the first size whose compile passes this.
const BUDGET_MS: f64 = 2000.0;

fn main() {
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
    let info = device.adapter_info();
    println!("{} on {:?}", info.name, info.backend);
    println!(
        "workgroup storage this device allows: {} bytes\n",
        device.limits().max_compute_workgroup_storage_size
    );

    // The control first, so a run that says nothing about the limit still says whether a refusal
    // is survivable. See the module comment.
    refused_is_an_error(&device, &queue);

    let arms: Vec<(&str, fn(usize) -> (Document, Graph))> =
        vec![("bodies", spheres), ("nested ops", nested_offsets)];
    for (label, build) in arms {
        println!("=== {label} ===");
        println!(
            "  {:>6} {:>8} {:>7} {:>7} {:>7} {:>11} {:>11}",
            "n", "KB", "bodies", "nodes", "points", "generate ms", "compile ms"
        );
        for n in [10usize, 25, 50, 100, 200, 400, 800, 1600] {
            let generating = std::time::Instant::now();
            let (doc, graph) = build(n);
            let program = match fieldform_gpu::generate(&doc, &graph) {
                Ok(p) => p,
                Err(e) => {
                    println!("  {n:>6} refused by codegen: {e}");
                    break;
                }
            };
            let generate_ms = generating.elapsed().as_secs_f64() * 1e3;
            let size = program.size;

            let compiling = std::time::Instant::now();
            let built = fieldform_gpu::Renderer::with_device(
                device.clone(),
                queue.clone(),
                &program,
                &[],
                &[],
                &[],
            );
            let compile_ms = compiling.elapsed().as_secs_f64() * 1e3;
            match built {
                Ok(_) => println!(
                    "  {n:>6} {:>8} {:>7} {:>7} {:>7} {generate_ms:>11.0} {compile_ms:>11.0}",
                    size.bytes / 1024,
                    size.bodies,
                    size.nodes,
                    size.deepest_point_chain,
                ),
                Err(e) => {
                    println!("  {n:>6} {:>8} REFUSED after {compile_ms:.0} ms", size.bytes / 1024);
                    println!("         {e}");
                    println!("  the documented limit for {label} is below {n}");
                    break;
                }
            }
            if compile_ms > BUDGET_MS {
                println!(
                    "  the documented limit for {label} is {n}: the first size past {BUDGET_MS:.0} ms"
                );
                break;
            }
        }
        println!();
    }

    println!(
        "Every compile figure is cold only the first time this runs: the driver serves a source it \n\
         has seen from its own cache in about twenty milliseconds. A re-run measures the cache."
    );
}

/// **The control.** A program whose source is valid WGSL and which the device cannot make a
/// pipeline from, so the one thing step 13 exists to prevent — a refusal reaching the process as a
/// panic — is exercised on every run rather than only on a graph big enough to find it.
///
/// A workgroup **size** past the device's invocation limit, rather than a huge graph: it is
/// instant, and it is *certain*, where "big enough that this driver gives up" is a number that
/// moves with the driver.
///
/// **It was oversized workgroup *storage* first, and that tested nothing.** An extra entry point
/// declaring a huge `var<workgroup>` was appended, and the control reported itself failed — as it is
/// written to. The reason is worth keeping: wgpu validates workgroup storage **per pipeline**, and
/// this renderer only ever makes pipelines for `march` and `skip_build`, so an entry point nothing
/// compiles is an entry point nothing checks. Patching `march`'s own attribute cannot miss for that
/// reason.
fn refused_is_an_error(device: &wgpu::Device, queue: &wgpu::Queue) {
    let allowed = device.limits().max_compute_invocations_per_workgroup;
    let mut program = fieldform_gpu::generate(
        &Document::new(),
        &{
            let mut g = Graph::empty();
            g.add(Node::Sphere {
                centre: Place::At([ParamRef::Literal(0.0); 3]),
                radius: ParamRef::Literal(10.0),
            });
            g
        },
    )
    .expect("a sphere generates");
    // The march's own workgroup, widened past what the device will run. `8, 8, 1` is what
    // `march.wgsl` declares; 1024 x 64 is 65 536 invocations against a limit that is 1 024 on every
    // device this runs on.
    //
    // Matched without the newline after it, because the sources are CRLF and a pattern carrying
    // `\n` matched nothing — which the assertion below caught rather than letting the control pass
    // on an unmodified program, for the wrong reason.
    //
    // **Anchored on the entry point's name, not on the attribute alone.** This asserted once that
    // the march was the only 8 x 8 workgroup in a program, and the draft-tier split (`21fca7a`)
    // made that false — `march_still` is 8 x 8 too — so from that commit the control panicked
    // before it tested anything. The step-11 design review found it, not a run. It is the pattern
    // `tile-mask-design-01941c5.md` Part 2 closes on: a second path added beside the first, and a
    // property only the first one had. So the patch now names `fn march(`, whose pipeline
    // `Renderer::build` makes eagerly, and replaces the attribute immediately before it and no
    // other.
    const MARCH_WORKGROUP: &str = "@compute @workgroup_size(8, 8, 1)";
    let entry = program
        .source
        .find("fn march(@builtin")
        .expect("the draft march is an entry point");
    let attribute = program.source[..entry]
        .rfind(MARCH_WORKGROUP)
        .expect("the draft march declares its workgroup just above itself");
    assert!(
        entry - attribute < 64,
        "the attribute found is the draft march's own, not one further up the file"
    );
    program
        .source
        .replace_range(attribute..attribute + MARCH_WORKGROUP.len(), "@compute @workgroup_size(1024, 64, 1)");
    match fieldform_gpu::Renderer::with_device(
        device.clone(),
        queue.clone(),
        &program,
        &[],
        &[],
        &[],
    ) {
        Err(e) => {
            println!("control: a refused pipeline came back as an error, not a panic.");
            println!("  {e}\n");
        }
        Ok(_) => {
            println!(
                "control: FAILED — the device made a pipeline for a workgroup of 65 536 \n\
                 invocations where it says it allows {allowed}. This arm is not testing what it \n\
                 claims; fix it before reading anything below.\n"
            );
        }
    }
}

/// `n` spheres, each left a **body** of its own — the cheapest body there is, so the count is what
/// varies.
///
/// Unconsumed rather than unioned. They were unioned first, and the size report said `bodies: 1` at
/// every count — correctly, since a union is one root and therefore one body. That arm was varying a
/// boolean's *operand* count, which is a different thing and is already measured in
/// `union_loop_cost.rs`. A body is a root nothing consumes, so leaving them alone is what makes this
/// the arm step 11 cares about.
fn spheres(n: usize) -> (Document, Graph) {
    let mut g = Graph::empty();
    let inputs = (0..n)
        .map(|i| {
            let (x, y) = ((i % 40) as f64 * 30.0, (i / 40) as f64 * 30.0);
            g.add(Node::Sphere {
                centre: Place::At([
                    ParamRef::Literal(x),
                    ParamRef::Literal(y),
                    ParamRef::Literal(0.0),
                ]),
                radius: ParamRef::Literal(12.0),
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(inputs.len(), n, "every sphere is its own body");
    (Document::new(), g)
}

/// `n` offsets stacked over one sphere — one body, one long expression, no extra point contexts.
fn nested_offsets(n: usize) -> (Document, Graph) {
    let mut g = Graph::empty();
    let mut acc = g.add(Node::Sphere {
        centre: Place::At([ParamRef::Literal(0.0); 3]),
        radius: ParamRef::Literal(10.0),
    });
    for _ in 0..n {
        // Alternating signs, so the body neither grows without bound nor shrinks away — a shape
        // that vanished would be measuring an empty field's compile.
        acc = g.add(Node::Offset { input: acc.into(), distance: ParamRef::Literal(0.01) });
        acc = g.add(Node::Offset { input: acc.into(), distance: ParamRef::Literal(-0.01) });
    }
    (Document::new(), g)
}
