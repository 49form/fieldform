// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Which way of growing a model makes the driver's compile blow up.
//!
//! **Written because a real document cost 3215 ms and nobody could say why.** The owner's wing
//! compiles in seconds; every other document on the machine is fast; and the difference between
//! a sphere (685 generated lines, 28 ms) and that wing (815 lines, 3215 ms) is a hundred and
//! thirty lines. A hundred and thirty lines are not worth three seconds unless something in them
//! is superlinear, and guessing which has already been wrong twice — it is not the prelude, and
//! it is not the normal differencing.
//!
//! So this grows one thing at a time and watches. Each row is a source the driver has never seen,
//! which is the only way to get a cold reading: a shader it has compiled before comes back from
//! its own disk cache in about ten milliseconds whatever it contains.
//!
//! The four ways a model gets bigger, kept apart:
//!
//! - **more bodies** — separate solids, unioned. The composition grows; nothing nests.
//! - **more blends** — a chain of filleted joins. Each one reads two fields and divides.
//! - **more profiles** — extrudes, which each carry a polyline and a central-differenced normal.
//! - **more point changes** — mirrors and twists stacked over one body. Every block that moves
//!   the point makes its whole subtree be written again, so this is the one that multiplies
//!   rather than adds, and the one to suspect.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example compile_scaling`

use fieldform_scene::{
    BooleanOp, Continuity, Curve, CurveId, Document, Fillet, Frame, Graph, Node, NodeId, ParamRef,
    Place, Turn,
};
use nalgebra::Point3;

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
    println!("{}\n", device.adapter_info().name);

    let series: Vec<(&str, fn(usize) -> (Document, Graph), Vec<usize>)> = vec![
        ("bodies unioned", bodies, vec![1, 2, 4, 8, 16]),
        ("blends chained", blends, vec![1, 2, 4, 8]),
        ("extrudes unioned", extrudes, vec![1, 2, 4, 8]),
        ("point changes stacked", point_changes, vec![0, 1, 2, 3, 4]),
        // **The wing, one construct at a time.** None of the series above reaches a tenth of
        // what the owner's document costs at the same size, so the cost is a particular block
        // rather than the amount of them. `n` selects how much of that document's shape is
        // rebuilt: 1 revolve, 2 + offset, 3 + extrude, 4 + twist, 5 + mirror, 6 + the blend and
        // the boolean that join them, which is the whole of it.
        ("the wing, built up", wing_up_to, vec![1, 2, 3, 4, 5, 6]),
    ];

    for (label, build, counts) in series {
        println!("{label}");
        println!("  {:>5} {:>8} {:>8} {:>11}", "n", "lines", "params", "driver ms");
        for n in counts {
            let (doc, graph) = build(n);
            let program = match fieldform_gpu::generate(&doc, &graph) {
                Ok(p) => p,
                Err(e) => {
                    println!("  {n:>5} refused: {e}");
                    continue;
                }
            };
            let cut = program
                .source
                .find("// ---- generated from the composition graph ----")
                .expect("marked");
            let t = std::time::Instant::now();
            let built = fieldform_gpu::Renderer::with_device(
                device.clone(),
                queue.clone(),
                &program,
                &[],
                &[],
                &[],
            );
            let ms = t.elapsed().as_secs_f64() * 1e3;
            match built {
                Ok(_) => println!(
                    "  {n:>5} {:>8} {:>8} {ms:>11.0}",
                    program.source[cut..].lines().count(),
                    program.params.len()
                ),
                Err(e) => println!("  {n:>5} refused by the driver after {ms:.0} ms: {e}"),
            }
        }
        println!();
    }
    // **What the operand masks cost the driver, cold, both ways.** No frame harness can answer
    // this any more: once a source has been compiled on this machine it comes back from the
    // driver's own disk cache in about ten milliseconds whatever it contains. So a series whose
    // every row is a shape this machine has never seen, each generated twice -- with the masks
    // and without -- and the two compiled one after the other on one device.
    //
    // Pipes on their own spokes, which is the shape the masks are for and the one the gate in
    // `OperandMasks` keeps: each pipe is read by the union alone, so each has an exclusive
    // weight of its own.
    println!("the masks' own compile, cold, both ways (pipes on spokes, unioned)");
    println!(
        "  {:>5} {:>8} {:>8} {:>11} {:>11} {:>8}",
        "pipes", "lines on", "off", "driver on", "driver off", "ratio"
    );
    for n in [2usize, 4, 8, 12, 16] {
        let (doc, graph) = spokes(n);
        let mut figures = Vec::new();
        for masked in [true, false] {
            let program = fieldform_gpu::generate_with_masks(&doc, &graph, masked)
                .expect("generates");
            let cut = program
                .source
                .find("// ---- generated from the composition graph ----")
                .expect("marked");
            let lines = program.source[cut..].lines().count();
            let t = std::time::Instant::now();
            let built = fieldform_gpu::Renderer::with_device(
                device.clone(),
                queue.clone(),
                &program,
                &[],
                &[],
                &[],
            );
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if built.is_err() {
                println!("  {n:>5} refused by the driver after {ms:.0} ms");
            }
            figures.push((lines, ms));
        }
        println!(
            "  {n:>5} {:>8} {:>8} {:>11.0} {:>11.0} {:>7.2}x",
            figures[0].0,
            figures[1].0,
            figures[0].1,
            figures[1].1,
            figures[0].1 / figures[1].1.max(1e-9),
        );
    }

    println!(
        "\nEvery row is a source this machine had never compiled, so every figure is cold. A\n\
         series that doubles its time when it doubles its size is linear and fine; one that\n\
         quadruples is the thing to fix."
    );
}

/// `n` bent pipes on their own spokes, meeting only near the hub, in one union -- the shape
/// `fieldform-io/examples/framed_capture.rs` measured the operand masks on. The spine's points
/// carry `n` so that no two counts generate a source the driver has already seen.
fn spokes(n: usize) -> (Document, Graph) {
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    let mut pipes = Vec::new();
    for i in 0..n {
        let turn = std::f64::consts::TAU * i as f64 / n as f64;
        let (c, s) = (turn.cos(), turn.sin());
        let id = CurveId::new();
        let at = |r: f64, z: f64| Point3::new(r * c, r * s, z);
        doc.curves.insert(
            id,
            Curve::polyline(
                id,
                "spine",
                vec![at(20.0, 0.0), at(80.0, 25.0), at(140.0, -10.0), at(200.0, 15.0)],
                false,
            )
            .expect("a bent line"),
        );
        pipes.push(graph.add(Node::Pipe { curve: id, radius: lit(9.0) }));
    }
    graph.add(Node::Boolean { op: BooleanOp::Union, inputs: pipes.into_iter().map(Into::into).collect() });
    (doc, graph)
}

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn ball(g: &mut Graph, i: usize) -> NodeId {
    g.add(Node::Sphere {
        centre: Place::At([lit(i as f64 * 30.0), lit(0.0), lit(0.0)]),
        radius: lit(20.0),
    })
}

fn bodies(n: usize) -> (Document, Graph) {
    let mut g = Graph::empty();
    let inputs: Vec<NodeId> = (0..n).map(|i| ball(&mut g, i)).collect();
    if inputs.len() > 1 {
        g.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: inputs.into_iter().map(Into::into).collect(),
        });
    }
    (Document::new(), g)
}

fn blends(n: usize) -> (Document, Graph) {
    let mut g = Graph::empty();
    let mut acc = ball(&mut g, 0);
    for i in 1..=n {
        let next = ball(&mut g, i);
        acc = g.add(Node::Blend {
            op: BooleanOp::Union,
            a: acc.into(),
            b: next.into(),
            fillet: Fillet { radius: lit(6.0), continuity: Continuity::G1, second: None },
        });
    }
    (Document::new(), g)
}

fn extrudes(n: usize) -> (Document, Graph) {
    let mut doc = Document::new();
    let mut g = Graph::empty();
    let mut inputs = Vec::new();
    for i in 0..n {
        let id = CurveId::new();
        let x = i as f64 * 80.0;
        doc.curves.insert(
            id,
            Curve::polyline(
                id,
                "section",
                vec![
                    Point3::new(x, -15.0, 0.0),
                    Point3::new(x + 30.0, -15.0, 0.0),
                    Point3::new(x + 30.0, 15.0, 0.0),
                    Point3::new(x, 15.0, 0.0),
                ],
                true,
            )
            .expect("a rectangle"),
        );
        inputs.push(g.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Normal,
            height: lit(40.0),
            draft: lit(0.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            corners: Default::default(),
        }));
    }
    if inputs.len() > 1 {
        g.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: inputs.into_iter().map(Into::into).collect(),
        });
    }
    (doc, g)
}

/// The owner's wing, rebuilt a block at a time, so the one that costs can be named.
fn wing_up_to(stage: usize) -> (Document, Graph) {
    let mut doc = Document::new();
    let mut g = Graph::empty();

    // A closed section in the XZ plane for the revolve to turn into a fuselage.
    let section = CurveId::new();
    doc.curves.insert(
        section,
        Curve::polyline(
            section,
            "fuselage section",
            vec![
                Point3::new(60.0, 0.0, -300.0),
                Point3::new(140.0, 0.0, -300.0),
                Point3::new(140.0, 0.0, 300.0),
                Point3::new(60.0, 0.0, 300.0),
            ],
            true,
        )
        .expect("a rectangle"),
    );
    let fuselage = g.add(Node::Revolve { rims: Default::default(), outline: fieldform_scene::Outline::closed(section), frame: Frame::default(), angle: fieldform_scene::ParamRef::Literal(360.0), sides: fieldform_scene::AngleSides::One });
    if stage == 1 {
        return (doc, g);
    }

    // A shell of it, which is what the wing's document does: an offset subtracted from the body.
    let bore = g.add(Node::Offset { input: fuselage.into(), distance: lit(-6.0) });
    if stage == 2 {
        g.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![fuselage.into(), bore.into()] });
        return (doc, g);
    }

    let wing_profile = CurveId::new();
    doc.curves.insert(
        wing_profile,
        Curve::polyline(
            wing_profile,
            "wing section",
            vec![
                Point3::new(0.0, 200.0, -20.0),
                Point3::new(400.0, 200.0, -20.0),
                Point3::new(400.0, 200.0, 20.0),
                Point3::new(0.0, 200.0, 20.0),
            ],
            true,
        )
        .expect("a rectangle"),
    );
    let wing = g.add(Node::Extrude { rims: Default::default(),
        outline: fieldform_scene::Outline::closed(wing_profile),
        sweep: fieldform_scene::Sweep::Normal,
        height: lit(1500.0),
        draft: lit(0.0),
        start: lit(0.0),
        sides: fieldform_scene::Sides::One,
        corners: Default::default(),
    });
    if stage == 3 {
        return (doc, g);
    }

    let twisted = g.add(Node::Twist {
        input: wing.into(),
        degrees_per_mm: lit(0.01),
        frame: Frame::default(),
    });
    if stage == 4 {
        return (doc, g);
    }

    let both = g.add(Node::Mirror {
        input: twisted.into(),
        frame: Frame::At {
            origin: [lit(0.0); 3],
            axis: [lit(0.0), lit(1.0), lit(0.0)],
        },
        join: true,
    });
    if stage == 5 {
        return (doc, g);
    }

    let joined = g.add(Node::Blend {
        op: BooleanOp::Union,
        a: fuselage.into(),
        b: both.into(),
        fillet: Fillet { radius: lit(30.0), continuity: Continuity::G1, second: None },
    });
    g.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![joined.into(), bore.into()] });
    (doc, g)
}

/// Mirrors and twists stacked over one solid. Every one of them moves the point, so the body
/// under it is written out again — this is the series that multiplies.
fn point_changes(n: usize) -> (Document, Graph) {
    let mut g = Graph::empty();
    let mut acc = g.add(Node::Cuboid { edge: Default::default(),
        centre: Place::At([lit(40.0), lit(0.0), lit(0.0)]),
        size: [lit(30.0), lit(30.0), lit(60.0)],
        turn: Turn::default(),
    });
    for i in 0..n {
        acc = if i % 2 == 0 {
            g.add(Node::Mirror {
                input: acc.into(),
                frame: Frame::At {
                    origin: [lit(0.0); 3],
                    axis: [lit(1.0), lit(0.0), lit(0.0)],
                },
                join: true,
            })
        } else {
            g.add(Node::Twist {
                input: acc.into(),
                degrees_per_mm: lit(0.3),
                frame: Frame::default(),
            })
        };
    }
    (Document::new(), g)
}
