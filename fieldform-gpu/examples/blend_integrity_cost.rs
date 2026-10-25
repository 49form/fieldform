// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What it costs the tracer for a blend to stop overstating its distance.
//!
//! `blend-integrity-spec.md` §3, option A, divides a rounded blend's value by 2^(1/n) and a
//! rounded pattern's by W^(1/n), W being the instances its window sums. Neither moves a surface;
//! both shorten every step the march takes, *everywhere* the body is traced and not only near
//! the fillet. This measures that, on the three shapes the spec names: a single blend, a blend
//! of blends (divided twice), and a rounded pattern of seven — the one likely to hurt, since
//! √7 ≈ 2.65.
//!
//! The scenes are the spec's own: slabs with coplanar tops, which is the case that overstated.
//! Written to build against the tree on either side of the change, so the same file gives both
//! columns of the ledger row in `performance-stability-loop-spec.md` §7.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example blend_integrity_cost`

use fieldform_gpu::{generate, Camera, Renderer};
use fieldform_scene::{BooleanOp, Continuity, Document, Fillet, Graph, Node, NodeId, ParamRef};
use nalgebra::Point3;

const K: f64 = 5.0;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn fillet() -> Fillet {
    Fillet { radius: lit(K), continuity: Continuity::G1, second: None }
}

fn slab(graph: &mut Graph, x_lo: f64, x_hi: f64) -> NodeId {
    graph.add(Node::Cuboid { edge: Default::default(),
        centre: fieldform_scene::Place::At([lit((x_lo + x_hi) / 2.0), lit(0.0), lit(-10.0)]),
        size: [lit(x_hi - x_lo), lit(40.0), lit(20.0)],
        turn: fieldform_scene::no_turn(),
    })
}

fn single() -> Graph {
    let mut graph = Graph::empty();
    let a = slab(&mut graph, -40.0, 10.0);
    let b = slab(&mut graph, -10.0, 40.0);
    graph.add(Node::Blend { op: BooleanOp::Union, a: a.into(), b: b.into(), fillet: fillet() });
    graph
}

/// The single blend, with a boss blended onto it and a bore blended out of that.
fn chained() -> Graph {
    let mut graph = single();
    let body = graph.roots()[0];
    let boss = graph.add(Node::Sphere {
        centre: fieldform_scene::Place::At([lit(0.0), lit(0.0), lit(4.0)]),
        radius: lit(12.0),
    });
    let bossed = graph.add(Node::Blend { op: BooleanOp::Union, a: body.into(), b: boss.into(), fillet: fillet() });
    let bore = graph.add(Node::Sphere {
        centre: fieldform_scene::Place::At([lit(-28.0), lit(0.0), lit(0.0)]),
        radius: lit(8.0),
    });
    graph.add(Node::Blend { op: BooleanOp::Difference, a: bossed.into(), b: bore.into(), fillet: fillet() });
    graph
}

/// The same, with the bore marked critical: every fillet tapers to nothing at it (§6), which
/// costs a second evaluation of the bore per sample, a divisor of 1.83 in place of 1.41, and a
/// differenced normal on the pixels where a fillet is thinning.
fn protected() -> Graph {
    let mut graph = chained();
    let bore = graph
        .nodes()
        .find(|(_, n)| matches!(n, Node::Sphere { radius: ParamRef::Literal(r), .. } if *r == 8.0))
        .map(|(id, _)| id)
        .expect("the bore");
    graph.mark_critical(fieldform_scene::Critical::new(bore));
    graph
}

fn patterned() -> Graph {
    let mut graph = Graph::empty();
    let body = slab(&mut graph, -55.0, -5.0);
    graph.add(Node::Pattern {
        options: Default::default(),
        input: body.into(),
        count: ParamRef::Literal(7.0),
        kind: fieldform_scene::PatternKind::Linear { step: [lit(10.0), lit(0.0), lit(0.0)] },
        join: true,
        fillet: fillet(),
    });
    graph
}

fn main() {
    let doc = Document::new();
    println!("{:<22} {:>10} {:>10}", "scene", "720", "1440");
    for (label, graph) in [
        ("a single blend", single()),
        ("a blend of blends", chained()),
        ("the same, bore critical", protected()),
        ("a rounded pattern of 7", patterned()),
    ] {
        let program = generate(&doc, &graph).expect("generates");
        let renderer = match pollster::block_on(Renderer::new(&program, &[], &[], &[], 64, 1.0)) {
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

        let centre = Point3::new(0.0, 0.0, -5.0);
        let reach = 60.0;
        let camera = Camera::looking_at(
            Point3::new(reach * 1.6, reach * 1.2, reach * 0.9),
            centre,
            nalgebra::Vector3::z(),
            (centre, reach),
            45.0,
            1.0,
        );
        let mut row = format!("{label:<22}");
        for edge in [720u32, 1440] {
            // Warm, then the median of fifteen: this driver scatters run to run, and a mean
            // carries one slow frame into the third figure.
            for _ in 0..3 {
                let _ = renderer.render(&camera, edge, edge, 512);
            }
            let mut frames: Vec<f64> = (0..15)
                .map(|_| {
                    let start = std::time::Instant::now();
                    renderer.render(&camera, edge, edge, 512).expect("renders");
                    start.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            frames.sort_by(|a, b| a.total_cmp(b));
            row.push_str(&format!(" {:>7.2} ms", frames[frames.len() / 2]));
        }
        println!("{row}");
    }
}
