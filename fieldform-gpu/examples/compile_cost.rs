// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the driver's pipeline build actually costs, and what that cost tracks.
//!
//! **Why this exists.** `rebuild_cost.rs` established that the driver's time is not the amount
//! of code: 158 664 characters with no pattern took 3265 ms and 160 259 with one took 662 ms on
//! an RTX A500. Its closing note proposed emitting each body as its own function; that has since
//! been built (`split_into_bodies`), and a patterned pipe still costs an M4000 **10.2 seconds**
//! under DX12 and 0.87 s under Vulkan, with the application's `BuildCost` naming the pipelines
//! rather than the module. So the question is no longer "is it big" but **what shape of source
//! the optimiser charges for**, and that is what this measures.
//!
//! The hypothesis it is built to test: **the cost is superlinear in the longest single
//! function**, not in the whole. Every row therefore prints the whole source's size *and* its
//! longest function's, so the two can be told apart by the numbers rather than by argument.
//!
//! **Reading it.** `module` is naga parsing and validating; `pipelines` is the driver turning
//! that module into the two compute pipelines. Only the second is the one in question.
//!
//! **A row is cold only once, and there is no salt that helps.** The driver caches compiled
//! shaders on disk across runs, and **the generated source is structure only** — every number is
//! a `params[..]` read, so moving a sphere or changing a radius does not alter one character of
//! it. A run of this a second time measures the cache: the first run here read 74/186/796 ms
//! where the second read 9/9/8. Comments and dead code do not help either; the driver ignores
//! both.
//!
//! So `FIELDFORM_COUNTS` takes the three counts to ladder, and the way to get a cold table is to
//! pass counts this machine has not compiled before. The default ladder is the one already
//! spent; 5/17/65 or 6/18/66 are each a fresh table. Only the *count* is structure, because a
//! pattern's count decides how its window is emitted and a chain's decides how deep it folds.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example compile_cost`

use fieldform_gpu::{generate, Renderer};
use fieldform_scene::{
    BooleanOp, CellMap, Continuity, Document, Fillet, Frame, Graph, LatticeCell, LatticePhase,
    Node, NodeId, ParamRef, PatternKind, Place,
};

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// The whole source, and the longest single function in it.
///
/// **Counted by brace depth rather than by a parser**, because the question is only how much
/// text sits inside one `fn`, and the generated source is emitted by us: every function starts
/// at column nought with `fn ` and ends at the matching brace on its own line.
fn shape_of(source: &str) -> (usize, usize, usize, usize, String) {
    let (mut functions, mut longest) = (0usize, 0usize);
    let (mut depth, mut start) = (0usize, None::<(usize, String)>);
    let mut named = String::new();
    for (at, line) in source.lines().enumerate() {
        if depth == 0 && line.starts_with("fn ") {
            functions += 1;
            let name =
                line[3..].split(['(', '<']).next().unwrap_or("?").trim().to_string();
            start = Some((at, name));
        }
        depth += line.matches('{').count();
        depth = depth.saturating_sub(line.matches('}').count());
        if depth == 0 {
            if let Some((from, name)) = start.take() {
                let size: usize =
                    source.lines().skip(from).take(at - from + 1).map(|l| l.len() + 1).sum();
                if size > longest {
                    longest = size;
                    named = name;
                }
            }
        }
    }
    let loops = source.matches("loop {").count()
        + source.matches("for (").count()
        + source.matches("while ").count();
    (source.len(), longest, functions, loops, named)
}

/// A ball, placed so that no two are the same text.
fn ball(graph: &mut Graph, x: f64, radius: f64) -> NodeId {
    graph.add(Node::Sphere {
        centre: Place::At([lit(x), lit(0.0), lit(0.0)]),
        radius: lit(radius),
    })
}

/// `count` balls as `count` separate bodies — one function each, by `split_into_bodies`.
fn bodies_apart(count: usize, salt: f64) -> (Document, Graph) {
    let mut graph = Graph::empty();
    for i in 0..count {
        ball(&mut graph, 30.0 * i as f64 + salt, 9.0);
    }
    (Document::new(), graph)
}

/// `count` balls folded into **one** body by a chain of blends: one function, `count` deep.
fn one_deep_body(count: usize, salt: f64) -> (Document, Graph) {
    let mut graph = Graph::empty();
    let mut joined = ball(&mut graph, salt, 9.0);
    for i in 1..count {
        let next = ball(&mut graph, 15.0 * i as f64 + salt, 9.0);
        joined = graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: joined.into(),
            b: next.into(),
            fillet: Fillet { radius: lit(3.0), continuity: Continuity::G1, second: None },
        });
    }
    (Document::new(), graph)
}

/// The same `count` copies as a pattern: one subtree inside a loop.
fn one_patterned_body(count: u32, salt: f64) -> (Document, Graph) {
    let mut graph = Graph::empty();
    let one = ball(&mut graph, salt, 9.0);
    graph.add(Node::Pattern {
        options: Default::default(),
        input: one.into(),
        count: ParamRef::Literal(f64::from(count)),
        kind: PatternKind::Linear { step: [lit(15.0), lit(0.0), lit(0.0)] },
        join: true,
        fillet: Fillet { radius: lit(0.0), continuity: Continuity::G1, second: None },
    });
    (Document::new(), graph)
}

/// A gyroid: a great deal of arithmetic in one function, and no repetition at all.
fn lattice(salt: f64) -> (Document, Graph) {
    let mut graph = Graph::empty();
    graph.add(Node::Lattice {
        cell: LatticeCell::Gyroid,
        frame: Frame::At {
            origin: [lit(salt), lit(0.0), lit(0.0)],
            axis: [lit(0.0), lit(0.0), lit(1.0)],
        },
        map: CellMap::Rectangular { cell: lit(20.0) },
        phase: LatticePhase::Sheet { thickness: lit(1.0), offset: lit(0.0) },
    });
    (Document::new(), graph)
}

fn main() {
    let salt: f64 = std::env::var("FIELDFORM_SALT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    println!("What the driver charges for, and what it tracks.  (salt {salt})\n");
    println!(
        "  {:>22}  {:>8}  {:>10}  {:>9}  {:>9}  {:>5}  {:>5}",
        "scene", "generate", "pipelines", "module", "chars", "fns", "loops"
    );
    println!(
        "  {:>22}  {:>8}  {:>10}  {:>9}  {:>9}  {:>5}  {:>5}",
        "", "", "", "", "longest fn", "", ""
    );
    println!();

    type Build = Box<dyn Fn(f64) -> (Document, Graph)>;
    let counts: Vec<usize> = std::env::var("FIELDFORM_COUNTS")
        .ok()
        .map(|v| v.split(',').filter_map(|n| n.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| v.len() == 3)
        .unwrap_or_else(|| vec![4, 16, 64]);
    let mut scenes: Vec<(String, Build)> = Vec::new();
    for &n in &counts {
        scenes.push((format!("{n} bodies apart"), Box::new(move |s| bodies_apart(n, s)) as Build));
    }
    for &n in &counts {
        scenes
            .push((format!("{n} blended, one body"), Box::new(move |s| one_deep_body(n, s))));
    }
    for &n in &counts[1..] {
        scenes.push((
            format!("{n} patterned (a loop)"),
            Box::new(move |s| one_patterned_body(n as u32, s)),
        ));
    }
    scenes.push(("a gyroid".into(), Box::new(lattice)));

    for (i, (label, build)) in scenes.iter().enumerate() {
        // A constant of its own per row, so no row is served the row above's compiled shader.
        let (doc, graph) = build(salt + 0.017 * (i as f64 + 1.0));
        let began = std::time::Instant::now();
        let program = match generate(&doc, &graph) {
            Ok(p) => p,
            Err(e) => {
                println!("  {label:>22}  could not generate: {e:?}");
                continue;
            }
        };
        let generate_ms = began.elapsed().as_secs_f64() * 1e3;
        let (chars, longest, functions, loops, biggest) = shape_of(&program.source);

        match pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0)) {
            Ok(renderer) => {
                let cost = renderer.build_cost();
                println!(
                    "  {label:>22}  {generate_ms:>6.0} ms  {:>8.0} ms  {:>7.0} ms  {chars:>9}  \
                     {functions:>5}  {loops:>5}",
                    cost.pipelines_ms, cost.module_ms
                );
                println!(
                    "  {:>22}  {:>8}  {:>10}  {:>9}  {longest:>9}   longest fn: {biggest}",
                    "", "", "", ""
                );
            }
            Err(e) => println!("  {label:>22}  could not build: {e:?}"),
        }
    }

    println!(
        "\n`pipelines` is the driver alone. If it tracks `longest fn` rather than `chars`, the\n\
         lever is splitting the biggest function; if it tracks neither, it is the shape of the\n\
         control flow inside it, and the pattern rows are the ones to read against the blended."
    );
}
