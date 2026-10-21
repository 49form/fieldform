// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Holds the pruned pattern loop to the unpruned one, bit for bit.
//!
//! Plan §5 V3: a pattern's loop skips an instance whose sub-field cannot read below the running
//! minimum, and "the pruned field equals the unpruned field **bitwise** at every sample. That is
//! the test, not an epsilon." A generated field runs nowhere but on the GPU, so that is where it
//! is evaluated: both programs on a 64³ lattice over the scene's box grown by a fifth, read back,
//! and compared by their bits. One point differing is a wrong prune, and the fix is the prune.
//!
//! Real documents are read from disk, so the viewport's own compile path is what is tested; the
//! synthetic scenes are here because a real document cannot be relied on to hold the cases that
//! catch a shortcut — a ring whose copies reach well into their neighbours' sectors, so that
//! "only the nearest instance" would be wrong, a row walked through its window, and a rounded
//! ring, whose fold is a sum and must be pruned by a different rule than a minimum.
//!
//! Run with:
//! `cargo run --release -p fieldform-io --features march-stats --example pattern_prune_equality -- <file.fieldform>…`

use fieldform_gpu::{generate_with_options, CurveUse, Renderer};
use fieldform_scene::{
    Continuity, Document, Fillet, Frame, Graph, Node, ParamRef, PatternKind, Place,
};
use nalgebra::Point3;

const GRID: u32 = 64;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// Eight long bars about Z, each reaching well into the sectors either side of its own.
fn overlapping_ring(fillet: Fillet) -> (Document, Graph) {
    let mut graph = Graph::empty();
    let bar = graph.add(Node::Cuboid {
        centre: Place::At([lit(40.0), lit(0.0), lit(0.0)]),
        // Sixty long across a sector that is thirty-one wide at this radius.
        size: [lit(10.0), lit(60.0), lit(10.0)],
        turn: fieldform_scene::no_turn(),
    });
    graph.add(Node::Pattern {
        options: Default::default(),
        input: bar.into(),
        count: ParamRef::Literal(8.0),
        kind: PatternKind::Radial { frame: Frame::default() },
        join: true,
        fillet,
    });
    (Document::new(), graph)
}

/// Four bars in a row, each six steps long, so the window covers the whole row and the fold
/// walks every instance — nearest first, outward, skipping what falls off either end.
fn short_row() -> (Document, Graph) {
    let mut graph = Graph::empty();
    let bar = graph.add(Node::Cuboid {
        centre: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
        size: [lit(60.0), lit(8.0), lit(8.0)],
        turn: fieldform_scene::no_turn(),
    });
    graph.add(Node::Pattern {
        options: Default::default(),
        input: bar.into(),
        count: ParamRef::Literal(4.0),
        kind: PatternKind::Linear { step: [lit(10.0), lit(3.0), lit(0.0)] },
        join: true,
        fillet: Fillet::hard(),
    });
    (Document::new(), graph)
}

/// Twelve bars in a row, three steps long each, so the fold walks a window rather than the row.
fn overlapping_row() -> (Document, Graph) {
    let mut graph = Graph::empty();
    let bar = graph.add(Node::Cuboid {
        centre: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
        size: [lit(30.0), lit(8.0), lit(8.0)],
        turn: fieldform_scene::no_turn(),
    });
    graph.add(Node::Pattern {
        options: Default::default(),
        input: bar.into(),
        count: ParamRef::Literal(12.0),
        kind: PatternKind::Linear { step: [lit(10.0), lit(3.0), lit(0.0)] },
        join: true,
        fillet: Fillet::hard(),
    });
    (Document::new(), graph)
}

/// Eight long bars over three quarters of a turn, centred on the body, two left out — the options
/// that change which copy a point belongs to (an arc does not wrap) and which are there at all. The
/// prune tests a copy's floor *after* the skip has passed it over, so a skipped copy must never be
/// the reason a pruned loop differs from an unpruned one.
fn skipping_arc(fillet: Fillet) -> (Document, Graph) {
    let mut graph = Graph::empty();
    let bar = graph.add(Node::Cuboid {
        centre: Place::At([lit(40.0), lit(0.0), lit(0.0)]),
        size: [lit(10.0), lit(60.0), lit(10.0)],
        turn: fieldform_scene::no_turn(),
    });
    graph.add(Node::Pattern {
        input: bar.into(),
        count: ParamRef::Literal(9.0),
        kind: PatternKind::Radial { frame: Frame::default() },
        join: false,
        fillet,
        options: fieldform_scene::PatternOptions {
            angle: lit(270.0),
            symmetric: true,
            // Nine is odd, so copy 4 stands on the body; these are two others.
            skip: vec![1, 6],
            ..Default::default()
        },
    });
    (Document::new(), graph)
}

/// Fourteen bars along a centred whole run with three left out: the window is a few cells wide
/// and the bias is part of every cell it names.
fn centred_run_with_gaps() -> (Document, Graph) {
    let mut graph = Graph::empty();
    let bar = graph.add(Node::Cuboid {
        centre: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
        size: [lit(30.0), lit(8.0), lit(8.0)],
        turn: fieldform_scene::no_turn(),
    });
    graph.add(Node::Pattern {
        input: bar.into(),
        count: ParamRef::Literal(14.0),
        kind: PatternKind::Linear { step: [lit(130.0), lit(39.0), lit(0.0)] },
        join: true,
        fillet: Fillet::hard(),
        options: fieldform_scene::PatternOptions {
            spacing: fieldform_scene::Spacing::Extent,
            symmetric: true,
            skip: vec![2, 3, 9],
            ..Default::default()
        },
    });
    (Document::new(), graph)
}

/// A grid of bars at two oblique steps, centred, with copies left out: both directions windowed,
/// and a loop that walks a window of a window. The prune tests each copy's floor after the skip has
/// passed it over, and the loop's order is rows then columns, which has to come out the same
/// pruned and not.
fn skewed_grid() -> (Document, Graph) {
    let mut graph = Graph::empty();
    let bar = graph.add(Node::Cuboid {
        centre: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
        size: [lit(14.0), lit(6.0), lit(6.0)],
        turn: fieldform_scene::no_turn(),
    });
    graph.add(Node::Pattern {
        input: bar.into(),
        count: ParamRef::Literal(11.0),
        kind: PatternKind::Linear { step: [lit(16.0), lit(3.0), lit(0.0)] },
        join: true,
        fillet: Fillet::hard(),
        options: fieldform_scene::PatternOptions {
            symmetric: true,
            second: Some(fieldform_scene::PatternSecond {
                step: [lit(4.0), lit(11.0), lit(2.0)],
                count: ParamRef::Literal(9.0),
            }),
            skip: vec![7, 40, 41],
            ..Default::default()
        },
    });
    (Document::new(), graph)
}

/// What the generator said of each pattern loop in a program's source.
fn said(source: &str) -> Vec<String> {
    let mut out: Vec<String> = source
        .lines()
        .filter_map(|l| l.trim().strip_prefix("// pattern ").map(str::to_string))
        .collect();
    out.dedup();
    out
}

/// The field of `graph` on the lattice, pruned or not.
///
/// `None` for a graph this check cannot feed: it supplies spines, and nothing baked.
fn sample(
    doc: &Document,
    graph: &Graph,
    prune: bool,
    bounds: (Point3<f64>, Point3<f64>),
    model: (Point3<f64>, Point3<f64>),
    rays: bool,
) -> Option<(Vec<f32>, Vec<String>, Option<fieldform_gpu::stats::MarchCapture>)> {
    let program = generate_with_options(doc, graph, true, prune).ok()?;
    if !program.curves.iter().all(|c| matches!(c, CurveUse::Spine(_))) {
        return None;
    }
    let size = (bounds.1 - bounds.0).norm();
    let floor = (size * 1e-5).max(1e-4);
    let curves: Vec<(Vec<Point3<f64>>, Option<fieldform_field::BakedProfile>)> = program
        .curves
        .iter()
        .map(|use_of| match use_of {
            // The viewport's own sampling of a spine at its middle rung.
            CurveUse::Spine(id) => (doc.curves[id].tessellate_to_angle(1.0, floor), None),
            other => panic!("this check supplies spines only, and the graph wants {other:?}"),
        })
        .collect();
    let renderer = pollster::block_on(Renderer::new(&program, &[], &curves, &[], 32, 1.0))
        .expect("a GPU, which is where a generated field runs");
    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
        .collect();
    renderer.set_params(&values);
    // The sphere each pruned loop tests against, read back out of the slots the generator named,
    // beside the scene it sits in: a sphere as wide as the scene can prune nothing inside it.
    let mut spheres = Vec::new();
    for (at, _) in program.source.match_indices("if (pattern_floor(q") {
        let call = &program.source[at..];
        let slot = call.find(", ").and_then(|c| call[c + 2..].split('u').next()).and_then(|n| n.parse::<usize>().ok());
        if let Some(slot) = slot {
            let v = &values[slot..slot + 5];
            spheres.push(format!(
                "sphere at ({:.1}, {:.1}, {:.1}) radius {:.1}, divisor {:.3}, in a lattice {:.1} corner to corner",
                v[0], v[1], v[2], v[3], v[4], size
            ));
        }
    }
    spheres.dedup();
    for sphere in &spheres {
        println!("    {sphere}");
    }
    let grid = renderer.field_on_grid(bounds.0, bounds.1, GRID).expect("the grid evaluates");
    // And the march itself, from the three-quarter view: what each ray read is a sample of the
    // field along a line no lattice visits, so two fields that agree to the bit take the same
    // steps on every ray. Only asked for on the model's own box.
    let march = rays.then(|| {
        let camera = fieldform_gpu::Camera::framing(model, 1.5);
        renderer.march_stats(&camera, 1080, 720, 1233).expect("the march runs")
    });
    Some((grid, said(&program.source), march))
}

/// Compares the two fields of one scene and says what it found. Returns how many points differ.
fn check(label: &str, doc: &Document, graph: &Graph, margin: f64) -> usize {
    let Some((lo, hi)) = fieldform_field::part_extent(doc, graph).as_box() else {
        println!("| {label} | — | — | skipped: the scene has no finite bounds |");
        return 0;
    };
    let rays = margin < 1.0;
    let margin = (hi - lo) * margin;
    let bounds = (lo - margin, hi + margin);
    let (Some((pruned, how, pruned_march)), Some((plain, _, plain_march))) = (
        sample(doc, graph, true, bounds, (lo, hi), rays),
        sample(doc, graph, false, bounds, (lo, hi), rays),
    )
    else {
        println!("| {label} | — | — | skipped: the graph wants more than spines, which is all this check supplies |");
        return 0;
    };
    let differ = pruned.iter().zip(&plain).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    // Where and by how much, for whoever has to fix the prune: a pruned value above the plain
    // one is an instance skipped that should have been taken.
    let n = GRID as usize;
    for (i, (a, b)) in pruned.iter().zip(&plain).enumerate().filter(|(_, (a, b))| a.to_bits() != b.to_bits()).take(6) {
        let (x, y, z) = (i % n, (i / n) % n, i / (n * n));
        let at = |k: usize, lo: f64, hi: f64| lo + (hi - lo) * k as f64 / (n - 1) as f64;
        println!(
            "    differs at ({:.2}, {:.2}, {:.2}): pruned {a} against {b}",
            at(x, bounds.0.x, bounds.1.x),
            at(y, bounds.0.y, bounds.1.y),
            at(z, bounds.0.z, bounds.1.z),
        );
    }
    if differ > 0 {
        let up = pruned.iter().zip(&plain).filter(|(a, b)| a > b).count();
        let down = pruned.iter().zip(&plain).filter(|(a, b)| a < b).count();
        let ulps = pruned.iter().zip(&plain).map(|(a, b)| (a.to_bits() as i64 - b.to_bits() as i64).abs()).max().unwrap_or(0);
        println!("    pruned above plain at {up}, below at {down}; the widest gap is {ulps} in the last place");
    }
    if let (Some(a), Some(b)) = (&pruned_march, &plain_march) {
        let rays_off = a.steps.iter().zip(&b.steps).filter(|(x, y)| x != y).count()
            + a.exits.iter().zip(&b.exits).filter(|(x, y)| x != y).count();
        let (ra, rb): (u64, u64) =
            (a.steps.iter().map(|s| u64::from(*s)).sum(), b.steps.iter().map(|s| u64::from(*s)).sum());
        let (ca, cb): (u64, u64) =
            (a.calls.iter().map(|s| u64::from(*s)).sum(), b.calls.iter().map(|s| u64::from(*s)).sum());
        println!(
            "    1080 × 720 rays: {rays_off} take different steps; readings {ra} pruned, {rb} plain;              sub-field calls a reading {:.2} pruned, {:.2} plain",
            ca as f64 / ra.max(1) as f64,
            cb as f64 / rb.max(1) as f64,
        );
    }
    let inside = plain.iter().filter(|v| **v < 0.0).count();
    println!(
        "| {label} | {GRID}³ = {} points, {inside} inside | **{differ} differ** | {} |",
        plain.len(),
        if how.is_empty() { "no pattern in this graph".to_string() } else { how.join("; ") }
    );
    differ
}

fn main() {
    let mut scenes: Vec<(String, Document, Graph)> = Vec::new();
    for path in std::env::args().skip(1) {
        match fieldform_io::open(std::path::Path::new(&path)) {
            Ok((doc, _)) => {
                let name = std::path::Path::new(&path)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or(path.clone());
                // Every part with anything in it, as the viewport would draw it.
                let graphs: Vec<Graph> =
                    doc.parts.values().filter(|p| !p.graph.roots().is_empty()).map(|p| p.graph.clone()).collect();
                for graph in graphs {
                    scenes.push((format!("`{name}`"), doc.clone(), graph));
                }
            }
            Err(e) => println!("could not open {path}: {e}"),
        }
    }
    let (doc, graph) = overlapping_ring(Fillet::hard());
    scenes.push(("synthetic: eight bars about Z, each overlapping its neighbours' sectors".into(), doc, graph));
    let (doc, graph) = overlapping_ring(Fillet { radius: lit(4.0), continuity: Continuity::G1 });
    scenes.push(("synthetic: the same ring, copies rounded into each other at 4 mm".into(), doc, graph));
    let (doc, graph) = short_row();
    scenes.push(("synthetic: four long bars in a short row".into(), doc, graph));
    let (doc, graph) = overlapping_row();
    scenes.push(("synthetic: twelve bars in a row, three steps long each".into(), doc, graph));
    let (doc, graph) = skipping_arc(Fillet::hard());
    scenes.push(("synthetic: nine bars over an arc, centred, two skipped".into(), doc, graph));
    let (doc, graph) = skipping_arc(Fillet { radius: lit(4.0), continuity: Continuity::G1 });
    scenes.push(("synthetic: the same arc, copies rounded into each other at 4 mm".into(), doc, graph));
    let (doc, graph) = centred_run_with_gaps();
    scenes.push(("synthetic: fourteen bars along a centred whole run, three skipped".into(), doc, graph));
    let (doc, graph) = skewed_grid();
    scenes.push(("synthetic: a centred grid of bars at two oblique steps, three skipped".into(), doc, graph));

    println!("| scene | lattice | pruned against unpruned | what the generator did |");
    println!("|---|---|---|---|");
    let mut wrong = 0;
    for (label, doc, graph) in &scenes {
        // The brief's fifth, and then a box seven times as wide: a ray that misses the model
        // goes on reading the field out to the far plane, a long way past any fifth.
        wrong += check(label, doc, graph, 0.2);
        wrong += check(&format!("{label}, box grown threefold each way"), doc, graph, 3.0);
    }
    assert_eq!(wrong, 0, "the pruned field differs from the unpruned one: the prune is wrong");
    println!("\nEvery point of every scene is the same to the bit.");
}
