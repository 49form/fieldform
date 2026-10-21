// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the block list works out from a part every frame, timed without the application.
//!
//! Refactor plan step 3.4. The list is drawn every frame, and until this step nothing it read was
//! kept from one frame to the next. Each quiet frame worked out the order of the rows three times
//! (`listed_blocks`: once for the rows, and once each for the part's summary line above them and
//! its hover), where each block is drawn once (`nesting_in`), and asked `Graph::roots` once each
//! for the summary and its hover and once more for every row it drew, to say whether the row is a
//! body. Each of those walks the whole part, so the last made the list quadratic in it.
//!
//! "A frame, afresh" is those asks, as the application made them before this step. "A frame,
//! kept" is the same frame asked of `KeptReadings`, as the application asks since: the readings
//! for the summary, its hover and the rows, the order copied out for the rows as
//! `App::ordered_nodes` copies it, and every row and every body of the summary looked up in the
//! kept root set. The two are timed in alternate rounds, so the machine's load falls on both
//! alike. "Kept, moving" is the control: its key moves every frame, so it keeps nothing and should
//! read as "built once" and the frame's lookups, which is what every frame of a drag pays.
//!
//! **What it found, 2026-10-12, dev build, medians of three runs:** a quiet frame of 200 pairs
//! took 16.7 ms afresh and 35 µs kept, and of 500 pairs 89.9 ms and 88 µs. The roots asked a row
//! were most of it: 500 walks of 0.14 ms. "Kept, moving" read 1.6 and 4.6 ms, as "built once"
//! does, so even a frame whose document moved, as every frame of a drag does, pays a tenth of what
//! it did at 200 pairs and a twentieth at 500; what is left there is `listed_blocks`, whose sort of
//! the branches walks a branch for every comparison. On the documents from the test-geometry
//! folder, of one to ten blocks, a frame went from 5 to 55 µs afresh to 0.5 to 1.2 µs.
//!
//! **Only what this step changes is timed.** The rest of the list's frame is the same in both
//! arms and is not here: the rows' grouping, labels and painting, which only the application can
//! run, and the summary's cycle check and the drop lines' walk of each body's branch, which read
//! the graph and are not kept. A row is counted as the application draws one with every block
//! closed: each block of the order nothing holds. An import's row read the order once more for
//! itself; no document here has one.
//!
//! The pairs are a sphere and an offset of it, repeated, so each offset is a body holding its
//! sphere: the documents step 2.17 measured the list's quadratic on.
//!
//! `cargo run -p fieldform-io --example list_cost -- <file.fform>…`, in the dev profile the
//! application is iterated in, where the scene's walks are optimized and the caller is not.

use std::time::Instant;

use fieldform_field::KeptReadings;
use fieldform_scene::{Document, Graph, ListReadings, Node, NodeId, ParamRef, Part, PartId, Place};

const ROUNDS: usize = 101;

fn median(mut us: Vec<f64>) -> f64 {
    us.sort_by(f64::total_cmp);
    us[us.len() / 2]
}

fn timed<T>(mut f: impl FnMut() -> T) -> f64 {
    let mut samples = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let began = Instant::now();
        std::hint::black_box(f());
        samples.push(began.elapsed().as_secs_f64() * 1e6);
    }
    median(samples)
}

/// `a` and `b` timed in alternate rounds, each going first in every other one, so neither has the
/// machine to itself while the other waits: medians of each.
fn interleaved<A, B>(mut a: impl FnMut() -> A, mut b: impl FnMut() -> B) -> (f64, f64) {
    let (mut a_us, mut b_us) = (Vec::with_capacity(ROUNDS), Vec::with_capacity(ROUNDS));
    for round in 0..ROUNDS {
        let mut time_a = || {
            let began = Instant::now();
            std::hint::black_box(a());
            a_us.push(began.elapsed().as_secs_f64() * 1e6);
        };
        if round % 2 == 0 {
            time_a();
        }
        let began = Instant::now();
        std::hint::black_box(b());
        b_us.push(began.elapsed().as_secs_f64() * 1e6);
        if round % 2 == 1 {
            time_a();
        }
    }
    (median(a_us), median(b_us))
}

/// `n` sphere-and-offset pairs in one part, stamped in the order they were made and settled, as
/// every edit leaves a document: each sphere is drawn inside the offset that reads it.
fn pairs(n: usize) -> Document {
    let lit = ParamRef::Literal;
    let mut doc = Document::new();
    let mut graph = Graph::empty();
    for i in 0..n {
        let ball = graph.add(Node::Sphere {
            centre: Place::at(i as f64 * 30.0, 0.0, 0.0),
            radius: lit(10.0),
        });
        let grown = graph.add(Node::Offset { input: ball.into(), distance: lit(1.0) });
        doc.created.insert(ball, 1_700_000_000_000_000 + 2 * i as u64);
        doc.created.insert(grown, 1_700_000_000_000_001 + 2 * i as u64);
    }
    doc.add_part(Part {
        id: PartId::new(),
        name: "Pairs".into(),
        material: None,
        graph,
        inputs: Vec::new(),
    });
    doc.settle_ownership();
    doc.pending_ownership_changes.clear();
    doc
}

/// The summary line's bodies, as `App::graph_summary` asked for them before step 3.4: the order,
/// and every block of it searched for among the roots.
fn summary_afresh(doc: &Document, graph: &Graph) -> Vec<NodeId> {
    let roots = graph.roots();
    doc.listed_blocks(graph).into_iter().filter(|id| roots.contains(id)).collect()
}

/// One quiet frame's asks, as the application made them before step 3.4: the summary and its
/// hover, the nesting and the order for the rows, and `Graph::roots` for every row drawn.
fn frame_afresh(doc: &Document, graph: &Graph) -> (Vec<NodeId>, Vec<bool>) {
    let summary = summary_afresh(doc, graph);
    let hover = summary_afresh(doc, graph);
    debug_assert_eq!(summary, hover);
    let nesting = doc.nesting_in(graph);
    let listed = doc.listed_blocks(graph);
    let rows = listed
        .iter()
        .filter(|id| !nesting.is_nested(**id))
        .map(|id| graph.roots().contains(id))
        .collect();
    (summary, rows)
}

/// The same frame asked of the kept readings, as the application asks since step 3.4: each ask
/// goes to the keeper, as each of the application's does.
fn frame_kept(
    kept: &mut KeptReadings<u64, ListReadings>,
    key: u64,
    read: impl Fn(&u64) -> ListReadings,
) -> (Vec<NodeId>, Vec<bool>) {
    let mut summary_kept = || {
        let list = kept.get(key, &read);
        list.listed.iter().copied().filter(|id| list.nesting.is_root(*id)).collect::<Vec<_>>()
    };
    let summary = summary_kept();
    let hover = summary_kept();
    debug_assert_eq!(summary, hover);
    let list = kept.get(key, &read);
    let listed = kept.get(key, &read).listed.clone();
    let rows = listed
        .iter()
        .filter(|id| !list.nesting.is_nested(**id))
        .map(|id| list.nesting.is_root(*id))
        .collect();
    (summary, rows)
}

fn measure(label: &str, doc: &Document, graph: &Graph) {
    let nesting_us = timed(|| doc.nesting_in(graph));
    let listed_us = timed(|| doc.listed_blocks(graph));
    let roots_us = timed(|| graph.roots());
    let once_us = timed(|| ListReadings::build(doc, graph));

    let read = |_: &u64| ListReadings::build(doc, graph);
    let mut kept = KeptReadings::default();
    let (afresh_us, kept_us) =
        interleaved(|| frame_afresh(doc, graph), || frame_kept(&mut kept, 0, read));
    // The control: a key that moves every frame keeps nothing, so it pays a build a frame.
    let mut moved = 0;
    let moving_us = timed(|| {
        moved += 1;
        frame_kept(&mut kept, moved, read)
    });

    // The kept frame has to say what the afresh one said, or it is cheaper by being wrong.
    let (summary, rows) = frame_afresh(doc, graph);
    let (summary_kept, rows_kept) = frame_kept(&mut kept, moved, read);
    assert!(summary_kept == summary, "{label}: the kept summary's bodies differ");
    assert!(rows_kept == rows, "{label}: the kept rows differ on which are bodies");

    println!(
        "| {label} | {} | {} | {} | {nesting_us:.1} | {listed_us:.1} | {roots_us:.1} | \
         {once_us:.1} | {afresh_us:.1} | {kept_us:.2} | {moving_us:.1} |",
        graph.nodes().count(),
        rows.len(),
        rows.iter().filter(|body| **body).count(),
    );
}

fn main() {
    println!("Medians of {ROUNDS}, microseconds of the interface's thread.\n");
    println!(
        "| document | blocks | rows | bodies | nesting_in | listed_blocks | roots | built once | \
         a frame, afresh | a frame, kept | kept, moving |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for n in [25, 200, 500] {
        let doc = pairs(n);
        let graph = &doc.parts.values().next().expect("one part").graph;
        measure(&format!("{n} pairs"), &doc, graph);
    }
    for path in std::env::args().skip(1) {
        match fieldform_io::open(std::path::Path::new(&path)) {
            Ok((doc, _)) => {
                let name = std::path::Path::new(&path)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or(path.clone());
                // Every part with anything in it, in name order so two runs print alike; the
                // application reads only the active one, so each is a row of its own.
                let mut parts: Vec<_> =
                    doc.parts.values().filter(|p| p.graph.nodes().next().is_some()).collect();
                parts.sort_by(|a, b| a.name.cmp(&b.name));
                let several = parts.len() > 1;
                for part in parts {
                    let label = if several {
                        format!("`{name}` {}", part.name)
                    } else {
                        format!("`{name}`")
                    };
                    measure(&label, &doc, &part.graph);
                }
            }
            Err(e) => println!("could not open {path}: {e}"),
        }
    }
}
