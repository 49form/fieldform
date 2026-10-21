// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the viewport's overlay and panels work out from a part every frame, timed without the
//! application.
//!
//! Refactor plan steps 3.2 and 3.3. Each frame paints every point and plane block over the model,
//! sizes the box picking aims into, and hands the trace every parameter's value, and until step
//! 3.3 none of it was kept from one frame to the next: the points were placed twice, since the pick
//! box asked for its own, and the points, the planes and the pick box each cloned the part's graph
//! first -- up to four whole-graph clones a frame. None of that could be timed while it ran only
//! inside a window.
//!
//! The piece columns are each reading as the application asked for it before step 3.3, with
//! whatever it worked out for itself. "A frame" is everything one quiet frame asked for that way.
//! "Built once" is `OverlayReadings::build`, every piece from one flattened graph: what each
//! revision of the document pays. "A frame, kept" is the same quiet frame asked of
//! `KeptReadings`, as the application asks since step 3.3, with nothing moved since the last;
//! "kept, moving" is its control, the key moving every frame, which keeps nothing and should read
//! as "built once". The two frame columns are timed in alternate rounds, so the machine's load
//! falls on both alike.
//!
//! **What it found the first time, 2026-10-12:** the clones are not what costs. Flattening a
//! graph of ten blocks takes a fifth of a microsecond. The pick box is nearly the whole frame,
//! because it asks every drawn body for its extent: 2.0 ms for one sweep, 1.7 ms for a vase, 17
//! ms for a wing of nine blocks, every frame. So sharing one flattened graph within a frame buys
//! nothing ("built once" reads what "a frame" does), and keeping the readings across frames buys
//! all of it.
//!
//! **And once they were kept, the same day:** a quiet frame asked of the kept readings reads 0.1 to
//! 0.2 µs on all four documents, which is the clock's own resolution, against 42 µs for the loft
//! and 1.7 to 17.5 ms for the other three asked afresh. "Kept, moving" reads as "built once", so
//! what a frame saves is exactly the build it no longer pays.
//!
//! The application's own view comes in as a document just opened has it: no body drawn as a
//! mesh, no profile row hidden, meshing at 64 cells and a 100 mm default body. The kept frame's
//! key is a bare count here; the application's also names that view, and builds it each ask.
//!
//! `cargo run --release -p fieldform-io --example overlay_cost -- <file.fform>…`

use std::time::Instant;

use fieldform_field::{
    display_graph_with_stand_ins, pick_bounds, plane_frames, point_positions,
    with_turns_flattened, KeptReadings, OverlayInputs, OverlayReadings,
};
use fieldform_scene::{Document, Graph, NodeId, ParamRef};

const ROUNDS: usize = 101;
/// The application's meshing resolution as it starts, one cell of which is the pick box's margin.
const RESOLUTION: usize = 64;
/// The application's default body, which is the box an empty scene is given.
const EMPTY_MM: f64 = 100.0;

fn median(mut us: Vec<f64>) -> f64 {
    us.sort_by(f64::total_cmp);
    us[us.len() / 2]
}

fn timed<T>(mut f: impl FnMut() -> T) -> (f64, T) {
    let mut samples = Vec::with_capacity(ROUNDS);
    let mut last = None;
    for _ in 0..ROUNDS {
        let began = Instant::now();
        let out = f();
        samples.push(began.elapsed().as_secs_f64() * 1e6);
        last = Some(out);
    }
    (median(samples), last.expect("at least one round"))
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

fn measure(label: &str, doc: &Document, part: &Graph) {
    // The application's reading of a number for drawing and sizing: one that cannot be resolved
    // is zero.
    let numbers = |r: &ParamRef| Some(doc.resolve(r).unwrap_or(0.0));
    let not_meshed = |_: NodeId| false;
    let inputs = OverlayInputs {
        drawn_as_mesh: &not_meshed,
        hidden_with_rows: &[],
        resolution: RESOLUTION,
        empty_mm: EMPTY_MM,
        numbers: &numbers,
    };

    // Each as the application asks for it: the points and the planes flatten the graph for
    // themselves, and the pick box builds the drawn graph and places the points again.
    let points_alone = || {
        point_positions(doc, part, &with_turns_flattened(doc, part, &numbers), &[], &numbers)
    };
    let planes_alone =
        || plane_frames(doc, part, &with_turns_flattened(doc, part, &numbers), &numbers);
    let bounds_alone = || {
        let (display, _) = display_graph_with_stand_ins(doc, part, &not_meshed);
        pick_bounds(doc, &display, &points_alone(), RESOLUTION, EMPTY_MM)
    };

    let (placed_us, _) = timed(|| with_turns_flattened(doc, part, &numbers));
    let (points_us, points) = timed(&points_alone);
    let (planes_us, planes) = timed(&planes_alone);
    let (display_us, _) = timed(|| display_graph_with_stand_ins(doc, part, &not_meshed));
    let (bounds_us, bounds) = timed(&bounds_alone);
    let (values_us, values) = timed(|| doc.parameter_values());
    let (once_us, once) = timed(|| OverlayReadings::build(doc, part, &inputs));

    // One quiet frame: the trace's parameter table, and the overlay's points, its pick box --
    // which placed the points again -- and its planes. Asked afresh, as before step 3.3, and of
    // the kept readings, as since: four asks, each copying out what the application copies out.
    let read = |_: &u64| OverlayReadings::build(doc, part, &inputs);
    let mut kept = KeptReadings::default();
    let mut kept_frame = |key: u64| {
        let values = kept.get(key, read);
        let points = kept.get(key, read).points.clone();
        let bounds = kept.get(key, read).pick_bounds;
        let planes = kept.get(key, read).planes.clone();
        (values, points, bounds, planes)
    };
    let (frame_us, kept_us) = interleaved(
        || (doc.parameter_values(), points_alone(), bounds_alone(), planes_alone()),
        || kept_frame(0),
    );
    // The control: a key that moves every frame keeps nothing, so it pays a build a frame.
    let mut moved = 0;
    let (moving_us, _) = timed(|| {
        moved += 1;
        kept_frame(moved)
    });
    let (values_kept, points_kept, bounds_kept, planes_kept) = kept_frame(moved);

    // Built once and kept have to say what the pieces said, or they are cheaper by being wrong.
    assert!(once.points == points, "{label}: the points built once differ");
    assert!(once.planes == planes, "{label}: the planes built once differ");
    assert!(once.pick_bounds == bounds, "{label}: the pick box built once differs");
    assert!(once.values == values, "{label}: the parameter table built once differs");
    assert!(points_kept == points, "{label}: the kept points differ");
    assert!(planes_kept == planes, "{label}: the kept planes differ");
    assert!(bounds_kept == bounds, "{label}: the kept pick box differs");
    assert!(values_kept.values == values, "{label}: the kept parameter table differs");

    println!(
        "| {label} | {} | {} | {} | {placed_us:.1} | {points_us:.1} | {planes_us:.1} | \
         {display_us:.1} | {bounds_us:.1} | {values_us:.1} | {frame_us:.1} | {once_us:.1} | \
         {kept_us:.2} | {moving_us:.1} |",
        part.nodes().count(),
        points.len(),
        planes.len(),
    );
}

fn main() {
    println!("Medians of {ROUNDS}, microseconds of the interface's thread.\n");
    println!(
        "| document | blocks | points | planes | flatten the turns | point positions | \
         plane frames | drawn graph | pick box | parameter table | a frame | built once | \
         a frame, kept | kept, moving |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
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
