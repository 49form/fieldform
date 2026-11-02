// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a tessellation holds at its peak, against what `estimate` said before it ran — backlog
//! item 53's "says a memory estimate before a run, and refuses what cannot fit".
//!
//! **The peak is counted, not inferred.** The example's allocator is the system's with two
//! counters beside it, so `peak` is the most heap the process held during the run, above what it
//! held when the run started: the tree, the mesh, and the largest working set of the passes after
//! it, whichever moment that was. The estimate is made first, from the same compiled field, and
//! knows nothing the run finds out.
//!
//! **What predicts it.** A run holds a few hundred bytes per surface leaf of the octree, whatever
//! the body — `peak per leaf` below is 267 to 721, and 277 to 612 on anything of a megabyte or
//! more — while the leaves themselves range over three orders of magnitude and are decided by the
//! surface, not by its box: the ball and the cage are the same size at every detail and differ
//! fifteenfold at the finest. So the estimate counts leaves, by walking the tree's own rule over a
//! thinned copy of it, and multiplies.
//!
//! Release build, 16 logical cores, 2026-10-24, run beside other builds (so read the times as
//! their shape). 100 mm ball; 100 mm cube cage; 100 mm cube of 20 mm gyroid cells with a 2 mm
//! sheet:
//!
//! | body | detail | pass | triangles | peak | peak per leaf | estimate | est / peak | est. time | run time |
//! |---|---|---|---|---|---|---|---|---|---|
//! | ball | Coarse | mesh | 732 | 0.1 MB | 267 B | 1.2 MB | 11.17 | 0.00 s | 0.0 s |
//! | ball | Coarse | mesh + improve | 732 | 0.3 MB | 721 B | 1.3 MB | 4.53 | 0.00 s | 0.0 s |
//! | ball | Coarse | mesh + reduce + improve | 732 | 0.2 MB | 437 B | 1.3 MB | 7.47 | 0.00 s | 0.0 s |
//! | ball | Normal | mesh | 11964 | 1.7 MB | 377 B | 3.1 MB | 1.80 | 0.02 s | 0.0 s |
//! | ball | Normal | mesh + improve | 11964 | 2.8 MB | 612 B | 4.4 MB | 1.58 | 0.02 s | 0.1 s |
//! | ball | Normal | mesh + reduce + improve | 11964 | 2.8 MB | 612 B | 4.4 MB | 1.58 | 0.02 s | 0.1 s |
//! | ball | Full | mesh | 32316 | 4.9 MB | 305 B | 8.4 MB | 1.71 | 0.02 s | 0.1 s |
//! | ball | Full | mesh + improve | 32316 | 5.4 MB | 333 B | 13.1 MB | 2.43 | 0.02 s | 0.2 s |
//! | ball | Enchilada | mesh | 75948 | 12.3 MB | 375 B | 16.0 MB | 1.31 | 0.03 s | 0.2 s |
//! | ball | Enchilada | mesh + improve | 75948 | 12.9 MB | 395 B | 25.4 MB | 1.97 | 0.02 s | 0.4 s |
//! | cage | Coarse | mesh | 2172 | 0.4 MB | 305 B | 1.6 MB | 4.14 | 0.03 s | 0.0 s |
//! | cage | Coarse | mesh + improve | 2172 | 0.4 MB | 309 B | 1.9 MB | 5.02 | 0.03 s | 0.0 s |
//! | cage | Coarse | mesh + reduce + improve | 2172 | 0.4 MB | 343 B | 1.9 MB | 4.53 | 0.03 s | 0.1 s |
//! | cage | Normal | mesh | 11556 | 1.7 MB | 357 B | 3.2 MB | 1.87 | 0.13 s | 0.2 s |
//! | cage | Normal | mesh + improve | 11556 | 2.6 MB | 544 B | 4.6 MB | 1.75 | 0.12 s | 0.3 s |
//! | cage | Normal | mesh + reduce + improve | 11556 | 2.6 MB | 544 B | 4.6 MB | 1.75 | 0.13 s | 0.3 s |
//! | cage | Full | mesh | 180276 | 25.7 MB | 327 B | 37.2 MB | 1.45 | 0.60 s | 5.3 s |
//! | cage | Full | mesh + improve | 180276 | 42.1 MB | 535 B | 59.8 MB | 1.42 | 0.67 s | 11.0 s |
//! | cage | Enchilada | mesh | 1014684 | 148.9 MB | 346 B | 198.8 MB | 1.33 | 0.98 s | 58.6 s |
//! | cage | Enchilada | mesh + improve | 1014684 | 180.6 MB | 420 B | 322.4 MB | 1.78 | 1.16 s | 94.9 s |
//! | gyroid | Coarse | mesh | 1243992 | 211.1 MB | 277 B | 351.9 MB | 1.67 | 0.49 s | 53.8 s |
//! | gyroid | Coarse | mesh + improve | 1243992 | 352.2 MB | 462 B | 571.1 MB | 1.62 | 0.93 s | 107.5 s |
//! | gyroid | Coarse | mesh + reduce + improve | 1243992 | 236.8 MB | 310 B | 571.1 MB | 2.41 | 0.37 s | 175.5 s |
//! | gyroid | Normal | mesh | 4364796 | 790.9 MB | 298 B | 1223.9 MB | 1.55 | 0.19 s | 50.8 s |
//! | gyroid | Normal | mesh + improve | 4364796 | 822.0 MB | 309 B | 1988.2 MB | 2.42 | 0.27 s | 212.1 s |
//!
//! **The estimate is never under, and is 1.3× to 2.4× over** on everything of a megabyte or more;
//! below that its one-megabyte floor is most of the number. Over, because each pass's bytes per
//! leaf is taken at the largest measured and the leaves 15% high; the spread is the passes' own —
//! the flip pass's table of edges grows in powers of two, so a mesh just past a power holds
//! nearly twice what one just under it does.
//!
//! `--leaves` measures the leaf count alone, against the exact count before balancing, which is
//! what the estimate samples for. Same machine and day:
//!
//! | body | detail | leaves | estimated | est / leaves | cells decided | est. time | count time |
//! |---|---|---|---|---|---|---|---|
//! | ball | Coarse | 416 | 416 | 1.000 | 713 | 0.00 s | 0.00 s |
//! | ball | Normal | 4760 | 4760 | 1.000 | 8393 | 0.03 s | 0.04 s |
//! | ball | Full | 15848 | 16908 | 1.067 | 12585 | 0.06 s | 0.18 s |
//! | ball | Enchilada | 35144 | 34293 | 0.976 | 14985 | 0.10 s | 0.32 s |
//! | cage | Coarse | 1304 | 1304 | 1.000 | 2441 | 0.08 s | 0.06 s |
//! | cage | Normal | 5072 | 5072 | 1.000 | 9449 | 0.34 s | 0.31 s |
//! | cage | Full | 76856 | 82515 | 1.074 | 26297 | 0.99 s | 4.99 s |
//! | cage | Enchilada | 478268 | 450828 | 0.943 | 30665 | 1.09 s | 20.54 s |
//! | gyroid | Coarse | 819518 | 799789 | 0.976 | 16969 | 0.28 s | 28.50 s |
//! | gyroid | Normal | 2870384 | 2787609 | 0.971 | 21065 | 0.52 s | 102.91 s |
//!
//! Within 8% everywhere, and exact where the tree is small enough never to be thinned. The first
//! version sampled one shallow level and built 128 of its cells in full; it read the cage at
//! Enchilada at 54 698 leaves of 478 268: they are concentrated under few of that level's seven
//! thousand cells, and the sample missed them — its fullest cell held 50 leaves where the level
//! averaged 69. Thinning every level instead finds them, since a region that keeps dividing
//! becomes a larger share of each level below it.
//!
//! Run with: `cargo run --release -p fieldform-field --example tessellation_memory
//! [ball|cage|gyroid] [--full] [--leaves]` — `--full` adds the gyroid at Full, fifteen million
//! triangles and several gigabytes.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use fieldform_field::{Cancel, GraphField, Progress, estimate, tessellate};
use fieldform_scene::{
    BooleanOp, CellMap, Document, Graph, LatticeCell, LatticePhase, MeshDetail, Node, ParamRef,
};

/// The system allocator, counting what is held and the most that has been.
struct Counting;
static NOW: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn held(more: usize) {
    let now = NOW.fetch_add(more, Ordering::Relaxed) + more;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            held(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        NOW.fetch_sub(l.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            if new > l.size() {
                held(new - l.size());
            } else {
                NOW.fetch_sub(l.size() - new, Ordering::Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// The peak heap `f` reached above what was held when it started.
fn peak_of<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = NOW.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let out = f();
    (out, PEAK.load(Ordering::Relaxed).saturating_sub(base))
}

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

/// A 100 mm ball.
fn sphere() -> (Document, Graph) {
    let mut g = Graph::empty();
    g.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(50.0) });
    (Document::new(), g)
}

/// A 100 mm cube cage — its limit surface, a rounded block.
fn cage() -> (Document, Graph) {
    let mut doc = Document::new();
    let id = doc.add_cage(fieldform_cage::Cage::cube_of(100.0));
    let mut g = Graph::empty();
    g.add(Node::Cage(id));
    (doc, g)
}

/// A 100 mm cube of 20 mm gyroid cells with a 2 mm sheet: the lattice `capped_lattice` meshes.
fn gyroid() -> (Document, Graph) {
    let mut graph = Graph::empty();
    let cube = graph.add(Node::Cuboid {
        edge: Default::default(),
        centre: fieldform_scene::Place::At([lit(0.0); 3]),
        size: [lit(100.0); 3],
        turn: fieldform_scene::no_turn(),
    });
    let lattice = graph.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell: LatticeCell::Gyroid,
        map: CellMap::Rectangular { cell: lit(20.0) },
        phase: LatticePhase::sheet(lit(2.0)),
    });
    graph.add(Node::Boolean {
        op: BooleanOp::Intersection,
        inputs: vec![cube.into(), lattice.into()],
    });
    (Document::new(), graph)
}

fn mb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let full = args.iter().any(|a| a == "--full");

    let which: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let all = [MeshDetail::Coarse, MeshDetail::Normal, MeshDetail::Full, MeshDetail::Enchilada];
    // The lattice at Full is 15 million triangles and ten minutes; asked for by name.
    let lattice: &[MeshDetail] = if full {
        &[MeshDetail::Coarse, MeshDetail::Normal, MeshDetail::Full]
    } else {
        &[MeshDetail::Coarse, MeshDetail::Normal]
    };
    let bodies: [(&str, fn() -> (Document, Graph), &[MeshDetail]); 3] =
        [("ball", sphere, &all), ("cage", cage, &all), ("gyroid", gyroid, lattice)];

    let leaves_only = args.iter().any(|a| a == "--leaves");
    if leaves_only {
        println!("| body | detail | leaves | estimated | est / leaves | cells decided | est. time | count time |");
        println!("|---|---|---|---|---|---|---|---|");
    } else {
        println!(
            "| body | detail | pass | triangles | peak | peak per leaf | estimate | est / peak | est. time | run time |"
        );
        println!("|---|---|---|---|---|---|---|---|---|---|");
    }
    for (name, make, details) in bodies {
        if !which.is_empty() && !which.iter().any(|w| *w == name) {
            continue;
        }
        let (doc, graph) = make();
        let root = graph.roots()[0];
        let (lo, hi) = fieldform_field::graph_extent(&doc, &graph, root).as_box().expect("bounded");
        let e = hi - lo;
        let longest = e.x.max(e.y).max(e.z);
        for detail in details {
            let tol = detail.tolerance(longest);
            let field =
                GraphField::compile_graph(&doc, &graph, (tol.min_cell_mm * 0.5).max(1e-4)).expect("compiles");
            // Reducing is ten times as slow as the rest put together, so it is measured where
            // that is minutes rather than hours.
            let reduce_too = matches!(detail, MeshDetail::Coarse)
                || (name != "gyroid" && matches!(detail, MeshDetail::Normal));
            let passes: Vec<(&str, bool, bool)> = [
                Some(("mesh", false, false)),
                Some(("mesh + improve", true, false)),
                reduce_too.then_some(("mesh + reduce + improve", true, true)),
            ]
            .into_iter()
            .flatten()
            .collect();
            if leaves_only {
                let t = std::time::Instant::now();
                let guess = estimate(&field, (lo, hi), &tol, false, false, &Cancel::new())
                    .expect("not cancelled");
                let guessed_in = t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                let exact = fieldform_field::surface_leaves(&field, (lo, hi), &tol, &Cancel::new())
                    .expect("not cancelled");
                let counted_in = t.elapsed().as_secs_f64();
                println!(
                    "| {name} | {} | {exact} | {:.0} | {:.3} | {} | {guessed_in:.2} s | {counted_in:.2} s |",
                    detail.label(),
                    guess.leaves.expected,
                    guess.leaves.expected / exact.max(1) as f64,
                    guess.leaves.cells_decided,
                );
                continue;
            }
            for (pass, improve, reduce) in passes {
                let t = std::time::Instant::now();
                let guess = estimate(&field, (lo, hi), &tol, improve, reduce, &Cancel::new())
                    .expect("not cancelled");
                let guessed_in = t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                let (out, peak) = peak_of(|| {
                    tessellate(&field, (lo, hi), &tol, improve, reduce, &Cancel::new(), &Progress::new())
                        .expect("not cancelled")
                });
                let took = t.elapsed().as_secs_f64();
                println!(
                    "| {name} | {} | {pass} | {} | {:.1} MB | {:.0} B | {:.1} MB | {:.2} | {:.2} s | {:.1} s |",
                    detail.label(),
                    out.meshed,
                    mb(peak),
                    peak as f64 / guess.leaves.expected.max(1.0),
                    mb(guess.bytes),
                    guess.bytes as f64 / peak.max(1) as f64,
                    guessed_in,
                    took,
                );
            }
        }
    }
}
