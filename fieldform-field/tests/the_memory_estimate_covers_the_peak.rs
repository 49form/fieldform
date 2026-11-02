// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The memory estimate a Tessellate block shows before it runs is at least what the run then
//! holds — measured, by counting every byte the allocator hands out while it runs.
//!
//! **One test in its own binary**, because the count is of the whole process: a second test
//! running beside it would add its allocations to this one's peak. The bodies are the three the
//! table in `examples/tessellation_memory.rs` was made from, cut down so the test takes seconds;
//! the example has the full sizes and the finer details.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use fieldform_field::{Cancel, GraphField, Progress, estimate, tessellate};
use fieldform_scene::{
    BooleanOp, CellMap, Document, Graph, LatticeCell, LatticePhase, MeshDetail, Node, ParamRef,
    Place,
};

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

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn ball() -> (Document, Graph) {
    let mut g = Graph::empty();
    g.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(50.0) });
    (Document::new(), g)
}

fn cage() -> (Document, Graph) {
    let mut doc = Document::new();
    let id = doc.add_cage(fieldform_cage::Cage::cube_of(100.0));
    let mut g = Graph::empty();
    g.add(Node::Cage(id));
    (doc, g)
}

/// Two gyroid cells each way, rather than the example's five.
fn gyroid() -> (Document, Graph) {
    let mut graph = Graph::empty();
    let cube = graph.add(Node::Cuboid {
        edge: Default::default(),
        centre: Place::At([lit(0.0); 3]),
        size: [lit(40.0); 3],
        turn: fieldform_scene::no_turn(),
    });
    let lattice = graph.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell: LatticeCell::Gyroid,
        map: CellMap::Rectangular { cell: lit(20.0) },
        phase: LatticePhase::sheet(lit(2.0)),
    });
    graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![cube.into(), lattice.into()] });
    (Document::new(), graph)
}

#[test]
fn the_estimate_is_at_least_the_peak_a_tessellation_reaches() {
    let cases: [(&str, fn() -> (Document, Graph), MeshDetail); 4] = [
        ("ball", ball, MeshDetail::Full),
        ("ball", ball, MeshDetail::Enchilada),
        ("cage", cage, MeshDetail::Normal),
        ("gyroid", gyroid, MeshDetail::Coarse),
    ];
    let mut readings = Vec::new();
    for (name, make, detail) in cases {
        let (doc, graph) = make();
        let root = graph.roots()[0];
        let (lo, hi) = fieldform_field::graph_extent(&doc, &graph, root).as_box().expect("bounded");
        let e = hi - lo;
        let tolerance = detail.tolerance(e.x.max(e.y).max(e.z));
        let field = GraphField::compile_graph(&doc, &graph, (tolerance.min_cell_mm * 0.5).max(1e-4))
            .expect("compiles");
        for (improve, reduce) in [(false, false), (true, false), (true, true)] {
            let guess = estimate(&field, (lo, hi), &tolerance, improve, reduce, &Cancel::new())
                .expect("not cancelled");
            let base = NOW.load(Ordering::Relaxed);
            PEAK.store(base, Ordering::Relaxed);
            let out = tessellate(&field, (lo, hi), &tolerance, improve, reduce, &Cancel::new(), &Progress::new())
                .expect("not cancelled");
            let peak = PEAK.load(Ordering::Relaxed).saturating_sub(base);
            drop(out);
            readings.push(format!(
                "{name} {} improve={improve} reduce={reduce}: estimated {} for a peak of {peak}",
                detail.label(),
                guess.bytes
            ));
            assert!(guess.bytes >= peak, "{}", readings.join("\n"));
        }
    }
    println!("{}", readings.join("\n"));
}
