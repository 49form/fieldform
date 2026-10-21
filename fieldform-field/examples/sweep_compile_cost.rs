// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a sweep costs the **field** to compile, piece by piece.
//!
//! Not the driver's shader compile — that is `fieldform-gpu/examples/wing_compile_cost.rs`. This
//! is `GraphField::compile_graph`, which runs on every rebuild, in front of everything else.
//!
//! **Written to gate a cache that turned out to be pointless, and it found the one that is not.**
//! The plan was to cache the spine's flattening: a sweep tessellates its spine on every compile,
//! and so does every `graph_extent` over it, and so does the shader's own reading — three
//! descriptions of one curve, rebuilt whenever anything changes. That sounds worth caching. It is
//! not, by three orders of magnitude.
//!
//! Measured on a swept aerofoil with a pipe and a cylinder beside it, on an RTX A500 laptop,
//! 2026-09-20:
//!
//! | | |
//! |---|---|
//! | `Curve::tessellate` at 0.5 mm | 0.0057 ms |
//! | `Spine::of_curve` at 0.5 mm | 0.0061 ms |
//! | `profile_loop` (the walk the compile makes) | 0.100 ms |
//! | `Profile::in_plane` | 0.000 ms |
//! | `Swept::held`, which builds every frame | 0.001 ms |
//! | **the profile's bake, 3395 cells** | **6.193 ms** |
//! | `GraphField::compile_graph`, the whole thing | 6.296 ms |
//!
//! So the spine is 0.1% of it and the bake is 98%. And the sweep is the whole compile: leaving
//! blocks out one at a time gives a cylinder alone at 0.004 ms, a pipe on the same span at 0.010,
//! and the swept aerofoil at 6.6.
//!
//! **Which made the cache worth having the profile's, not the spine's** — and it pays where the
//! editing happens, because dragging a *spine* does not change a *profile*. Built, and measured
//! here:
//!
//! | | |
//! |---|---|
//! | the first compile | 6.599 ms |
//! | the same graph again | 0.462 ms |
//! | **the spine dragged between compiles** | **0.529 ms** |
//!
//! See `LeafCache::profiles` for what the key is and why each part of it is there — it is not
//! the key this started with, and `a_cached_compile_answers_what_an_uncached_one_would` is what
//! holds it.
//!
//! Run with: `cargo run --release -p fieldform-field --example sweep_compile_cost`

use fieldform_scene::{Curve, CurveId, Document, Graph, Node, ParamRef, Place};
use nalgebra::Point3;
use std::time::Instant;

const ROUNDS: u32 = 200;

fn main() {
    let (doc, graph) = wing_like();
    let spine = *doc.curves.keys().find(|id| doc.curves[id].name == "Span").expect("the span");
    let curve = &doc.curves[&spine];

    println!("a spine of {} control points\n", curve.control_points().len());

    // What one flattening costs, at the two tolerances the two sides use.
    for tolerance in [0.1, 0.5, 5.0] {
        let t = Instant::now();
        let mut points = 0usize;
        for _ in 0..ROUNDS {
            points = curve.tessellate(tolerance).len();
        }
        let each = t.elapsed().as_secs_f64() * 1e6 / ROUNDS as f64;
        println!("  tessellate at {tolerance:>4} mm: {points:>5} points, {each:>8.1} us each");
    }

    // And what the spine costs as the field builds it, which measures as it flattens.
    let t = Instant::now();
    for _ in 0..ROUNDS {
        std::hint::black_box(fieldform_field::Spine::of_curve(curve, 0.5));
    }
    println!(
        "  Spine::of_curve at  0.5 mm: {:>21.1} us each\n",
        t.elapsed().as_secs_f64() * 1e6 / ROUNDS as f64
    );

    // The two things that do the flattening in anger.
    let t = Instant::now();
    for _ in 0..ROUNDS {
        std::hint::black_box(
            fieldform_field::GraphField::compile_graph(&doc, &graph, 0.5).expect("compiles"),
        );
    }
    let compile = t.elapsed().as_secs_f64() * 1e3 / ROUNDS as f64;

    let t = Instant::now();
    for _ in 0..ROUNDS {
        std::hint::black_box(fieldform_field::part_extent(&doc, &graph));
    }
    let extent = t.elapsed().as_secs_f64() * 1e3 / ROUNDS as f64;

    let t = Instant::now();
    for _ in 0..20 {
        std::hint::black_box(fieldform_gpu_generate(&doc, &graph));
    }
    let generate = t.elapsed().as_secs_f64() * 1e3 / 20.0;

    // Where a sweep's own compile goes, piece by piece, since the whole is six milliseconds and
    // a flattening is six microseconds of it.
    let profile_id =
        *doc.curves.keys().find(|id| doc.curves[id].name == "Section").expect("the section");
    let t = Instant::now();
    for _ in 0..ROUNDS {
        let _ = std::hint::black_box(fieldform_field::profile_loop_turning(
            &doc, profile_id, 1.0, Some(spine),
        ));
    }
    let loop_ms = t.elapsed().as_secs_f64() * 1e3 / ROUNDS as f64;

    let world = fieldform_field::profile_loop_turning(&doc, profile_id, 1.0, Some(spine))
        .expect("a closed loop")
        .points;
    let (origin, normal) = fieldform_field::plane_of(&world).expect("planar");
    let rotation = fieldform_field::rotation_from_z(normal);
    let into = move |p: Point3<f64>| {
        Point3::from(rotation.inverse_transform_vector(&(p.coords - origin.coords)))
    };
    let t = Instant::now();
    for _ in 0..ROUNDS {
        let _ = std::hint::black_box(fieldform_field::Profile::in_plane(&world, into, 0.5));
    }
    let profile_ms = t.elapsed().as_secs_f64() * 1e3 / ROUNDS as f64;

    let built = fieldform_field::Profile::in_plane(&world, into, 0.5).expect("planar");
    let path = fieldform_field::Spine::of_curve(curve, 0.5);
    let seed = rotation * nalgebra::Vector3::x();
    let t = Instant::now();
    for _ in 0..ROUNDS {
        std::hint::black_box(
            fieldform_field::Swept::held(
                built.clone(),
                path.clone(),
                seed,
                fieldform_field::Held::Square,
            )
            .expect("a sweep"),
        );
    }
    let swept_ms = t.elapsed().as_secs_f64() * 1e3 / ROUNDS as f64;

    // The call the compile actually makes, at the kernel's own turn: `profile_loop`, not the
    // one-degree reading the viewport uses.
    let t = Instant::now();
    for _ in 0..ROUNDS {
        let _ = std::hint::black_box(fieldform_field::profile_loop(&doc, profile_id, Some(spine)));
    }
    let kernel_loop = t.elapsed().as_secs_f64() * 1e3 / ROUNDS as f64;

    // The one thing in the sweep's arm not yet measured: the profile's bake.
    let reads = fieldform_field::profile_reads_out_to(
        &doc,
        &graph,
        graph.nodes().find(|(_, n)| matches!(n, Node::Sweep { .. })).expect("a sweep").0,
    );
    let (plo, phi) = built.bounds();
    let cells = fieldform_field::bake_cells((phi - plo).max(), 0.5, reads);
    let t = Instant::now();
    for _ in 0..ROUNDS {
        std::hint::black_box(cells.map(|c| built.clone().baked(c, reads)));
    }
    let bake_ms = t.elapsed().as_secs_f64() * 1e3 / ROUNDS as f64;

    println!("  profile bake ({cells:?} cells):{bake_ms:>7.3} ms");
    println!("  profile_loop_turning(1 deg):{loop_ms:>8.3} ms");
    println!("  profile_loop (the compile's):{kernel_loop:>7.3} ms");
    println!("  Profile::in_plane:          {profile_ms:>8.3} ms");
    println!("  Swept::held (the frames):   {swept_ms:>8.3} ms");
    println!("  GraphField::compile_graph:  {compile:>8.3} ms");
    println!("  part_extent:                {extent:>8.3} ms");
    println!("  fieldform_gpu::generate:    {generate:>8.3} ms");

    // And what a cache does for it -- first the same graph twice, then the case that matters:
    // the spine dragged between compiles, which changes the path and not the profile.
    let mut cache = fieldform_field::LeafCache::default();
    let t = Instant::now();
    std::hint::black_box(
        fieldform_field::GraphField::compile_cached(&doc, &graph, 0.5, &mut cache)
            .expect("compiles"),
    );
    let cold = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    for _ in 0..ROUNDS {
        std::hint::black_box(
            fieldform_field::GraphField::compile_cached(&doc, &graph, 0.5, &mut cache)
                .expect("compiles"),
        );
    }
    let warm = t.elapsed().as_secs_f64() * 1e3 / ROUNDS as f64;

    // A drag of the path: a new spine every time, the same profile every time.
    let mut cache = fieldform_field::LeafCache::default();
    let t = Instant::now();
    for step in 0..ROUNDS {
        let (mut doc, graph) = wing_like();
        let id = *doc.curves.keys().find(|i| doc.curves[i].name == "Span").expect("the span");
        let tip = doc.curves[&id].control_points()[3] + nalgebra::Vector3::z() * step as f64;
        doc.curves.get_mut(&id).expect("the span").set_control_point(3, tip);
        std::hint::black_box(
            fieldform_field::GraphField::compile_cached(&doc, &graph, 0.5, &mut cache)
                .expect("compiles"),
        );
    }
    let dragged = t.elapsed().as_secs_f64() * 1e3 / ROUNDS as f64;

    println!("
  with LeafCache:");
    println!("    first compile             {cold:>8.3} ms");
    println!("    the same graph again      {warm:>8.3} ms");
    println!("    the spine dragged         {dragged:>8.3} ms");

    // And which block the compile is actually spent in, by leaving them out one at a time.
    println!("\n  what the compile is spent on, by difference:");
    let mut last = 0.0;
    for (label, keep) in [
        ("a cylinder alone", 1usize),
        ("  + a pipe on the span", 2),
        ("  + the swept aerofoil", 3),
    ] {
        let (doc, graph) = wing_like_with(keep);
        let t = Instant::now();
        for _ in 0..ROUNDS {
            std::hint::black_box(
                fieldform_field::GraphField::compile_graph(&doc, &graph, 0.5).expect("compiles"),
            );
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / ROUNDS as f64;
        println!("    {label:<24} {ms:>8.3} ms   (+{:>7.3})", ms - last);
        last = ms;
    }

    println!(
        "\nCaching a flattening is worth having only if it is a real share of those.\n\
         The shader is generated once per rebuild; the extent is asked many times."
    );
}

/// `generate` lives in the gpu crate, which this one cannot see -- so the shader's own share is
/// measured by the example in that crate. Kept here as a stub so the shape of the question is
/// visible in one place.
fn fieldform_gpu_generate(_doc: &Document, _graph: &Graph) -> usize {
    0
}

/// A wing-like part: a swept aerofoil, a pipe, and a cylinder, so the walk has curves in it.
fn wing_like() -> (Document, Graph) {
    wing_like_with(3)
}

/// The same part with `keep` blocks: 1 the cylinder, 2 adds the pipe, 3 adds the sweep.
fn wing_like_with(keep: usize) -> (Document, Graph) {
    let lit = ParamRef::Literal;
    let mut doc = Document::new();

    let section = CurveId::new();
    let chord = 400.0;
    let mut points = Vec::new();
    for i in 0..24 {
        let u = i as f64 / 24.0;
        let x = chord * (1.0 - (u * std::f64::consts::TAU).cos()) * 0.5;
        let t = 5.0 * chord * 0.12
            * (0.2969 * (x / chord).sqrt() - 0.1260 * (x / chord)
                - 0.3516 * (x / chord).powi(2)
                + 0.2843 * (x / chord).powi(3)
                - 0.1015 * (x / chord).powi(4));
        let z = if (u * std::f64::consts::TAU).sin() >= 0.0 { t } else { -t };
        points.push(Point3::new(x, 200.0, z));
    }
    doc.curves.insert(
        section,
        Curve::polyline(section, "Section", points, true).expect("a closed section"),
    );

    let spine = CurveId::new();
    doc.curves.insert(
        spine,
        Curve::new(
            spine,
            "Span",
            3,
            vec![
                Point3::new(0.0, 200.0, 0.0),
                Point3::new(40.0, 700.0, 20.0),
                Point3::new(90.0, 1300.0, 60.0),
                Point3::new(140.0, 1900.0, 120.0),
            ],
            vec![1.0; 4],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            false,
        )
        .expect("a cubic"),
    );

    let mut graph = Graph::empty();
    let body = graph.add(Node::Cylinder { edge: Default::default(),
        ends: fieldform_scene::Ends::At {
            from: Place::At([lit(-600.0), lit(0.0), lit(0.0)]),
            to: Place::At([lit(1400.0), lit(0.0), lit(0.0)]),
        },
        radius: lit(180.0),
    });
    let mut inputs = vec![body];
    if keep >= 2 {
        inputs.push(graph.add(Node::Pipe { curve: spine, radius: lit(12.0) }));
    }
    if keep >= 3 {
        inputs.push(graph.add(Node::Sweep { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(section),
            spine,
            section: Default::default(),
            law: Default::default(),
        }));
    }
    graph.add(Node::Boolean { op: fieldform_scene::BooleanOp::Union, inputs: inputs.into_iter().map(Into::into).collect() });
    (doc, graph)
}
