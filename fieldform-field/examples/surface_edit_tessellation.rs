// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Whether a conformal lattice still meshes after one of its surface's points is moved.
//!
//! **Written from a crash.** The owner opened `conformal-cell-map-tpms.fform`, made its surface
//! the control mesh, moved one point up — and the Tessellate block, re-running on the edit, came
//! back with 0 triangles in 0.0 s, which the preview then tried to upload and wgpu refused. The
//! upload is fixed separately; this asks why the mesh was empty.
//!
//! **It was not the edit.** The same saved part meshes to nothing with no geometry touched at
//! all, if the box it is meshed in is grown 28.35 mm — which is what moving a point up did. At
//! Coarse, before the mesher was told the lattice's wall thickness and after:
//!
//! | box grown up by | before | after |
//! |---|---|---|
//! | 0 mm | 130 576 | 131 644 |
//! | 1 mm | 129 436 | 131 468 |
//! | 5 mm | 130 048 | 131 548 |
//! | 10 mm | 128 772 | 130 420 |
//! | 20 mm | 130 760 | 131 732 |
//! | 28.35 mm | **0** | 130 576 |
//!
//! and the edited part, which meshed to 0 triangles in 0.01 s, comes out at 132 592 in 7.5 s
//! against 7.0 s for the part as saved. See `adaptive.rs`'s module note for the rule.
//!
//! **The edited part is not closed, and that is the surface's fault rather than the mesher's**:
//! 132 open edges, every one of their 264 ends within 38.3 mm of the spike the moved point makes.
//! A conformal map is single-valued only inside its surface's curvature radius
//! (`conformal_agreement.rs`), and a 28 mm spike on a 200 mm panel is well outside it.
//!
//! **And the holes at the spike are not the curvature limit**, which is worth recording because
//! the curvature is what everyone reaches for first. Measured on this same file, with the middle
//! point pulled up 28 mm:
//!
//! | | as saved | spiked |
//! |---|---|---|
//! | tightest fold, control mesh | 225.1 mm | 40.3 mm |
//! | tightest fold, limit surface | 136.1 mm | 27.6 mm |
//! | smallest cell | 33.33 mm | 33.33 mm |
//! | fastest the field changes | — | **2.86 mm per mm**, at (−5.5, 0, 19.5) |
//!
//! The shell reaches 10 mm either side, which is well inside every fold radius here, and the
//! cells do not get smaller. What breaks is the promise a distance field makes: it changes by
//! 2.86 mm per millimetre travelled beside the spike, where the mesher's pruning believes 1.0 —
//! so cells that do hold surface are cut away. The jump is in `(s, t)`, the surface's own
//! coordinates, near where the spiked faces meet. A measure that predicts *that* from the
//! surface alone is not built: the panel warns about the fold radius, which is a real limit and
//! a different one. A probe that walks each sample along its normal and asks where the map sends
//! it was tried and does not predict it either — 36.9 mm on the spiked panel, against a 10 mm
//! shell that already tears.
//!
//! **What does find it is `verify`.** `find_overstep` ranks sample pairs by steepness or bend,
//! then refines each suspect by quartering toward its steepest part. On this file: nothing on
//! the version as saved (1.1 s), and on the spiked one a genuine discontinuity — the value
//! **steps 0.354 mm across a span of nothing** at (23.9, −26.7, 28.3), in 1.7 s. That is the
//! fault: not the fold radius, not the cell size, but the surface's own coordinates jumping
//! where the spiked faces meet, which the mesher's pruning cannot survive.
//!
//! Run with: `cargo run --release -p fieldform-field --example surface_edit_tessellation -- <file>`

use fieldform_field::{Cancel, GraphField, Progress, ScalarField, graph_extent, tessellate};
use fieldform_scene::MeshDetail;

fn main() {
    let path = std::env::args().nth(1).expect("a .fform file");
    let (mut doc, _) = fieldform_io::open(std::path::Path::new(&path)).expect("opens");
    let graph = doc.parts_ordered()[0].graph.clone();
    for (id, node) in graph.nodes() {
        println!("node {id:?}: {node:?}");
    }
    println!("roots: {:?}", graph.roots());
    for t in &doc.tessellations {
        println!("tessellation {:?} body {:?} detail {:?}", t.name, t.body, t.detail);
    }
    let surface = *doc.quad_meshes.keys().next().expect("a surface");

    let run = |doc: &fieldform_scene::Document, label: &str| {
        for root in graph.roots() {
            let extent = graph_extent(doc, &graph, root);
            let Some((lo, hi)) = extent.as_box() else {
                println!("{label}: root {root:?} unbounded");
                continue;
            };
            let e = hi - lo;
            let tolerance = MeshDetail::Coarse.tolerance(e.x.max(e.y).max(e.z));
            let Some(sub) = graph.subtree(root) else { continue };
            let field = match GraphField::compile_graph(doc, &sub, (tolerance.min_cell_mm * 0.5).max(1e-4)) {
                Ok(f) => f,
                Err(e) => {
                    println!("{label}: root {root:?} does not compile: {e}");
                    continue;
                }
            };
            let centre = nalgebra::center(&lo, &hi);
            let (ilo, ihi) = field.interval(lo, hi);
            println!(
                "{label}: root {root:?} box {lo:?}..{hi:?}  interval over it [{ilo:.3}, {ihi:.3}]  value at centre {:.3}",
                field.distance(centre)
            );
            // The octree's root, as the mesher lays it out, and its eight children: the value
            // at each child's centre against the half-diagonal it has to beat to be pruned.
            let size = e.x.max(e.y).max(e.z) + 4.0 * tolerance.min_cell_mm;
            let half = size / 4.0;
            for k in 0..8 {
                let c = centre
                    + nalgebra::Vector3::new(
                        if k & 1 == 0 { -half } else { half },
                        if k & 2 == 0 { -half } else { half },
                        if k & 4 == 0 { -half } else { half },
                    );
                let h = nalgebra::Vector3::repeat(half);
                let (a, b) = field.interval(c - h, c + h);
                println!(
                    "{label}:   child {k} centre {:?}  value {:.3}  interval [{a:.3}, {b:.3}]",
                    [c.x, c.y, c.z],
                    field.distance(c),
                );
            }
            let started = std::time::Instant::now();
            let out = tessellate(&field, (lo, hi), &tolerance, false, false, &Cancel::new(), &Progress::new())
                .expect("not cancelled");
            let mut edges = std::collections::HashMap::new();
            for t in &out.mesh.triangles {
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    *edges.entry((a.min(b), a.max(b))).or_insert(0u32) += 1;
                }
            }
            let open = edges.values().filter(|&&n| n == 1).count();
            // Where the holes are: a spread of them says the mesher, a cluster says the field.
            let mut far: f64 = 0.0;
            let mut near_peak = 0;
            for ((a, b), n) in &edges {
                if *n != 1 {
                    continue;
                }
                for v in [a, b] {
                    let q = out.mesh.positions[*v as usize];
                    let r = (q.x * q.x + q.y * q.y).sqrt();
                    far = far.max(r);
                    if r < 40.0 {
                        near_peak += 1;
                    }
                }
            }
            if open > 0 {
                println!(
                    "{label}:   open-edge ends within 40 mm of the axis: {near_peak} of {}; farthest {far:.1} mm out",
                    open * 2
                );
            }
            println!(
                "{label}: root {root:?} {} triangles, {open} open edges, in {:.2}s",
                out.mesh.triangles.len(),
                started.elapsed().as_secs_f64()
            );
        }
    };

    // What the panel now says beside the shell thickness: the surface's tightest fold, and
    // whether the shell reaches past it.
    let fold_now = |doc: &fieldform_scene::Document| {
        fieldform_field::mesh_fold_mm(doc.quad_meshes.get(&surface).expect("there"))
    };
    // How small the cells get: one period of the pattern is 1/(N*|grad s|) across, so the worst
    // gradient anywhere is the smallest cell anywhere. A mesher cannot resolve what is finer
    // than its own floor, and a cell it cannot resolve is a hole.
    let cells_now = |doc: &fieldform_scene::Document, cells_per_face: f64| {
        let mesh = doc.quad_meshes.get(&surface).expect("there");
        let map = fieldform_field::SurfaceMap::new(mesh, 4).expect("tessellates");
        let g = map.worst_parametric_gradient();
        (map.tightest_fold_mm(), 1.0 / (cells_per_face * g))
    };
    {
        let (limit_fold, smallest) = cells_now(&doc, 1.0);
        println!(
            "as saved: fold {:.1} mm control / {limit_fold:.1} mm limit; smallest cell {smallest:.2} mm",
            fold_now(&doc)
        );
    }

    // The same check on the file as saved, which must come back clean.
    {
        let root = graph.roots()[0];
        let sub = graph.subtree(root).expect("a root");
        let (lo, hi) = graph_extent(&doc, &graph, root).as_box().expect("bounded");
        let field = GraphField::compile_graph(&doc, &sub, ((hi - lo).norm() * 1e-3).max(1e-4))
            .expect("compiles");
        let margin = (hi - lo) * 0.02;
        let started = std::time::Instant::now();
        match fieldform_field::find_overstep(&field, (lo - margin, hi + margin), 64) {
            Some(j) => println!(
                "as saved: verify finds {:.2} mm per mm at {:?} in {:.1} s",
                j.rate,
                [j.at.x, j.at.y, j.at.z],
                started.elapsed().as_secs_f64()
            ),
            None => println!(
                "as saved: verify finds nothing too steep, in {:.1} s",
                started.elapsed().as_secs_f64()
            ),
        }
    }

    run(&doc, "as saved");

    // The same part in boxes grown upward a little, with nothing about the geometry changed:
    // if the count swings, it is where the octree's corners land that decides, not the edit.
    {
        let root = graph.roots()[0];
        let (lo, hi) = graph_extent(&doc, &graph, root).as_box().expect("bounded");
        let sub = graph.subtree(root).expect("a root");
        for grow in [0.0, 1.0, 5.0, 10.0, 20.0, 28.35] {
            let hi = hi + nalgebra::Vector3::new(0.0, 0.0, grow);
            let e = hi - lo;
            let tolerance = MeshDetail::Coarse.tolerance(e.x.max(e.y).max(e.z));
            let field = GraphField::compile_graph(&doc, &sub, (tolerance.min_cell_mm * 0.5).max(1e-4))
                .expect("compiles");
            let out = tessellate(&field, (lo, hi), &tolerance, false, false, &Cancel::new(), &Progress::new())
                .expect("not cancelled");
            println!("as saved, box grown up by {grow:>5.2} mm: {} triangles", out.mesh.triangles.len());
        }
    }

    // Up by a tenth of the surface's size, at the point nearest its middle — "moved a point up".
    let mesh = doc.quad_meshes.get_mut(&surface).expect("there");
    let ids: Vec<_> = mesh.vertex_ids().collect();
    let (mut lo, mut hi) = (mesh.vertex_position(ids[0]), mesh.vertex_position(ids[0]));
    for &v in &ids {
        let p = mesh.vertex_position(v);
        lo = lo.inf(&p);
        hi = hi.sup(&p);
    }
    let middle = nalgebra::center(&lo, &hi);
    let pick = *ids
        .iter()
        .min_by(|a, b| {
            (mesh.vertex_position(**a) - middle)
                .norm()
                .total_cmp(&(mesh.vertex_position(**b) - middle).norm())
        })
        .unwrap();
    let size = (hi - lo).norm();
    let was = mesh.vertex_position(pick);
    mesh.set_vertex_position(pick, was + nalgebra::Vector3::new(0.0, 0.0, size * 0.1));
    println!("moved point {} from {was:?} up by {:.2} mm (surface box {lo:?}..{hi:?})", pick.index(), size * 0.1);

    {
        let (limit_fold, smallest) = cells_now(&doc, 1.0);
        println!(
            "after the move: fold {:.1} mm control / {limit_fold:.1} mm limit; smallest cell {smallest:.2} mm",
            fold_now(&doc)
        );
    }

    // What `verify` now reports, run exactly as the application runs it.
    {
        let root = graph.roots()[0];
        let sub = graph.subtree(root).expect("a root");
        let (lo, hi) = graph_extent(&doc, &graph, root).as_box().expect("bounded");
        let field = GraphField::compile_graph(&doc, &sub, ((hi - lo).norm() * 1e-3).max(1e-4))
            .expect("compiles");
        let margin = (hi - lo) * 0.02;
        let started = std::time::Instant::now();
        let found = fieldform_field::find_overstep(&field, (lo - margin, hi + margin), 64);
        match found {
            Some(j) => println!(
                "after the move: verify finds {:.2} mm per mm at {:?} (moved {:.3} mm across {:.5} mm), in {:.1} s",
                j.rate,
                [j.at.x, j.at.y, j.at.z],
                j.moved,
                j.across,
                started.elapsed().as_secs_f64()
            ),
            None => println!(
                "after the move: verify finds nothing too steep, in {:.1} s",
                started.elapsed().as_secs_f64()
            ),
        }
    }

    // **Is the field actually discontinuous at the spike?** A lower-bound field promises to
    // change by no more than a millimetre per millimetre travelled. Where it breaks that, the
    // mesher's pruning is unsound and the surface tears -- which is what holes are. Swept over a
    // plane through the spike, fine enough that a real wall is several samples thick.
    {
        let root = graph.roots()[0];
        let sub = graph.subtree(root).expect("a root");
        let field = GraphField::compile_graph(&doc, &sub, 1e-3).expect("compiles");
        let step = 0.25f64;
        let mut worst = 0.0f64;
        let mut worst_at = nalgebra::Point3::origin();
        let mut z = 0.0;
        while z <= 60.0 {
            let mut x = -60.0;
            let mut last: Option<(f64, f64)> = None;
            while x <= 60.0 {
                let p = nalgebra::Point3::new(x, 0.0, z);
                let v = field.distance(p);
                if let Some((px, pv)) = last {
                    let rate = (v - pv).abs() / (x - px).abs();
                    if rate > worst {
                        worst = rate;
                        worst_at = p;
                    }
                }
                last = Some((x, v));
                x += step;
            }
            z += step;
        }
        println!(
            "after the move: the field changes at most {worst:.2} mm per mm, worst at {:?}              (a distance field promises 1.0)",
            [worst_at.x, worst_at.y, worst_at.z]
        );
    }

    run(&doc, "after the move");
}
