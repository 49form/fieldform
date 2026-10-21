// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What an export of a real document costs and how sound it is, at several cell sizes.
//!
//! Run against the file that prompted the question rather than a synthetic stand-in, because
//! the thing being asked about — a blend seam crossing a lattice — is not something a test
//! field reproduces by accident.
//!
//! Reports the three faults that matter in a file handed to a slicer, separately, because they
//! have different fixes:
//!
//! - **Slivers.** Triangles with almost no area. They carry no reliable normal, so a flat-shaded
//!   viewer lights them arbitrarily — the white starbursts.
//! - **Open edges.** An edge used by other than two triangles is a hole, and a slicer will say
//!   so before it says anything else.
//! - **Cost.** Triangles, megabytes, seconds.
//!
//! `cargo run --release -p fieldform-io --example export_quality -- <file.fieldform>`

use std::collections::HashMap;

use fieldform_field::ScalarField;

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: export_quality <file.fieldform> [cell_mm ...]");
        std::process::exit(2);
    };
    let sizes: Vec<f64> = {
        let given: Vec<f64> = std::env::args().skip(2).filter_map(|a| a.parse().ok()).collect();
        if given.is_empty() { vec![0.8, 0.4, 0.2] } else { given }
    };

    let (doc, _archive) = fieldform_io::open(std::path::Path::new(&path)).expect("opens");
    let part = doc.parts.values().next().expect("a part");
    let graph = &part.graph;

    // The box every root reaches, which is what the application exports within.
    let mut lo = nalgebra::Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut hi = nalgebra::Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for r in graph.roots() {
        if let Some((a, b)) = fieldform_field::graph_extent(&doc, graph, r).as_box() {
            lo = nalgebra::Point3::new(lo.x.min(a.x), lo.y.min(a.y), lo.z.min(a.z));
            hi = nalgebra::Point3::new(hi.x.max(b.x), hi.y.max(b.y), hi.z.max(b.z));
        }
    }
    let longest = (hi - lo).x.max((hi - lo).y).max((hi - lo).z);
    println!("part box {longest:.1} mm across\n");
    println!(
        "{:>8}  {:>7}  {:>11}  {:>9}  {:>8}  {:>8}  {:>10}",
        "cell mm", "across", "triangles", "STL", "time", "slivers", "open edges"
    );

    for cell_mm in sizes {

        // Widened on demand. A blend pushes the surface *outward* by up to its radius, so a
        // box taken from the operands' extents can cut through geometry the blend created --
        // and a clipped surface is left open where it leaves the box, which is a hole along a
        // curve rather than over an area. That is the shape of the open-edge count, so it is
        // worth being able to rule in or out.
        let margin = cell_mm
            * std::env::var("FIELDFORM_MARGIN_CELLS")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(2.0);
        let bounds = (
            nalgebra::Point3::new(lo.x - margin, lo.y - margin, lo.z - margin),
            nalgebra::Point3::new(hi.x + margin, hi.y + margin, hi.z + margin),
        );
        // From the box actually being sampled, so widening the margin does not silently
        // coarsen the cells -- which made a first comparison between margins meaningless.
        let span = bounds.1 - bounds.0;
        let cells =
            ((span.x.max(span.y).max(span.z) / cell_mm).ceil() as usize).clamp(8, 2048);
        let field =
            fieldform_field::GraphField::compile_graph(&doc, graph, (cell_mm * 0.5).max(1e-4))
                .expect("compiles");

        let started = std::time::Instant::now();
        let mesh = if std::env::var("FIELDFORM_ADAPTIVE").is_ok() {
            // Matched to the uniform run as closely as the two can be: the chord tolerance is
            // what a cell of this size buys on a curved surface, and the floor is that cell.
            // The angular tolerance has no uniform counterpart at all -- it is the control the
            // grid cannot express.
            fieldform_field::mesh_field_adaptive(
                &field,
                bounds,
                &fieldform_field::MeshTolerance {
                    chord_mm: std::env::var("FIELDFORM_CHORD")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(cell_mm * 0.25),
                    angle_deg: std::env::var("FIELDFORM_ANGLE")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(20.0),
                    max_edge_mm: 0.0,
                    min_cell_mm: cell_mm,
                },
            )
        } else {
            fieldform_field::mesh_field(&field, bounds, cells)
        };
        // Optional, so the same harness reports both sides of the question.
        let mut mesh = mesh;
        if let Ok(t) = std::env::var("FIELDFORM_SIMPLIFY") {
            let tol: f64 = t.parse().unwrap_or(cell_mm * 0.25);
            let report = fieldform_field::simplify(&mut mesh, &field, tol);
            eprintln!(
                "   simplified at {tol:.3} mm: {} -> {} triangles ({:.1}% saved),                  {} refused for error, {} for topology",
                report.triangles_before,
                report.triangles_after,
                report.saved() * 100.0,
                report.refused_for_error,
                report.refused_for_topology,
            );
        }
        if std::env::var("FIELDFORM_WELD").is_ok() {
            let dropped = fieldform_field::weld(&mut mesh, cell_mm * 1e-6);
            eprintln!("   welded away {dropped} degenerate triangles");
        }
        if let Ok(t) = std::env::var("FIELDFORM_RELAX") {
            let tol: f64 = t.parse().unwrap_or(cell_mm * 0.25);
            let moved = fieldform_field::relax(&mut mesh, &field, tol);
            eprintln!("   relaxed {moved} vertex moves at {tol:.3} mm");
        }
        if let Ok(t) = std::env::var("FIELDFORM_FLIP") {
            let tol: f64 = t.parse().unwrap_or(cell_mm * 0.25);
            let made = fieldform_field::improve_shape(&mut mesh, &field, tol);
            eprintln!("   flipped {made} edges at {tol:.3} mm");
        }
        let took = started.elapsed().as_secs_f64();

        let mut slivers = 0usize;
        // Which kind of degeneracy: two corners in the same place, or three distinct
        // corners in a line. They need different fixes — coincident corners can be welded and
        // the quad emitted as one triangle, which keeps the surface closed; collinear ones are
        // a real if thin piece of surface and may not be dropped at all.
        let (mut coincident, mut collinear) = (0usize, 0usize);
        // Triangles wound the wrong way round. A flat-shaded viewer lights these from the
        // inside, which reads as a bright speck on an otherwise smooth surface -- the
        // starbursts. Judged against the mesh's own vertex normals, which come from the field
        // rather than from the winding, so the two are independent answers.
        let mut flipped = 0usize;
        for t in &mesh.triangles {
            let (a, b, c) = (
                mesh.positions[t[0] as usize],
                mesh.positions[t[1] as usize],
                mesh.positions[t[2] as usize],
            );
            // A thousandth of a cell's area: small enough that nothing deliberate is this size,
            // large enough to catch what a shading pass will struggle with.
            let winding = (b - a).cross(&(c - a));
            let field_says = mesh.normals[t[0] as usize]
                + mesh.normals[t[1] as usize]
                + mesh.normals[t[2] as usize];
            // Only where the question is meaningful. A triangle with almost no area has a
            // winding normal made of rounding error, and one spanning a wall thinner than a
            // cell has vertex normals pointing opposite ways whose sum means nothing. Counting
            // those inflates the number with cases no shading pass can get right anyway.
            let na = mesh.normals[t[0] as usize];
            let nb = mesh.normals[t[1] as usize];
            let nc = mesh.normals[t[2] as usize];
            let coherent = na.dot(&nb) > 0.5 && nb.dot(&nc) > 0.5 && nc.dot(&na) > 0.5;
            let big = winding.norm() * 0.5 > cell_mm * cell_mm * 1e-2;
            if coherent && big && winding.dot(&field_says) < 0.0 {
                flipped += 1;
            }
            if (b - a).cross(&(c - a)).norm() * 0.5 < cell_mm * cell_mm * 1e-3 {
                slivers += 1;
                let eps = cell_mm * 1e-6;
                if (a - b).norm() < eps || (b - c).norm() < eps || (c - a).norm() < eps {
                    coincident += 1;
                } else {
                    collinear += 1;
                }
            }
        }

        let mut uses: HashMap<(u32, u32), i32> = HashMap::new();
        for t in &mesh.triangles {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *uses.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        // Split, because the two failures are opposite. An edge used *once* is a hole: a quad
        // that should have been emitted was not. An edge used *three or more* times is
        // non-manifold: more surface meets there than a surface may, which is what one vertex
        // per cell produces where two sheets pass through the same cell -- the ambiguous case
        // that a manifold criterion exists to catch. A slicer objects to both and they are
        // fixed in different places.
        let holes = uses.values().filter(|n| **n == 1).count();
        let nonmanifold = uses.values().filter(|n| **n > 2).count();
        let open = holes + nonmanifold;

        // How far the mesh strays from the field, sampled *over the triangles* rather than at
        // their corners. A vertex check cannot see this: a triangle can have all three corners
        // on the surface and bulge away from it in the middle, which is exactly what an
        // over-eager simplifier produces.
        let mut worst_off = 0.0f64;
        let mut worst_at: Option<(nalgebra::Point3<f64>, f64)> = None;
        let mut mean_off = 0.0f64;
        let mut samples = 0usize;
        for t in &mesh.triangles {
            let (a, b, c) = (
                mesh.positions[t[0] as usize],
                mesh.positions[t[1] as usize],
                mesh.positions[t[2] as usize],
            );
            // Barycentric points including the middle and the edges, which is where a flat
            // triangle strays furthest from a curved surface.
            for (u, v) in [
                (1.0 / 3.0, 1.0 / 3.0),
                (0.5, 0.5),
                (0.5, 0.0),
                (0.0, 0.5),
                (0.25, 0.25),
                (0.5, 0.25),
                (0.25, 0.5),
            ] {
                let p = a + (b - a) * u + (c - a) * v;
                let d = field.distance(p).abs();
                if d > worst_off {
                    worst_off = d;
                    worst_at = Some((p, (b - a).norm().max((c - b).norm()).max((a - c).norm())));
                }
                mean_off += d;
                samples += 1;
            }
        }
        if samples > 0 {
            mean_off /= samples as f64;
        }
        // **The knife-edge metric, and it was computed and thrown away.** Backlog 27 reads the
        // collinear count as the size of the fringe standing off a taper -- three distinct
        // corners in a line is what a cell straddling a vanishing wall produces -- and cites it
        // as going 229 to 4. The summary line only ever printed the total.
        eprintln!("   slivers: {coincident} coincident, {collinear} collinear (the knife-edge fringe)");
        eprintln!(
            "   off the field: mean {mean_off:.4} mm, worst {worst_off:.4} mm over {samples}              samples on {} triangles",
            mesh.triangles.len(),
        );
        if let Some((p, longest)) = worst_at {
            eprintln!(
                "   worst at ({:.2}, {:.2}, {:.2}) on a triangle {longest:.3} mm across",
                p.x, p.y, p.z,
            );
        }

        // Triangle *shape*, which is what the comparison with other CAD software was about: at
        // the same triangle
        // count a wireframe of near-equilateral triangles reads as a considered mesh and a
        // wireframe of slivers does not. The smallest angle is the usual measure — an
        // equilateral triangle has sixty degrees, and much under twenty is a sliver.
        let mut worst_angle = 180.0f64;
        let mut mean_angle = 0.0f64;
        let mut under_20 = 0usize;
        for t in &mesh.triangles {
            let (a, b, c) = (
                mesh.positions[t[0] as usize],
                mesh.positions[t[1] as usize],
                mesh.positions[t[2] as usize],
            );
            let mut smallest = 180.0f64;
            for (p, q, r) in [(a, b, c), (b, c, a), (c, a, b)] {
                if let (Some(u), Some(v)) =
                    ((q - p).try_normalize(1e-12), (r - p).try_normalize(1e-12))
                {
                    smallest = smallest.min(u.dot(&v).clamp(-1.0, 1.0).acos().to_degrees());
                }
            }
            worst_angle = worst_angle.min(smallest);
            mean_angle += smallest;
            if smallest < 20.0 {
                under_20 += 1;
            }
        }
        if !mesh.triangles.is_empty() {
            mean_angle /= mesh.triangles.len() as f64;
        }
        eprintln!(
            "   triangle shape: smallest angle {worst_angle:.2} deg, mean smallest \
             {mean_angle:.1} deg, {under_20} under 20 deg ({:.1}%)",
            100.0 * under_20 as f64 / mesh.triangles.len().max(1) as f64,
        );

        let bytes = 84 + mesh.triangles.len() * 50;
        println!(
            "{cell_mm:>8.3}  {cells:>7}  {:>11}  {:>7.1} MB  {took:>7.1}s  {slivers:>8}  \
             {open:>10}   holes {holes}, non-manifold {nonmanifold}, flipped {flipped}",
            mesh.triangles.len(),
            bytes as f64 / (1024.0 * 1024.0),
        );
    }
}
