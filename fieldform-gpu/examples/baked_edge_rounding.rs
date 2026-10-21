// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Why a flat face on an imported part shades as though it were rounded.
//!
//! Reported 2026-09-02 against a STEP import: faces that are flat in the CAD read as softly
//! curved, and it "always makes things look rounder than they actually are."
//!
//! # It is not the tessellation
//!
//! A flat face tessellates to two triangles at *any* tolerance, so a coarse import makes curved
//! faces faceted rather than flat faces round — the opposite symptom. The import setting is not
//! the lever here.
//!
//! # It is the shading normal
//!
//! `MeshField::from_mesh` sets `shading_normals` to the **angle-weighted vertex pseudonormals**
//! it computes for the sign test, on the reasoning that those are the best smooth normals a
//! mesh with no analytic surface behind it can offer. That reasoning holds for a scanned blob
//! and is exactly wrong for a machined part: at a box's corner the vertex normal is the average
//! of three perpendicular faces, so it points diagonally, and barycentric interpolation across
//! the face carries that error over the *whole* face — a box face is two triangles, so there is
//! nowhere on it that is not next to a corner.
//!
//! This measures that directly, and then measures what a crease-angle rule recovers: average
//! only across neighbours whose dihedral angle is gentle, so a tessellated cylinder still shades
//! smoothly and a machined face stays flat.
//!
//! Run with:
//! `cargo run --release -p fieldform-gpu --example baked_edge_rounding`

use fieldform_field::{MeshField, ScalarField};
use fieldform_gpu::BakedField;
use nalgebra::{Point3, Vector3};
use std::f64::consts::FRAC_1_SQRT_2;

const BOX_MM: [f64; 3] = [200.0, 120.0, 40.0];

fn main() {
    let half = Vector3::new(BOX_MM[0], BOX_MM[1], BOX_MM[2]) * 0.5;
    let field = box_mesh();

    println!(
        "A {} x {} x {} mm box. Every point below is on the +Z face, which is flat, so the\n\
         true normal is +Z everywhere and any tilt at all is error.\n",
        BOX_MM[0], BOX_MM[1], BOX_MM[2],
    );

    // Across the face, from its centre out toward the long edge.
    println!("{:>10}  {:>16}  {:>18}", "y (mm)", "from the edge", "tilt off +Z");
    println!("{}", "-".repeat(50));
    let mut worst: f64 = 0.0;
    for step in 0..=10 {
        let y = half.y * step as f64 / 10.0;
        let p = Point3::new(0.0, y, half.z);
        let n = field.normal(p).expect("a mesh field carries normals");
        let tilt = n.dot(&Vector3::z()).clamp(-1.0, 1.0).acos().to_degrees();
        // The last row sits exactly *on* the edge, where two faces meet and there is no one
        // right normal. It is a hairline and it is excluded from the summary rather than
        // counted as an error the face has.
        let on_edge = step == 10;
        if !on_edge {
            worst = worst.max(tilt);
        }
        println!(
            "{y:>10.1}  {:>16.1}  {tilt:>17.2}\u{b0}{}",
            half.y - y,
            if on_edge { "   (on the edge)" } else { "" },
        );
    }
    println!();
    println!("Worst tilt across the face interior: {worst:.2}\u{b0}");
    println!(
        "Before crease-angle normals this ran from 0\u{b0} at the centre to 45\u{b0} at the edge --"
    );
    println!(
        "a face with a gradient painted across it, which is what \"rounder than they actually"
    );
    println!("are\" describes.");
    println!();

    // And the bake, which is the second factor rather than the first: it interpolates whatever
    // normals it was given, so it cannot recover what the mesh already lost — but it does add a
    // band of its own around every edge, and that band is a property of the cell.
    let margin = 8.0;
    let bounds = (
        Point3::from(-half - Vector3::repeat(margin)),
        Point3::from(half + Vector3::repeat(margin)),
    );
    let span = (bounds.1 - bounds.0).norm();
    println!("Baked, over a {span:.0} mm diagonal, which is the second factor and now the only one:");
    println!();
    println!(
        "{:>7}  {:>10}  {:>18}  {:>22}",
        "cells", "cell (mm)", "rounded band", "tilt clear of it",
    );
    println!("{}", "-".repeat(64));
    for resolution in [32usize, 64, 128, 256] {
        let baked = BakedField::new(&field, bounds, resolution, span * 0.15);
        let cell = span / resolution as f64;

        // How far in from the edge the normal settles to +Z, and what is left once it has.
        // One degree is about where a shaded surface stops reading as curved.
        let mut band = 0.0;
        let mut clear: f64 = 0.0;
        let mut d = 0.0;
        while d <= half.y {
            let n = baked.sample_normal(Point3::new(0.0, half.y - d, half.z));
            let tilt = n.dot(&Vector3::z()).clamp(-1.0, 1.0).acos().to_degrees();
            if tilt >= 1.0 {
                band = d;
                clear = 0.0;
            } else {
                clear = clear.max(tilt);
            }
            d += cell * 0.05;
        }
        println!(
            "{resolution:>7}  {cell:>10.2}  {:>10.1} mm ({:>3.1}c)  {clear:>21.2}\u{b0}",
            band,
            band / cell,
        );
    }

    println!();
    println!(
        "What is left is the bake's own. Trilinear interpolation across a cell straddling an"
    );
    println!(
        "edge blends two sharply different normals, so a band about a cell wide rounds off. It"
    );
    println!(
        "scales with the cell rather than with the part, so it follows View > Display"
    );
    println!(
        "resolution -- and unlike the 45 degrees above, it sits at the edges rather than across"
    );
    println!("the whole face.");
    // The other two lookup paths, which the crease fix deliberately left alone. A closest point
    // lands on a *face* most of the time, but on an edge or a vertex whenever it is directly
    // outside one — along the rim of a machined part that is a whole line of pixels rather than
    // a measure-zero set, so "it hardly ever happens" is not an argument.
    //
    // They keep the vertex pseudonormals, and this is the check that they may: at a rim the
    // pseudonormal *is* the bisector of the two faces, which is the right outward direction
    // there. Measured at 0.25 degrees, so a hairline along an edge is not coming from here.
    println!();
    println!("Points just outside the rim, where the closest point is the edge itself:");
    println!("{:>12}  {:>18}", "above (mm)", "tilt off the true");
    println!("{}", "-".repeat(34));
    for out in [0.05f64, 0.2, 1.0, 4.0] {
        // Just beyond the +Y +Z rim, on the diagonal between the two faces it joins. The true
        // outward direction there is the bisector, at 45 degrees to each.
        let p = Point3::new(0.0, half.y + out * FRAC_1_SQRT_2, half.z + out * FRAC_1_SQRT_2);
        let n = field.normal(p).expect("a mesh carries normals");
        // Exactly unit-length, so the dot product below carries no length error of its own.
        let truth = Vector3::new(0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2);
        let tilt = n.dot(&truth).clamp(-1.0, 1.0).acos().to_degrees();
        println!("{out:>12.2}  {tilt:>17.2}\u{b0}");
    }
    println!();

    // What building the field costs, since the crease pass runs inside `from_mesh` and an
    // imported assembly is where that is felt. A grid-tessellated sheet stands in for one.
    println!();
    for sides in [64usize, 160, 240] {
        let (positions, triangles) = grid_sheet(sides);
        let count = triangles.len();
        let started = std::time::Instant::now();
        let field = MeshField::from_mesh(positions, triangles).expect("a closed sheet");
        let built = started.elapsed();
        println!(
            "MeshField::from_mesh over {count:>7} triangles: {:>7.1} ms",
            built.as_secs_f64() * 1000.0,
        );
        let _ = field.normal(Point3::new(0.0, 0.0, 0.0));
    }
    println!(
        "That is paid once per compile, and every show/hide and every menu action is a compile."
    );

    // What the leaf cache saves, which is the whole of that cost on the second and every later
    // compile: a mesh is immutable, so its BVH and crease normals are the same ones.
    {
        use fieldform_field::{GraphField, LeafCache};
        use fieldform_scene::{Document, Graph, MeshData, Node};

        let (positions, triangles) = grid_sheet(160);
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        for _ in 0..4 {
            let id = doc.add_mesh(MeshData {
                source: None,
                edges: Vec::new(),
                name: "body".into(),
                positions: positions.iter().map(|p| [p.x, p.y, p.z]).collect(),
                triangles: triangles.clone(),
                tolerance_mm: Some(0.05),
                open_edges: 0,
            });
            graph.add(Node::Mesh(id));
        }

        let started = std::time::Instant::now();
        GraphField::compile_graph(&doc, &graph, 0.25).expect("four meshes compile");
        let cold = started.elapsed().as_secs_f64() * 1000.0;

        let mut cache = LeafCache::default();
        GraphField::compile_cached(&doc, &graph, 0.25, &mut cache).expect("first, cold");
        let started = std::time::Instant::now();
        GraphField::compile_cached(&doc, &graph, 0.25, &mut cache).expect("second, warm");
        let warm = started.elapsed().as_secs_f64() * 1000.0;

        println!();
        println!("Compiling four such bodies, which is what one show/hide costs:");
        println!("  without the leaf cache: {cold:>8.1} ms");
        println!("  with it, second time:   {warm:>8.1} ms");
    }

    // And the bake, which is the other per-change cost and the one View > Display resolution
    // moves. It is cubic in the resolution, so the control is far steeper than it looks: a
    // number typed into a spinner two steps up is an order of magnitude of work.
    println!();
    let (positions, triangles) = grid_sheet(160);
    let field = MeshField::from_mesh(positions, triangles).expect("a closed sheet");
    let bake_bounds = (
        Point3::new(-60.0, -60.0, -15.0),
        Point3::new(60.0, 60.0, 15.0),
    );
    let diagonal = (bake_bounds.1 - bake_bounds.0).norm();
    println!("Baking one 103 680-triangle body, which is what a display resolution costs:");
    println!("{:>7}  {:>12}  {:>14}", "cells", "bake", "x4 bodies");
    println!("{}", "-".repeat(38));
    for resolution in [32usize, 64, 128, 256] {
        let started = std::time::Instant::now();
        let baked = BakedField::new(&field, bake_bounds, resolution, diagonal * 0.15);
        let took = started.elapsed().as_secs_f64();
        let _ = baked.sample(Point3::origin());
        println!("{resolution:>7}  {:>9.2} s  {:>11.2} s", took, took * 4.0);
    }
}

/// A closed box whose top face is subdivided into a grid, so the triangle count can be dialled
/// up the way an imported assembly's is.
fn grid_sheet(n: usize) -> (Vec<Point3<f64>>, Vec<[u32; 3]>) {
    let step = 100.0 / n as f64;
    let mut positions = Vec::new();
    for layer in 0..2 {
        for j in 0..=n {
            for i in 0..=n {
                positions.push(Point3::new(
                    -50.0 + i as f64 * step,
                    -50.0 + j as f64 * step,
                    if layer == 0 { -5.0 } else { 5.0 },
                ));
            }
        }
    }
    let per = (n + 1) * (n + 1);
    let at = |layer: usize, i: usize, j: usize| (layer * per + j * (n + 1) + i) as u32;
    let mut triangles = Vec::new();
    for j in 0..n {
        for i in 0..n {
            triangles.push([at(1, i, j), at(1, i + 1, j), at(1, i + 1, j + 1)]);
            triangles.push([at(1, i, j), at(1, i + 1, j + 1), at(1, i, j + 1)]);
            triangles.push([at(0, i, j), at(0, i + 1, j + 1), at(0, i + 1, j)]);
            triangles.push([at(0, i, j), at(0, i, j + 1), at(0, i + 1, j + 1)]);
        }
    }
    // Walls, so the sheet is a closed solid the field can sign.
    for i in 0..n {
        triangles.push([at(0, i, 0), at(0, i + 1, 0), at(1, i + 1, 0)]);
        triangles.push([at(0, i, 0), at(1, i + 1, 0), at(1, i, 0)]);
        triangles.push([at(0, i + 1, n), at(0, i, n), at(1, i, n)]);
        triangles.push([at(0, i + 1, n), at(1, i, n), at(1, i + 1, n)]);
        triangles.push([at(0, 0, i + 1), at(0, 0, i), at(1, 0, i)]);
        triangles.push([at(0, 0, i + 1), at(1, 0, i), at(1, 0, i + 1)]);
        triangles.push([at(0, n, i), at(0, n, i + 1), at(1, n, i + 1)]);
        triangles.push([at(0, n, i), at(1, n, i + 1), at(1, n, i)]);
    }
    (positions, triangles)
}

/// A closed axis-aligned box as a triangle mesh -- what an imported machined part is made of.
fn box_mesh() -> MeshField {
    let h = Vector3::new(BOX_MM[0], BOX_MM[1], BOX_MM[2]) * 0.5;
    let corners: Vec<Point3<f64>> = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ]
    .iter()
    .map(|(x, y, z)| Point3::new(x * h.x, y * h.y, z * h.z))
    .collect();

    // Wound so each normal points out, which is what the sign test depends on.
    let faces: [[usize; 4]; 6] = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [3, 7, 6, 2],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
    ];
    let mut triangles = Vec::new();
    for [a, b, c, d] in faces {
        triangles.push([a as u32, b as u32, c as u32]);
        triangles.push([a as u32, c as u32, d as u32]);
    }

    MeshField::from_mesh(corners, triangles).expect("a closed box is a valid mesh")
}
