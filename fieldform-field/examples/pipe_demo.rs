// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Imports an IGES curve, sweeps it into a pipe, and booleans it against a cage — printing
//! cross-sections so the result can be checked without a viewport.
//!
//! The app draws this interactively now, so this exists as the headless check: the slice
//! confirms the boolean went the right way at a glance, and the STL confirms the mesh is
//! real geometry rather than something that only looks right in one projection.
//!
//! Run with:
//! `cargo run --release -p fieldform-field --example pipe_demo -- <file.igs> [radius_mm]`

use fieldform_cage::Cage;
use fieldform_field::{FieldMesh, GraphField, ScalarField, mesh_field};
use fieldform_scene::{BooleanOp, CageId, Document, Graph, Node, ParamRef};
use nalgebra::Point3;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: pipe_demo <file.igs> [radius_mm]");
        std::process::exit(2);
    };
    let radius: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(100.0);

    let import = match fieldform_io::import_iges(std::path::Path::new(&path)) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        }
    };
    println!("{}", import.summary());

    let Some(curve) = import.curves.into_iter().next() else {
        eprintln!("no curves in that file");
        std::process::exit(1);
    };
    let (lo, hi) = curve.domain();
    println!(
        "using \"{}\": {:.1} mm long, pipe radius {radius} mm\n",
        curve.name,
        curve.length(0.01)
    );

    let mut doc = Document::new();
    let curve_id = curve.id;
    doc.curves.insert(curve_id, curve);

    // A cage to boolean against, scaled up so it actually meets a 2 m curve.
    let cage_id = CageId::new();
    doc.cages.insert(cage_id, scaled_cube(600.0));

    for (label, op) in [
        ("pipe alone", None),
        ("cage UNION pipe", Some(BooleanOp::Union)),
        ("cage MINUS pipe", Some(BooleanOp::Difference)),
    ] {
        let mut graph = Graph::from_cage(cage_id);
        let cage_node = graph.roots()[0];
        let pipe = graph.add(Node::Pipe {
            curve: curve_id,
            radius: ParamRef::Literal(radius),
        });
        match op {
            None => pipe,
            Some(op) => graph.add(Node::Boolean {
                op,
                inputs: vec![cage_node.into(), pipe.into()],
            }),
        };

        let field = GraphField::compile_graph(&doc, &graph, 0.01).expect("valid graph");
        println!("--- {label} ---");
        slice(&field, lo, hi);

        // Mesh it too, and write an STL — the slice shows the shape is right, the mesh shows
        // it is watertight and openable in something that renders triangles.
        let bounds = (
            Point3::new(-1400.0, -900.0, -900.0),
            Point3::new(1400.0, 900.0, 900.0),
        );
        let started = std::time::Instant::now();
        let mesh = mesh_field(&field, bounds, 96);
        let path = format!("pipe_demo_{}.stl", label.replace(' ', "_"));
        write_stl(&mesh, &path);
        println!(
            "{} triangles in {:.2} s -> {path}",
            mesh.triangle_count(),
            started.elapsed().as_secs_f64()
        );
        println!();
    }
}

/// Writes a binary STL. Deliberately here rather than in `fieldform-io`: this is a diagnostic
/// dump, and the real export path (M4) should go through Fidget's mesher with its own
/// resolution and units handling rather than inherit this example's choices.
fn write_stl(mesh: &FieldMesh, path: &str) {
    use std::io::Write;
    let file = std::fs::File::create(path).expect("cannot create STL");
    let mut w = std::io::BufWriter::new(file);

    w.write_all(&[0u8; 80]).expect("header");
    w.write_all(&(mesh.triangles.len() as u32).to_le_bytes())
        .expect("count");

    for t in &mesh.triangles {
        let p: Vec<_> = t.iter().map(|&i| mesh.positions[i as usize]).collect();
        let n = (p[1] - p[0]).cross(&(p[2] - p[0]));
        let n = n.try_normalize(1e-12).unwrap_or_else(nalgebra::Vector3::z);
        for v in [n.x, n.y, n.z] {
            w.write_all(&(v as f32).to_le_bytes()).expect("normal");
        }
        for q in &p {
            for v in [q.x, q.y, q.z] {
                w.write_all(&(v as f32).to_le_bytes()).expect("vertex");
            }
        }
        w.write_all(&0u16.to_le_bytes()).expect("attribute");
    }
    w.flush().expect("flush");
}

/// A cube cage of the given size, since `Cage::cube()` is a unit cube and the test curve is
/// two metres long.
fn scaled_cube(size: f64) -> Cage {
    let cube = Cage::cube();
    let positions: Vec<Point3<f64>> = cube
        .vertex_ids()
        .map(|v| Point3::from(cube.vertex_position(v).coords * size))
        .collect();
    let faces: Vec<Vec<u32>> = cube
        .face_ids()
        .map(|f| {
            cube.face_vertices(f)
                .iter()
                .map(|v| v.index() as u32)
                .collect()
        })
        .collect();
    Cage::from_faces(positions, &faces).expect("scaling preserves the cage invariants")
}

/// Prints the Z = 0 slice of a field as ASCII: `#` inside the solid, `.` outside.
///
/// Crude on purpose. The point is to confirm the pipe follows the curve and that the boolean
/// went the right way, both of which are obvious at a glance and neither of which a single
/// distance value would show.
fn slice(field: &dyn ScalarField, _lo: f64, _hi: f64) {
    const WIDTH: usize = 100;
    const HEIGHT: usize = 25;
    const EXTENT: f64 = 1300.0; // a little past the 2000 mm curve's ends

    for row in 0..HEIGHT {
        let y = EXTENT * 0.55 * (1.0 - 2.0 * row as f64 / (HEIGHT - 1) as f64);
        let line: String = (0..WIDTH)
            .map(|col| {
                let x = -EXTENT + 2.0 * EXTENT * col as f64 / (WIDTH - 1) as f64;
                if field.distance(Point3::new(x, y, 0.0)) <= 0.0 {
                    '#'
                } else {
                    '.'
                }
            })
            .collect();
        println!("{line}");
    }
}
