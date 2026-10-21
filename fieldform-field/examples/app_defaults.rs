// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Reproduces exactly what the app does with its default settings when you import a curve and
//! press Apply — headlessly, so the numbers can be looked at.
//!
//! Run with:
//! `cargo run --release -p fieldform-field --example app_defaults -- <file.igs>`

use fieldform_cage::Cage;
use fieldform_field::{GraphField, ScalarField, mesh_field};
use fieldform_scene::{BooleanOp, CageId, Document, Graph, Node, ParamRef};
use nalgebra::Point3;

// The app's defaults, copied from main.rs.
// Before: a fixed 5 mm, unrelated to anything imported.
const OLD_RADIUS_MM: f64 = 5.0;
// After: 2.5% of the curve's bounding diagonal, tidied.
fn new_radius(size: f64) -> f64 {
    let v = size * 0.025;
    let magnitude = 10f64.powf(v.log10().floor());
    (v / magnitude * 10.0).round() / 10.0 * magnitude
}
const DEFAULT_RESOLUTION: usize = 64;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: app_defaults <file.igs>");
    let import = fieldform_io::import_iges(std::path::Path::new(&path)).expect("valid IGES");
    let curve = import.curves.into_iter().next().expect("a curve");

    println!("curve: {:.1} mm long", curve.length(0.01));

    // The app starts with Cage::cube(), which is a UNIT cube — 1 mm across.
    let mut cmin = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut cmax = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for p in curve.tessellate(0.5) {
        cmin = Point3::new(cmin.x.min(p.x), cmin.y.min(p.y), cmin.z.min(p.z));
        cmax = Point3::new(cmax.x.max(p.x), cmax.y.max(p.y), cmax.z.max(p.z));
    }
    let curve_size = (cmax - cmin).norm();
    let radius_mm: f64 = if std::env::args().any(|a| a == "--old") {
        OLD_RADIUS_MM
    } else {
        new_radius(curve_size)
    };
    println!("curve bbox diagonal: {curve_size:.1} mm -> radius {radius_mm} mm");

    let cage = Cage::cube();
    let mut cage_lo = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut cage_hi = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for v in cage.vertex_ids() {
        let p = cage.vertex_position(v);
        cage_lo = Point3::new(cage_lo.x.min(p.x), cage_lo.y.min(p.y), cage_lo.z.min(p.z));
        cage_hi = Point3::new(cage_hi.x.max(p.x), cage_hi.y.max(p.y), cage_hi.z.max(p.z));
    }
    println!("default cage: {:.3} mm across", (cage_hi - cage_lo).norm());

    let mut doc = Document::new();
    let cage_id = CageId::new();
    doc.cages.insert(cage_id, cage);
    let curve_id = curve.id;
    doc.curves.insert(curve_id, curve);

    // The app's composed_bounds: cage vertices, plus the pipe when not subtracting.
    let mut lo = cage_lo;
    let mut hi = cage_hi;
    for p in doc.curves[&curve_id].tessellate(0.1) {
        let r = radius_mm;
        lo = Point3::new(lo.x.min(p.x - r), lo.y.min(p.y - r), lo.z.min(p.z - r));
        hi = Point3::new(hi.x.max(p.x + r), hi.y.max(p.y + r), hi.z.max(p.z + r));
    }
    let extent = hi - lo;
    let longest = extent.x.max(extent.y).max(extent.z);
    let cell = longest / DEFAULT_RESOLUTION as f64;

    println!("bounds: {:.1} mm on the longest axis", longest);
    println!("cell size at resolution {DEFAULT_RESOLUTION}: {cell:.1} mm");
    println!("pipe radius: {radius_mm} mm");
    println!(
        "-> the pipe is {:.2} cells across; below about 2 there is nothing for the mesher to \
         find",
        2.0 * radius_mm / cell
    );

    let mut graph = Graph::from_cage(cage_id);
    let cage_node = graph.roots()[0];
    let pipe = graph.add(Node::Pipe {
        curve: curve_id,
        radius: ParamRef::Literal(radius_mm),
    });
    graph.add(Node::Boolean {
        op: BooleanOp::Union,
        inputs: vec![cage_node.into(), pipe.into()],
    });

    let field = GraphField::compile_graph(&doc, &graph, 0.01).expect("valid graph");
    let mesh = mesh_field(&field, (lo, hi), DEFAULT_RESOLUTION);
    println!("\nmeshed: {} triangles", mesh.triangle_count());

    // How many grid samples actually landed inside anything?
    let mut inside = 0;
    let n = 40;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let p = Point3::new(
                    lo.x + extent.x * i as f64 / n as f64,
                    lo.y + extent.y * j as f64 / n as f64,
                    lo.z + extent.z * k as f64 / n as f64,
                );
                if field.distance(p) <= 0.0 {
                    inside += 1;
                }
            }
        }
    }
    println!(
        "{inside} of {} probe points are inside the solid",
        n * n * n
    );
}
