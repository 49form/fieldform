// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where a composed, meshed surface goes wrong, measured rather than looked at.
//!
//! Reproduces what the app draws in Field mode — a cage unioned with a pipe swept along a
//! curve, meshed at the viewport's own settings — and reports the three things that can be
//! wrong with the result, separately, because they have different fixes:
//!
//! - **Holes.** Every undirected edge of a closed surface belongs to exactly two triangles.
//!   Anything else is a tear, and reads as a bite taken out of the model.
//! - **Position.** A vertex should sit on the surface, so `|distance|` at it should be near
//!   zero. Reported against the cell size, which is what the mesher is entitled to be wrong by.
//! - **Normals.** What the surface is shaded with, against what the field says it should be.
//!   This is what faceting is, and it does not have to improve when the mesh gets finer.
//!
//! Run with:
//! `cargo run --release -p fieldform-field --example composed_mesh_quality`

use fieldform_cage::Cage;
use fieldform_field::{FieldMesh, GraphField, ScalarField, mesh_field};
use fieldform_scene::{BooleanOp, CageId, Curve, CurveId, Document, Graph, Node, ParamRef};
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use std::collections::HashMap;

/// The app's defaults, as shown in its own controls.
const CAGE_MM: f64 = 1000.0;
const PIPE_RADIUS_MM: f64 = 51.0;
const VIEWPORT_CELLS: usize = 64;

fn main() {
    println!(
        "A {CAGE_MM} mm cage unioned with a {PIPE_RADIUS_MM} mm pipe — what the app draws in \
         Field mode.\n"
    );

    println!(
        "  cells    cell     tris    holes   folds   position (mm)      spine facet (deg)  pipe"
    );
    println!("                                            worst    mean    worst    mean   cells");
    for cells in [VIEWPORT_CELLS, 96, 128, 192] {
        report(cells, curve_tolerance_from_cells(cells));
    }

    // Faceting along the pipe, which the mesh resolution cannot touch because it is in the
    // field: the spine is a polyline and the tube creases at each joint.
    let (doc, _) = scene();
    println!(
        "
  spine tolerance     points   worst joint (deg)"
    );
    for cells in [VIEWPORT_CELLS, 124, 192] {
        let t = curve_tolerance_from_cells(cells);
        let (n, worst) = spine_shape(&doc, t);
        println!("  {t:>8.2} mm ({cells:>3})   {n:>6}   {worst:>10.1}");
    }
    for t in [1.0, 0.1, 0.02] {
        let (n, worst) = spine_shape(&doc, t);
        println!("  {t:>8.2} mm        {n:>6}   {worst:>10.1}");
    }

    // Faceting *around* the pipe: the silhouette, which smooth vertex normals cannot hide.
    println!(
        "
  cells   pipe cells across   silhouette facet (deg)  worst    mean"
    );
    for cells in [64usize, 124, 192, 256] {
        let (doc, graph) = scene();
        let t = curve_tolerance_from_cells(cells);
        let field = GraphField::compile_graph(&doc, &graph, t).expect("valid graph");
        let bounds = composed_bounds();
        let extent = bounds.1 - bounds.0;
        let cell = extent.x.max(extent.y).max(extent.z) / cells as f64;
        let mesh = mesh_field(&field, bounds, cells);
        let (worst, mean) = facet_angle(&mesh);
        println!(
            "  {cells:>5}   {:>17.1}                           {worst:>6.1}  {mean:>6.1}",
            2.0 * PIPE_RADIUS_MM / cell
        );
    }

    println!(
        "\nThe same, with the curve tessellated finely instead of at the mesh's cell size.\n\
         A pipe's surface is a tube around a *polyline*, so its normal is perpendicular to\n\
         whichever segment is nearest and turns a corner at every joint. That crease is in the\n\
         field itself, so no amount of meshing resolution removes it — only a finer spine does."
    );
    println!("\n  cells    cell     tris    holes   folds   position (mm)      normal (deg)");
    println!("                                            worst    mean    worst    mean");
    for cells in [VIEWPORT_CELLS, 96, 128, 192] {
        report(cells, 0.05);
    }
}

/// What `build_composed_mesh` uses today: half a sampling cell.
fn curve_tolerance_from_cells(cells: usize) -> f64 {
    let bounds = composed_bounds();
    let extent = bounds.1 - bounds.0;
    let cell = extent.x.max(extent.y).max(extent.z) / cells as f64;
    (cell * 0.5).max(1e-4)
}

fn report(cells: usize, tolerance: f64) {
    let (doc, graph) = scene();
    let field = GraphField::compile_graph(&doc, &graph, tolerance).expect("valid graph");

    let bounds = composed_bounds();
    let extent = bounds.1 - bounds.0;
    let cell = extent.x.max(extent.y).max(extent.z) / cells as f64;

    let mesh = mesh_field(&field, bounds, cells);
    let (holes, folds) = topology(&mesh);
    let (pos_worst, pos_mean) = position_error(&mesh, &field);
    let (n_worst, n_mean) = spine_faceting(&doc, &graph, tolerance);
    // How many sampling cells the pipe is across. The app warns below two; three is not a
    // warning and is still barely a tube.
    let across = 2.0 * PIPE_RADIUS_MM / cell;

    println!(
        "  {cells:>5}  {cell:>6.2}  {:>7}  {holes:>7}  {folds:>6}  {pos_worst:>6.2}  \
         {pos_mean:>6.2}  {n_worst:>7.2}  {n_mean:>6.2}  {across:>6.1}",
        mesh.triangle_count()
    );
}

/// A cage unioned with a pipe, matching the app's field generator.
fn scene() -> (Document, Graph) {
    let mut doc = Document::new();
    let cage_id = CageId::new();
    doc.cages.insert(cage_id, Cage::cube_of(CAGE_MM).into());

    let curve_id = CurveId::new();
    let s = CAGE_MM;
    let curve = Curve::polyline(
        curve_id,
        "spine",
        vec![
            Point3::new(-s, -0.18 * s, -0.10 * s),
            Point3::new(-0.3 * s, -0.05 * s, 0.02 * s),
            Point3::new(0.3 * s, 0.10 * s, 0.05 * s),
            Point3::new(s, 0.22 * s, 0.12 * s),
        ],
        false,
    )
    .expect("valid polyline");
    doc.curves.insert(curve_id, curve);

    let mut graph = Graph::from_cage(cage_id);
    let cage_node = graph.roots()[0];
    let pipe = graph.add(Node::Pipe {
        curve: curve_id,
        radius: ParamRef::Literal(PIPE_RADIUS_MM),
    });
    graph.add(Node::Boolean {
        op: BooleanOp::Union,
        inputs: vec![cage_node.into(), pipe.into()],
    });

    (doc, graph)
}

/// The box the app would sample: cage vertices and the swept curve, padded.
fn composed_bounds() -> (Point3<f64>, Point3<f64>) {
    let s = CAGE_MM;
    let h = s * 0.5;
    let mut lo = Point3::new(-h, -h, -h);
    let mut hi = Point3::new(h, h, h);
    for p in [
        Point3::new(-s, -0.18 * s, -0.10 * s),
        Point3::new(s, 0.22 * s, 0.12 * s),
    ] {
        let pad = PIPE_RADIUS_MM;
        lo = Point3::new(
            lo.x.min(p.x - pad),
            lo.y.min(p.y - pad),
            lo.z.min(p.z - pad),
        );
        hi = Point3::new(
            hi.x.max(p.x + pad),
            hi.y.max(p.y + pad),
            hi.z.max(p.z + pad),
        );
    }
    (lo, hi)
}

/// Segment count and worst joint angle of the spine at a given tolerance.
///
/// A tube's normal points away from the nearest spine segment, so it turns a corner at every
/// joint. The joint angle *is* the crease angle on the tube's surface — this is faceting
/// along the pipe's length, as distinct from faceting around its circumference, which is the
/// mesh's business rather than the field's.
fn spine_shape(doc: &Document, tolerance: f64) -> (usize, f64) {
    let curve = doc.curves.values().next().expect("one curve");
    let pts = curve.tessellate(tolerance);
    let mut worst: f64 = 0.0;
    for w in pts.windows(3) {
        let a = (w[1] - w[0]).try_normalize(1e-12);
        let b = (w[2] - w[1]).try_normalize(1e-12);
        if let (Some(a), Some(b)) = (a, b) {
            worst = worst.max(a.dot(&b).clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    (pts.len(), worst)
}

/// Worst and mean angle between the face normals of triangles sharing an edge, restricted to
/// the pipe.
///
/// This is what faceting looks like on a silhouette. Vertex normals are analytic here, so the
/// *interior* of the tube shades smoothly however coarse the mesh — but the outline is a
/// polygon with one flat per cell around the circumference, and no amount of smooth shading
/// hides an outline.
fn facet_angle(mesh: &FieldMesh) -> (f64, f64) {
    let s = CAGE_MM;
    let near_pipe = |p: Point3<f64>| {
        // Well away from the cage, so this measures the tube rather than the sphere.
        p.x.abs() > 0.62 * s
    };
    let face_normal = |t: &[u32; 3]| {
        let a = mesh.positions[t[0] as usize];
        let b = mesh.positions[t[1] as usize];
        let c = mesh.positions[t[2] as usize];
        (b - a).cross(&(c - a)).try_normalize(1e-12)
    };

    let mut by_edge: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, t) in mesh.triangles.iter().enumerate() {
        let centre = mesh.positions[t[0] as usize];
        if !near_pipe(centre) {
            continue;
        }
        for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            by_edge
                .entry(if a < b { (a, b) } else { (b, a) })
                .or_default()
                .push(i);
        }
    }

    let mut angles = Vec::new();
    for tris in by_edge.values() {
        if tris.len() != 2 {
            continue;
        }
        let (Some(n0), Some(n1)) = (
            face_normal(&mesh.triangles[tris[0]]),
            face_normal(&mesh.triangles[tris[1]]),
        ) else {
            continue;
        };
        angles.push(n0.dot(&n1).clamp(-1.0, 1.0).acos().to_degrees());
    }
    if angles.is_empty() {
        return (0.0, 0.0);
    }
    (
        angles.iter().cloned().fold(0.0f64, f64::max),
        angles.iter().sum::<f64>() / angles.len() as f64,
    )
}

/// Boundary edges (holes) and edges shared by more than two triangles (folds).
fn topology(mesh: &FieldMesh) -> (usize, usize) {
    let mut uses: HashMap<(u32, u32), u32> = HashMap::new();
    for t in &mesh.triangles {
        for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *uses.entry(if a < b { (a, b) } else { (b, a) }).or_insert(0) += 1;
        }
    }
    (
        uses.values().filter(|&&c| c == 1).count(),
        uses.values().filter(|&&c| c > 2).count(),
    )
}

fn position_error(mesh: &FieldMesh, field: &GraphField) -> (f64, f64) {
    let d: Vec<f64> = mesh
        .positions
        .par_iter()
        .map(|p| field.distance(*p).abs())
        .collect();
    (
        d.iter().cloned().fold(0.0f64, f64::max),
        d.iter().sum::<f64>() / d.len().max(1) as f64,
    )
}

/// How far the pipe's normal is from what a smooth curve would give.
///
/// Not "mesh normal vs field normal": `mesh_field` shades *from* `field.normal`, so comparing
/// the two is comparing a value with itself and always reports zero. That is what this
/// measurement did at first, and it hid the answer.
///
/// The honest comparison is coarse spine against fine. A pipe is a tube around a polyline, so
/// its normal points away from whichever segment is nearest and turns a corner at every joint.
/// That crease lives in the field, so meshing more finely cannot remove it.
fn spine_faceting(doc: &Document, graph: &Graph, coarse: f64) -> (f64, f64) {
    let rough = GraphField::compile_graph(doc, graph, coarse).expect("valid graph");
    let fine = GraphField::compile_graph(doc, graph, 0.02).expect("valid graph");

    // Points on the tube, found by stepping around the spine at one radius.
    let s = CAGE_MM;
    let mut angles = Vec::new();
    for i in 0..400 {
        let t = i as f64 / 399.0;
        let along = Point3::new(
            -s + 2.0 * s * t,
            -0.18 * s + 0.40 * s * t,
            -0.10 * s + 0.22 * s * t,
        );
        for j in 0..16 {
            let a = j as f64 / 16.0 * std::f64::consts::TAU;
            let p = along + Vector3::new(0.0, a.cos(), a.sin()) * PIPE_RADIUS_MM;
            let (Some(r), Some(f)) = (rough.normal(p), fine.normal(p)) else {
                continue;
            };
            angles.push(r.dot(&f).clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    if angles.is_empty() {
        return (0.0, 0.0);
    }
    (
        angles.iter().cloned().fold(0.0f64, f64::max),
        angles.iter().sum::<f64>() / angles.len() as f64,
    )
}
