// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A lattice that follows an open surface has to stay a field past that surface's edge.
//!
//! Over the surface the closest point is reached along the normal, and everything about the map
//! is continuous. Past the rim the closest point is on the boundary, and the *signed distance*
//! flips sign across the surface's own extension while the point is still millimetres from the
//! edge. The lattice's third argument used to be that signed distance, so it jumped by twice the
//! distance to the edge, the cells jumped with it, and the field read 3.9 mm of room a hundredth
//! of a millimetre from material 8 mm deep. `gpu_matches_cpu`'s shell scene carried that as an
//! unexplained 8.35 mm for five days: both sides had the same discontinuity, so they agreed to
//! the last bit while the marcher stepped across it. The third argument is now the displacement
//! *along the normal* (`SurfacePoint::along`), which is the same number over the surface and
//! passes through zero past it.

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{
    CellMap, Document, Graph, LatticeCell, LatticePhase, Node, ParamRef, Part, PartId,
};
use nalgebra::{Point3, Vector3};

/// The domed panel `gpu_matches_cpu` shells: 800 mm across, 50 mm of rise, its rim at z = 0.
fn shelled_dome() -> (Document, PartId) {
    let cage = 1000.0;
    let (n, span, rise) = (4usize, cage * 0.8, cage * 0.05);
    let step = span / n as f64;
    let mut positions = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let (x, y) = (i as f64 * step - span * 0.5, j as f64 * step - span * 0.5);
            let (u, v) = (x / (span * 0.5), y / (span * 0.5));
            positions.push(Point3::new(x, y, rise * (1.0 - u * u) * (1.0 - v * v)));
        }
    }
    let at = |i: usize, j: usize| (j * (n + 1) + i) as u32;
    let mut faces = Vec::new();
    for j in 0..n {
        for i in 0..n {
            faces.push(vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
        }
    }
    let mut doc = Document::new();
    let surface = doc.add_quad_mesh(
        fieldform_cage::QuadMesh::from_faces(positions, &faces).expect("a domed panel"),
        "dome",
    );
    let mut graph = Graph::empty();
    graph.add(Node::Lattice {
        frame: fieldform_scene::Frame::default(),
        cell: LatticeCell::Gyroid,
        map: CellMap::Conformal {
            surface,
            cells_per_face: 2,
            depth: ParamRef::Literal(cage * 0.12),
            thickness: Some(ParamRef::Literal(cage * 0.05)),
        },
        phase: LatticePhase::sheet(ParamRef::Literal(cage * 0.02)),
    });
    let part =
        Part { id: PartId::new(), name: "shell".into(), material: None, graph, inputs: Vec::new() };
    let id = part.id;
    doc.add_part(part);
    (doc, id)
}

/// Lines that cross the surface's extension beyond each kind of edge: past a side, past a
/// corner, and the one `gpu_matches_cpu` found.
fn probes() -> Vec<(Point3<f64>, Vector3<f64>)> {
    let z = Vector3::z();
    vec![
        (Point3::new(413.79, -282.18, -15.0), z),
        // Off y = 0, which is a shared face edge and a different question -- see the last test.
        (Point3::new(405.0, 37.0, -15.0), z),
        (Point3::new(420.0, 150.0, -15.0), z),
        (Point3::new(0.0, -410.0, -15.0), z),
        (Point3::new(410.0, 410.0, -15.0), z),
        // Across the edge itself, a little above and below the rim's plane.
        (Point3::new(380.0, 100.0, 4.0), Vector3::x()),
        (Point3::new(380.0, 100.0, -4.0), Vector3::x()),
    ]
}

#[test]
fn the_value_never_moves_faster_than_the_point_does() {
    let (doc, part) = shelled_dome();
    let field = GraphField::compile(&doc, &doc.parts[&part], 0.25).expect("compiles");
    let travel = 0.01;
    for (from, dir) in probes() {
        let mut last = field.distance(from);
        let mut worst = (0.0f64, 0.0f64);
        let mut t = travel;
        while t <= 30.0 {
            let v = field.distance(from + dir * t);
            if (v - last).abs() > worst.0 {
                worst = ((v - last).abs(), t);
            }
            last = v;
            t += travel;
        }
        // A lower-bound field moves no faster than the point; the jump was 12.18 mm here.
        assert!(
            worst.0 <= travel * 1.05,
            "from {from:?} along {dir:?}: the value moved {:.4} mm over {travel} mm of travel, \
             {:.2} mm along",
            worst.0,
            worst.1
        );
    }
}

#[test]
fn it_never_overstates_the_room_before_its_material_past_the_rim() {
    let (doc, part) = shelled_dome();
    let field = GraphField::compile(&doc, &doc.parts[&part], 0.25).expect("compiles");
    for (from, dir) in probes() {
        let mut t = 0.0;
        while t <= 30.0 {
            let at = from + dir * t;
            let claimed = field.distance(at);
            if claimed > 0.0 {
                for step in 1..=16 {
                    let s = claimed * step as f64 / 16.0;
                    assert!(
                        field.distance(at + dir * s) > -1e-9,
                        "crossed material {s:.3} mm into a claimed {claimed:.3} mm at {at:?}"
                    );
                }
            }
            t += 0.5;
        }
    }
}

/// **A smaller step that is not the rim's, measured so it cannot grow unnoticed.** Each face
/// carries its own `(s, t)`, continuous across a shared edge but with a gradient that is not
/// (`conformal-cell-map-spec.md` section 6: the parametrization is C0). A wall's isovalue is
/// scaled by that gradient, so it steps a little from one face to the next, and the value with
/// it. On this dome that is under a tenth of a millimetre; on faces more dissimilar than these
/// it would be more, and a remesh toward even quads is what brings it down.
#[test]
fn the_step_across_a_shared_face_edge_is_small_and_known() {
    let (doc, part) = shelled_dome();
    let field = GraphField::compile(&doc, &doc.parts[&part], 0.25).expect("compiles");
    let travel = 0.01;
    let mut worst = 0.0f64;
    // Across y = 0, the edge the middle two rows of faces share, at three places along it.
    for x in [-250.0, 100.0, 330.0] {
        for z in [20.0, 45.0] {
            let from = Point3::new(x, -3.0, z);
            let mut last = field.distance(from);
            let mut t = travel;
            while t <= 6.0 {
                let v = field.distance(from + Vector3::y() * t);
                worst = worst.max((v - last).abs() - travel);
                last = v;
                t += travel;
            }
        }
    }
    println!("largest step across a shared face edge, beyond the travel itself: {worst:.4} mm");
    assert!(worst < 0.15, "the step across a face edge has grown to {worst:.4} mm");
}
