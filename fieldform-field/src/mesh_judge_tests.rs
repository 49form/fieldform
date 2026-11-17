// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
use super::*;
use crate::{tessellate, Cancel, FieldMesh, MeshTolerance, Progress, Sphere};

fn ball() -> Sphere {
    Sphere { centre: Point3::origin(), radius: 10.0 }
}

fn bounds() -> (Point3<f64>, Point3<f64>) {
    (Point3::new(-10.0, -10.0, -10.0), Point3::new(10.0, 10.0, 10.0))
}

const CHORD: f64 = 0.02;

fn meshed(field: &(dyn ScalarField + Sync), bounds: (Point3<f64>, Point3<f64>)) -> FieldMesh {
    let tolerance =
        MeshTolerance { chord_mm: CHORD, angle_deg: 20.0, max_edge_mm: 0.0, min_cell_mm: 0.1 };
    tessellate(field, bounds, &tolerance, true, false, &Cancel::new(), &Progress::default())
        .expect("not cancelled")
        .mesh
}

fn judged(field: &(dyn ScalarField + Sync), mesh: &FieldMesh) -> MeshJudgement {
    judge_mesh(field, bounds(), &mesh.positions, &mesh.triangles, CHORD)
}

/// The control: what the exporter makes of a ball passes every reading, and its volume is the
/// ball's to within what its distance allows.
#[test]
fn a_tessellated_ball_is_closed_outward_and_within_its_chord() {
    let mesh = meshed(&ball(), bounds());
    let j = judged(&ball(), &mesh);
    assert!(j.is_closed() && j.is_outward() && j.is_settled(), "{j:#?}");
    assert_eq!(j.shells, 1);
    assert!(j.hausdorff() <= CHORD, "{j:#?}");
    let exact = 4.0 / 3.0 * std::f64::consts::PI * 1000.0;
    assert!((j.volume_mm3 - exact).abs() <= j.volume_slack(), "{} against {exact}", j.volume_mm3);
}

/// Every triangle wound the other way: no edge disagrees with its neighbour, so only the shell's
/// volume and the field's gradient can tell -- and both do.
#[test]
fn a_ball_wound_the_wrong_way_reads_inside_out_though_every_edge_agrees() {
    let mut mesh = meshed(&ball(), bounds());
    for t in &mut mesh.triangles {
        t.swap(1, 2);
    }
    let j = judged(&ball(), &mesh);
    assert!(j.is_closed());
    assert_eq!(j.misoriented_edges, 0);
    assert_eq!(j.inward_shells, 1);
    assert_eq!(j.facing_in, j.facing_checked);
    assert!(j.volume_mm3 < 0.0);
}

/// One triangle flipped among thousands: its three edges disagree with their neighbours, and it
/// is the one triangle that faces in.
#[test]
fn one_flipped_triangle_is_three_misoriented_edges_and_one_facing_in() {
    let mut mesh = meshed(&ball(), bounds());
    mesh.triangles[0].swap(1, 2);
    let j = judged(&ball(), &mesh);
    assert_eq!(j.misoriented_edges, 3, "{j:#?}");
    assert_eq!(j.facing_in, 1, "{j:#?}");
    assert_eq!(j.inward_shells, 0);
}

/// A triangle taken out leaves its three edges with one triangle each.
#[test]
fn a_missing_triangle_is_a_tear_of_three_edges() {
    let mut mesh = meshed(&ball(), bounds());
    mesh.triangles.remove(0);
    let j = judged(&ball(), &mesh);
    assert_eq!(j.open_edges, 3);
    assert!(!j.is_closed());
}

/// A body the mesh left out has no triangles to be off the surface, so only the distance from
/// the field to the mesh can find it -- and it reads as far as the missing body is.
#[test]
fn a_body_left_out_of_the_mesh_is_found_from_the_field_side() {
    struct TwoBalls;
    impl ScalarField for TwoBalls {
        fn distance(&self, p: Point3<f64>) -> f64 {
            ((p - Point3::new(0.0, 0.0, 0.0)).norm() - 10.0)
                .min((p - Point3::new(40.0, 0.0, 0.0)).norm() - 5.0)
        }
    }
    let mesh = meshed(&ball(), bounds());
    let both = (Point3::new(-10.0, -10.0, -10.0), Point3::new(45.0, 10.0, 10.0));
    let j = judge_mesh(&TwoBalls, both, &mesh.positions, &mesh.triangles, CHORD);
    assert!(j.is_closed() && j.is_outward());
    assert!(j.off_surface.worst < 1.0, "the mesh is of the ball it has: {j:#?}");
    // The far side of the small ball is 45 from the origin, so 35 from the big one's surface.
    assert!((j.uncovered.worst - 35.0).abs() < 0.5, "{j:#?}");
}

/// Two tetrahedra touching at one corner have no open edge and are still not one closed
/// surface: the corner is two fans.
#[test]
fn two_solids_touching_at_a_corner_are_pinched_there() {
    let tetra = |o: Vector3<f64>, s: f64| {
        [Vector3::zeros(), Vector3::x() * s, Vector3::y() * s, Vector3::z() * s]
            .map(|v| Point3::from(o + v))
    };
    let mut positions: Vec<Point3<f64>> = tetra(Vector3::zeros(), 1.0).to_vec();
    // The second shares vertex 0 and points the other way.
    positions.extend(tetra(Vector3::zeros(), -1.0)[1..].iter());
    let faces = |o: u32, a: u32, b: u32, c: u32| [[o, b, a], [o, c, b], [o, a, c], [a, b, c]];
    let mut triangles = faces(0, 1, 2, 3).to_vec();
    triangles.extend(faces(0, 4, 5, 6).iter().map(|t| [t[0], t[2], t[1]]));
    let t = topology(&positions, &triangles);
    assert_eq!(t.open, 0);
    assert_eq!(t.overloaded, 0);
    assert_eq!(t.misoriented, 0);
    assert_eq!(t.pinched, 1);
    assert_eq!(t.shells, 1, "joined at the corner, so one piece by vertices");
}
