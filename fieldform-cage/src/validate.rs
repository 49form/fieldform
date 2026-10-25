// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What is wrong with a mesh that the constructor could still build: the red faults that stop a
//! cage being a solid's field, and the amber ones that are legal and likely a mistake.
//!
//! Three tiers (the cage-validation notes §1). The *structural* tier -- bowties, flips, a face
//! naming one vertex twice -- is the constructor's and never reaches here: a mesh with one of
//! those cannot exist. This module is the other two. **Red** is what makes a closed mesh still not
//! a solid: an open rim, a zero-length edge, a collapsed face, two vertices in one place. **Amber**
//! is what gives a poor surface: a star point, a short edge, a folded or twisted face. The
//! thresholds for red are "cannot be meant" and fixed; the amber ones are taste, relative to the
//! mesh's own median edge, and a setting.
//!
//! The one red check not here is the self-intersecting limit surface, which needs the tessellated
//! surface and a triangle test, and lives beside the field that tessellates it.

use crate::{FaceId, QuadMesh, VertexId};
use nalgebra::{Point3, Vector3};

/// Which check a fault came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Check {
    // Red.
    Open,
    ZeroLengthEdge,
    CollapsedFace,
    CoincidentVertices,
    SelfIntersection,
    // Amber.
    StarPoint,
    TwoFacePoint,
    ShortEdge,
    NearCoincidentVertices,
    FoldedFace,
    TwistedFace,
}

impl Check {
    pub fn is_red(self) -> bool {
        matches!(
            self,
            Check::Open
                | Check::ZeroLengthEdge
                | Check::CollapsedFace
                | Check::CoincidentVertices
                | Check::SelfIntersection
        )
    }

    /// The check's name as a person reads it in a list.
    pub fn label(self) -> &'static str {
        match self {
            Check::Open => "open rim",
            Check::ZeroLengthEdge => "zero-length edge",
            Check::CollapsedFace => "collapsed face",
            Check::CoincidentVertices => "coincident vertices",
            Check::SelfIntersection => "self-intersecting surface",
            Check::StarPoint => "star point",
            Check::TwoFacePoint => "two-face point",
            Check::ShortEdge => "short edge",
            Check::NearCoincidentVertices => "near-coincident vertices",
            Check::FoldedFace => "folded face",
            Check::TwistedFace => "twisted face",
        }
    }
}

/// One thing found, and where to show it.
#[derive(Clone, Debug, PartialEq)]
pub struct Fault {
    pub check: Check,
    pub vertices: Vec<VertexId>,
    pub edges: Vec<(VertexId, VertexId)>,
    pub faces: Vec<FaceId>,
    /// A number worth saying beside it -- the valence of a star point, the angle of a fold.
    pub measure: Option<f64>,
}

impl Fault {
    fn at(check: Check) -> Self {
        Self { check, vertices: Vec::new(), edges: Vec::new(), faces: Vec::new(), measure: None }
    }
}

/// The amber checks' settings, each relative to the mesh's median edge length `m` where it is a
/// length, so the same model at 1:1 and at 1000:1 gives the same warnings.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Thresholds {
    /// A vertex with this many faces or more is a star point. 3 and 5 are ordinary in all-quad
    /// work, so the default is 6.
    pub star_valence: usize,
    /// Show every extraordinary vertex, for study.
    pub show_all_extraordinary: bool,
    /// An edge shorter than this many median edges.
    pub short_edge: f64,
    /// Two vertices nearer than this many median edges, joined or not.
    pub near_coincident: f64,
    /// Two neighbouring faces whose normals differ by more than this many degrees.
    pub fold_degrees: f64,
    /// A quad whose two triangle normals differ by more than this many degrees.
    pub twist_degrees: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            star_valence: 6,
            show_all_extraordinary: false,
            short_edge: 0.1,
            near_coincident: 0.05,
            fold_degrees: 160.0,
            twist_degrees: 90.0,
        }
    }
}

/// The red threshold: `1e-6` of the bounding-box diagonal. "Cannot be meant", not taste, so it is
/// not a setting.
pub fn epsilon_length(mesh: &QuadMesh) -> f64 {
    let (lo, hi) = bounds(mesh);
    let diagonal = (hi - lo).norm();
    if diagonal > 0.0 { diagonal * 1e-6 } else { 1e-9 }
}

fn bounds(mesh: &QuadMesh) -> (Point3<f64>, Point3<f64>) {
    let mut lo = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut hi = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for v in mesh.vertex_ids() {
        let p = mesh.vertex_position(v);
        lo = lo.inf(&p);
        hi = hi.sup(&p);
    }
    (lo, hi)
}

/// The median edge length `m`, which every amber length is relative to.
pub fn median_edge(mesh: &QuadMesh) -> f64 {
    let mut lengths: Vec<f64> = mesh
        .edges()
        .map(|(a, b)| (mesh.vertex_position(a) - mesh.vertex_position(b)).norm())
        .collect();
    if lengths.is_empty() {
        return 0.0;
    }
    lengths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    lengths[lengths.len() / 2]
}

/// A quad's normal: the cross product of its diagonals, which is robust for a non-planar quad.
pub fn face_normal(mesh: &QuadMesh, f: FaceId) -> Vector3<f64> {
    let [a, b, c, d] = mesh.face_vertices(f).map(|v| mesh.vertex_position(v));
    (c - a).cross(&(d - b))
}

/// Half the norm of the diagonals' cross product: the quad's area, robust for a non-planar one.
pub fn face_area(mesh: &QuadMesh, f: FaceId) -> f64 {
    face_normal(mesh, f).norm() * 0.5
}

/// The red faults of the geometry: zero-length edges, collapsed faces and coincident vertices,
/// at `epsilon_length`. The open rim is the body's to report, since it knows its holes.
pub fn red_faults(mesh: &QuadMesh) -> Vec<Fault> {
    let eps = epsilon_length(mesh);
    let mut out = Vec::new();

    let short: Vec<(VertexId, VertexId)> = mesh
        .edges()
        .filter(|&(a, b)| (mesh.vertex_position(a) - mesh.vertex_position(b)).norm() < eps)
        .collect();
    if !short.is_empty() {
        out.push(Fault { edges: short, ..Fault::at(Check::ZeroLengthEdge) });
    }

    let collapsed: Vec<FaceId> = mesh.face_ids().filter(|&f| face_area(mesh, f) < eps * eps).collect();
    if !collapsed.is_empty() {
        out.push(Fault { faces: collapsed, ..Fault::at(Check::CollapsedFace) });
    }

    let close = vertices_within(mesh, eps, false);
    if !close.is_empty() {
        let mut vertices: Vec<VertexId> = close.iter().flat_map(|&(a, b)| [a, b]).collect();
        vertices.sort();
        vertices.dedup();
        out.push(Fault { vertices, ..Fault::at(Check::CoincidentVertices) });
    }
    out
}

/// Every pair of vertices nearer than `within`, by a grid hash rather than every pair against
/// every other. `joined` says whether a pair sharing an edge counts: the red check wants only the
/// unjoined ones (a joined pair is a zero-length edge, reported as that), the amber one wants both.
fn vertices_within(mesh: &QuadMesh, within: f64, joined: bool) -> Vec<(VertexId, VertexId)> {
    use std::collections::HashMap;
    if within <= 0.0 || mesh.vertex_count() < 2 {
        return Vec::new();
    }
    let cell = |p: Point3<f64>| -> [i64; 3] {
        [(p.x / within).floor() as i64, (p.y / within).floor() as i64, (p.z / within).floor() as i64]
    };
    let mut grid: HashMap<[i64; 3], Vec<VertexId>> = HashMap::new();
    for v in mesh.vertex_ids() {
        grid.entry(cell(mesh.vertex_position(v))).or_default().push(v);
    }
    let adjacent = |a: VertexId, b: VertexId| mesh.adjacent_vertices(a).contains(&b);
    let mut out = Vec::new();
    for v in mesh.vertex_ids() {
        let p = mesh.vertex_position(v);
        let c = cell(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(others) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else { continue };
                    for &o in others {
                        if o <= v {
                            continue;
                        }
                        if (mesh.vertex_position(o) - p).norm() < within && (joined || !adjacent(v, o)) {
                            out.push((v, o));
                        }
                    }
                }
            }
        }
    }
    out
}

/// The amber faults, under `thresholds`.
pub fn amber_faults(mesh: &QuadMesh, thresholds: &Thresholds) -> Vec<Fault> {
    let m = median_edge(mesh);
    let mut out = Vec::new();

    let stars: Vec<(VertexId, usize)> = mesh
        .vertex_ids()
        .map(|v| (v, mesh.valence(v)))
        .filter(|&(v, n)| {
            !mesh.vertex_on_boundary(v)
                && (n >= thresholds.star_valence || (thresholds.show_all_extraordinary && n != 4 && n > 2))
        })
        .collect();
    for (v, n) in stars {
        out.push(Fault { vertices: vec![v], measure: Some(n as f64), ..Fault::at(Check::StarPoint) });
    }

    // An interior vertex of two faces: what a diagonal cut makes. Its surface is unmeasured, so it
    // is always shown.
    let two: Vec<VertexId> =
        mesh.vertex_ids().filter(|&v| !mesh.vertex_on_boundary(v) && mesh.valence(v) == 2).collect();
    if !two.is_empty() {
        out.push(Fault { vertices: two, ..Fault::at(Check::TwoFacePoint) });
    }

    let short: Vec<(VertexId, VertexId)> = mesh
        .edges()
        .filter(|&(a, b)| (mesh.vertex_position(a) - mesh.vertex_position(b)).norm() < thresholds.short_edge * m)
        .collect();
    if !short.is_empty() {
        out.push(Fault { edges: short, ..Fault::at(Check::ShortEdge) });
    }

    let near = vertices_within(mesh, thresholds.near_coincident * m, true);
    if !near.is_empty() {
        let mut vertices: Vec<VertexId> = near.iter().flat_map(|&(a, b)| [a, b]).collect();
        vertices.sort();
        vertices.dedup();
        out.push(Fault { vertices, ..Fault::at(Check::NearCoincidentVertices) });
    }

    // Folded: a face and a neighbour across an edge whose normals disagree by more than the
    // threshold -- the faces lean back onto each other.
    let normals: Vec<Option<Vector3<f64>>> =
        mesh.face_ids().map(|f| face_normal(mesh, f).try_normalize(1e-12)).collect();
    let fold_cos = thresholds.fold_degrees.to_radians().cos();
    let mut folded: Vec<FaceId> = Vec::new();
    let mut worst = 0.0f64;
    for (a, b) in mesh.edges() {
        let Some((f, g)) = mesh.edge_faces(a, b) else { continue };
        if let (Some(nf), Some(ng)) = (normals[f.index()], normals[g.index()]) {
            let cos = nf.dot(&ng);
            if cos < fold_cos {
                worst = worst.max(cos.clamp(-1.0, 1.0).acos().to_degrees());
                folded.push(f);
                folded.push(g);
            }
        }
    }
    if !folded.is_empty() {
        folded.sort();
        folded.dedup();
        out.push(Fault { faces: folded, measure: Some(worst), ..Fault::at(Check::FoldedFace) });
    }

    // Twisted: the two triangles of a quad, split on either diagonal, disagree. A split that
    // yields a zero-area triangle has no normal and is skipped, not read as twisted: a diagonal
    // cut's faces have a straight corner at the centre vertex, so one of their splits always is.
    let twist_cos = thresholds.twist_degrees.to_radians().cos();
    let mut twisted: Vec<FaceId> = Vec::new();
    for f in mesh.face_ids() {
        let [a, b, c, d] = mesh.face_vertices(f).map(|v| mesh.vertex_position(v));
        let tri = |p: Point3<f64>, q: Point3<f64>, r: Point3<f64>| (q - p).cross(&(r - p)).try_normalize(1e-12);
        let splits = [(tri(a, b, c), tri(a, c, d)), (tri(a, b, d), tri(b, c, d))];
        if splits.iter().any(|(n1, n2)| matches!((n1, n2), (Some(n1), Some(n2)) if n1.dot(n2) < twist_cos)) {
            twisted.push(f);
        }
    }
    if !twisted.is_empty() {
        out.push(Fault { faces: twisted, ..Fault::at(Check::TwistedFace) });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cage;

    fn cube() -> QuadMesh {
        Cage::cube_of(10.0).mesh().clone()
    }

    #[test]
    fn a_clean_cube_has_no_faults_at_either_tier() {
        let cube = cube();
        assert!(red_faults(&cube).is_empty());
        assert!(amber_faults(&cube, &Thresholds::default()).is_empty(), "valence 3 is ordinary");
        let all = Thresholds { show_all_extraordinary: true, ..Thresholds::default() };
        assert_eq!(amber_faults(&cube, &all).len(), 8, "every corner, when asked for every one");
    }

    /// **The red threshold is a fraction of the model, not millimetres**: the same collapse is
    /// red at 1:1 and at 1000:1, and a 0.01 mm gap on a 10 mm model is not.
    #[test]
    fn red_faults_scale_with_the_model() {
        for scale in [1.0, 1000.0] {
            let mut m = Cage::cube_of(10.0 * scale).mesh().clone();
            // Vertex 1 onto vertex 0: a zero-length edge, four collapsed faces -- no: two faces
            // lose area to zero only if degenerate; here the -Z and -Y faces fold to triangles,
            // which still have area. So this makes one zero-length edge and nothing else red.
            m.set_vertex_position(VertexId::from_index(1), m.vertex_position(VertexId::from_index(0)));
            let faults = red_faults(&m);
            assert_eq!(faults.len(), 1, "{faults:?}");
            assert_eq!(faults[0].check, Check::ZeroLengthEdge);
            assert_eq!(faults[0].edges.len(), 1);
        }
        let mut m = cube();
        // A gap of a thousandth of the model is a gap, not a coincidence.
        let p = m.vertex_position(VertexId::from_index(0));
        m.set_vertex_position(VertexId::from_index(6), p + Vector3::new(0.01, 0.0, 0.0));
        assert!(red_faults(&m).iter().all(|f| f.check != Check::CoincidentVertices));
        // A gap of a millionth is not.
        m.set_vertex_position(VertexId::from_index(6), p + Vector3::new(1e-7, 0.0, 0.0));
        let faults = red_faults(&m);
        assert!(faults.iter().any(|f| f.check == Check::CoincidentVertices && f.vertices == vec![VertexId::from_index(0), VertexId::from_index(6)]), "{faults:?}");
    }

    #[test]
    fn a_collapsed_face_is_red() {
        let mut m = cube();
        // The whole -Z face onto one line: vertices 0..4 onto the x axis at two points.
        let line = |x: f64| Point3::new(x, -5.0, -5.0);
        for (v, x) in [(0, -5.0), (1, 5.0), (2, 5.0), (3, -5.0)] {
            m.set_vertex_position(VertexId::from_index(v), line(x));
        }
        let faults = red_faults(&m);
        assert!(faults.iter().any(|f| f.check == Check::CollapsedFace && f.faces == vec![FaceId::from_index(0)]), "{faults:?}");
    }

    /// **The amber thresholds are relative to the median edge**, so the same model at 1:1 and
    /// 1000:1 gives identical warnings (the notes' test 7).
    #[test]
    fn amber_warnings_are_the_same_at_any_scale() {
        let warnings = |scale: f64| {
            let mut m = Cage::cube_of(10.0 * scale).mesh().clone();
            // Vertex 6 pulled most of the way to vertex 2: a short edge, and a near pair.
            let (p2, p6) = (m.vertex_position(VertexId::from_index(2)), m.vertex_position(VertexId::from_index(6)));
            m.set_vertex_position(VertexId::from_index(6), p2 + (p6 - p2) * 0.03);
            amber_faults(&m, &Thresholds::default()).into_iter().map(|f| f.check).collect::<Vec<_>>()
        };
        let at_one = warnings(1.0);
        assert!(at_one.contains(&Check::ShortEdge) && at_one.contains(&Check::NearCoincidentVertices), "{at_one:?}");
        assert_eq!(at_one, warnings(1000.0));
    }

    /// A face folded back over its neighbour -- the second quad of a strip turned up and over the
    /// first -- is amber, with the angle between their normals quoted.
    #[test]
    fn a_face_folded_back_over_its_neighbour_is_amber() {
        let strip = |fold: f64| {
            QuadMesh::from_faces(
                vec![
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(1.0, 0.0, 0.0),
                    Point3::new(1.0 + fold.cos(), 0.0, fold.sin()),
                    Point3::new(1.0 + fold.cos(), 1.0, fold.sin()),
                    Point3::new(1.0, 1.0, 0.0),
                    Point3::new(0.0, 1.0, 0.0),
                ],
                &[vec![0, 1, 4, 5], vec![1, 2, 3, 4]],
            )
            .unwrap()
        };
        let flat = amber_faults(&strip(0.0), &Thresholds::default());
        assert!(flat.iter().all(|f| f.check != Check::FoldedFace), "{flat:?}");
        let faults = amber_faults(&strip(170f64.to_radians()), &Thresholds::default());
        let fold = faults.iter().find(|f| f.check == Check::FoldedFace).unwrap_or_else(|| panic!("folded: {faults:?}"));
        assert!((fold.measure.unwrap() - 170.0).abs() < 1e-6, "{:?}", fold.measure);
        assert_eq!(fold.faces, vec![FaceId::from_index(0), FaceId::from_index(1)]);
    }

    /// A quad with a straight corner is not twisted: one of its splits has no area and is skipped.
    #[test]
    fn a_straight_corner_is_not_a_twist_and_a_real_twist_is() {
        let flat = QuadMesh::from_faces(
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            &[vec![0, 1, 2, 3]],
        )
        .unwrap();
        assert!(amber_faults(&flat, &Thresholds::default()).iter().all(|f| f.check != Check::TwistedFace));

        let twisted = QuadMesh::from_faces(
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 1.0), Point3::new(1.0, 1.0, -1.0)],
            &[vec![0, 1, 3, 2]],
        )
        .unwrap();
        let faults = amber_faults(&twisted, &Thresholds::default());
        assert!(faults.iter().any(|f| f.check == Check::TwistedFace), "{faults:?}");
    }
}
