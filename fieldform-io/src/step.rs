// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Reading a solid STEP file into a triangle mesh the document can hold.
//!
//! Stage 1 of the two `import.md` describes: tessellate the B-rep and treat the result as a
//! closed mesh. Stage 2 — the surfaces themselves as an exact field leaf — is backlog item 19,
//! and the same kernel underneath can answer it.
//!
//! # Why OpenCASCADE, after trying not to
//!
//! `truck-stepio` was tried first and was the wrong call, which is worth recording because the
//! reasoning that led there looked sound. Its evaluation spike round-tripped a solid through
//! truck's *own* writer, and passing that says only that a library agrees with itself. Against
//! three real exports it read none of them: a 25 MB AP214 file refused outright, a part from another CAD system
//! panicked the process and then lost 1571 of 6637 faces, and a sound solid lost 114 faces —
//! more at a finer tolerance than at a coarse one.
//!
//! OCCT is the reader nearly every open-source CAD tool uses, and it is the same code path
//! those tools trust. The cost is licensing rather than engineering: **LGPL-2.1 with a header
//! exception**, which obliges this project to ship OCCT as a replaceable library, to say in its
//! documentation that it uses it, and — exactly as with `vendor/fidget/` and for a different
//! reason — **never to modify it**, since changes to OCCT must be published under LGPL.
//!
//! # What still has to be done to what it returns
//!
//! OCCT hands back one triangulation per face, correctly wound for each face's own orientation.
//! Two things remain, and both were learned from the truck attempt rather than assumed:
//!
//! 1. **Welding.** Per-face triangulations share no vertices, so every edge is used once and
//!    the solid reads as an open surface. [`MeshField`] would refuse it, correctly, being
//!    unable to tell that from a genuine hole.
//! 2. **Global orientation, measured rather than trusted.** A solid should come back wound
//!    outward; the signed volume says whether it did. Getting this wrong swaps inside for
//!    outside while leaving a mesh that passes every structural check, so it is checked rather
//!    than assumed even though OCCT is reliable about it.
//!
//! [`MeshField`]: https://docs.rs/fieldform-field

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use fieldform_scene::StepOptions;
use fieldform_scene::{Curve, CurveId, MeshData};
use nalgebra::Point3;
use opencascade::{mesh::Mesher, primitives::Shape};

/// Why a STEP file could not be imported.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StepError {
    #[error("the file could not be read as STEP: {0}")]
    Read(String),
    #[error("the file was read but could not be tessellated: {0}")]
    Tessellate(String),
    #[error("the file was read but holds no solids and no curves")]
    NoGeometry,
}

/// What one import produced, beside the mesh itself.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepReport {
    /// The chord height used.
    pub tolerance_used_mm: f64,
    /// Vertices merged by welding. Large by design — per-face triangulations share none — and
    /// reported so a number near zero can be recognised as the warning sign it is.
    pub vertices_welded: usize,
    /// True when the mesh had to be turned inside out to face the right way.
    pub reversed: bool,
    /// Edges used by other than exactly two triangles. Non-zero means visible geometry that is
    /// not a solid, and so cannot become a signed field.
    pub open_edges: usize,
    /// Solids the file held, and how many became meshes. They differ when one would not
    /// tessellate, which is worth surfacing rather than quietly dropping.
    ///
    /// **Zero found is a finding, not a failure.** A file can hold surfaces and no solids at
    /// all — a styling model exported as a shell, say — and then nothing in it was ever going
    /// to be closed. That is worth telling the user plainly, because the remedy is not a
    /// healing setting: there is no solid to repair.
    pub solids_found: usize,
    pub solids_converted: usize,
    /// Curves in the file that bound no face, and how many became curves in the document.
    ///
    /// They differ when one carries no geometry to read — a degenerate seam has topology and
    /// no curve — or when its numbers do not make a curve this document can hold.
    pub curves_found: usize,
    pub curves_converted: usize,
    /// What the reader was told.
    ///
    /// Reported because these are the first things to vary when a file will not come in as a
    /// solid, and because a user who raised the sewing tolerance deserves to see whether it
    /// helped — a report that does not carry its own settings cannot be compared with the next
    /// one.
    pub options: StepOptions,
    /// Open edges *before* healing, so the effect of healing is visible rather than inferred.
    pub open_edges_before_healing: Option<usize>,
    /// Kinds of thing this build did not convert, by name and count.
    ///
    /// `import.md`'s standing requirement, and only truthful if written at import time: the
    /// record of what a later build stands to recover from the same source.
    pub unconverted: BTreeMap<String, usize>,
}

/// Reads a STEP file into a triangle mesh.
///
/// Takes a path rather than the file's text because OCCT's reader does its own IO. That is
/// worth noting rather than hiding: it means this cannot import from bytes already in memory,
/// which will matter the day a document re-imports from its own `sources/` entry.
///
/// This forwards to [`import_step_with`] unchanged, and the importer binary and the diagnosis
/// example both call that directly, so only the tests use this name and only the tests compile
/// it.
#[cfg(test)]
fn import_step(
    path: &Path,
    name: impl Into<String>,
    options: StepOptions,
) -> Result<(Vec<MeshData>, Vec<Curve>, StepReport), StepError> {
    import_step_with(path, name, options)
}

/// The curves a shape holds that bound no face, as document curves.
///
/// **Exact, not sampled.** Every edge is asked for the NURBS it lies on, and OCCT's conics are
/// rational B-splines of degree two — so a circle arrives as a circle rather than as a polyline
/// that looks like one. `Edge::nurbs` is where the conversion happens and why it costs nothing
/// in accuracy.
///
/// An edge that yields nothing is counted and skipped rather than failing the import: a
/// degenerate seam has topology and no geometry, and a file that holds one still has solids
/// worth reading.
fn read_free_curves(shape: &Shape, name: &str, report: &mut StepReport) -> Vec<Curve> {
    let free = shape.free_edges();
    report.curves_found = free.len();
    let mut out = Vec::new();
    for edge in &free {
        let Some(nurbs) = edge.nurbs() else { continue };
        let points: Vec<Point3<f64>> =
            nurbs.poles.iter().map(|p| Point3::new(p.x, p.y, p.z)).collect();
        // Numbered only when there is more than one, so a file with a single curve keeps the
        // name the user recognises -- the same rule the meshes above follow.
        let curve_name = match free.len() {
            1 => name.to_string(),
            _ => format!("{name} curve {}", out.len() + 1),
        };
        // **Measured, not taken from the flag.** OCCT reports periodicity, but the curve has to
        // be trimmed to the edge's own parameter range before it can become a B-spline at all
        // -- a straight edge lies on an *infinite* line -- and a trimmed curve is never
        // periodic, however far round it goes. So the flag is false for a full circle and the
        // only honest answer is whether the two ends are in the same place.
        let closed = match (points.first(), points.last()) {
            (Some(a), Some(b)) => (a - b).norm() < 1e-9,
            _ => false,
        };
        match Curve::new(
            CurveId::new(),
            curve_name,
            nurbs.degree,
            points,
            nurbs.weights.clone(),
            nurbs.knots.clone(),
            closed,
        ) {
            Ok(curve) => out.push(curve),
            // Kept as a count rather than an error. What arrives from another program is not
            // this document's to vouch for, and one curve it cannot hold is not a reason to
            // refuse the solids beside it.
            Err(_) => continue,
        }
    }
    report.curves_converted = out.len();
    out
}

/// Serialises every entry into OCCT, of any kind.
///
/// **OCCT is not thread-safe**, and the way it says so is an access violation rather than a
/// wrong answer. Two imports at once in one process reliably crash — found because four tests
/// that each import a small box faulted the harness, and passed one at a time.
///
/// The lock lives here rather than in the tests because the tests were not the bug, they were
/// the messenger: any caller that imported two files at once would meet the same fault. Poison
/// is ignored deliberately — a panicking import must not leave the application unable to
/// import anything ever again, and the lock guards a library's global state rather than data
/// whose invariants a panic could have broken.
static OCCT: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Held for as long as OCCT is being called.
///
/// Every entry, not only reading: writing faulted the same way once a second test wrote a STEP
/// file beside the first, and the shared state is the library's, not the reader's. So the rule
/// is the whole library rather than the one function that happened to crash first — a lock
/// that covers only the paths already known to fault has to be rediscovered by each new one.
fn occt() -> std::sync::MutexGuard<'static, ()> {
    OCCT.lock().unwrap_or_else(|e| e.into_inner())
}

/// Reads a STEP file's solids as meshes and the curves that bound no face as document curves,
/// with a report of what was read and what was not.
pub fn import_step_with(
    path: &Path,
    name: impl Into<String>,
    options: StepOptions,
) -> Result<(Vec<MeshData>, Vec<Curve>, StepReport), StepError> {
    // Clamped once, here, rather than trusted from a caller or re-checked at each use: the
    // bounds are the mesher's and this is the only door into it.
    let options = options.sane();
    let _occt = occt();
    // OCCT reports failure by return value, but a malformed file can still assert deep in the
    // reader. An importer reads files it did not write, so a third-party crash is treated as
    // data rather than as a fault — the alternative is losing the user's document to someone
    // else's bad export.
    let shape = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Shape::read_step(path)
    }))
    .map_err(|_| StepError::Read("the reader crashed on this file".to_owned()))?
    .map_err(|e| StepError::Read(format!("{e:?}")))?;

    let mut report = StepReport {
        tolerance_used_mm: options.linear_deflection_mm,
        options,
        ..Default::default()
    };

    // Split into solids *before* healing, and heal each one separately. Two reasons, and the
    // first was found by a test rather than reasoned about: sewing returns a **shell**, not a
    // solid, so exploring for solids afterwards finds none and the split silently never
    // happens. The second is the one that matters more — sewing a whole assembly can stitch
    // two parts that merely touch into one shape, which is a change to the geometry that
    // nothing downstream could notice.
    //
    // A STEP assembly is a compound of many solids, and tessellating it whole merges them:
    // every face where two parts meet becomes an edge shared by four triangles instead of two,
    // so nothing in the result is closed. A real assembly measured 46 483 such edges
    // against 6 genuine holes — the merging was the entire problem, and no healing addresses
    // it.
    let name: String = name.into();
    let solids: Vec<Shape> = shape.solids().map(Shape::from).collect();
    report.solids_found = solids.len();

    // **The curves that are not the boundary of anything**, read before `shape` is consumed
    // below. A STEP file holds these natively -- construction lines, centrelines, the sketch a
    // part was drawn from -- and they are the ones somebody put there on purpose. Every other
    // edge in the file is a face boundary, and a real part has thousands.
    let curves = read_free_curves(&shape, &name, &mut report);

    // A file with no solids at all still yields a mesh, from the shape itself: it may hold
    // loose surfaces, and refusing to show the user what they have is worse than showing them
    // something open.
    let parts: Vec<Shape> = if solids.is_empty() { vec![shape] } else { solids };
    let mut meshes = Vec::new();

    for (i, part) in parts.iter().enumerate() {
        // Numbered only when there is more than one, so a single-solid file keeps the name the
        // user recognises.
        let part_name = if parts.len() == 1 {
            name.clone()
        } else {
            format!("{name} ({})", i + 1)
        };

        // Healed per part, and a part whose repair pass crashes is skipped rather than losing
        // the import. Repair runs before tessellation because sewing joins *faces*, and once
        // they are triangles the topology it needs is gone — welding a mesh afterwards looks
        // similar and is not the same thing, since it can only merge vertices that already
        // coincide exactly, where sewing closes a gap.
        let healed: Option<Shape> = match (options.sew_tolerance_mm > 0.0)
            .then_some(options.sew_tolerance_mm)
        {
            None => None,
            Some(tolerance) => {
                let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if options.shape_fix {
                        part.healed().sew(tolerance)
                    } else {
                        part.sew(tolerance)
                    }
                }));
                match attempt {
                    Ok(shape) => Some(shape),
                    Err(_) => {
                        *report
                            .unconverted
                            .entry("solid (repair crashed)".to_owned())
                            .or_insert(0) += 1;
                        continue;
                    }
                }
            }
        };
        // The repaired shape when there is one, otherwise the part untouched. Borrowed rather
        // than copied: `Shape` owns an OCCT handle and has no cheap clone.
        let target = healed.as_ref().unwrap_or(part);

        if let Some(mesh) = mesh_one(target, options, &part_name, &mut report) {
            meshes.push(mesh);
        }
    }

    // **Curves alone are geometry.** This refused any file that produced no mesh, which was
    // right while curves were not read: a file with nothing in it and a file whose contents
    // this could not see were the same outcome. They are not any more -- a sketch exported as
    // STEP holds no solid at all and is exactly what somebody meant to import.
    if meshes.is_empty() && curves.is_empty() {
        return Err(StepError::NoGeometry);
    }
    report.solids_converted = meshes.len();
    Ok((meshes, curves, report))
}

/// Tessellates one shape into a mesh, or `None` if it produced no geometry.
///
/// Failures here are counted rather than propagated: one solid of four hundred that will not
/// mesh should cost that solid, not the import.
fn mesh_one(
    shape: &Shape,
    options: StepOptions,
    name: &str,
    report: &mut StepReport,
) -> Option<MeshData> {
    let mesh = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Both criteria. A linear deflection alone leaves the angular one at OCCT's 0.5 rad
        // which is 28 degrees between facet normals and is what faceting on a curved face
        // actually is. See [`StepOptions::angular_deflection_deg`] for the measurements.
        //
        // Threaded: a shape meshes face by face and the faces are independent. The `OCCT` lock
        // an import holds is about *our* calls being serialised with each other, not about
        // denying the library its own internal parallelism.
        Mesher::try_new_angular(
            shape,
            options.linear_deflection_mm,
            options.angular_deflection_rad(),
            options.parallel,
        )
        .and_then(|m| m.mesh())
    }));
    let mesh = match mesh {
        Ok(Ok(mesh)) => mesh,
        Ok(Err(_)) | Err(_) => {
            *report.unconverted.entry("solid (would not mesh)".to_owned()).or_insert(0) += 1;
            return None;
        }
    };
    if mesh.skipped_faces > 0 {
        // A face OCCT could not triangulate leaves a hole, and holes are what stop a mesh
        // becoming a field. Named rather than left to be inferred from an edge count: this is
        // the record `import.md` asks for of what a later build could recover.
        *report
            .unconverted
            .entry("face (no triangulation)".to_owned())
            .or_insert(0) += mesh.skipped_faces;
    }
    if mesh.indices.len() < 3 {
        return None;
    }

    let positions: Vec<[f64; 3]> = mesh.vertices.iter().map(|v| [v.x, v.y, v.z]).collect();
    let triangles: Vec<[u32; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| [t[0] as u32, t[1] as u32, t[2] as u32])
        .collect();

    let before = positions.len();
    let (positions, mut triangles) = weld(positions, triangles, WELD_TOLERANCE_MM);
    report.vertices_welded += before - positions.len();

    // Measured, not assumed. See the module docs.
    if signed_volume(&positions, &triangles) < 0.0 {
        report.reversed = true;
        for t in &mut triangles {
            t.swap(1, 2);
        }
    }

    let open_edges = MeshData::count_open_edges(&triangles);
    report.open_edges += open_edges;
    Some(MeshData {
        edges: brep_edges(shape),
        // Tagged by the caller, which is the one that knows which file this run was reading.
        source: None,
        name: name.to_owned(),
        positions,
        triangles,
        tolerance_mm: Some(options.linear_deflection_mm),
        open_edges,
    })
}

/// How close two vertices must be to be the same one.
///
/// Well below any chord height worth using, so it cannot merge points the tessellator meant to
/// keep apart, and far above the agreement between two faces evaluated from one shared edge.
const WELD_TOLERANCE_MM: f64 = 1e-6;

/// Merges coincident vertices so triangles from separate faces share them.
///
/// Rounded onto a grid rather than compared pairwise: one pass instead of N², which matters at
/// import sizes, and a tolerance is what "the same vertex" means here anyway.
fn weld(
    positions: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    tolerance: f64,
) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let key = |p: &[f64; 3]| {
        [
            (p[0] / tolerance).round() as i64,
            (p[1] / tolerance).round() as i64,
            (p[2] / tolerance).round() as i64,
        ]
    };
    let mut seen: HashMap<[i64; 3], u32> = HashMap::new();
    let mut out_positions: Vec<[f64; 3]> = Vec::new();
    let mut remap = vec![0u32; positions.len()];
    for (i, p) in positions.iter().enumerate() {
        let at = *seen.entry(key(p)).or_insert_with(|| {
            out_positions.push(*p);
            (out_positions.len() - 1) as u32
        });
        remap[i] = at;
    }
    let out_triangles = triangles
        .into_iter()
        .map(|t| [remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]])
        // A triangle whose corners welded together has no area, and no business in a mesh
        // whose edges are about to be counted: it would contribute two edges used once each
        // and make a closed solid look open.
        .filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
        .collect();
    (out_positions, out_triangles)
}

/// The solid's own edges, tessellated into polylines.
///
/// Read after the mesh pass rather than instead of it: `approximation_segments` walks the curve
/// with `GCPnts_TangentialDeflection`, which is a traversal rather than a computation, and the
/// points it returns are on the *exact* curve — so the lines are truer than the triangles they
/// are drawn over, which is the whole reason they read as CAD.
///
/// A two-point edge is kept. It is a straight line, and a machined part is mostly those.
fn brep_edges(shape: &Shape) -> Vec<Vec<[f64; 3]>> {
    shape
        .edges()
        .map(|edge| {
            edge.approximation_segments()
                .map(|p| [p.x, p.y, p.z])
                .collect::<Vec<[f64; 3]>>()
        })
        .filter(|line| line.len() >= 2)
        .collect()
}

/// The signed volume of a closed mesh, positive exactly when its triangles wind outward.
///
/// The divergence theorem over the surface: each triangle contributes the signed volume of the
/// tetrahedron it makes with the origin, and for a closed surface those sum to the enclosed
/// volume wherever the origin happens to sit.
fn signed_volume(positions: &[[f64; 3]], triangles: &[[u32; 3]]) -> f64 {
    let dot_cross = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| {
        a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
    };
    triangles
        .iter()
        .map(|t| {
            dot_cross(
                positions[t[0] as usize],
                positions[t[1] as usize],
                positions[t[2] as usize],
            )
        })
        .sum::<f64>()
        / 6.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A STEP file for a box, written by OCCT so the test needs no fixture.
    ///
    /// Round-tripping rather than checking in a `.stp`: a fixture only ever tests the one
    /// exporter that produced it, and a mis-read then has to survive being compared against
    /// known geometry rather than merely not crashing.
    fn box_step(dir: &std::path::Path, w: f64, d: f64, h: f64) -> std::path::PathBuf {
        let path = dir.join("box.step");
        let _occt = occt();
        let shape = opencascade::primitives::Shape::box_with_dimensions(w, d, h);
        shape.write_step(&path).expect("OCCT writes its own format");
        path
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("fieldform-step-{name}"));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    /// Two boxes far apart, written into one file the way an assembly is.
    fn two_box_step(dir: &std::path::Path) -> std::path::PathBuf {
        use opencascade::primitives::Shape;
        let path = dir.join("two.step");
        let _occt = occt();
        let a = Shape::box_with_dimensions(10.0, 10.0, 10.0);
        let b = Shape::box_with_dimensions(10.0, 10.0, 10.0).translated(opencascade::glam::dvec3(50.0, 0.0, 0.0));
        Shape::write_all_step([&a, &b], &path).expect("OCCT writes its own format");
        path
    }

    /// A STEP file's loose curves come back, and a circle comes back as a circle.
    ///
    /// **The thing that was missing.** STEP holds curves that bound no face, and the importer
    /// read only solids — so a file of construction geometry imported as nothing at all.
    ///
    /// Checked by *measuring the curve*, not by counting it. A polyline through the same points
    /// would pass any test that only asserts one curve arrived, and would be the wrong answer:
    /// the whole reason to read the NURBS is that a circle stays exact. So this samples the
    /// curve that came back and asserts every sample is the right distance from the centre.
    #[test]
    fn a_free_standing_curve_is_read_as_the_nurbs_it_is() {
        use opencascade::primitives::{Edge, Shape};

        let dir = scratch("free_curves");
        std::fs::create_dir_all(&dir).expect("scratch");
        let path = dir.join("curves.step");

        // The same guard every other fixture here takes. OCCT is not thread-safe, and building
        // shapes and writing them is entry into it just as much as reading is -- without this
        // the test races the ones beside it and the harness dies of an access violation rather
        // than failing.
        let guard = occt();
        let radius = 25.0;
        let circle = Shape::from(Edge::circle(
            opencascade::glam::dvec3(0.0, 0.0, 0.0),
            opencascade::glam::dvec3(0.0, 0.0, 1.0),
            radius,
        ));
        let line = Shape::from(Edge::segment(
            opencascade::glam::dvec3(-40.0, 0.0, 0.0),
            opencascade::glam::dvec3(40.0, 0.0, 0.0),
        ));
        Shape::write_all_step([&circle, &line], &path).expect("OCCT writes its own format");
        // Released before importing, which takes the lock for itself.
        drop(guard);

        let (meshes, curves, report) =
            import_step(&path, "sketch", StepOptions::default()).expect("the file reads");

        assert!(meshes.is_empty(), "a file of curves should not produce solids");
        assert_eq!(report.curves_found, 2, "both edges belong to no face");
        assert_eq!(curves.len(), 2, "both should have become curves");

        // The circle, found by which one closes rather than by position in the list — the order
        // OCCT hands edges back in is its business, not a thing to assert.
        let ring = curves.iter().find(|c| c.is_closed()).expect("the circle came back closed");
        assert!(
            ring.degree() >= 2,
            "a circle is a rational quadratic at least, not a polyline: got degree {}",
            ring.degree()
        );
        // Over the curve's own parameter range, which is OCCT's and is arc length here rather
        // than a normalised nought-to-one — `Curve::evaluate` takes the parameter, not a
        // fraction, and sampling 0..1 on an 80 mm line covers the first millimetre of it.
        let (lo, hi) = ring.domain();
        let mut worst = 0.0f64;
        for i in 0..=200 {
            let p = ring.evaluate(lo + (hi - lo) * i as f64 / 200.0);
            worst = worst.max((p.coords.xy().norm() - radius).abs());
            worst = worst.max(p.z.abs());
        }
        assert!(worst < 1e-9, "the circle is off by {worst} mm somewhere, so it is not a circle");

        // And the line, which should be exactly a segment between the two points asked for.
        let straight = curves.iter().find(|c| !c.is_closed()).expect("the line came back");
        let (lo, hi) = straight.domain();
        let (a, b) = (straight.evaluate(lo), straight.evaluate(hi));
        assert!(
            (a.x.abs() - 40.0).abs() < 1e-9 && (b.x.abs() - 40.0).abs() < 1e-9,
            "the line runs {a:?} to {b:?}, not -40 to 40"
        );
    }

    /// An assembly arrives as one body per solid, and each of them is closed.
    ///
    /// Tessellating the compound whole is what this replaced, and it merged them: a real
    /// assembly came back as a single mesh with 46 483 open edges where the six parts touched.
    /// Two boxes that do not even touch still merge into one mesh under the old path, so this
    /// is the smallest case that tells the two apart.
    #[test]
    fn an_assembly_arrives_as_one_closed_body_per_solid() {
        let dir = scratch("two-solids");
        let path = two_box_step(&dir);

        let (meshes, _, report) = import_step(&path, "two", StepOptions::default()).expect("OCCT reads its own STEP");

        assert_eq!(report.solids_found, 2);
        assert_eq!(report.solids_converted, 2);
        assert_eq!(meshes.len(), 2);
        for mesh in &meshes {
            assert!(mesh.is_closed(), "{} left {} open edges", mesh.name, mesh.open_edges);
        }
    }

    /// Numbered, and only when there is more than one.
    ///
    /// A single-solid file keeps the name the user chose the file for; there is nothing to
    /// disambiguate, and "body (1)" alone in a tree reads as if something is missing.
    #[test]
    fn parts_are_numbered_only_when_the_file_holds_more_than_one() {
        let dir = scratch("naming");
        let (many, _, _) = import_step(&two_box_step(&dir), "asm", StepOptions::default()).expect("two solids");
        let names: Vec<&str> = many.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["asm (1)", "asm (2)"]);

        let (one, _, _) = import_step(&box_step(&dir, 10.0, 10.0, 10.0), "part", StepOptions::default()).expect("one");
        assert_eq!(one[0].name, "part");
    }

    /// The whole point: a solid goes in and a closed mesh of the same solid comes out.
    ///
    /// Checked by *volume* rather than triangle count, because the count depends on the
    /// tessellator's choices and the volume does not — and it catches a mesh that is the right
    /// size and inside out as readily as one that is the wrong size.
    #[test]
    fn a_step_box_comes_back_as_a_closed_mesh_of_the_same_box() {
        let dir = scratch("roundtrip");
        let path = box_step(&dir, 2.0, 3.0, 4.0);
        let (meshes, _, report) = import_step(&path, "box", StepOptions::default()).expect("OCCT reads its own STEP");
        assert_eq!(meshes.len(), 1, "one box is one solid");
        assert_eq!(report.solids_found, 1);
        let mesh = &meshes[0];

        let (lo, hi) = mesh.bounds().expect("a box has bounds");
        for (k, want) in [2.0, 3.0, 4.0].into_iter().enumerate() {
            assert!((hi[k] - lo[k] - want).abs() < 1e-6, "axis {k}: {}", hi[k] - lo[k]);
        }

        // Closed is the property that decides whether it can become a field at all.
        assert_eq!(mesh.open_edges, 0, "a box is a closed solid");
        assert!(mesh.is_closed());

        // Positive volume means the triangles wind outward, which everything downstream needs.
        let volume = signed_volume(&mesh.positions, &mesh.triangles);
        assert!((volume - 24.0).abs() < 1e-6, "volume {volume}, expected 24");
        assert!(report.vertices_welded > 0, "per-face triangulations always need welding");
    }

    /// The tessellation tolerance is recorded, because a document that cannot say how faithful
    /// its mesh is has nothing to compare a later exact re-import against.
    #[test]
    fn the_chord_height_the_source_was_read_at_is_kept() {
        let dir = scratch("tolerance");
        let path = box_step(&dir, 1.0, 1.0, 1.0);
        let (meshes, _, report) = import_step(&path, "cube", StepOptions { linear_deflection_mm: 0.125, ..StepOptions::default() }).expect("OCCT reads its own STEP");
        let mesh = &meshes[0];
        assert_eq!(mesh.tolerance_mm, Some(0.125));
        assert_eq!(report.tolerance_used_mm, 0.125);
        assert_eq!(mesh.name, "cube");
    }

    /// A file that is not STEP is refused with a message rather than a panic. Importing
    /// someone else's file is the likeliest way to meet a bad one.
    #[test]
    fn something_that_is_not_step_is_refused() {
        let dir = scratch("junk");
        let path = dir.join("junk.step");
        std::fs::write(&path, "this is not a STEP file").expect("scratch is writable");
        let err = import_step(&path, "junk", StepOptions::default()).expect_err("not STEP");
        assert!(matches!(err, StepError::Read(_) | StepError::NoGeometry), "got {err:?}");
    }

    /// The linear deflections are the CAD-conventional ones and are absolute, not relative.
    /// Pinned because the whole reason they exist is that a fraction of the bounding box was
    /// wrong.
    #[test]
    fn the_presets_use_absolute_millimetres() {
        assert_eq!(StepOptions::coarse().linear_deflection_mm, 0.1);
        assert_eq!(StepOptions::normal().linear_deflection_mm, 0.05);
        assert_eq!(StepOptions::fine().linear_deflection_mm, 0.01);
        assert_eq!(StepOptions::default(), StepOptions::normal());
    }

    /// A preset is a set of values, not a mode — so every one of them reports itself, and a
    /// nudged setting reports none.
    ///
    /// This is what keeps the dialog honest: the buttons fill the fields in, and nothing stores
    /// "which preset" beside the fields where the two could disagree.
    #[test]
    fn a_preset_is_recognised_by_its_values_and_a_changed_one_is_not() {
        for (name, preset) in StepOptions::presets() {
            assert_eq!(preset.preset_name(), Some(name), "{name} should recognise itself");
        }
        let nudged = StepOptions { angular_deflection_deg: 9.0, ..StepOptions::normal() };
        assert_eq!(nudged.preset_name(), None);
    }

    /// The settings are clamped to what the mesher can deliver, at both ends.
    ///
    /// The lower angular bound is measured rather than chosen: below about 3 degrees the mesher
    /// emits slivers whose creases come out *worse* than the coarsest setting, so accepting a
    /// smaller number would be promising something it cannot do.
    #[test]
    fn the_settings_are_clamped_to_what_the_mesher_can_deliver() {
        let absurd = StepOptions {
            linear_deflection_mm: 0.0,
            angular_deflection_deg: 0.1,
            sew_tolerance_mm: -1.0,
            shape_fix: false,
            parallel: true,
        }
        .sane();
        assert!(absurd.linear_deflection_mm >= 0.001);
        assert!(absurd.angular_deflection_deg >= 3.0);
        assert_eq!(absurd.sew_tolerance_mm, 0.0);
    }

    /// Repair is the preset that reaches further, and it says so in both of the ways it can.
    #[test]
    fn the_repair_preset_both_repairs_and_sews_further() {
        let repair = StepOptions::repair();
        assert!(repair.shape_fix);
        assert!(repair.sew_tolerance_mm > StepOptions::normal().sew_tolerance_mm);
    }
}
