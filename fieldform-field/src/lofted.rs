// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Sections passed through in turn along a centerline — the loft. the loft-spec design note.
//!
//! # Correspondence, not value mixing (ruled by the owner 2026-10-04, spec §13)
//!
//! A loft interpolates *boundaries*, vertex to vertex, the way a NURBS loft does. Every section
//! is resampled to one vertex count with its corners matched to the other sections' corners
//! ([`correspond`]), and the section at a station between two listed ones is the polygon whose
//! vertex `j` lies between their vertex `j`s — straight between them at G0, on a Hermite rail
//! through every listed position at G1. The field is the signed distance to that polygon, read
//! in the frame of the query's own station.
//!
//! **Why not mix the two sections' distance values**, which is what this leaf did first:
//! mixing two distance fields does not carry a shape from one to the other, it shrinks it where
//! the two do not overlap (`morph-spec.md` §2); the creases left by two sections' corners run
//! across the surface without meeting each other, which is the first thing anyone lofting a
//! drawn profile sees; and a smooth mix bulges past the sections it passes through. A polygon
//! whose vertices travel does none of those: a corner runs to the corner it was matched with,
//! the section is a real section the whole way, and its distance is exact in the plane, so what
//! the field is divided by is the sweep's own bound rather than one that doubled the viewport's
//! steps — measured in `fieldform-gpu/examples/loft_cost.rs` before the change and again after.
//!
//! # The centerline
//!
//! Every section is read in the query's own station frame, relative to the spine point — the
//! sweep's arrangement, so a loft bends with its path and its sections turn with the frame. The
//! centerline is also where the correspondence is decided: corners are matched in frame
//! coordinates, where "no twist" means what a person means by it.
//!
//! # What it shares with a sweep
//!
//! Everything but the section: the spine, the rotation-minimising [`crate::SpineFrame`], the
//! slab at the ends, and the bend's own stretch.

use crate::{Normalization, ScalarField, Spine, SpineFrame};
use nalgebra::{Point2, Point3, Vector2, Vector3};

/// The most vertices a section is resampled to, and so the most segments one reading of a
/// loft walks whatever the sections were drawn with.
///
/// **A number the field depends on, so it is one number, and it was chosen by measurement.**
/// The surface between two sections is the polygon through the resampled vertices, and a
/// circle resampled to `N` sits inside its arc by `r(1 − cos(π/N))`: at sixty-four, 0.05 mm on
/// a 40 mm radius and 0.11 mm on a 90 mm one — under the quarter millimetre the viewport draws
/// at, and far under what the sections' own tessellation at a degree of turn was costing: three
/// hundred and sixty points, walked two or four times a step. The cost is linear in the count;
/// on the owner's three-section loft (`fieldform-gpu/examples/document_cost.rs`, draft frame at
/// 720², still at 1600×900): 96 → 39.1 / 74.5 ms, **64 → 27.9 / 56.9 ms**, 48 → 23.1 / 47.7 ms.
/// Sixty-four is where the shape is still the drawn shape to the eye and the frame is a third
/// cheaper; a count that follows the boundary's turn rather than its length — a square needs
/// four — is the next refinement, and would make this a ceiling rather than the figure.
/// Corners are kept exactly whatever the count, and sections drawn with fewer points than this
/// are resampled to no more than the most any of them has, so two squares stay eight vertices.
pub(crate) const LOFT_SAMPLES: usize = 64;

/// The turn at a vertex that makes it a corner the correspondence keeps and matches, in degrees.
///
/// A curve tessellated at [`crate::PROFILE_TURN_DEGREES`] turns a degree or so at every joint;
/// a drawn corner turns tens. Twenty is well clear of both.
pub(crate) const CORNER_DEGREES: f64 = 20.0;

/// How a loft gets from one section to the next.
///
/// **Its own type rather than [`fieldform_scene::RampShape`]**, which spells a *law's* easing
/// along a path. This is the continuity of an interpolation *between sections*, which is a
/// different question with the same two answers' names; sharing the type would be D9 applied
/// to two things that merely sound alike. `fieldform_scene::LoftSmooth` is its half.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Smooth {
    /// Each vertex runs straight from one section to the next. Creased at every interior
    /// section, which is sometimes exactly what was wanted.
    #[default]
    Linear,
    /// Each vertex runs on a Hermite rail through every listed position, so the solid passes
    /// through each section without a crease. The rails' slopes are one-sided at the two ends.
    Hermite,
}

/// Why a loft could not be built.
#[derive(Clone, Debug, PartialEq)]
pub enum LoftError {
    /// Fewer than two sections is not an interpolation.
    TooFewSections(usize),
    /// Two sections station at the same place or out of the order they are listed in, by index.
    ///
    /// The station is the arc of the spine point nearest the section's centroid, so this is how
    /// a section dragged past its neighbour — or a spine reshaped under them — reports itself.
    /// Named by index rather than refused silently, because which two crossed is the whole of
    /// what the person needs to know.
    SectionsOutOfOrder { earlier: usize, later: usize },
    /// A spine with fewer than two distinct points has nothing to station along.
    SpineHasNoLength,
    /// The sections reach further from the spine than its tightest bend is wide, so the solid
    /// would fold through itself. The sweep's own refusal, for the same reason.
    FoldsThroughItself { curvature: f64, reach: f64 },
}

impl std::fmt::Display for LoftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoftError::TooFewSections(n) => write!(
                f,
                "a loft needs two sections to interpolate between and has {n}"
            ),
            LoftError::SectionsOutOfOrder { earlier, later } => write!(
                f,
                "section {later} stations before section {earlier} along the centerline, so the \
                 solid would double back on itself \u{2014} reorder them, or move one along the \
                 path"
            ),
            LoftError::SpineHasNoLength => {
                write!(f, "the centerline has no length, so there is nowhere to station the sections")
            }
            LoftError::FoldsThroughItself { curvature, reach } => write!(
                f,
                "the sections reach {reach:.2} mm from the centerline and its tightest bend has \
                 a radius of {:.2} mm, so the loft would fold through itself \u{2014} shrink the \
                 sections or ease the bend",
                1.0 / curvature
            ),
        }
    }
}

impl std::error::Error for LoftError {}

/// One section, placed: where it sits along the spine and its vertices in the frame there.
#[derive(Clone, Debug)]
struct Station {
    /// Arc length along the spine, in millimetres.
    at: f64,
    /// The section resampled to [`LOFT_SAMPLES`] vertices in correspondence with every other
    /// station's — vertex `j` here runs to vertex `j` there — in the plane square to the spine
    /// at `at`, relative to the spine point, counter-clockwise.
    vertices: Vec<Point2<f64>>,
    /// Each vertex's rail slope here, per millimetre of arc: central over the neighbouring
    /// stations, one-sided at the ends. Empty for a linear loft, which has no rails.
    slopes: Vec<Vector2<f64>>,
}

/// Sections passed through in turn along a centerline.
#[derive(Clone, Debug)]
pub struct Lofted {
    stations: Vec<Station>,
    pub spine: Spine,
    frame: SpineFrame,
    smooth: Smooth,
    /// `√(1 + along²) / (1 − κ·ρ)`: see [`Lofted::stretch`].
    stretch: f64,
    /// How far each section's own plane is from square to the spine at its station, in
    /// degrees -- the **tilt reading** of `loft-spec.md` §3, one per section.
    tilts: Vec<f64>,
    /// How far past the furthest vertex of any listed section a smooth rail can bulge between
    /// two stations. See [`Lofted::overshoot`].
    overshoot: f64,
    /// How far along toward folding through itself the loft is: the spine's tightest curvature
    /// times the furthest the interpolated section reaches, which is refused at one.
    bend: f64,
    /// How the rims are cut where the side meets the first and the last section. Hard unless
    /// [`Lofted::with_rims`] says otherwise; kept apart from the build, which is shared by every
    /// reader of the same sections and does not depend on it.
    rims: crate::graph_field::Rim,
}

/// The centerline a loft builds for itself when it is given none: a cubic from the first
/// section's centroid to the last's, setting off along each one's own plane normal.
///
/// **Derived at compile from the sections, and never stored.** A stored spine would be a
/// second description of where the sections are, and the two would drift the moment one was
/// edited — the rule a sweep's frame and an extrude's plane already follow. Handle lengths are
/// a third of the chord, which is the usual choice and keeps the curve inside the hull of the
/// two tangents.
///
/// The normals are oriented first-to-last, so a section drawn the other way round does not
/// send the path backwards; a normal square to the chord has nothing to say about direction
/// and the chord is used instead.
///
/// **It is a guess, and the panel says so** by drawing it dashed and offering to make it a
/// real curve. `loft-spec.md` §3 carries the ruling and the trade-off.
pub(crate) fn derived_centerline(sections: &[Vec<Point3<f64>>], samples: usize) -> Option<Vec<Point3<f64>>> {
    let [a, c0, c1, b] = derived_centerline_cubic(sections)?;
    let n = samples.max(2);
    Some(
        (0..=n)
            .map(|i| {
                let t = i as f64 / n as f64;
                let (u, t2, u2) = (1.0 - t, t * t, (1.0 - t) * (1.0 - t));
                Point3::from(
                    a.coords * (u2 * u)
                        + c0.coords * (3.0 * u2 * t)
                        + c1.coords * (3.0 * u * t2)
                        + b.coords * (t2 * t),
                )
            })
            .collect(),
    )
}

/// The derived centerline as the four control points of the cubic it is, in order -- what
/// "make it a real curve" writes into the document, so that the curve a person then drags is
/// the very path the loft was following and not a fit to it.
pub fn derived_centerline_cubic(sections: &[Vec<Point3<f64>>]) -> Option<[Point3<f64>; 4]> {
    let first = sections.first()?;
    let last = sections.last()?;
    let centroid = |points: &Vec<Point3<f64>>| centroid_of(without_seam(points));
    let (a, b) = (centroid(first), centroid(last));
    let chord = b - a;
    let span = chord.norm();
    if span <= 1e-9 {
        return None;
    }
    let along = chord / span;
    // Each end's own plane normal, turned to point the way the path travels. A section whose
    // normal is square to the chord says nothing about direction, so the chord answers.
    let facing = |points: &Vec<Point3<f64>>| {
        crate::plane_of(points)
            .map(|(_, n)| n)
            .filter(|n| n.dot(&along).abs() > 1e-6)
            .map(|n| if n.dot(&along) < 0.0 { -n } else { n })
            .unwrap_or(along)
    };
    let handle = span / 3.0;
    Some([a, a + facing(first) * handle, b - facing(last) * handle, b])
}

/// A closed polyline without the point a closed curve's tessellation repeats at its seam --
/// `Profile::new`'s own rule, so the centroid a section is stationed by is the centroid of the
/// boundary the section is read as. A rational circle begins and ends on one control point,
/// and counting it twice pulled every such section's station a little toward that point.
/// Found in review.
fn without_seam(points: &[Point3<f64>]) -> &[Point3<f64>] {
    match (points.first(), points.last()) {
        (Some(a), Some(b)) if points.len() > 1 && (a - b).norm() <= 1e-12 => {
            &points[..points.len() - 1]
        }
        _ => points,
    }
}

fn centroid_of(points: &[Point3<f64>]) -> Point3<f64> {
    Point3::from(
        points.iter().fold(Vector3::zeros(), |sum, p| sum + p.coords) / points.len().max(1) as f64,
    )
}

/// A closed polygon, read by arc length.
struct Loop {
    points: Vec<Point2<f64>>,
    /// Arc length from vertex 0 to each vertex, and the whole perimeter last.
    arc: Vec<f64>,
}

impl Loop {
    /// Counter-clockwise, with no repeated points. Orientation is normalised so that vertex
    /// `j` of every section travels the same way round, which the correspondence depends on.
    fn new(points: &[Point2<f64>]) -> Self {
        let mut kept: Vec<Point2<f64>> = Vec::with_capacity(points.len());
        for p in points {
            if kept.last().is_none_or(|q| (p - q).norm() > 1e-12) {
                kept.push(*p);
            }
        }
        if kept.len() > 1 && (kept[0] - kept[kept.len() - 1]).norm() <= 1e-12 {
            kept.pop();
        }
        let area: f64 = (0..kept.len())
            .map(|i| {
                let (a, b) = (kept[i], kept[(i + 1) % kept.len()]);
                a.x * b.y - b.x * a.y
            })
            .sum();
        if area < 0.0 {
            kept.reverse();
        }
        let mut arc = Vec::with_capacity(kept.len() + 1);
        let mut total = 0.0;
        for i in 0..kept.len() {
            arc.push(total);
            total += (kept[(i + 1) % kept.len()] - kept[i]).norm();
        }
        arc.push(total);
        Self { points: kept, arc }
    }

    fn perimeter(&self) -> f64 {
        *self.arc.last().unwrap_or(&0.0)
    }

    /// The vertices where the boundary turns by more than [`CORNER_DEGREES`], in order.
    fn corners(&self) -> Vec<usize> {
        let n = self.points.len();
        let limit = CORNER_DEGREES.to_radians();
        (0..n)
            .filter(|&i| {
                let before = self.points[i] - self.points[(i + n - 1) % n];
                let after = self.points[(i + 1) % n] - self.points[i];
                let turn = (before.x * after.y - before.y * after.x).atan2(before.dot(&after));
                turn.abs() > limit
            })
            .collect()
    }

    /// The point `s` millimetres of arc round from vertex 0, wrapping.
    fn at(&self, s: f64) -> Point2<f64> {
        let n = self.points.len();
        let total = self.perimeter();
        if n == 0 || total <= 0.0 {
            return self.points.first().copied().unwrap_or_else(Point2::origin);
        }
        let s = s.rem_euclid(total);
        // The segment holding `s`: the last vertex whose arc is not past it.
        let i = match self.arc[..n].binary_search_by(|a| a.partial_cmp(&s).expect("finite")) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let (a, b) = (self.points[i], self.points[(i + 1) % n]);
        let run = self.arc[i + 1] - self.arc[i];
        let t = if run > 0.0 { (s - self.arc[i]) / run } else { 0.0 };
        a + (b - a) * t
    }
}

/// Every section resampled to one vertex count, vertex `j` of each in correspondence with
/// vertex `j` of the others — the whole of what makes this a loft rather than a morph.
///
/// **Corners to corners when the counts agree, arc length otherwise**, which is the rule a
/// NURBS loft follows: where every section has the same number of corners, corner `j` of each
/// is matched to corner `j` of the next — the cyclic shift chosen so the matched corners move
/// least in frame coordinates, so nothing twists that was not drawn twisted — and the run between
/// two corners is sampled at the same fractions of its own length in every section, so corners
/// land on corners and straight runs stay straight. Where the counts differ, or no section has
/// a corner, every section keeps its own corners exactly and spreads the rest of the budget
/// evenly round its perimeter, with its start chosen to move least against the section before
/// it; corner `j` then runs to whatever lies at the same share of the way round, which is what
/// lofting a square into a circle means.
///
/// Frame coordinates throughout: the frame is rotation-minimising, so "the same place" at two
/// stations is the place a person would point at.
fn correspond(sections: &[Vec<Point2<f64>>]) -> Vec<Vec<Point2<f64>>> {
    let loops: Vec<Loop> = sections.iter().map(|s| Loop::new(s)).collect();
    let corners: Vec<Vec<usize>> = loops.iter().map(Loop::corners).collect();
    let most = corners.iter().map(Vec::len).max().unwrap_or(0);
    // No denser than the densest section was drawn, no sparser than twice the corners, and
    // never under four -- the count a polygon needs to be walked at all.
    let drawn = loops.iter().map(|l| l.points.len()).max().unwrap_or(0);
    let samples = LOFT_SAMPLES.min(drawn).max(2 * most).max(4);
    let c = corners[0].len();
    if c > 0 && corners.iter().all(|k| k.len() == c) {
        by_corners(&loops, &corners, samples)
    } else {
        by_arc(&loops, &corners, samples)
    }
}

/// [`correspond`] where every section has `c` corners: matched in order, shifted to move least.
fn by_corners(loops: &[Loop], corners: &[Vec<usize>], samples: usize) -> Vec<Vec<Point2<f64>>> {
    let c = corners[0].len();
    // Which of each section's corners is corner 0, chained from the section before.
    let mut shifts = vec![0usize; loops.len()];
    for k in 1..loops.len() {
        let previous = |j: usize| loops[k - 1].points[corners[k - 1][(j + shifts[k - 1]) % c]];
        let cost = |s: usize| -> f64 {
            (0..c).map(|j| (loops[k].points[corners[k][(j + s) % c]] - previous(j)).norm_squared()).sum()
        };
        shifts[k] = (0..c).min_by(|a, b| cost(*a).total_cmp(&cost(*b))).unwrap_or(0);
    }
    // Each run between two corners gets vertices in proportion to its share of the perimeter,
    // averaged over the sections, and never fewer than its own corner.
    let run_length = |k: usize, j: usize| -> f64 {
        let (from, to) = (corners[k][(j + shifts[k]) % c], corners[k][(j + 1 + shifts[k]) % c]);
        let (a, b) = (loops[k].arc[from], loops[k].arc[to]);
        (b - a).rem_euclid(loops[k].perimeter().max(1e-12)).max(if c == 1 { loops[k].perimeter() } else { 0.0 })
    };
    let shares: Vec<f64> = (0..c)
        .map(|j| {
            (0..loops.len())
                .map(|k| run_length(k, j) / loops[k].perimeter().max(1e-12))
                .sum::<f64>()
                / loops.len() as f64
        })
        .collect();
    let mut counts: Vec<usize> = shares.iter().map(|s| ((s * samples as f64).floor() as usize).max(1)).collect();
    let mut remaining = samples.saturating_sub(counts.iter().sum());
    while remaining > 0 {
        // To the run whose share is least served by what it has.
        let j = (0..c)
            .max_by(|a, b| {
                (shares[*a] / counts[*a] as f64).total_cmp(&(shares[*b] / counts[*b] as f64))
            })
            .unwrap_or(0);
        counts[j] += 1;
        remaining -= 1;
    }
    (0..loops.len())
        .map(|k| {
            let mut out = Vec::with_capacity(samples);
            for j in 0..c {
                let from = loops[k].arc[corners[k][(j + shifts[k]) % c]];
                let length = run_length(k, j);
                for i in 0..counts[j] {
                    out.push(loops[k].at(from + length * i as f64 / counts[j] as f64));
                }
            }
            out
        })
        .collect()
}

/// [`correspond`] where the corner counts differ: each section keeps its own corners and
/// spreads the rest evenly, starting where it moves least against the section before.
fn by_arc(loops: &[Loop], corners: &[Vec<usize>], samples: usize) -> Vec<Vec<Point2<f64>>> {
    // A section's vertices from a start `s0` millimetres round: its corners where they are, and
    // the rest at even fractions of the perimeter from the start, all in arc order from it.
    let sampled = |k: usize, s0: f64| -> Vec<Point2<f64>> {
        let total = loops[k].perimeter().max(1e-12);
        let even = samples - corners[k].len();
        let mut at: Vec<f64> = corners[k]
            .iter()
            .map(|&i| (loops[k].arc[i] - s0).rem_euclid(total))
            .chain((0..even).map(|i| total * (i as f64 + 0.5) / even as f64))
            .collect();
        at.sort_by(|a, b| a.total_cmp(b));
        at.into_iter().map(|s| loops[k].at(s0 + s)).collect()
    };
    let mut out: Vec<Vec<Point2<f64>>> = Vec::with_capacity(loops.len());
    for k in 0..loops.len() {
        let start = if k == 0 {
            // Where the first section crosses its own +u axis, or the nearest vertex to it: a
            // start that depends on the frame and not on where the curve happened to begin.
            let n = loops[0].points.len();
            let best = (0..n)
                .max_by(|a, b| {
                    let dir = |i: usize| {
                        let d = loops[0].points[i].coords;
                        d.x / d.norm().max(1e-12)
                    };
                    dir(*a).total_cmp(&dir(*b))
                })
                .unwrap_or(0);
            loops[0].arc[best]
        } else {
            // The start that moves the vertices least from the section before, out of twice as
            // many candidates as vertices.
            let total = loops[k].perimeter();
            let candidates = 2 * samples;
            (0..candidates)
                .map(|n| total * n as f64 / candidates as f64)
                .min_by(|a, b| {
                    let cost = |s0: f64| -> f64 {
                        sampled(k, s0)
                            .iter()
                            .zip(&out[k - 1])
                            .map(|(p, q)| (p - q).norm_squared())
                            .sum()
                    };
                    cost(*a).total_cmp(&cost(*b))
                })
                .unwrap_or(0.0)
        };
        out.push(sampled(k, start));
    }
    out
}

impl Lofted {
    /// A loft, or the reason there cannot be one.
    ///
    /// `sections` are the world-space polylines in the order the block lists them; `seed` is
    /// the direction the first section's own `+u` axis should point, as the sweep's is.
    pub(crate) fn new(
        sections: &[Vec<Point3<f64>>],
        spine: Spine,
        seed: Vector3<f64>,
        smooth: Smooth,
    ) -> Result<Self, LoftError> {
        if sections.len() < 2 {
            return Err(LoftError::TooFewSections(sections.len()));
        }
        if spine.length() <= 0.0 {
            return Err(LoftError::SpineHasNoLength);
        }
        let frame = SpineFrame::along(&spine, seed).ok_or(LoftError::SpineHasNoLength)?;

        // **Each section stations where the spine passes nearest its centroid**, and is then
        // read in the plane square to the spine there. A section whose own plane is tilted from
        // that one is read as its projection; how far it is tilted is a reading the panel
        // shows, never a tolerance.
        let mut ats: Vec<f64> = Vec::with_capacity(sections.len());
        let mut flats: Vec<Vec<Point2<f64>>> = Vec::with_capacity(sections.len());
        let mut tilts: Vec<f64> = Vec::with_capacity(sections.len());
        for points in sections {
            let points = without_seam(points);
            if points.len() < 3 {
                return Err(LoftError::TooFewSections(0));
            }
            let near = spine.nearest(centroid_of(points)).ok_or(LoftError::SpineHasNoLength)?;
            let (tangent, normal) = frame.at(&spine, near.arc);
            let binormal = tangent.cross(&normal);
            flats.push(
                points
                    .iter()
                    .map(|p| {
                        let d = p - near.point;
                        Point2::new(d.dot(&normal), d.dot(&binormal))
                    })
                    .collect(),
            );
            ats.push(near.arc);
            // The angle between the section's own plane normal and the path, either way round:
            // nought is square, and a section drawn flat across a tilted path reads its tilt.
            // A section too thin to have a plane reads as square, which is what its
            // projection is.
            tilts.push(
                crate::plane_of(points)
                    .map(|(_, n)| n.dot(&tangent).abs().clamp(0.0, 1.0).acos().to_degrees())
                    .unwrap_or(0.0),
            );
        }

        // Strictly increasing, in the order the block lists them. A section that stations
        // before its predecessor is a solid that doubles back, and which two crossed is what
        // the message says.
        for i in 1..ats.len() {
            if ats[i] <= ats[i - 1] {
                return Err(LoftError::SectionsOutOfOrder { earlier: i - 1, later: i });
            }
        }

        let resampled = correspond(&flats);
        let mut stations: Vec<Station> = ats
            .iter()
            .zip(resampled)
            .map(|(at, vertices)| Station { at: *at, vertices, slopes: Vec::new() })
            .collect();
        if smooth == Smooth::Hermite {
            let last = stations.len() - 1;
            for k in 0..=last {
                let (lo, hi) = if k == 0 { (0, 1) } else if k == last { (last - 1, last) } else { (k - 1, k + 1) };
                let h = stations[hi].at - stations[lo].at;
                let slopes: Vec<Vector2<f64>> = (0..stations[k].vertices.len())
                    .map(|j| {
                        let d = stations[hi].vertices[j] - stations[lo].vertices[j];
                        if h > 0.0 { d / h } else { Vector2::zeros() }
                    })
                    .collect();
                stations[k].slopes = slopes;
            }
        }

        // **The tube the solid lies inside, rails included.** A listed vertex is where it is;
        // a smooth rail between two stations lies inside the hull of its Bézier points, so the
        // furthest the surface can get from the spine is the furthest of those -- exact for a
        // linear loft and a bound for a smooth one, and what the extent and the fold test read.
        let reach = stations
            .iter()
            .flat_map(|s| s.vertices.iter().map(|v| v.coords.norm()))
            .fold(0.0f64, f64::max);
        let mut hull = reach;
        let mut along: f64 = 0.0;
        for i in 0..stations.len() - 1 {
            let h = stations[i + 1].at - stations[i].at;
            for j in 0..stations[i].vertices.len() {
                let (a, b) = (stations[i].vertices[j], stations[i + 1].vertices[j]);
                match smooth {
                    Smooth::Linear => along = along.max((b - a).norm() / h),
                    Smooth::Hermite => {
                        let (b1, b2) = (
                            a + stations[i].slopes[j] * (h / 3.0),
                            b - stations[i + 1].slopes[j] * (h / 3.0),
                        );
                        hull = hull.max(b1.coords.norm()).max(b2.coords.norm());
                        // The rail's speed is bounded by its hodograph's control points.
                        along = along
                            .max(3.0 * (b1 - a).norm() / h)
                            .max(3.0 * (b2 - b1).norm() / h)
                            .max(3.0 * (b - b2).norm() / h);
                    }
                }
            }
        }
        let overshoot = (hull - reach).max(0.0);
        let curvature = crate::max_curvature(spine.points());
        let product = curvature * hull;
        if product >= 1.0 {
            return Err(LoftError::FoldsThroughItself { curvature, reach: hull });
        }
        Ok(Self {
            stations,
            spine,
            frame,
            smooth,
            // **The sweep's bound.** In the plane the reading is a true distance to the
            // section, rate one; along the path the section's vertices move at most `along`
            // millimetres per millimetre of arc, and a distance to a thing whose every point
            // moves that fast changes that fast; and the bend stretches the station planes as
            // it stretches a sweep's. No grid is sampled and nothing is estimated.
            stretch: (1.0 + along * along).sqrt() / (1.0 - product),
            tilts,
            overshoot,
            bend: product,
            rims: crate::graph_field::Rim::default(),
        })
    }

    /// The same loft with its rims cut as told. A copy, because the build it starts from is the
    /// one every reader of these sections shares (`build_loft`'s memo), and rims are the block's.
    pub fn with_rims(&self, rims: crate::graph_field::Rim) -> Self {
        Lofted { rims, ..self.clone() }
    }

    /// How deep the material goes in the first and the last section, the shallower of the two --
    /// what a rim at either end is held to (`round-chamfer-spec.md` §4). `Footprint::depth`, the
    /// sampled inradius, of each end's own polygon.
    pub fn end_depth(&self) -> f64 {
        let depth = |k: usize| {
            crate::Profile::new(self.stations[k].vertices.clone())
                .map(|p| crate::Footprint::Closed(p).depth())
                .unwrap_or(0.0)
        };
        depth(0).min(depth(self.stations.len() - 1))
    }

    /// What the reading is divided by to stay a lower bound.
    pub fn stretch(&self) -> f64 {
        self.stretch
    }

    /// Where each section sits along the spine, in millimetres of arc.
    pub fn stations(&self) -> Vec<f64> {
        self.stations.iter().map(|s| s.at).collect()
    }

    /// How far outside every listed section the surface can lie, in the plane.
    ///
    /// **Nought for G0 and not for G1.** A vertex running straight between two stations is
    /// never further out than at one of them. A smooth rail through a small section, two large
    /// ones and a small one overshoots the large pair -- the ordinary bottle -- and the extent
    /// grows by this, or the per-tile cull would clip the bulge. Found in review on the value-
    /// mixing loft and kept across the change to rails, which bulge the same way: radii 10, 30,
    /// 30, 10 over 10, 100 and 10 mm reach 34.55 mm. Bounded by the rails' Bézier hulls.
    pub fn overshoot(&self) -> f64 {
        self.overshoot
    }

    /// How far toward folding through itself the loft is, as a share of the refusal: the
    /// spine's tightest curvature times the furthest the interpolated section reaches from it.
    /// Nought is straight; one is refused. The bend's part of [`Lofted::stretch`] is
    /// `1 / (1 - bend_share)`.
    pub fn bend_share(&self) -> f64 {
        self.bend
    }

    /// How many sections it passes through.
    pub fn section_count(&self) -> usize {
        self.stations.len()
    }

    /// How far each section's own plane is from square to the path at its station, in degrees
    /// and in the order the sections are listed. A reading the panel shows, never a tolerance:
    /// the section is read as its projection whatever the angle.
    pub fn tilts(&self) -> &[f64] {
        &self.tilts
    }

    /// Where section `k` sits, in millimetres of arc along the spine.
    pub fn station(&self, k: usize) -> f64 {
        self.stations[k].at
    }

    /// Section `k` as the leaf reads it: projected square to the spine at its own station,
    /// relative to the spine point there, and resampled into correspondence with the others.
    /// What the shader is handed, so that it walks the very vertices the field walks.
    pub fn section_points(&self, k: usize) -> &[Point2<f64>] {
        &self.stations[k].vertices
    }

    /// Each vertex's rail slope at section `k`, per millimetre of arc, in the vertices' order
    /// -- what the shader is handed beside the section at G1, so a vertex on a rail is two
    /// points and two slopes read rather than four points and two stations computed. Empty
    /// for a linear loft, which has no rails.
    pub fn section_slopes(&self, k: usize) -> &[Vector2<f64>] {
        &self.stations[k].slopes
    }

    /// The rotation-minimising normal at each sample of the spine, in order — what the shader
    /// is handed as the frame, computed on the very polyline it is handed as the spine.
    pub fn frame_normals(&self) -> &[Vector3<f64>] {
        self.frame.normals()
    }

    /// The least area the section has between two listed ones, as a share of the smaller of
    /// the two — the **neck reading** of `loft-spec.md` §6, as §13 reads it.
    ///
    /// **A reading, not a refusal and not a knob.** A section whose vertices travel keeps its
    /// area where a mix of distances lost it, so this no longer fires for two sections apart;
    /// what it reports now is a correspondence that crosses itself on the way -- a section
    /// twisted against its neighbour further than its corners can be matched -- whose polygon
    /// pinches to a bow tie between them. Under a half the panel says so in a sentence.
    pub fn neck(&self) -> Option<(usize, f64)> {
        let mut worst: Option<(usize, f64)> = None;
        for i in 0..self.stations.len().saturating_sub(1) {
            let ends = self.area(i, 0.0).min(self.area(i, 1.0));
            if ends <= 0.0 {
                continue;
            }
            let least = (1..8).map(|s| self.area(i, s as f64 / 8.0)).fold(f64::INFINITY, f64::min);
            let share = least / ends;
            if worst.is_none_or(|(_, w)| share < w) {
                worst = Some((i, share));
            }
        }
        worst
    }

    /// The area the section at `t` through span `i` encloses, by the shoelace -- signed, so a
    /// polygon that crosses itself counts its two lobes against each other, which is what a
    /// neck is.
    fn area(&self, i: usize, t: f64) -> f64 {
        let n = self.stations[i].vertices.len();
        let mut twice = 0.0;
        let mut a = self.vertex(i, t, n - 1);
        for j in 0..n {
            let b = self.vertex(i, t, j);
            twice += a.x * b.y - b.x * a.y;
            a = b;
        }
        0.5 * twice
    }

    /// Which span an arc falls in, and how far along it: `(i, t)` with `t` in `[0, 1]`.
    ///
    /// Clamped at both ends, so past the last section the interpolation holds that section —
    /// the slab is what cuts the solid, not this.
    fn span_at(&self, arc: f64) -> (usize, f64) {
        let last = self.stations.len() - 1;
        if arc <= self.stations[0].at {
            return (0, 0.0);
        }
        if arc >= self.stations[last].at {
            return (last - 1, 1.0);
        }
        let i = self
            .stations
            .windows(2)
            .position(|w| arc >= w[0].at && arc <= w[1].at)
            .unwrap_or(last - 1);
        let (a, b) = (self.stations[i].at, self.stations[i + 1].at);
        let h = b - a;
        (i, if h > 0.0 { (arc - a) / h } else { 0.0 })
    }

    /// Vertex `j` of the section at `t` through span `i`: on the straight run between its two
    /// listed positions, or on the Hermite rail through them with the slopes the stations hold.
    fn vertex(&self, i: usize, t: f64, j: usize) -> Point2<f64> {
        let (a, b) = (self.stations[i].vertices[j], self.stations[i + 1].vertices[j]);
        match self.smooth {
            Smooth::Linear => a + (b - a) * t,
            Smooth::Hermite => {
                let h = self.stations[i + 1].at - self.stations[i].at;
                let (m0, m1) = (self.stations[i].slopes[j], self.stations[i + 1].slopes[j]);
                let (t2, t3) = (t * t, t * t * t);
                let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                let h10 = t3 - 2.0 * t2 + t;
                let h01 = -2.0 * t3 + 3.0 * t2;
                let h11 = t3 - t2;
                Point2::from(a.coords * h00 + m0 * (h10 * h) + b.coords * h01 + m1 * (h11 * h))
            }
        }
    }

    /// The signed distance from `q` to the section at `t` through span `i` — `Profile`'s own
    /// walk, the magnitude the nearest segment and the sign crossing parity by the same
    /// half-open rule, over vertices made on the way rather than stored. The shader's
    /// `loft_section_distance` is this, line for line.
    fn section_distance(&self, i: usize, t: f64, q: Point2<f64>) -> f64 {
        let n = self.stations[i].vertices.len();
        let mut nearest = f64::INFINITY;
        let mut inside = false;
        let mut a = self.vertex(i, t, n - 1);
        for j in 0..n {
            let b = self.vertex(i, t, j);
            let ab = b - a;
            let len2 = ab.norm_squared();
            let s = if len2 > 0.0 { ((q - a).dot(&ab) / len2).clamp(0.0, 1.0) } else { 0.0 };
            nearest = nearest.min((q - (a + ab * s)).norm());
            if (a.y > q.y) != (b.y > q.y) {
                let x = a.x + (q.y - a.y) / (b.y - a.y) * (b.x - a.x);
                if q.x < x {
                    inside = !inside;
                }
            }
            a = b;
        }
        if inside { -nearest } else { nearest }
    }
}

impl ScalarField for Lofted {
    fn distance(&self, p: Point3<f64>) -> f64 {
        let Some(near) = self.spine.nearest(p) else {
            return f64::INFINITY;
        };
        let (tangent, normal) = self.frame.at(&self.spine, near.arc);
        let binormal = tangent.cross(&normal);
        let d = p - near.point;
        let q = Point2::new(d.dot(&normal), d.dot(&binormal));
        // **The arc, unclamped.** `Spine::nearest` clamps to the ends, so past either one it
        // reports that end's own arc and leaves the overshoot in the tangential part of `d`.
        // Adding it back is what gives a cap its distance — without it the slab read nought
        // everywhere beyond the solid, and a point ten millimetres under the base measured
        // only its distance across the section. The sweep carries the same overshoot in its
        // slide, for the same reason.
        let w = near.arc + d.dot(&tangent);
        let (i, t) = self.span_at(w);
        let section = self.section_distance(i, t, q);

        // The slab over the stretch the sections occupy, as a sweep's is over its window.
        let (first, last) = (
            self.stations[0].at,
            self.stations[self.stations.len() - 1].at,
        );
        let slab = (w - 0.5 * (first + last)).abs() - 0.5 * (last - first);
        // A rim through the blend's bound, which needs no divisor the loft did not already take
        // (`Rim::join`) -- the section's rate along the path is in the stretch already.
        let combined = if !self.rims.is_hard() {
            self.rims.join_ends(section, slab, w - 0.5 * (first + last))
        } else if section > 0.0 && slab > 0.0 {
            section.hypot(slab)
        } else {
            section.max(slab)
        };
        combined / self.stretch
    }

    fn normalization(&self) -> Normalization {
        // A section travelling along a path is read in planes the bend stretches, and its own
        // travel adds a rate of its own; both are divided out, and what is left is a bound.
        // A blend against a loft is marked approximate, as one against a sweep is.
        Normalization::LowerBound
    }

    fn lipschitz(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cone, Depth, Extrusion, Profile};
    use std::f64::consts::TAU;

    /// A circle of `r` about the origin in the z = `at` plane, as world points.
    fn circle(r: f64, at: f64, n: usize) -> Vec<Point3<f64>> {
        (0..n)
            .map(|i| {
                let a = TAU * i as f64 / n as f64;
                Point3::new(r * a.cos(), r * a.sin(), at)
            })
            .collect()
    }

    fn square(half: f64, at: f64) -> Vec<Point3<f64>> {
        vec![
            Point3::new(-half, -half, at),
            Point3::new(half, -half, at),
            Point3::new(half, half, at),
            Point3::new(-half, half, at),
        ]
    }

    /// **The fold test counts the bulge.** The bottle of radii 10, 30, 30, 10 bent round a
    /// 33 mm radius: its listed sections reach 30, inside the bend's centre, but the Hermite
    /// between the two 30s reaches past 34 and folds through it -- refused, where the sections'
    /// own reach would have let it build. At 40 mm it is clear of the fold either way. Found
    /// in review, measured by its probe.
    #[test]
    fn the_fold_test_counts_a_smooth_lofts_bulge() {
        let bent = |radius: f64| {
            // An arc in the xz-plane setting off along +z, with each section square to it.
            let at = |s: f64| {
                let theta = s / radius;
                (
                    Point3::new(radius - radius * theta.cos(), 0.0, radius * theta.sin()),
                    Vector3::new(theta.cos(), 0.0, -theta.sin()),
                )
            };
            let sections: Vec<Vec<Point3<f64>>> = [(10.0, 0.0), (30.0, 10.0), (30.0, 110.0), (10.0, 120.0)]
                .into_iter()
                .map(|(r, s): (f64, f64)| {
                    let (centre, u) = at(s);
                    (0..64)
                        .map(|i| {
                            let a = TAU * i as f64 / 64.0;
                            centre + u * (r * a.cos()) + Vector3::y() * (r * a.sin())
                        })
                        .collect()
                })
                .collect();
            let spine = Spine::new((0..=240).map(|i| at(120.0 * i as f64 / 240.0).0).collect());
            Lofted::new(&sections, spine, Vector3::x(), Smooth::Hermite)
        };
        assert!(
            matches!(bent(33.0), Err(LoftError::FoldsThroughItself { .. })),
            "the bulge reaches past the bend's centre"
        );
        let clear = bent(40.0).expect("clear of the fold");
        assert!(clear.overshoot() > 4.0 && clear.overshoot() < 7.0, "a bound on a 4.55 mm bulge: {}", clear.overshoot());
        assert!(clear.bend_share() < 1.0 && clear.bend_share() > 0.8, "{}", clear.bend_share());
    }

    fn straight(length: f64) -> Spine {
        Spine::new(vec![Point3::origin(), Point3::new(0.0, 0.0, length)])
    }

    /// **Two concentric circles along a straight spine are a cone**, which is the acceptance
    /// that checks the stationing, the projection, the interpolation and the slab at once
    /// against a shape this code did not produce.
    ///
    /// Exactly a cone, in fact, and that is worth saying: for two circles centred on the spine
    /// the interpolated reading `(1-t)(|q| - r0) + t(|q| - r1)` is `|q| - ((1-t) r0 + t r1)`,
    /// which is the circle of the interpolated radius. Linear radius against height is a cone.
    #[test]
    fn two_concentric_circles_along_a_straight_spine_are_a_cone() {
        let lofted = Lofted::new(
            &[circle(30.0, 0.0, 720), circle(10.0, 80.0, 720)],
            straight(80.0),
            Vector3::x(),
            Smooth::Linear,
        )
        .expect("two circles loft");
        let cone = Cone {
            base: Point3::origin(),
            top: Point3::new(0.0, 0.0, 80.0),
            base_radius: 30.0,
            top_radius: 10.0,
        };
        let mut worst: f64 = 0.0;
        for i in 0..500 {
            let p = Point3::new(
                -45.0 + (i % 19) as f64 * 5.0,
                -45.0 + (i % 23) as f64 * 4.1,
                -12.0 + (i % 29) as f64 * 3.7,
            );
            let want = cone.distance(p);
            let got = lofted.distance(p) * lofted.stretch();
            if want.abs() > 0.3 {
                assert_eq!(
                    got < 0.0,
                    want < 0.0,
                    "at {p:?} the loft reads {got} and the cone {want}"
                );
            }
            // Never further from the surface than the cone is: the contract, before the
            // division as well as after it.
            assert!(
                lofted.distance(p).abs() <= want.abs() + 0.3,
                "at {p:?} the loft overstates: {} against {want}",
                lofted.distance(p)
            );
            worst = worst.max((got - want).abs());
        }
        // The surface itself, along a radial at three heights.
        for z in [10.0, 40.0, 70.0] {
            let r = 30.0 - 20.0 * z / 80.0;
            assert!(lofted.distance(Point3::new(r - 0.2, 0.0, z)) < 0.0, "inside at {z}");
            assert!(lofted.distance(Point3::new(r + 0.2, 0.0, z)) > 0.0, "outside at {z}");
        }
    }

    /// **Two equal squares along a straight spine are an extrude**, which says the
    /// interpolation does nothing when there is nothing to interpolate -- and that the slab and
    /// the corner are the sweep's own.
    #[test]
    fn two_equal_squares_along_a_straight_spine_are_an_extrude() {
        let lofted = Lofted::new(
            &[square(10.0, 0.0), square(10.0, 50.0)],
            straight(50.0),
            Vector3::x(),
            Smooth::Linear,
        )
        .expect("two squares loft");
        assert!(
            (lofted.stretch() - 1.0).abs() < 1e-9,
            "nothing varies, so nothing is divided: {}",
            lofted.stretch()
        );
        let extrude = Extrusion { rims: Default::default(),
            footprint: Profile::new(vec![
                Point2::new(-10.0, -10.0),
                Point2::new(10.0, -10.0),
                Point2::new(10.0, 10.0),
                Point2::new(-10.0, 10.0),
            ])
            .expect("a square")
            .into(),
            start: 0.0,
            height: 50.0,
            depth: Depth::One,
            draft_tangent: 0.0,
            corners: Default::default(),
        };
        for x in -3..=3 {
            for y in -3..=3 {
                for z in -2..=12 {
                    let p = Point3::new(x as f64 * 6.0, y as f64 * 6.0, z as f64 * 5.0);
                    let (a, b) = (lofted.distance(p), extrude.distance(p));
                    assert!((a - b).abs() < 1e-9, "at {p:?}: loft {a}, extrude {b}");
                }
            }
        }
    }

    /// **Two equal squares lofted with rounded rims are the extrude with the same rims**, to the
    /// bit, at every shape -- the loft's rim is the extrude's join over the same two readings,
    /// and with nothing varying along the path neither divides by anything.
    #[test]
    fn two_equal_squares_lofted_with_rims_are_the_rimmed_extrude() {
        use crate::graph_field::Rim;
        let plain = Lofted::new(
            &[square(10.0, 0.0), square(10.0, 50.0)],
            straight(50.0),
            Vector3::x(),
            Smooth::Linear,
        )
        .expect("two squares loft");
        for exponent in [2.0, 3.0, 1.0] {
            let rims = Rim { reach: 3.0, exponent, ends: Default::default(), square: None, second: None };
            let lofted = plain.with_rims(rims);
            let extrude = Extrusion {
                rims,
                footprint: Profile::new(vec![
                    Point2::new(-10.0, -10.0),
                    Point2::new(10.0, -10.0),
                    Point2::new(10.0, 10.0),
                    Point2::new(-10.0, 10.0),
                ])
                .expect("a square")
                .into(),
                start: 0.0,
                height: 50.0,
                depth: Depth::One,
                draft_tangent: 0.0,
                corners: Default::default(),
            };
            for x in -3..=3 {
                for y in -3..=3 {
                    for z in -2..=12 {
                        let p = Point3::new(x as f64 * 3.7, y as f64 * 3.3, z as f64 * 4.3);
                        let (a, b) = (lofted.distance(p), extrude.distance(p));
                        assert!((a - b).abs() < 1e-9, "n = {exponent} at {p:?}: loft {a}, extrude {b}");
                    }
                }
            }
            assert!(lofted.distance(Point3::new(9.9, 0.0, 49.9)) > 0.0, "n = {exponent}: the rim is cut");
        }
        // The end sections are ten deep, from the middle to an edge.
        assert!((plain.end_depth() - 10.0).abs() < 0.5, "{}", plain.end_depth());
    }

    /// **A section stationed before its predecessor is refused by name**, saying which two
    /// crossed -- a solid that doubles back is not a loft, and which pair did it is the whole
    /// of what the person needs.
    #[test]
    fn a_section_out_of_order_is_refused_by_name() {
        let out = Lofted::new(
            &[circle(10.0, 0.0, 64), circle(10.0, 60.0, 64), circle(10.0, 30.0, 64)],
            straight(80.0),
            Vector3::x(),
            Smooth::Linear,
        );
        assert_eq!(
            out.expect_err("the third section is behind the second"),
            LoftError::SectionsOutOfOrder { earlier: 1, later: 2 }
        );
        // And in order it is ordinary.
        Lofted::new(
            &[circle(10.0, 0.0, 64), circle(10.0, 30.0, 64), circle(10.0, 60.0, 64)],
            straight(80.0),
            Vector3::x(),
            Smooth::Linear,
        )
        .expect("in order, three sections loft");
        // Two sections are the fewest there can be.
        assert_eq!(
            Lofted::new(&[circle(10.0, 0.0, 64)], straight(50.0), Vector3::x(), Smooth::Linear)
                .expect_err("one section is not an interpolation"),
            LoftError::TooFewSections(1)
        );
    }

    /// **Hermite passes through its interior sections without a crease, and G0 does not.**
    /// Differenced as the section's own half-width along the spine either side of the middle
    /// station: a crease is a step in that rate, and a smooth pass is not.
    #[test]
    fn hermite_has_no_crease_at_an_interior_section_where_linear_does() {
        let sections = [circle(20.0, 0.0, 360), circle(10.0, 50.0, 360), circle(20.0, 100.0, 360)];
        let half_width = |l: &Lofted, z: f64| {
            (0..4000)
                .map(|i| i as f64 * 0.01)
                .find(|x| l.distance(Point3::new(*x, 0.0, z)) > 0.0)
                .unwrap_or(0.0)
        };
        let rate = |l: &Lofted, z: f64| (half_width(l, z + 1.0) - half_width(l, z - 1.0)) / 2.0;
        for (smooth, creased) in [(Smooth::Linear, true), (Smooth::Hermite, false)] {
            let l = Lofted::new(&sections, straight(100.0), Vector3::x(), smooth)
                .expect("three circles loft");
            // The rate a little either side of the middle station at 50.
            // **Close to the station, because a crease is a *step* in the rate and this
            // shape turns its rate through zero there either way.** Five millimetres out
            // the smooth pass already reads +/-0.075 for the shape's own curvature, which
            // is not a crease and the first draft of this test called one.
            let (before, after) = (rate(&l, 49.0), rate(&l, 51.0));
            let step = (after - before).abs();
            if creased {
                assert!(
                    step > 0.3,
                    "G0 should crease at its middle section: the rate goes {before} -> {after}"
                );
            } else {
                assert!(
                    step < 0.08,
                    "G1 should pass through smoothly: the rate goes {before} -> {after}"
                );
            }
        }
    }

    /// **A section carried across the spine keeps its area**, which mixing two distance
    /// fields could not do (`morph-spec.md` §2) and vertex correspondence does by
    /// construction: the circle travels. The neck reading, which fired for this under the
    /// first design, is quiet -- and the surface really is there in the middle.
    #[test]
    fn a_section_carried_across_the_spine_no_longer_necks() {
        let apart: Vec<Vec<Point3<f64>>> = vec![
            circle(12.0, 0.0, 180).iter().map(|p| p + Vector3::new(-16.0, 0.0, 0.0)).collect(),
            circle(12.0, 60.0, 180).iter().map(|p| p + Vector3::new(16.0, 0.0, 0.0)).collect(),
        ];
        let split = Lofted::new(&apart, straight(60.0), Vector3::x(), Smooth::Linear)
            .expect("lofts");
        let (_, share) = split.neck().expect("a span to measure");
        assert!(share > 0.95, "a travelling circle keeps its area: {share}");
        // Half way the circle is centred on the spine, radius 12.
        assert!(split.distance(Point3::new(0.0, 0.0, 30.0)) < 0.0, "solid at the middle");
        assert!(split.distance(Point3::new(11.0, 0.0, 30.0)) < 0.0, "inside its radius");
        assert!(split.distance(Point3::new(13.0, 0.0, 30.0)) > 0.0, "and outside it");
    }

    /// **Corners run to corners.** A square lofted to a square turned forty degrees about
    /// the spine: each corner is matched to the corner it is nearest, so half way the section
    /// is the square turned twenty degrees -- a corner is still a corner -- and not the
    /// rounded blur a mix of the two distances gives, whose creases miss each other. The
    /// owner's report that opened §13.
    #[test]
    fn corners_run_to_corners_and_stay_corners_between() {
        let turned = |half: f64, at: f64, degrees: f64| -> Vec<Point3<f64>> {
            let a = degrees.to_radians();
            square(half, at)
                .iter()
                .map(|p| Point3::new(p.x * a.cos() - p.y * a.sin(), p.x * a.sin() + p.y * a.cos(), p.z))
                .collect()
        };
        let lofted = Lofted::new(
            &[turned(20.0, 0.0, 0.0), turned(20.0, 60.0, 40.0)],
            straight(60.0),
            Vector3::x(),
            Smooth::Linear,
        )
        .expect("two squares loft");
        // The section half way: each corner is half way along the chord between its two
        // listed positions, so it sits on the diagonal turned twenty degrees at the chord's
        // radius, 20√2·cos 20° -- a square turned twenty degrees and scaled by cos 20°, which
        // is what a NURBS loft of the same two squares gives too.
        let scale = 20.0f64.to_radians().cos();
        let diagonal = (45.0f64 + 20.0).to_radians();
        let reach = 28.28 * scale;
        let corner = Point3::new(reach * diagonal.cos(), reach * diagonal.sin(), 30.0);
        assert!(lofted.distance(corner * 0.98) < 0.0, "just inside the corner");
        assert!(lofted.distance(corner * 1.03) > 0.0, "just outside it");
        let side = 20.0f64.to_radians();
        let mid = Point3::new(20.0 * scale * side.cos(), 20.0 * scale * side.sin(), 30.0);
        assert!(lofted.distance(mid * 0.98) < 0.0 && lofted.distance(mid * 1.03) > 0.0, "the side's middle");
        // And it is exactly that square: its edges' midpoints read nought.
        let twenty = turned(20.0 * scale, 30.0, 20.0);
        for (k, v) in twenty.iter().enumerate() {
            let next = twenty[(k + 1) % 4];
            let on_edge = Point3::from((v.coords + next.coords) * 0.5);
            assert!(lofted.distance(on_edge).abs() * lofted.stretch() < 0.05, "on the edge: {}", lofted.distance(on_edge));
        }
    }

    /// **Never overstates**, which is the contract everything else rests on: along a bent
    /// spine, Hermite, four dissimilar sections, in six directions.
    #[test]
    fn a_loft_never_overstates_the_room_before_its_surface() {
        let bend: Vec<Point3<f64>> = (0..=60)
            .map(|i| {
                let a = 0.9 * std::f64::consts::PI * i as f64 / 60.0;
                Point3::new(40.0 * a.sin(), 6.0 * a.sin() * a.sin(), 40.0 * (1.0 - a.cos()))
            })
            .collect();
        let spine = Spine::new(bend);
        // Four sections placed along it by eye, each a different shape.
        let at_station = |k: usize| {
            let n = spine.points().len();
            spine.points()[(n - 1) * k / 3]
        };
        let ring = |c: Point3<f64>, r: f64, n: usize| -> Vec<Point3<f64>> {
            (0..n)
                .map(|i| {
                    let a = TAU * i as f64 / n as f64;
                    c + Vector3::new(r * a.cos(), r * a.sin(), 0.0)
                })
                .collect()
        };
        let sections = [
            ring(at_station(0), 9.0, 48),
            ring(at_station(1), 5.0, 4),
            ring(at_station(2), 8.0, 48),
            ring(at_station(3), 6.0, 6),
        ];
        let lofted = Lofted::new(&sections, spine, Vector3::y(), Smooth::Hermite)
            .expect("four sections loft");
        let dirs = [
            Vector3::x(),
            Vector3::y(),
            Vector3::z(),
            -Vector3::x(),
            -Vector3::y(),
            -Vector3::z(),
        ];
        for x in -4..=8 {
            for y in -3..=3 {
                for z in -2..=10 {
                    let from = Point3::new(x as f64 * 8.0, y as f64 * 5.0, z as f64 * 9.0);
                    let claimed = lofted.distance(from);
                    if claimed <= 0.0 {
                        continue;
                    }
                    for d in dirs {
                        for step in 1..=12 {
                            let at = from + d * (claimed * step as f64 / 12.0);
                            assert!(
                                lofted.distance(at) > -1e-9,
                                "from {from:?} along {d:?}: crossed the surface inside a \
                                 claimed {claimed}",
                            );
                        }
                    }
                }
            }
        }
    }

    /// **The derived centerline sets off along each end section's own normal**, which is what
    /// makes two parallel circles offset sideways produce an S rather than a straight line
    /// through them. `loft-spec.md` §3's acceptance.
    #[test]
    fn a_derived_centerline_leaves_each_section_along_its_own_normal() {
        // Two circles in parallel planes (both normal to +z), offset in x: the path has to
        // leave the first going up, bend across, and arrive at the second going up.
        let a = circle(10.0, 0.0, 64);
        let b: Vec<Point3<f64>> =
            circle(10.0, 60.0, 64).iter().map(|p| p + Vector3::new(40.0, 0.0, 0.0)).collect();
        let path = derived_centerline(&[a, b], 48).expect("two sections give a path");
        let n = path.len();
        let start = (path[1] - path[0]).normalize();
        let end = (path[n - 1] - path[n - 2]).normalize();
        // Both ends run along the sections' own normal, +z, to within the sampling.
        assert!(start.dot(&Vector3::z()) > 0.999, "it leaves along the first normal: {start:?}");
        assert!(end.dot(&Vector3::z()) > 0.999, "and arrives along the last: {end:?}");
        // And it is an S: the middle has crossed most of the way in x while the ends have not.
        let middle = path[n / 2];
        assert!(
            middle.x > 15.0 && middle.x < 25.0,
            "the path should cross in the middle, not at an end: {middle:?}"
        );
        // It starts and ends at the two centroids.
        assert!((path[0] - Point3::origin()).norm() < 1e-9);
        assert!((path[n - 1] - Point3::new(40.0, 0.0, 60.0)).norm() < 1e-9);
    }

    /// A loft on its own derived centerline is an ordinary loft: the two circles above station
    /// in order and build.
    #[test]
    fn a_loft_can_stand_on_the_centerline_it_derives() {
        let a = circle(12.0, 0.0, 120);
        let b: Vec<Point3<f64>> =
            circle(6.0, 60.0, 120).iter().map(|p| p + Vector3::new(30.0, 0.0, 0.0)).collect();
        let path = derived_centerline(&[a.clone(), b.clone()], 64).expect("a path");
        let lofted = Lofted::new(&[a, b], Spine::new(path), Vector3::x(), Smooth::Linear)
            .expect("it lofts on its own path");
        // Solid just inside both ends, which is what "passes through each section" means.
        // **Just inside, not on them**: a section sits on its own cap, where the slab reads
        // nought and the solid is not negative — the same thing an extrude does at its plane.
        assert!(lofted.distance(Point3::new(0.0, 0.0, 2.0)) < 0.0, "inside the first section");
        assert!(
            lofted.distance(Point3::new(30.0, 0.0, 58.0)) < 0.0,
            "and inside the last"
        );
        // And the neck reading is healthy: a derived centerline keeps the sections overlapping,
        // which is the whole argument for having one (`morph-spec.md` section 2).
        let (_, share) = lofted.neck().expect("a span");
        assert!(share > 0.5, "its own centerline should not neck: {share}");
    }

    /// Sections wider than the bend are refused by name, as a sweep's section is.
    #[test]
    fn sections_wider_than_the_bend_are_refused() {
        let arc: Vec<Point3<f64>> = (0..=40)
            .map(|i| {
                let a = 0.5 * std::f64::consts::PI * i as f64 / 40.0;
                Point3::new(10.0 * a.cos(), 0.0, 10.0 * a.sin())
            })
            .collect();
        let spine = Spine::new(arc);
        let wide = [
            circle(14.0, 0.0, 64).iter().map(|p| Point3::new(p.x + 10.0, p.z, p.y)).collect(),
            circle(14.0, 0.0, 64).iter().map(|p| Point3::new(p.x, p.z, p.y + 10.0)).collect(),
        ];
        let err = Lofted::new(&wide, spine, Vector3::y(), Smooth::Linear)
            .expect_err("sections wider than the bend fold");
        assert!(
            matches!(err, LoftError::FoldsThroughItself { .. }),
            "refused as {err:?} rather than for folding"
        );
        assert!(err.to_string().contains("fold"), "{err}");
    }
}

