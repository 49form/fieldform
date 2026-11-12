// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A curvature comb: a block that reads a curve and draws its curvature.
//!
//! **A reading, kept as a block.** At samples along the curve a line normal to it whose length is
//! the curvature there, with the tips joined: a smooth envelope is a fair curve, a step in it a
//! tangent-only joint, a kink a corner, a crossing an inflection. It has inputs, a name, an eye
//! and a row, so it is a block; it composes nothing and makes nothing, so it is a record of the
//! document beside the Save Image blocks and the sections and never a node of a part's graph.
//!
//! **The scale is fitted once and then it is a number.** A comb that re-fitted itself to the
//! curve on every edit would draw a flattened bulge exactly as it drew the bulge, which is the
//! thing it exists to show. So it is fitted when the comb is made, stored, and re-fitted only on
//! request ([`fitted_scale`]).
//!
//! **Curvature from the curve, not from a picture of it.** The derivatives are the rational
//! curve's own at each parameter ([`Curve::derivatives`]): one read from the control polygon is
//! wrong on every weighted arc, and one read from a tessellation moves with its tolerance.

use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};

use crate::curve::Curve;
use crate::id::{CombId, CurveId, PartId};

fn is_false(b: &bool) -> bool {
    !*b
}

/// One curvature comb.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurvatureComb {
    pub id: CombId,
    /// A name given to it, or `None` for [`Self::DEFAULT_NAME`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The part whose list shows it: the part active when it was made, as a Save Image block's.
    pub part: PartId,
    /// The curve it reads.
    pub curve: CurveId,
    /// Reads the whole profile the curve is in -- a closed loop or a held open chain -- rather
    /// than the curve alone. The profile is found again every time it is drawn, so a comb whose
    /// profile comes apart reads its curve alone until it is put back together.
    #[serde(default, skip_serializing_if = "is_false")]
    pub whole_loop: bool,
    /// Tooth length per unit curvature, in mm² (a curvature is 1/mm). Fitted when made, then a
    /// number the person owns. Never negative: `flip` is the sign, so that a negative scale and
    /// a flip cannot be two spellings of one fact.
    pub scale: f64,
    /// Samples along the curve, both ends included: 2 to [`Self::MAX_TEETH`].
    pub teeth: u32,
    /// Teeth toward the centre of curvature instead of away from it.
    #[serde(default, skip_serializing_if = "is_false")]
    pub flip: bool,
    /// Hidden by its eye. A hidden curve hides its comb as well, whatever this says.
    #[serde(default, skip_serializing_if = "is_false")]
    pub hidden: bool,
}

impl CurvatureComb {
    pub const DEFAULT_NAME: &'static str = "Curvature Comb";
    pub const DEFAULT_TEETH: u32 = 48;
    pub const MAX_TEETH: u32 = 2000;
    /// The longest tooth, as a fraction of the curve's bounding-box diagonal, when fitted.
    pub const FIT_FRACTION: f64 = 0.15;

    /// A comb on `curve`, fitted to it.
    pub fn new(part: PartId, curve: &Curve) -> Self {
        let teeth = Self::DEFAULT_TEETH;
        Self {
            id: CombId::new(),
            name: None,
            part,
            curve: curve.id,
            whole_loop: false,
            scale: fitted_scale(curve, teeth),
            teeth,
            flip: false,
            hidden: false,
        }
    }

    /// What its row is called.
    pub fn title(&self) -> &str {
        self.name.as_deref().unwrap_or(Self::DEFAULT_NAME)
    }

    /// The teeth asked for, held to what can be drawn.
    pub fn teeth_drawn(&self) -> u32 {
        self.teeth.clamp(2, Self::MAX_TEETH)
    }

    /// What it draws on `curve` now.
    pub fn read(&self, curve: &Curve) -> CombReading {
        read_comb(curve, self.scale.max(0.0), self.teeth_drawn(), self.flip)
    }
}

/// One sample of a comb: where on which curve, its tooth's tip, the curvature there and the way
/// the comb is travelling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tooth {
    /// The curve it is on.
    pub curve: CurveId,
    /// The curve's own parameter, whichever way the comb runs along it.
    pub t: f64,
    pub at: Point3<f64>,
    pub tip: Point3<f64>,
    /// The unsigned curvature, 1/mm.
    pub curvature: f64,
    /// The unit tangent in the direction the comb travels, which is the curve's own unless the
    /// comb runs along the curve backwards.
    pub tangent: Vector3<f64>,
}

/// Where two curves of a profile meet, measured from the samples either side of the meeting: the
/// numbers a joint's continuity is read from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointReading {
    /// The curve arriving, by its place along the profile; the one leaving is the next, or the
    /// first where a loop closes.
    pub after: usize,
    pub at: Point3<f64>,
    /// The angle between the tangent arriving and the tangent leaving, in degrees: nought where
    /// the joint is tangent-continuous.
    pub angle_degrees: f64,
    /// The curvature leaving less the curvature arriving, 1/mm: nought where it is
    /// curvature-continuous as well.
    pub curvature_step: f64,
}

/// What a comb draws and what its panel reads out, computed from the curve every time.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CombReading {
    /// Every tooth, in order along the curve. At a knot where the curve may break, two teeth at
    /// one point: the curvature arriving and the curvature leaving. A profile's joints are the
    /// same: the end of one curve and the start of the next, two teeth at one place.
    pub teeth: Vec<Tooth>,
    /// The tips joined in order, as runs: a sample with no tangent breaks the envelope there,
    /// so a cusp shows as a gap rather than as a line drawn across it.
    pub envelope: Vec<Vec<Point3<f64>>>,
    /// The largest curvature among the samples and where it was, or `None` with no tooth.
    pub max: Option<Tooth>,
    /// Samples where the curve has no tangent (coincident control points, a cusp).
    pub skipped: usize,
    /// Each meeting of two curves of a profile, in order along it. None for one curve.
    pub joints: Vec<JointReading>,
}

impl CombReading {
    /// The radius of the tightest bend sampled, in mm: `None` with nothing bent.
    pub fn tightest_radius(&self) -> Option<f64> {
        self.max.filter(|m| m.curvature > 0.0).map(|m| 1.0 / m.curvature)
    }
}

/// The curvature vector at a sample, toward the centre of curvature: `((P'×P'')×P')/|P'|⁴`, whose
/// length is `|P'×P''|/|P'|³`. `None` with no tangent.
///
/// **No Frenet frame**, so a straight stretch is a vector of length nought rather than a normal
/// divided by nothing, and the vector reverses with the turn by itself -- which is what carries a
/// planar S-bend's comb across the curve at its inflection instead of drawing two bulges. And it
/// is the same whichever way the curve is run: turning `P'` round turns both crosses round.
pub fn curvature_vector(d1: Vector3<f64>, d2: Vector3<f64>) -> Option<Vector3<f64>> {
    let speed = d1.norm();
    if speed < 1e-12 {
        return None;
    }
    Some(d1.cross(&d2).cross(&d1) / speed.powi(4))
}

/// The parameters a comb samples at, each with the span it is read in: `teeth` spaced evenly in
/// the parameter, both ends included, and a pair at every interior knot where the curve may
/// break, one from each side, so a corner is drawn as a step rather than stepped over.
fn samples(curve: &Curve, teeth: u32) -> Vec<(Option<usize>, f64)> {
    let (lo, hi) = curve.domain();
    let n = teeth.max(2) as usize;
    let breaks = curve.breaks();
    let mut out: Vec<(Option<usize>, f64)> = (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        // A sample exactly on a break would be a third tooth at it, read from one side only.
        .filter(|t| !breaks.iter().any(|(_, _, at)| at == t))
        .map(|t| (None, t))
        .collect();
    for (left, right, at) in breaks {
        out.push((Some(left), at));
        out.push((Some(right), at));
    }
    // Stable, so a break's pair keeps its left-then-right order.
    out.sort_by(|a, b| a.1.total_cmp(&b.1));
    out
}

/// One curve's samples in the order the comb travels, `None` where there is no tangent.
fn sample_curve(curve: &Curve, scale: f64, teeth: u32, flip: bool, backwards: bool) -> Vec<Option<Tooth>> {
    let sign = if flip { 1.0 } else { -1.0 };
    let travel = if backwards { -1.0 } else { 1.0 };
    let mut out: Vec<Option<Tooth>> = samples(curve, teeth)
        .into_iter()
        .map(|(span, t)| {
            let (at, d1, d2) = match span {
                Some(span) => curve.derivatives_in_span(span, t),
                None => curve.derivatives(t),
            };
            let k = curvature_vector(d1, d2)?;
            Some(Tooth {
                curve: curve.id,
                t,
                at,
                tip: at + k * (sign * scale),
                curvature: k.norm(),
                tangent: d1.normalize() * travel,
            })
        })
        .collect();
    if backwards {
        out.reverse();
    }
    out
}

/// Teeth, envelope, maximum and skips from samples in order. `close` is a sample the envelope
/// returns to at the end -- a loop's first -- and is not a tooth of its own.
fn assemble(samples: Vec<Option<Tooth>>, close: Option<Tooth>) -> CombReading {
    let mut reading = CombReading::default();
    let mut run: Vec<Point3<f64>> = Vec::new();
    for sample in samples {
        let Some(tooth) = sample else {
            reading.skipped += 1;
            if run.len() > 1 {
                reading.envelope.push(std::mem::take(&mut run));
            }
            run.clear();
            continue;
        };
        if reading.max.is_none_or(|m| tooth.curvature > m.curvature) {
            reading.max = Some(tooth);
        }
        run.push(tooth.tip);
        reading.teeth.push(tooth);
    }
    if let (Some(first), false) = (close, run.is_empty()) {
        run.push(first.tip);
    }
    if run.len() > 1 {
        reading.envelope.push(run);
    }
    reading
}

/// The comb of `curve` at `scale` (mm²) with `teeth` samples.
pub fn read_comb(curve: &Curve, scale: f64, teeth: u32, flip: bool) -> CombReading {
    assemble(sample_curve(curve, scale, teeth, flip, false), None)
}

/// Which of a profile's curves the comb runs along backwards, the profile being in order with
/// each curve's end meeting the next one's.
///
/// **Read from where the ends are, not stored**, as the profile itself is: the first curve runs
/// forwards unless only its start meets the second, and every later one runs from whichever of
/// its ends meets the end the last one arrived at.
pub fn path_backwards(pieces: &[&Curve]) -> Vec<bool> {
    let meets = |a: Point3<f64>, b: Point3<f64>| (a - b).norm() <= crate::JOINED_MM;
    let mut out = Vec::with_capacity(pieces.len());
    let mut arrived: Option<Point3<f64>> = None;
    for (i, piece) in pieces.iter().enumerate() {
        let (start, end) = piece.ends();
        let backwards = match arrived {
            Some(at) => !meets(start, at) && meets(end, at),
            None => match pieces.get(i + 1).map(|next| next.ends()) {
                Some((s, e)) => !meets(end, s) && !meets(end, e) && (meets(start, s) || meets(start, e)),
                None => false,
            },
        };
        arrived = Some(if backwards { start } else { end });
        out.push(backwards);
    }
    out
}

/// The comb of a profile: its curves in order, each sampled `teeth` times in the direction the
/// profile runs, with the envelope carried across every joint -- and round from the last curve
/// to the first where it closes -- so a G2 joint reads as one smooth envelope, a G1 joint as a
/// step in it, and a corner as a kink in the teeth themselves. One curve is [`read_comb`].
pub fn read_comb_path(pieces: &[&Curve], closed: bool, scale: f64, teeth: u32, flip: bool) -> CombReading {
    if pieces.len() < 2 {
        return pieces.first().map(|c| read_comb(c, scale, teeth, flip)).unwrap_or_default();
    }
    let backwards = path_backwards(pieces);
    let per_piece: Vec<Vec<Option<Tooth>>> = pieces
        .iter()
        .zip(&backwards)
        .map(|(c, b)| sample_curve(c, scale, teeth, flip, *b))
        .collect();
    let mut joints = Vec::new();
    let meetings = if closed { pieces.len() } else { pieces.len() - 1 };
    for i in 0..meetings {
        let arriving = per_piece[i].last().copied().flatten();
        let leaving = per_piece[(i + 1) % pieces.len()].first().copied().flatten();
        if let (Some(a), Some(l)) = (arriving, leaving) {
            joints.push(JointReading {
                after: i,
                at: a.at,
                angle_degrees: a.tangent.dot(&l.tangent).clamp(-1.0, 1.0).acos().to_degrees(),
                curvature_step: l.curvature - a.curvature,
            });
        }
    }
    let close = if closed { per_piece[0].first().copied().flatten() } else { None };
    let mut reading = assemble(per_piece.into_iter().flatten().collect(), close);
    reading.joints = joints;
    reading
}

/// The scale whose longest tooth is [`CurvatureComb::FIT_FRACTION`] of the curve's bounding-box
/// diagonal, read over `teeth` samples -- the ones the comb draws. A curve with no bend gets the
/// diagonal itself, which draws nothing and is a sane number to scrub from; a curve of no size,
/// one.
pub fn fitted_scale(curve: &Curve, teeth: u32) -> f64 {
    fitted_scale_path(&[curve], teeth)
}

/// [`fitted_scale`] over a profile's curves together: one box, one largest curvature.
pub fn fitted_scale_path(pieces: &[&Curve], teeth: u32) -> f64 {
    let mut lo = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut hi = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut most: f64 = 0.0;
    for curve in pieces {
        // The box of a dense sample rather than of a tessellation: a fit is one number, and a
        // tolerance search for it would cost more than the comb.
        let (t0, t1) = curve.domain();
        for i in 0..=256 {
            let p = curve.evaluate(t0 + (t1 - t0) * i as f64 / 256.0);
            lo = lo.inf(&p);
            hi = hi.sup(&p);
        }
        if let Some(m) = read_comb(curve, 0.0, teeth, false).max {
            most = most.max(m.curvature);
        }
    }
    let diagonal = if lo.x.is_finite() { (hi - lo).norm() } else { 0.0 };
    if diagonal <= 0.0 {
        return 1.0;
    }
    if most > 0.0 { CurvatureComb::FIT_FRACTION * diagonal / most } else { diagonal }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quarter_circle(r: f64) -> Curve {
        let w = 1.0 / 2.0_f64.sqrt();
        Curve::new(
            CurveId::new(),
            "arc",
            2,
            vec![Point3::new(r, 0.0, 0.0), Point3::new(r, r, 0.0), Point3::new(0.0, r, 0.0)],
            vec![1.0, w, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            false,
        )
        .unwrap()
    }

    /// The rational case the control polygon gets wrong: a weighted quarter circle reads `1/r`
    /// at every one of its 48 samples, so its comb is flat.
    #[test]
    fn a_weighted_quarter_circle_has_curvature_one_over_r_at_every_sample() {
        let r = 25.0;
        let reading = read_comb(&quarter_circle(r), 100.0, 48, false);
        assert_eq!(reading.teeth.len(), 48);
        for tooth in &reading.teeth {
            assert!((tooth.curvature - 1.0 / r).abs() < 1e-12, "at t={} read {}", tooth.t, tooth.curvature);
        }
        assert!((reading.tightest_radius().unwrap() - r).abs() < 1e-9);
    }

    #[test]
    fn a_line_has_zero_curvature_everywhere_and_a_zero_length_comb() {
        let line = Curve::polyline(
            CurveId::new(),
            "line",
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(30.0, 4.0, 1.0)],
            false,
        )
        .unwrap();
        let reading = read_comb(&line, 1000.0, 48, false);
        assert_eq!(reading.teeth.len(), 48);
        assert!(reading.teeth.iter().all(|t| t.curvature == 0.0 && (t.tip - t.at).norm() == 0.0));
        assert_eq!(reading.tightest_radius(), None, "nothing bent");
        assert!((fitted_scale(&line, 48) - (30.0f64.powi(2) + 16.0 + 1.0).sqrt()).abs() < 1e-9, "the diagonal");
    }

    /// On a circle the comb sits outside, where surfacing tools draw it: each tip is further
    /// from the centre than the curve.
    #[test]
    fn teeth_point_away_from_the_centre_of_curvature() {
        let r = 25.0;
        let reading = read_comb(&quarter_circle(r), 100.0, 16, false);
        for tooth in &reading.teeth {
            let (tip, at) = (tooth.tip.coords, tooth.at.coords);
            assert!(tip.dot(&at) > at.norm_squared(), "at t={}", tooth.t);
            assert!((tip.norm() - (r + 100.0 / r)).abs() < 1e-9, "a tooth is scale times curvature long");
        }
    }

    #[test]
    fn flip_negates_every_tooth_and_nothing_else() {
        let arc = quarter_circle(25.0);
        let out = read_comb(&arc, 40.0, 20, false);
        let back = read_comb(&arc, 40.0, 20, true);
        assert_eq!(out.teeth.len(), back.teeth.len());
        for (a, b) in out.teeth.iter().zip(&back.teeth) {
            assert_eq!((a.t, a.at, a.curvature), (b.t, b.at, b.curvature));
            assert!(((a.tip - a.at) + (b.tip - b.at)).norm() < 1e-12);
        }
        assert_eq!(out.max.map(|m| m.t), back.max.map(|m| m.t));
    }

    /// A polyline's corner is two teeth at one point, one read from each side, joined by the
    /// envelope -- so the step is drawn rather than averaged away.
    #[test]
    fn an_interior_knot_with_a_corner_draws_both_one_sided_tips() {
        let corner = Point3::new(10.0, 0.0, 0.0);
        let polyline = Curve::polyline(
            CurveId::new(),
            "corner",
            vec![Point3::new(0.0, 0.0, 0.0), corner, Point3::new(10.0, 10.0, 0.0)],
            false,
        )
        .unwrap();
        // 5 teeth over [0, 2] would put one on the knot at 1: it gives way to the pair.
        let reading = read_comb(&polyline, 10.0, 5, false);
        let at_corner: Vec<_> = reading.teeth.iter().filter(|t| (t.at - corner).norm() < 1e-12).collect();
        assert_eq!(at_corner.len(), 2, "one from each side, and not a third");
        assert_eq!(reading.teeth.len(), 6);
        assert_eq!(reading.envelope.len(), 1, "one run, through both");
        assert_eq!(reading.envelope[0].len(), 6);

        // A cubic with a curvature step at an interior knot: the two sides read differently.
        let cubic = Curve::new(
            CurveId::new(),
            "kinked",
            2,
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(5.0, 5.0, 0.0),
                Point3::new(10.0, 5.0, 0.0),
                Point3::new(20.0, -10.0, 0.0),
            ],
            vec![1.0; 4],
            vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            false,
        )
        .unwrap();
        let reading = read_comb(&cubic, 10.0, 4, false);
        let knot = cubic.evaluate(0.5);
        let pair: Vec<_> = reading.teeth.iter().filter(|t| (t.at - knot).norm() < 1e-12).collect();
        assert_eq!(pair.len(), 2);
        assert!((pair[0].curvature - pair[1].curvature).abs() > 1e-3, "a quadratic's curvature steps at its knot");
    }

    /// A comb whose curve has a point with no tangent skips it and says so, and the envelope
    /// breaks there rather than being drawn across it.
    #[test]
    fn a_sample_with_no_tangent_is_skipped_and_breaks_the_envelope() {
        let p = Point3::new(5.0, 5.0, 0.0);
        let cusp = Curve::new(
            CurveId::new(),
            "cusp",
            2,
            vec![p, p, Point3::new(10.0, 0.0, 0.0)],
            vec![1.0; 3],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            false,
        )
        .unwrap();
        let reading = read_comb(&cusp, 1.0, 9, false);
        assert_eq!(reading.skipped, 1, "the start, where the first two points coincide");
        assert_eq!(reading.teeth.len(), 8);
    }

    #[test]
    fn a_comb_round_trips_and_writes_nothing_it_was_not_given() {
        let comb = CurvatureComb::new(PartId::new(), &quarter_circle(10.0));
        let text = serde_json::to_string(&comb).unwrap();
        assert!(!text.contains("flip") && !text.contains("hidden") && !text.contains("whole_loop") && !text.contains("name"));
        let back: CurvatureComb = serde_json::from_str(&text).unwrap();
        assert_eq!(back, comb);
    }

    fn line(a: [f64; 3], b: [f64; 3]) -> Curve {
        Curve::polyline(CurveId::new(), "line", vec![Point3::from(a), Point3::from(b)], false).unwrap()
    }

    /// A second quarter of the same circle, from (0, r) round to (-r, 0), drawn the wrong way
    /// round on purpose: from its far end back to the joint.
    fn second_quarter_backwards(r: f64) -> Curve {
        let w = 1.0 / 2.0_f64.sqrt();
        Curve::new(
            CurveId::new(),
            "arc2",
            2,
            vec![Point3::new(-r, 0.0, 0.0), Point3::new(-r, r, 0.0), Point3::new(0.0, r, 0.0)],
            vec![1.0, w, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            false,
        )
        .unwrap()
    }

    /// Two quarters of one circle meet G2: the comb reads no turn and no curvature step there,
    /// though the second is drawn backwards -- which the comb runs in the profile's direction.
    #[test]
    fn a_g2_joint_reads_zero_tangent_angle_and_zero_curvature_step() {
        let r = 25.0;
        let (a, b) = (quarter_circle(r), second_quarter_backwards(r));
        assert_eq!(path_backwards(&[&a, &b]), vec![false, true]);
        let reading = read_comb_path(&[&a, &b], false, 50.0, 24, false);
        assert_eq!(reading.joints.len(), 1);
        let joint = reading.joints[0];
        assert!((joint.at - Point3::new(0.0, r, 0.0)).norm() < 1e-9);
        assert!(joint.angle_degrees < 1e-6, "tangent-continuous: {}", joint.angle_degrees);
        assert!(joint.curvature_step.abs() < 1e-9, "curvature-continuous: {}", joint.curvature_step);
        assert_eq!(reading.envelope.len(), 1, "one envelope across the joint");
        assert_eq!(reading.teeth.len(), 48);
        // The backwards curve's teeth run from the joint outward, the way the profile goes.
        assert!((reading.teeth[24].at - Point3::new(0.0, r, 0.0)).norm() < 1e-9);
    }

    /// An arc into the line tangent to it meets G1: no turn, and the curvature steps from 1/r to
    /// nought, which is the step a comb exists to show.
    #[test]
    fn a_g1_joint_reads_zero_tangent_angle_and_a_nonzero_curvature_step() {
        let r = 25.0;
        let (a, b) = (quarter_circle(r), line([0.0, r, 0.0], [-30.0, r, 0.0]));
        let reading = read_comb_path(&[&a, &b], false, 50.0, 12, false);
        let joint = reading.joints[0];
        assert!(joint.angle_degrees < 1e-6);
        assert!((joint.curvature_step + 1.0 / r).abs() < 1e-12, "{}", joint.curvature_step);
    }

    /// A square of four lines, closed: four right angles, and the envelope carried round from the
    /// last side back to the first.
    #[test]
    fn a_closed_profile_reads_every_joint_and_closes_its_envelope() {
        let sides = [
            line([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]),
            line([10.0, 0.0, 0.0], [10.0, 10.0, 0.0]),
            line([0.0, 10.0, 0.0], [10.0, 10.0, 0.0]), // drawn backwards
            line([0.0, 10.0, 0.0], [0.0, 0.0, 0.0]),
        ];
        let pieces: Vec<&Curve> = sides.iter().collect();
        let reading = read_comb_path(&pieces, true, 1.0, 3, false);
        assert_eq!(reading.joints.len(), 4);
        for joint in &reading.joints {
            assert!((joint.angle_degrees - 90.0).abs() < 1e-9, "{}", joint.angle_degrees);
            assert_eq!(joint.curvature_step, 0.0);
        }
        let envelope = &reading.envelope[0];
        assert_eq!(envelope.len(), 13, "twelve tips and the first again");
        assert_eq!(envelope[0], envelope[12]);
    }
}
