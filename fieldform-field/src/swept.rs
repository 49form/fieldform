// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A profile carried along a curve — the general case of a pipe.
//!
//! `Node::Pipe` sweeps a circle and never has to ask which way is up, because a circle is the
//! same whichever way it is held. Any other profile has to be *oriented* at every point of the
//! spine, continuously, and that orientation is the whole of what this module adds to the two
//! things it composes: a [`Profile`], which already answers a 2D signed distance, and a
//! [`Spine`], which already answers the nearest point and how far along it is.
//! the sweep-spec design note carries the decisions.
//!
//! # The frame is rotation-minimising, and the alternatives were rejected by name
//!
//! Frenet's normal flips at every inflection. A fixed up-vector spins the section through a full
//! turn wherever the spine's tangent passes through it — plausible geometry, wrong shape, nothing
//! reporting it, which is the failure D8 exists to name. A rotation-minimising frame is carried
//! along the curve by parallel transport, so the section turns only as much as the curve forces
//! it to: what a hand holding a profile does. Computed once per spine sample by double reflection
//! (Wang, Jüttler, Zheng and Bosch, 2008), which is exact for a polyline.
//!
//! # The bound
//!
//! Section planes are parallel on a straight run and converge on the inside of a bend, so the map
//! from a point to how far along the path it is stretches by up to `1 / (1 − κ·r)` — tube
//! coordinates' own Jacobian. Past `κ·r = 1` the section is wider than the bend and the sweep
//! folds through itself; that is refused rather than bounded.
//!
//! **The stretch is the arc's alone.** In a rotation-minimising frame the section's own two
//! coordinates change at exactly the rate a point moves, however the path bends, so a reading
//! taken in the section is a true distance and only what reads the arc -- the caps, and a law
//! that changes the section along the way -- is stretched. A section held square under the plain
//! law is therefore read as it is, with the caps divided where they are read from inside, in the
//! tube about the spine where every point has one nearest place ([`tube_radius`]); beyond the
//! tube it reads no more than the tube allows. A polyline's frame leans off its segments toward
//! the joints and, off a plane, twists about them, and the reading gives up what those can
//! account for ([`frame_tilt`]). Everything else divides its whole value by the stretch, as a
//! twist and an attractor divide by theirs. Both report [`Normalization::LowerBound`].
//! `sweep-spec.md` §4 carries the argument.

use crate::{Normalization, ScalarField, Spine};
use nalgebra::{Matrix3, Point2, Point3, Vector3};

/// How the section is held as it travels — the kernel's side of `fieldform_scene::Section`.
///
/// **Two shapes of answer, not three.** Square is the original: the section is perpendicular to
/// the path everywhere and the evaluation can find it from the *nearest point* on the spine,
/// which is one profile reading per sample. The other two hold the section at some angle to the
/// path, and then the nearest point is no longer the section a point belongs to — what is, is
/// the section plane that *contains* it, which has to be solved for segment by segment. So they
/// share one evaluation and Square keeps its own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Held {
    /// Perpendicular to the path, turning with it.
    Square,
    /// Flush with the profile's own plane at the start, carrying that tilt relative to the path
    /// the whole way. Given as the profile's basis; the tilt itself is worked out once, against
    /// the frame's first sample, and carried by the frame from there.
    Flush { normal: Vector3<f64>, u: Vector3<f64>, v: Vector3<f64> },
    /// Never turns. One section basis for the whole path: the profile's own.
    Parallel { normal: Vector3<f64>, u: Vector3<f64>, v: Vector3<f64> },
}

/// How a section changes size along the path -- `sweep-v2-spec.md` §2.
///
/// **Two, because they answer different questions and each is one line in the leaf.** Scale
/// keeps the section's shape and is what a guide rail will drive; a draft leans every wall at
/// one angle, which is what a moulded part wants and what the extrude's draft already is --
/// the two are **one construction**, so a drafted sweep along a straight path is the drafted
/// extrude of the same section.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Taper {
    None,
    /// The section is `end` times its drawn size where the path ends, following the law.
    Scale { end: f64 },
    /// Every wall leans out by this angle's tangent per millimetre of path. A constant angle
    /// by definition, so it does not follow the law.
    Draft { tangent: f64 },
}

/// How a section changes as it travels: its taper, its turn about the path, the law that
/// distributes both, and the stretch of path the solid occupies.
///
/// The one mechanism of `sweep-v2-spec.md` §2: with `f` the fraction of the way along the
/// window, the section at `f` is the outline scaled by `s(f) = 1 + (end − 1)·shape(f)` about
/// the point the path carries it by, then turned by `θ(f) = twist·shape(f)`. Built as a
/// function of arc fraction and not baked into the field, so a guide rail (V2) can drive the
/// same two numbers from a curve instead.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SectionLaw {
    pub taper: Taper,
    /// Whether a draft rounds the section's corners or mitres them -- the same choice an
    /// extrude carries, because a drafted sweep along a straight path *is* the drafted
    /// extrude. Nothing for a scale, which moves no wall along its own normal.
    pub corners: fieldform_scene::Corners,
    /// How far the section turns about the path over the window, in radians, positive from
    /// the section's `+u` toward its `+v`.
    pub twist: f64,
    /// How the scale and the turn are distributed along the window.
    pub shape: fieldform_scene::RampShape,
    /// Where along the spine the solid runs, as fractions of its length: `(0, 1)` is all of it.
    pub window: (f64, f64),
}

impl Default for SectionLaw {
    fn default() -> Self {
        Self {
            taper: Taper::None,
            corners: fieldform_scene::Corners::default(),
            twist: 0.0,
            shape: fieldform_scene::RampShape::Linear,
            window: (0.0, 1.0),
        }
    }
}

impl SectionLaw {
    /// Whether this is the law every sweep had before there was one: no taper, no turn, the
    /// whole path. A sweep under it is the sweep it always was, byte for byte in the shader.
    pub(crate) fn is_plain(&self) -> bool {
        *self == Self::default()
    }

    /// The scale at a fraction `f` of the way along the window.
    pub(crate) fn scale_at(&self, f: f64) -> f64 {
        match self.taper {
            Taper::Scale { end } => 1.0 + (end - 1.0) * self.shape.apply(f),
            _ => 1.0,
        }
    }

    /// The turn at a fraction `f` of the way along the window, in radians.
    pub(crate) fn turn_at(&self, f: f64) -> f64 {
        self.twist * self.shape.apply(f)
    }

    /// The scale at the two ends of the window, least first: the scale is monotone in the
    /// law, so these are its extremes.
    fn scale_range(&self) -> (f64, f64) {
        match self.taper {
            Taper::Scale { end } => (end.min(1.0), end.max(1.0)),
            _ => (1.0, 1.0),
        }
    }

    /// How far a section that reaches `reach` from the point carrying it can get from that
    /// point under this law, over a window `run` long: the furthest the scale takes it, plus
    /// the whole of a draft's offset outward. **What the fold refusal, the bound and the
    /// extent all measure by**, which is why it is one function.
    pub(crate) fn reach_under(&self, reach: f64, run: f64) -> f64 {
        let grown = match self.taper {
            Taper::Draft { tangent } => (run * tangent).max(0.0),
            _ => 0.0,
        };
        self.scale_range().1 * reach + grown
    }

    /// What the law adds to the bend's stretch: `√(1 + K²)`, with `K` how fast the reading can
    /// change along the path -- `sweep-v2-spec.md` §3: `|s′|·r + |θ′|ρ + |tan α|`, the law's
    /// slopes bounded by its shape's steepest over the window's length. One under the plain law.
    ///
    /// **The scale's term is the reach, proved 2026-10-16, and was `2ρ/s_min + r` before.** A
    /// scaled section reads `g = s·D(q/s)`, so `∂g/∂s = D(x) − x·∇D(x)` at `x = q/s`; with `c` the
    /// nearest point of the outline to `x` and `n` the unit vector from it, `D = ±|x − c|` and
    /// `∇D = ±n`, so `D − x·∇D = ∓n·c`, whose size is at most `|c|`, the outline's reach. That
    /// holds wherever `∇D` exists, outside and in, far from the section and at it, for a closed
    /// outline and for a wall (whose band moves the same way and whose reach counts its
    /// thickness). The old term bounded `D(x)` and `x·∇D` each by `|x| + r` and took `|x|` up to
    /// the solid's reach over its least scale -- two places that never coincide -- and on the
    /// owner's wing, narrowing to a hundredth, divided the reading by 82.9 where this divides by
    /// 1.05 (backlog item 114): the body was drawn 82 thresholds proud of itself, the seam with
    /// its neighbour stepped, and every sharp edge wore a tube of false hits. The twist's term
    /// still takes `ρ` to the solid's reach, which is the one place a local bound (backlog item 50)
    /// is still owed here.
    pub(crate) fn stretch_along(&self, reach: f64, run: f64) -> f64 {
        if !(run > 0.0) {
            return 1.0;
        }
        let k = self.shape.slope_factor();
        let rho_max = self.reach_under(reach, run);
        let scale_rate = match self.taper {
            Taper::Scale { end } => k * (end - 1.0).abs() / run,
            _ => 0.0,
        };
        let turn_rate = k * self.twist.abs() / run;
        let lean = match self.taper {
            Taper::Draft { tangent } => tangent.abs(),
            _ => 0.0,
        };
        let along = scale_rate * reach + turn_rate * rho_max + lean;
        (1.0 + along * along).sqrt()
    }
}

/// Why a sweep could not be built.
#[derive(Clone, Debug, PartialEq)]
pub enum SweepError {
    /// A taper that makes the section nothing before the path ends: a scale at or below nought,
    /// or a draft inward by more than the section is deep.
    TaperClosesTheSection {
        /// What was asked for, as the panel shows it.
        asked: String,
    },
    /// A section held parallel never turns, so it has no axis to twist about.
    TwistOnAParallelSection,
    /// A path that closes on itself has no start for a law to run from and no end for it to
    /// reach, so its section cannot taper, turn or be cut short.
    LawOnAClosedPath,
    /// A window that starts where it ends, runs backwards, or lies off the path.
    EmptyWindow { start: f64, end: f64 },
    /// A draft pushes a section's boundary out from its inside, and an open wall has none.
    DraftOnAnOpenWall,
    /// A draft asked to mitre the corners of a **wall**, which V1 does not do.
    ///
    /// A band has two faces, and a draft moves both outward along their own normals. Mitring
    /// means mitring both, which is two displaced walks per query on each side and wants its
    /// own spec. The version that shipped briefly mitred the loop and folded an unchanged band
    /// about it, which slid the band instead of growing it — so a wall drafted the same amount
    /// was a different thickness depending on a corner toggle. Refused rather than rounded
    /// quietly, because a block that says Sharp and draws Round is the kind of wrong nothing
    /// reports.
    SharpCornersOnAWall,
    /// A spine with fewer than two distinct points has no direction to carry anything along.
    SpineHasNoLength,
    /// The profile reaches further from the spine than the spine's tightest bend is wide.
    ///
    /// Past this the sweep folds through itself and the nearest spine point stops being unique,
    /// so there is no field to state a bound for. Both numbers are reported because either one
    /// is the fix: shrink the profile, or ease the bend.
    FoldsThroughItself {
        /// The tightest curvature on the spine, per millimetre.
        curvature: f64,
        /// How far the profile reaches from its origin, in millimetres.
        reach: f64,
    },
    /// The path runs too nearly *along* the section rather than through it.
    ///
    /// A section held at an angle to the path is swept by sliding it along the path, so the path
    /// has to cross it: as the angle approaches a right angle the same profile smears out
    /// without bound and the sweep has no thickness to speak of. Refused rather than bounded,
    /// the way a fold is.
    SectionAlmostAlongThePath {
        /// The worst angle anywhere on the path, in degrees from square to the section.
        degrees: f64,
    },
}

impl std::fmt::Display for SweepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SweepError::SpineHasNoLength => {
                write!(f, "the spine has no length, so there is nothing to sweep along")
            }
            SweepError::TaperClosesTheSection { asked } => write!(
                f,
                "the taper ({asked}) closes the section to nothing before the end of the path \
                 \u{2014} shorten the path or ease the taper"
            ),
            SweepError::TwistOnAParallelSection => write!(
                f,
                "a section held parallel never turns, so it has no axis to twist about \u{2014} \
                 hold it square to the path, or flush with its start"
            ),
            SweepError::LawOnAClosedPath => write!(
                f,
                "a path that closes on itself has no start for a taper, a twist or an extent \
                 to run from"
            ),
            SweepError::EmptyWindow { start, end } => write!(
                f,
                "the extent runs from {start:.0}% to {end:.0}% of the path, which is nothing of \
                 it \u{2014} the start must come before the end, both between 0 and 100"
            ),
            SweepError::SharpCornersOnAWall => write!(
                f,
                "a draft can only mitre the corners of a closed profile, not of a wall \u{2014} \
                 a band has two faces and mitring both is not built yet; round its corners, or \
                 draft the profile the wall runs along instead"
            ),
            SweepError::DraftOnAnOpenWall => write!(
                f,
                "a draft pushes a section's walls out from its inside, and an open wall has \
                 none \u{2014} scale it instead, or close the chain"
            ),
            SweepError::SectionAlmostAlongThePath { degrees } => write!(
                f,
                "the path runs {degrees:.0}\u{00b0} off square to the section it carries, which \
                 is so nearly along it that the sweep has no thickness \u{2014} turn the profile \
                 toward the path, or hold the section square to it"
            ),
            SweepError::FoldsThroughItself { curvature, reach } => write!(
                f,
                "the profile reaches {reach:.2} mm from the spine and the spine's tightest bend \
                 has a radius of {:.2} mm, so the sweep would fold through itself — shrink the \
                 profile or ease the bend",
                1.0 / curvature
            ),
        }
    }
}

impl std::error::Error for SweepError {}

/// A rotation-minimising frame along a spine: one tangent and one normal per sample.
///
/// Built once, beside the spine, and read by segment. The binormal is not stored because it is
/// `T × N` and storing it would be a third number that could disagree with the other two.
#[derive(Clone, Debug)]
pub struct SpineFrame {
    tangents: Vec<Vector3<f64>>,
    normals: Vec<Vector3<f64>>,
}

impl SpineFrame {
    /// The frame along `spine`, seeded so the normal at the start is as close to `seed` as the
    /// tangent allows.
    ///
    /// `None` for a spine with no direction. The seed is what lets a profile start as it was
    /// drawn: its own first in-plane axis, projected out of the start tangent.
    pub(crate) fn along(spine: &Spine, seed: Vector3<f64>) -> Option<Self> {
        let points = spine.points();
        if points.len() < 2 {
            return None;
        }
        let tangents = sample_tangents(points)?;

        // The seed, made perpendicular to the first tangent. If it lay along the tangent there
        // is no preferred direction and any perpendicular will do — the frame is then carried
        // from an arbitrary but *continuous* start, which is the property that matters.
        let t0 = tangents[0];
        let mut n0 = seed - t0 * seed.dot(&t0);
        if n0.norm_squared() < 1e-12 {
            n0 = any_perpendicular(t0);
        }
        let mut normals = Vec::with_capacity(points.len());
        normals.push(n0.normalize());

        // Double reflection. Each step reflects the previous normal across the plane bisecting
        // the chord, then across the plane bisecting the tangents' difference — two reflections
        // make a rotation, and this one is the rotation that carries `t_i` to `t_{i+1}` with no
        // component about the tangent, which is what "minimising" means. Exact on a polyline.
        for i in 0..points.len() - 1 {
            let v1 = points[i + 1] - points[i];
            let c1 = v1.norm_squared();
            let r = normals[i];
            if c1 < 1e-18 {
                normals.push(r);
                continue;
            }
            let r_l = r - v1 * (2.0 / c1 * v1.dot(&r));
            let t_l = tangents[i] - v1 * (2.0 / c1 * v1.dot(&tangents[i]));
            let v2 = tangents[i + 1] - t_l;
            let c2 = v2.norm_squared();
            let next = if c2 < 1e-18 { r_l } else { r_l - v2 * (2.0 / c2 * v2.dot(&r_l)) };
            // Re-squared against the tangent it belongs with: the arithmetic above is exact in
            // exact arithmetic, and a thousand samples of rounding are not.
            let t = tangents[i + 1];
            let next = next - t * next.dot(&t);
            normals.push(next.try_normalize(1e-12).unwrap_or(r));
        }
        Some(Self { tangents, normals })
    }

    /// The tangent and normal at an arc length, interpolated within the segment it falls in.
    ///
    /// Read by the same segment `Spine::nearest` reported its arc length from, so the frame and
    /// the nearest point cannot disagree about which piece of the curve they are on.
    pub(crate) fn at(&self, spine: &Spine, arc: f64) -> (Vector3<f64>, Vector3<f64>) {
        let (i, f) = spine.segment_at(arc);
        self.within(i, f)
    }

    /// The tangent and normal a fraction `f` of the way along segment `i`: what [`Self::at`]
    /// reads, and what [`frame_tilt`] samples.
    fn within(&self, i: usize, f: f64) -> (Vector3<f64>, Vector3<f64>) {
        let j = (i + 1).min(self.tangents.len() - 1);
        let t = (self.tangents[i] * (1.0 - f) + self.tangents[j] * f)
            .try_normalize(1e-12)
            .unwrap_or(self.tangents[i]);
        let n = self.normals[i] * (1.0 - f) + self.normals[j] * f;
        let n = n - t * n.dot(&t);
        let n = n.try_normalize(1e-12).unwrap_or(self.normals[i]);
        (t, n)
    }

    /// The normal at each sample, in order — what the shader is handed.
    pub(crate) fn normals(&self) -> &[Vector3<f64>] {
        &self.normals
    }
}

/// A unit tangent at each sample: the segment's direction at the ends, and the mean of the two
/// adjacent segments' directions in between, so the frame turns smoothly through a joint rather
/// than jumping at it.
fn sample_tangents(points: &[Point3<f64>]) -> Option<Vec<Vector3<f64>>> {
    let n = points.len();
    let seg = |i: usize| (points[i + 1] - points[i]).try_normalize(1e-12);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let before = if i > 0 { seg(i - 1) } else { None };
        let after = if i + 1 < n { seg(i) } else { None };
        let t = match (before, after) {
            (Some(a), Some(b)) => (a + b).try_normalize(1e-12).or(Some(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => out.last().copied(),
        };
        out.push(t?);
    }
    Some(out)
}

fn any_perpendicular(t: Vector3<f64>) -> Vector3<f64> {
    let axis = if t.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
    t.cross(&axis)
}

/// The tightest curvature along a polyline: the turning angle at each interior sample over the
/// mean length of the two segments meeting there.
///
/// A polyline concentrates its curvature at its joints, which makes this an overestimate for a
/// coarse tessellation and an honest figure for a fine one — and the spine is tessellated to the
/// compile's own tolerance, which is fine. Over rather than under, which is the safe direction
/// for a number the bound divides by.
pub(crate) fn max_curvature(points: &[Point3<f64>]) -> f64 {
    let mut worst: f64 = 0.0;
    for i in 1..points.len().saturating_sub(1) {
        worst = worst.max(joint_curvature(points[i - 1], points[i], points[i + 1]));
    }
    worst
}

/// The curvature a polyline concentrates at `at`, between the leg from `before` and the leg to
/// `after`: the angle it turns over the mean of the two lengths. Nought where a leg has none.
fn joint_curvature(before: Point3<f64>, at: Point3<f64>, after: Point3<f64>) -> f64 {
    let (a, b) = (at - before, after - at);
    let (la, lb) = (a.norm(), b.norm());
    if la < 1e-12 || lb < 1e-12 {
        return 0.0;
    }
    let cos = (a.dot(&b) / (la * lb)).clamp(-1.0, 1.0);
    cos.acos() / (0.5 * (la + lb))
}

/// How far the frame's tangent leans off the segments it is read on, at most, how far the frame
/// twists about them, and where the first segment's own region closes -- what the reading in
/// the tube has to know about the polyline itself rather than the curve it stands for
/// (`sweep-spec.md` §4.1).
///
/// All are measured against **the frame a rigid segment would carry**: the frame pulled back
/// onto the segment's own normal plane by the least rotation that takes its tangent to the
/// segment's direction. Carried from segment to segment by the least rotation between their
/// directions, that frame meets itself across every joint's inner bisector, so the section read
/// in it is continuous where the nearest place is unique and moves from a segment to its
/// neighbour, and true.
///
/// **The lean is half the turn at a joint**, because the frame takes the bisector there, so a
/// section read across the frame's plane is shortened by up to its cosine: the solid reaches
/// `reach / cos` from the spine, and every reading in the section is off the rigid one by up to
/// `1 − cos` of the distance from the spine. Read off the frame's own tangents rather than the
/// turns, because round a closed spine the frame does not wrap: it leaves along the first
/// segment and arrives along the last, so the corner where the spine closes is leaned off whole.
///
/// **The twist is the frame's turn about a segment against the rigid one.** In a plane it is
/// only what interpolating the normal leaves: none with the normal in the plane or square to
/// it, 0.00001 radians round a ring with it held at a slant. Off a plane the frame's tangent
/// swings across the segment out of the plane of the lean, and the pulled-back frame turns with
/// it -- 0.00014 radians over the whole of the glyph's tight path walked at two degrees, 0.017
/// over a coil cut five to one. It is continuous through every joint, so what a reading can be
/// off by is how far it ranges over the spine, times the distance from the spine of the point
/// it is turned at. Sampled at sixteen places a segment, which reads that coil's range
/// within 0.06 % of a hundred and twenty-eight; the ends alone read it 7 % short. `None` where a
/// closed spine's frame does not meet itself where it closes -- carried round a loop out of a
/// plane, it comes back turned, and the section steps there -- which leaves only the divided
/// reading.
///
/// Read off `frame`, which must be the frame along `points` the reading uses: the twist is a
/// property of how the frame is interpolated as well as of the polyline. `None` for a spine
/// with no direction.
pub(crate) fn frame_tilt(points: &[Point3<f64>], frame: &SpineFrame) -> Option<FrameTilt> {
    let n = points.len();
    let directions: Vec<Option<Vector3<f64>>> = (0..n.saturating_sub(1))
        .map(|k| (points[k + 1] - points[k]).try_normalize(1e-12))
        .collect();
    let first = directions.iter().flatten().next().copied()?;
    let last = directions.iter().rev().flatten().next().copied()?;
    let tangents = &frame.tangents;
    let mut lean: f64 = 0.0;
    for (k, direction) in directions.iter().enumerate() {
        if let Some(e) = direction {
            lean = lean.max(tangents[k].angle(e)).max(tangents[k + 1].angle(e));
        }
    }
    let closed = n > 2 && (points[0] - points[n - 1]).norm_squared() < 1e-18;
    if closed {
        lean = lean.max(tangents[0].angle(&last)).max(tangents[n - 1].angle(&first));
    }
    let cos = lean.cos();
    // A lean of a right angle or more has no tube, and the least rotation is not defined past
    // a half turn.
    if !(cos > 0.0) {
        return Some(FrameTilt { cos, twist: None, closes: 0.0 });
    }
    // The rigid frame's `u`, carried along, and the range of the real frame's turn against it.
    let mut rigid: Option<(Vector3<f64>, Vector3<f64>)> = None;
    let (mut low, mut high) = (0.0f64, 0.0f64);
    for (i, direction) in directions.iter().enumerate() {
        let Some(e) = *direction else { continue };
        let u = match rigid {
            Some((before, u)) => least_turn(before, e, u),
            None => least_turn(tangents[i], e, frame.normals[i]),
        };
        let u = (u - e * u.dot(&e)).normalize();
        for k in 0..=TWIST_SAMPLES {
            let (t, normal) = frame.within(i, k as f64 / TWIST_SAMPLES as f64);
            let back = least_turn(t, e, normal);
            let turned = u.cross(&back).dot(&e).atan2(u.dot(&back));
            low = low.min(turned);
            high = high.max(turned);
        }
        rigid = Some((e, u));
    }
    let meets = |(before, u): (Vector3<f64>, Vector3<f64>)| {
        // Carried round the corner where the spine closes, against where the frame starts.
        let carried = least_turn(before, first, u);
        let start = least_turn(tangents[0], first, frame.normals[0]);
        carried.cross(&start).dot(&first).atan2(carried.dot(&start)).abs() < 1e-9
    };
    let twist = match (closed, rigid) {
        (false, _) => Some(high - low),
        (true, Some(end)) if meets(end) => Some(high - low),
        (true, _) => None,
    };
    // **Where a segment's own region closes over it.** The inner bisectors at a segment's two
    // ends meet `L / (tan(φ₁/2) + tan(φ₂/2))` out from it, and past that a point's nearest
    // place jumps across it, from the segment before to the one after. In a plane those two
    // rigid frames meet there as neighbours' do. Off a plane they do not -- carried round the
    // joints between, the rigid frame comes back turned -- and past an open end the reading is
    // the cap's, which never meets the next segment's: an end segment's region closes
    // `L / tan(φ/2)` out, its outer end square to it. So the tube stops at the first of these.
    let legs: Vec<(Vector3<f64>, f64)> = (0..n - 1)
        .filter_map(|k| {
            let leg = points[k + 1] - points[k];
            let length = leg.norm();
            (length > 1e-12).then(|| (leg / length, length))
        })
        .collect();
    let in_a_plane = lies_in_a_plane(points);
    let m = legs.len();
    let mut closes = f64::INFINITY;
    for k in 0..m {
        let before = if k > 0 { Some(k - 1) } else { closed.then_some(m - 1) };
        let after = if k + 1 < m { Some(k + 1) } else { closed.then_some(0) };
        if in_a_plane && before.is_some() && after.is_some() {
            continue;
        }
        let half_turn =
            |j: Option<usize>| j.map_or(0.0, |j| (0.5 * legs[j].0.angle(&legs[k].0)).tan());
        let closing = half_turn(before) + half_turn(after);
        if closing > 0.0 {
            closes = closes.min(legs[k].1 / closing);
        }
    }
    Some(FrameTilt { cos, twist, closes })
}

/// Whether every point of a polyline lies in the plane its most turning joint names, to a
/// billionth of its size.
fn lies_in_a_plane(points: &[Point3<f64>]) -> bool {
    let mut normal = Vector3::zeros();
    for w in points.windows(3) {
        let across = (w[1] - w[0]).cross(&(w[2] - w[1]));
        if across.norm_squared() > normal.norm_squared() {
            normal = across;
        }
    }
    let size = points.iter().fold(0.0f64, |m, p| m.max((p - points[0]).norm()));
    match normal.try_normalize(1e-12) {
        None => true,
        Some(normal) => {
            points.iter().all(|p| (p - points[0]).dot(&normal).abs() <= 1e-9 * (1.0 + size))
        }
    }
}

/// Where along a segment [`frame_tilt`] reads the frame's twist. See there.
const TWIST_SAMPLES: usize = 16;

/// `v` turned by the least rotation that takes the unit vector `from` to the unit vector `to`
/// -- about their common normal, by the angle between them. Not defined for opposite vectors,
/// which no caller here passes.
fn least_turn(from: Vector3<f64>, to: Vector3<f64>, v: Vector3<f64>) -> Vector3<f64> {
    let k = from.cross(&to);
    v + k.cross(&v) + k.cross(&k.cross(&v)) / (1.0 + from.dot(&to))
}

/// What [`frame_tilt`] finds: the cosine of the frame's greatest lean off a segment, the range
/// of its twist about them in radians (`None` where only the divided reading holds), and how
/// far out the first segment's own region closes over it, which the tube may not pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FrameTilt {
    pub cos: f64,
    pub twist: Option<f64>,
    pub closes: f64,
}

/// Where a square section's own reading is trusted: the tube about the spine, the reach of the
/// solid inside it, and how much of a reading the frame's lean and twist can account for there
/// (`sweep-spec.md` §4.1).
#[derive(Clone, Copy, Debug, PartialEq)]
struct InTube {
    /// The furthest any of the solid is from the spine: the section's reach over the cosine of
    /// the frame's lean.
    reach: f64,
    /// [`tube_radius`], or less where a segment's own region closes first ([`frame_tilt`]).
    radius: f64,
    /// `1 − cos` of the frame's lean.
    tilt: f64,
    /// The range of the frame's twist ([`frame_tilt`]), or `None` where the section may not be
    /// read as it is at all: then only the tube's own bound applies, to the divided reading.
    twist: Option<f64>,
}

impl InTube {
    /// The tube for a section reaching `reach` round the polyline `points` with `frame` along
    /// it, or `None` where the tube is no wider than the solid reaches -- which a lean of a
    /// right angle or more always is.
    fn of(points: &[Point3<f64>], frame: &SpineFrame, reach: f64) -> Option<Self> {
        let tilt = frame_tilt(points, frame)?;
        let radius = tube_radius(points).min(tilt.closes);
        (radius * tilt.cos > reach).then(|| Self {
            reach: reach / tilt.cos,
            radius,
            tilt: 1.0 - tilt.cos,
            twist: tilt.twist,
        })
    }

    /// What the frame's lean and twist can take off a reading `rho` from the spine: the lean's
    /// share of the distance here and at the surface, no further than the reach, and the twist
    /// turning a point at the surface. `None` where the section is not read as it is.
    fn allowance(&self, rho: f64) -> Option<f64> {
        self.twist.map(|twist| self.tilt * (rho + self.reach) + self.reach * twist)
    }
}

/// How far from a spine a point can be and still read the section at its own nearest place
/// truthfully: the radius of the tube a square section's reading is trusted in, and infinite
/// for a straight spine (`sweep-spec.md` §4).
///
/// Two things close the tube, and it is the nearer of them.
///
/// **The bend.** Past `1/κ` on the inside of a bend neighbouring section planes cross, with `κ`
/// read at the joints as the stretch reads it -- the corner where a closed spine meets its own
/// start among them. A polyline also turns its frame *within* a segment, faster than `κ` where a
/// short segment sits between long ones, and leans the frame off the segment as it goes. That
/// is not read here: the reading allows for the lean itself, against the reading a rigid
/// segment's frame would give, which a turn within a segment does not disturb ([`frame_tilt`],
/// `sweep-spec.md` §4.1). Read here instead, it shrank the glyph's tube from 481 to 386, inside
/// the 447 its section reaches, which would have kept the whole stretch on the scene this was
/// built for.
///
/// **A far part of the spine coming back.** Two places can only both be nearest to a point across
/// a gap that leaves the spine square at both ends, and the gap is the integral of the spine's
/// direction between them: its component along the direction at one end can only vanish if the
/// direction somewhere between points against it. So a gap can be square to the spine at both of
/// its ends only between segments whose directions turn a right angle from each end on the way
/// across -- along both ways round a closed spine -- and the tube is at most half the nearest such
/// gap. The rule admits pairs that hold no such gap (a quarter of a circle qualifies, which keeps
/// a ring's tube to 0.71 of its radius) and excludes none that do, which is the direction a
/// bound has to err in.
pub(crate) fn tube_radius(points: &[Point3<f64>]) -> f64 {
    // A repeated point has no direction, and the gaps are measured between segments that do.
    let mut pts: Vec<Point3<f64>> = Vec::with_capacity(points.len());
    for &p in points {
        if pts.last().is_none_or(|q: &Point3<f64>| (p - q).norm_squared() > 1e-18) {
            pts.push(p);
        }
    }
    let n = pts.len();
    if n < 2 {
        return 0.0;
    }
    let closed = n > 3 && (pts[0] - pts[n - 1]).norm_squared() < 1e-18;
    // The corner where a closed spine meets its own start is a joint like every other, and
    // `max_curvature` reads only those between the first point and the last.
    let mut curvature = max_curvature(&pts);
    if closed {
        curvature = curvature.max(joint_curvature(pts[n - 2], pts[0], pts[1]));
    }
    let bend = if curvature > 0.0 { 1.0 / curvature } else { f64::INFINITY };
    bend.min(0.5 * nearest_return(&pts, closed, 2.0 * bend))
}

/// The shortest gap between two segments of a spine that could be square to it at both ends,
/// looked for only below `within`; infinite when there is none. See [`tube_radius`].
fn nearest_return(points: &[Point3<f64>], closed: bool, within: f64) -> f64 {
    let segments = points.len() - 1;
    let dir: Vec<Vector3<f64>> =
        (0..segments).map(|k| (points[k + 1] - points[k]).normalize()).collect();
    // A path whose directions all lie within 45 degrees of their mean never turns a right angle
    // anywhere, so no pair qualifies -- which is every gentle path, answered without a search.
    let mean = (0..segments)
        .fold(Vector3::zeros(), |m, k| m + (points[k + 1] - points[k]))
        .try_normalize(1e-12);
    if let Some(mean) = mean
        && dir.iter().all(|d| d.dot(&mean) > std::f64::consts::FRAC_1_SQRT_2)
    {
        return f64::INFINITY;
    }
    // How many segments on, and how many back, the direction first turns a right angle from a
    // segment's own -- within one lap of a closed spine, and to the end of an open one.
    const NEVER: i64 = i64::MAX / 4;
    let n = segments as i64;
    let at = |k: i64| k.rem_euclid(n) as usize;
    // The angle between two directions is at most the turning between them, so the search for
    // a right angle may start where the turning first reaches one -- which on a planar bend is
    // where it ends, and spared a ring of two thousand segments a scan of half of them each.
    // Turning summed over three laps, so a closed spine can be read from any segment both ways.
    let laps = if closed { 3 } else { 1 };
    let mut turned = vec![0.0f64; laps * segments];
    for k in 1..laps * segments {
        let (prev, here) = (at(k as i64 - 1), at(k as i64));
        turned[k] = turned[k - 1] + dir[prev].angle(&dir[here]);
    }
    let offset = if closed { n } else { 0 };
    let turning = |from: i64, to: i64| turned[(to + offset) as usize] - turned[(from + offset) as usize];
    let first_turn = |a: usize, step: i64| -> i64 {
        let a = a as i64;
        let reach = |d: i64| if step > 0 { turning(a, a + d) } else { turning(a - d, a) };
        let (mut lo, mut hi) = (1, n - 1);
        if !closed {
            hi = if step > 0 { n - 1 - a } else { a };
        }
        if hi < 1 || reach(hi) < std::f64::consts::FRAC_PI_2 - 1e-9 {
            return NEVER;
        }
        // The first `d` the turning reaches a right angle by.
        while lo < hi {
            let mid = (lo + hi) / 2;
            if reach(mid) >= std::f64::consts::FRAC_PI_2 - 1e-9 { hi = mid } else { lo = mid + 1 }
        }
        let last = if closed { n - 1 } else if step > 0 { n - 1 - a } else { a };
        for d in lo..=last {
            if dir[at(a + step * d)].dot(&dir[a as usize]) <= 0.0 {
                return d;
            }
        }
        NEVER
    };
    let ahead: Vec<i64> = (0..segments).map(|a| first_turn(a, 1)).collect();
    let behind: Vec<i64> = (0..segments).map(|a| first_turn(a, -1)).collect();
    // **Which pairs could hold such a gap, as a range of how far apart they are.** A pair is
    // `i < j`, `d = j − i` segments on, with the foot on `i` covering its start and the foot on
    // `j` its end, so every joint is the start of one segment or the end of another. The way
    // across must turn a right angle from `i`'s direction and from `j`'s: `d` at least
    // `ahead[i]` and `behind[j]`. Round a closed spine the way back must too, `n − d` at least
    // what each needs looking the other way, and the way that arrives at a joint from behind
    // reads the direction it arrives along -- so either of the two directions at that joint is
    // let count, the generous side. So each segment admits a range of `d` as the near end of a
    // pair and another as the far end, and a pair qualifies where the two ranges meet.
    let back = |a: i64| if closed { n - a } else { NEVER };
    let near_from: Vec<i64> = ahead.clone();
    let near_to: Vec<i64> =
        (0..segments).map(|i| back(behind[i].min(behind[at(i as i64 - 1)]))).collect();
    let far_from: Vec<i64> = behind.clone();
    let far_to: Vec<i64> =
        (0..segments).map(|j| back(ahead[j].min(ahead[at(j as i64 + 1)]))).collect();
    let qualifies = |i: usize, j: usize| -> bool {
        let d = (j - i) as i64;
        d >= near_from[i].max(far_from[j]) && d <= near_to[i].min(far_to[j])
    };
    // **Runs of segments first, nearest first.** Every pair is a quadratic search, and a ring of
    // two thousand segments took 35 ms that way, asked every frame the block is shown. So the
    // segments are taken in runs, a run's box bounds every gap from it below, and a run's pair
    // is ruled out at once where no `d` between them can meet both ranges at their widest. What
    // is left is visited by its boxes' distance, nearest first, and the visit stops at the first
    // pair of runs no nearer than the nearest gap found.
    let run = segments.div_ceil(256).max(8);
    let runs = segments.div_ceil(run);
    let span = |r: usize| (r * run, ((r + 1) * run).min(segments) - 1);
    let boxes: Vec<(Point3<f64>, Point3<f64>)> = (0..runs)
        .map(|r| {
            let (first, last) = span(r);
            points[first..=last + 1].iter().fold(
                (points[first], points[first]),
                |(lo, hi), p| (lo.inf(p), hi.sup(p)),
            )
        })
        .collect();
    // Each run's ranges at their widest: the least any of its segments needs, and the most any
    // of them allows.
    let widest = |of: &[i64], low: bool| -> Vec<i64> {
        (0..runs)
            .map(|r| {
                let (first, last) = span(r);
                let values = of[first..=last].iter().copied();
                if low { values.min().unwrap_or(NEVER) } else { values.max().unwrap_or(-NEVER) }
            })
            .collect()
    };
    let (near_least, near_most) = (widest(&near_from, true), widest(&near_to, false));
    let (far_least, far_most) = (widest(&far_from, true), widest(&far_to, false));
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
    for a in 0..runs {
        let (a0, a1) = span(a);
        for b in a..runs {
            let (b0, b1) = span(b);
            let nearest_d = (b0 as i64 - a1 as i64).max(1);
            let furthest_d = b1 as i64 - a0 as i64;
            let from = near_least[a].max(far_least[b]).max(nearest_d);
            let to = near_most[a].min(far_most[b]).min(furthest_d);
            if from > to {
                continue;
            }
            let ((alo, ahi), (blo, bhi)) = (boxes[a], boxes[b]);
            let apart = (alo - bhi).sup(&(blo - ahi)).sup(&Vector3::zeros()).norm();
            if apart < within {
                candidates.push((apart, a, b));
            }
        }
    }
    candidates.sort_by(|x, y| x.0.total_cmp(&y.0));
    let middles: Vec<Point3<f64>> =
        (0..segments).map(|k| nalgebra::center(&points[k], &points[k + 1])).collect();
    let halves: Vec<f64> = (0..segments).map(|k| 0.5 * (points[k + 1] - points[k]).norm()).collect();
    let mut nearest = within;
    for (apart, a, b) in candidates {
        if apart >= nearest {
            break;
        }
        let ((a0, a1), (b0, b1)) = (span(a), span(b));
        for i in a0..=a1 {
            for j in b0.max(i + 1)..=b1 {
                // Two segments are no nearer than their middles less their half lengths, which
                // rules out most of a run's pairs before the exact gap is asked for.
                if (middles[i] - middles[j]).norm() - halves[i] - halves[j] >= nearest {
                    continue;
                }
                if qualifies(i, j) {
                    let gap = segment_gap(points[i], points[i + 1], points[j], points[j + 1]);
                    nearest = nearest.min(gap);
                }
            }
        }
    }
    if nearest < within { nearest } else { f64::INFINITY }
}

/// The shortest distance between two segments: the closest pair of points, one on each, by the
/// standard clamped solution of the two-parameter least squares.
fn segment_gap(p1: Point3<f64>, q1: Point3<f64>, p2: Point3<f64>, q2: Point3<f64>) -> f64 {
    let (d1, d2, r) = (q1 - p1, q2 - p2, p1 - p2);
    let (a, e, f) = (d1.dot(&d1), d2.dot(&d2), d2.dot(&r));
    let (s, t) = if a <= 1e-18 && e <= 1e-18 {
        (0.0, 0.0)
    } else if a <= 1e-18 {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = d1.dot(&r);
        if e <= 1e-18 {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = d1.dot(&d2);
            let denom = a * e - b * b;
            let s = if denom > 1e-18 { ((b * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };
            let t = (b * s + f) / e;
            if t < 0.0 {
                ((-c / a).clamp(0.0, 1.0), 0.0)
            } else if t > 1.0 {
                (((b - c) / a).clamp(0.0, 1.0), 1.0)
            } else {
                (s, t)
            }
        }
    };
    ((p1 + d1 * s) - (p2 + d2 * t)).norm()
}

/// A section carried along a spine, held square to it: a profile, or a wall.
#[derive(Clone, Debug)]
pub struct Swept {
    pub footprint: crate::Footprint,
    pub spine: Spine,
    pub frame: SpineFrame,
    /// Whether the spine closes on itself, in which case there are no end caps.
    closed: bool,
    /// `1 / (1 − κ_max · r_max)`: what the reading is divided by to stay a lower bound -- all of
    /// it, or only the caps read from inside where `tube` says the section reads truly.
    stretch: f64,
    /// The tube about the spine, where the section is held square under the plain law and the
    /// tube is wider than the solid reaches: there the reading is the section's own, less what
    /// the frame's lean and twist can account for, and the stretch divides only the caps
    /// (`sweep-spec.md` §4). `None` everywhere else, which divides the whole reading as every
    /// sweep did before.
    tube: Option<InTube>,
    /// Whether the section's axes are coefficients in the frame (so they are carried along the
    /// path) or world vectors (so they never turn).
    follow: bool,
    /// The section's `u` axis: in the frame's basis when `follow`, in the world otherwise.
    ua: Vector3<f64>,
    /// The same for `v`.
    vb: Vector3<f64>,
    /// How the section changes along the path. Plain for every sweep that has not asked.
    law: SectionLaw,
    /// The stretch of path the solid occupies, in millimetres of arc: the window applied to
    /// the spine's length.
    window: (f64, f64),
    /// How the rims are cut where the side meets each end of an open path. Hard unless
    /// [`Swept::with_rims`] says otherwise.
    rims: crate::graph_field::Rim,
}

impl Swept {
    /// The same sweep with its rims cut as told -- each end of an open path, where the section
    /// meets its cap. A closed path has no ends, and the compile refuses a rim on one by name.
    pub fn with_rims(mut self, rims: crate::graph_field::Rim) -> Self {
        self.rims = rims;
        self
    }

    /// How long the run of path the solid occupies is, in millimetres of arc: what half of is
    /// the most a rim may take from each end.
    pub fn run(&self) -> f64 {
        self.window.1 - self.window.0
    }

    /// A sweep holding its section square, or the reason there cannot be one.
    ///
    /// The graph builds every sweep through `under_law`, with the hold and the law its block
    /// says, so only the tests here build one square by default, and only the tests compile
    /// this (decision 1.4).
    #[cfg(test)]
    fn new(
        footprint: impl Into<crate::Footprint>,
        spine: Spine,
        seed: Vector3<f64>,
    ) -> Result<Self, SweepError> {
        Self::held(footprint, spine, seed, Held::Square)
    }

    /// A sweep holding the section as `held` says, or the reason there cannot be one. See
    /// [`Held`].
    ///
    /// `seed` is the direction the profile's own `+u` axis should point at the start of the
    /// spine, as near as the start tangent allows.
    pub fn held(
        footprint: impl Into<crate::Footprint>,
        spine: Spine,
        seed: Vector3<f64>,
        held: Held,
    ) -> Result<Self, SweepError> {
        Self::under_law(footprint, spine, seed, held, SectionLaw::default())
    }

    /// The same, with the section changing along the path as `law` says. See [`SectionLaw`].
    pub(crate) fn under_law(
        footprint: impl Into<crate::Footprint>,
        spine: Spine,
        seed: Vector3<f64>,
        held: Held,
        law: SectionLaw,
    ) -> Result<Self, SweepError> {
        let footprint = footprint.into();
        let frame = SpineFrame::along(&spine, seed).ok_or(SweepError::SpineHasNoLength)?;
        if spine.length() <= 0.0 {
            return Err(SweepError::SpineHasNoLength);
        }
        let points = spine.points();
        let closed = points.len() > 2
            && (points[0] - points[points.len() - 1]).norm_squared() < 1e-18;
        if closed && !law.is_plain() {
            return Err(SweepError::LawOnAClosedPath);
        }
        // **The law's refusals, by name, before any number is derived from it.**
        let (p0, p1) = law.window;
        if !(p0.is_finite() && p1.is_finite()) || p0 < 0.0 || p1 > 1.0 || p0 >= p1 {
            return Err(SweepError::EmptyWindow { start: p0 * 100.0, end: p1 * 100.0 });
        }
        let length = spine.length();
        let window = (p0 * length, p1 * length);
        let run = window.1 - window.0;
        if law.twist != 0.0 && matches!(held, Held::Parallel { .. }) {
            return Err(SweepError::TwistOnAParallelSection);
        }
        // The taper's own refusals, before any number is derived from it.
        match law.taper {
            Taper::None => {}
            Taper::Scale { end } => {
                if !(end > 0.0) || !end.is_finite() {
                    return Err(SweepError::TaperClosesTheSection {
                        asked: format!("scale {end}"),
                    });
                }
            }
            Taper::Draft { tangent } => {
                if !tangent.is_finite() {
                    return Err(SweepError::TaperClosesTheSection {
                        asked: format!("draft {tangent}"),
                    });
                }
                if footprint.is_open_wall() && tangent != 0.0 {
                    return Err(SweepError::DraftOnAnOpenWall);
                }
                // And a wall of any kind cannot be mitred yet; see the variant.
                if footprint.is_wall()
                    && tangent != 0.0
                    && !law.corners.is_default()
                {
                    return Err(SweepError::SharpCornersOnAWall);
                }
                // Inward by more than the section is deep is a section that has closed.
                if tangent < 0.0 && run * -tangent >= footprint.depth() {
                    return Err(SweepError::TaperClosesTheSection {
                        asked: format!("draft {:.1}\u{00b0} inward", tangent.abs().atan().to_degrees()),
                    });
                }
            }
        }
        // How far the section reaches from the point that carries it, at its largest under
        // the law, which is what the bend is measured against.
        let reach = footprint.reach_from_origin();
        let rho_max = law.reach_under(reach, run);
        let curvature = max_curvature(points);
        let product = curvature * rho_max;
        if product >= 1.0 {
            return Err(SweepError::FoldsThroughItself { curvature, reach: rho_max });
        }
        // **The section's axes, worked out once.** A square section is the frame's own normal
        // and binormal; a flush one is the profile's own pair written in the frame *at the
        // start*, which the frame then carries so the tilt stays the same relative to the path;
        // a parallel one is the profile's own pair in the world, which never turns.
        let (follow, ua, vb) = match held {
            Held::Square => (true, Vector3::y(), Vector3::z()),
            Held::Flush { u, v, .. } => {
                let (t0, n0) = (frame.tangents[0], frame.normals[0]);
                let b0 = t0.cross(&n0);
                let into = Matrix3::from_columns(&[t0, n0, b0]).transpose();
                (true, into * u, into * v)
            }
            Held::Parallel { u, v, .. } => (false, u, v),
        };
        // A held section is swept by sliding it along the path, so the path has to cross it.
        // Checked here rather than at every sample, because it is a property of the two shapes.
        if let Some(worst) = worst_obliquity(follow, ua, vb, &frame, points)
            && worst < COS_FLOOR
        {
            return Err(SweepError::SectionAlmostAlongThePath {
                degrees: worst.clamp(-1.0, 1.0).acos().to_degrees(),
            });
        }
        // **Where the section's own reading can be trusted.** Only a square section under the
        // plain law reads nothing of the arc in its section; a held one is slid along the path
        // and a law moves the section with the arc. And only where the tube is wider than the
        // solid reaches, since the argument needs every point of the solid inside it.
        // `sweep_tube` in `graph_field` asks the same of the graph for the shader.
        let tube = (matches!(held, Held::Square) && law.is_plain())
            .then(|| InTube::of(points, &frame, rho_max))
            .flatten();
        Ok(Self {
            footprint,
            spine,
            frame,
            closed,
            // `√(1 + K²) / (1 − κ·ρ)`: the bend's own stretch, as v1 had it, times what the
            // law adds along the path. Exactly v1's number under the plain law, which a test
            // asserts. `sweep_stretch` in `graph_field.rs` computes the same from the graph.
            stretch: law.stretch_along(reach, run) / (1.0 - product),
            tube,
            follow,
            ua,
            vb,
            law,
            window,
            rims: crate::graph_field::Rim::default(),
        })
    }

    /// What the reading is divided by.
    ///
    /// The shader is handed the same number by `sweep_stretch`, which asks the graph rather than
    /// a built sweep, so only the tests here read it off one, and only the tests compile this
    /// (decision 1.4).
    #[cfg(test)]
    fn stretch(&self) -> f64 {
        self.stretch
    }

    /// How the section changes along the path. Test-only for the same reason as `stretch`: the
    /// shader's law comes from `sweep_law`.
    #[cfg(test)]
    fn law(&self) -> &SectionLaw {
        &self.law
    }

    /// Whether the path closes on itself. Only a test asks a built sweep, so only the tests
    /// compile this (decision 1.4).
    #[cfg(test)]
    fn is_closed(&self) -> bool {
        self.closed
    }

    /// The section's reading at `q`, at arc `w` along the path: the outline scaled and turned
    /// as the law says it is there -- `sweep-v2-spec.md` §2's one mechanism.
    ///
    /// `q = R(−θ)·x / s` and `d₂ = s·outline(q)`: scaling a distance's argument and its value
    /// by the same number is the exact distance to the scaled outline, and the rotation is an
    /// isometry, so in its own plane the section is still exactly what it was. A draft is the
    /// outline's contour `(w − a₀)·tan α` out, which is the extrude's own draft and so the same
    /// solid along a straight path. Past either end of the window the law holds its end value,
    /// so each cap is a flat section of the state the law reached there.
    ///
    /// Returned with whether the walls stand square to the caps, which decides how the reading
    /// is combined with the slab: Pythagoras is exact between two perpendicular measurements
    /// and overstates between a leaning wall and its cap, so a drafted section takes the larger
    /// of the two, **exactly as the extrude does** -- and a drafted sweep along a straight path
    /// is then the drafted extrude pointwise, which is what "one construction" has to mean. A
    /// draft's offset is therefore also left unclamped past the ends, as the extrude's is: the
    /// contour keeps growing and the slab cuts it. A scaled or turned section holds its end
    /// state past the caps instead, so the cap is a flat section of what the law reached, and
    /// the corner to its rim is the exact distance.
    fn section_at(&self, x: Point2<f64>, w: f64) -> (f64, bool) {
        if self.law.is_plain() {
            return (self.footprint.distance(x), true);
        }
        let (a0, a1) = self.window;
        let f = if a1 > a0 { ((w - a0) / (a1 - a0)).clamp(0.0, 1.0) } else { 0.0 };
        let theta = self.law.turn_at(f);
        let turned = if theta != 0.0 {
            let (sin, cos) = (-theta).sin_cos();
            Point2::new(cos * x.x - sin * x.y, sin * x.x + cos * x.y)
        } else {
            x
        };
        match self.law.taper {
            Taper::None => (self.footprint.distance_tapered(turned, 1.0, 0.0), true),
            Taper::Scale { .. } => {
                (self.footprint.distance_tapered(turned, self.law.scale_at(f), 0.0), true)
            }
            Taper::Draft { tangent } => (
                // Through the section's own offset, so the corners are cut the way the block
                // says -- the extrude's arithmetic, which is what makes the two one
                // construction.
                self.footprint.offset_by(turned, (w - a0) * tangent, self.law.corners),
                tangent == 0.0,
            ),
        }
    }

    /// The section's two axes at an arc length, in the world.
    ///
    /// **Held sections differ only in what these two vectors are.** A square section's are the
    /// frame's own normal and binormal; a flush one's are the same two turned by the tilt the
    /// profile had at the start, which the frame then carries; a parallel one's are the
    /// profile's own, the same everywhere. Written as coefficients in the frame (`follow`) or as
    /// world vectors, so one evaluation serves all three.
    fn axes(&self, arc: f64) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let (t, n) = self.frame.at(&self.spine, arc);
        if !self.follow {
            return (t, self.ua, self.vb);
        }
        let b = t.cross(&n);
        let f = Matrix3::from_columns(&[t, n, b]);
        (t, f * self.ua, f * self.vb)
    }
}

impl ScalarField for Swept {
    fn distance(&self, p: Point3<f64>) -> f64 {
        // Refused at construction for an empty spine, so this cannot fail here.
        let Some(near) = self.spine.nearest(p) else {
            return f64::INFINITY;
        };
        let (tangent, u, v) = self.axes(near.arc);
        // The section's own plane normal. For a square section this is the tangent and every
        // line below reduces to what this function was before held sections existed.
        let m = u.cross(&v);
        let d = p - near.point;
        // **Slid onto the section, not projected across it.** The section a point belongs to is
        // the one that *contains* it, so the point travels along the path's own direction until
        // it reaches that plane — the leaning extrude's arithmetic (`SkewExtrusion`), with the
        // spine's frame in place of the plane's. Projecting instead would measure a point
        // against a section it is not in, which is what re-centres a tilted sweep.
        let square = tangent.dot(&m);
        if square.abs() <= 1e-9 {
            // Refused at construction, so this is the belt to those braces: a path running
            // along its own section has no sweep to measure.
            return f64::INFINITY;
        }
        let slide = d.dot(&m) / square;
        let landed = d - tangent * slide;
        let w = near.arc + slide;
        let (section, upright) = self.section_at(Point2::new(landed.dot(&u), landed.dot(&v)), w);
        // **And the reading is shortened by how far the section leans.** A wall `w` from the
        // middle of a leaning prism is `w·cos θ` away square to its face, so a reading taken in
        // the section overstates by `1/cos θ` — and overstating is the one thing the contract
        // forbids. `square` is that cosine, and it is 1 for a section held square to the path.
        let lean = square.abs();
        // The slab is the window: where the solid runs along the path, which is all of it
        // unless an extent says otherwise. A closed path has no ends to cap.
        let slab = (!self.closed).then(|| {
            let (a0, a1) = self.window;
            (w - (a0 + a1) * 0.5).abs() - (a1 - a0) * 0.5
        });
        // The corner is Pythagoras between two *perpendicular* measurements. A leaning section
        // and its own end cap are not perpendicular to each other, and taking the corner anyway
        // overstated a 45-degree lean by 8 mm where `SkewExtrusion` first met it — so it is
        // taken only where the two are square, exactly as `extrude_distance` guards it, and
        // not for a drafted section's walls either, as the extrude does not (`section_at`).
        let hard = |slab: Option<f64>| match slab {
            None => section * lean,
            Some(slab) if section > 0.0 && slab > 0.0 && lean >= 1.0 - 1e-9 && upright => {
                section.hypot(slab)
            }
            Some(slab) => (section * lean).max(slab),
        };
        // **A rim is the blend's bound across the two**, which holds at any angle a lean or a
        // taper puts between them, so it takes the divisor the sweep already does and no other
        // (`Rim::join`).
        // Which end is nearer is how far past the window's middle the point landed.
        let toward = w - (self.window.0 + self.window.1) * 0.5;
        let divided = match slab {
            Some(slab) if !self.rims.is_hard() => self.rims.join_ends(section * lean, slab, toward),
            _ => hard(slab),
        } / self.stretch;
        let Some(tube) = self.tube else {
            return divided;
        };
        // **The section read as it is, and the caps divided where they are read from
        // inside** (`sweep-spec.md` §4). The frame turns about no axis of its own, so the
        // section's coordinates change at the rate the point moves. Only the arc is stretched,
        // and only the slab reads it: past an end the nearest place is the end itself and the
        // map is rigid, so a positive slab is exact too, and a negative one is read inside the
        // section, within the reach, where the stretch bounds it.
        //
        // **And it is the hard corner's even when the rims are rounded.** Dividing only the
        // negative slab leaves a corner's zero set where it was, since a `max` reads only which
        // side is larger, but it moves a rim's: the arc lies where both readings are negative,
        // so this one would put the rim a stretch further along the path than the divided
        // reading does, and the `max` below would draw two surfaces at once -- 0.4 mm off the
        // shader on the bent sweep. A rim only takes material away, so the rounded solid lies
        // inside the hard one, and outside it the hard reading less its allowance is still no
        // more than there is. Inside, the hard solid's depth says nothing of the rounded one's,
        // so a rimmed sweep reads the divided reading there.
        let own = hard(slab.map(|slab| if slab > 0.0 { slab } else { slab / self.stretch }));
        // **Less what the frame's lean and twist can account for** (`sweep-spec.md` §4.1). A
        // polyline's frame leans off each segment toward the bisectors at its ends, so the
        // section is read across a plane tilted by up to half a joint's turn, which shortens a
        // reading `ρ` from the spine by up to `(1 − cos)·ρ` against the one a rigid segment's
        // frame would give ([`frame_tilt`]). That one is continuous and true where the nearest
        // place is unique; this one bulges at every joint and steps across a joint's inner
        // bisector wherever the legs either side turn their frames at different rates -- 2.3 mm
        // past the surface beside a 2 mm leg between 44° joints, read as it was. The two differ
        // by that much here, by as much again at the surface, no further than the reach from
        // the spine, and by the frame's twist turning a point there, so the reading less all
        // three claims no more than there is, outside or in. Where that leaves less than
        // dividing does -- within about that much of the surface -- the divided reading stands,
        // and where the frame does not meet itself round a closed spine it stands throughout.
        let rho = near.distance;
        let reading = match (tube.allowance(rho), own > 0.0) {
            (None, _) => divided,
            (Some(allowance), true) => divided.max(own - allowance),
            (Some(_), false) if !self.rims.is_hard() => divided,
            (Some(allowance), false) => divided.min(own + allowance),
        };
        // **And no more than the tube allows.** Within it every point has one nearest place, so
        // a reading that stays inside it on the way to the surface is held as above; one whose
        // way would leave it is answered by what is known without it: the surface is at least
        // `tube − ρ` away, or the reading would have been held, and at least `ρ − reach`, since
        // nothing of the solid is further than its reach from the spine. A coil read 8.4 mm long
        // between its turns before this.
        reading.min((rho - tube.reach).max(tube.radius - rho))
    }

    fn normalization(&self) -> Normalization {
        // Always: a straight spine would be exact, and a straight spine is what an extrude is
        // for. Saying `LowerBound` here is what marks a blend against a sweep approximate.
        Normalization::LowerBound
    }

    fn lipschitz(&self) -> f64 {
        1.0
    }
}

/// How square the path stays to the section it carries: the smallest `|tangent · m|` anywhere,
/// where `m` is the section's own plane normal.
///
/// `None` for a section that is square by construction. Below [`COS_FLOOR`] the sweep is refused
/// — see [`SweepError::SectionAlmostAlongThePath`].
fn worst_obliquity(
    follow: bool,
    ua: Vector3<f64>,
    vb: Vector3<f64>,
    frame: &SpineFrame,
    points: &[Point3<f64>],
) -> Option<f64> {
    // A section written in the frame is square to the path wherever its `u` and `v` are the
    // frame's own two, which is the one case with nothing to check.
    if follow && (ua - Vector3::y()).norm() < 1e-12 && (vb - Vector3::z()).norm() < 1e-12 {
        return None;
    }
    let mut worst = f64::INFINITY;
    for i in 0..points.len() {
        let t = frame.tangents[i];
        let n = frame.normals[i];
        let (u, v) = if follow {
            let f = Matrix3::from_columns(&[t, n, t.cross(&n)]);
            (f * ua, f * vb)
        } else {
            (ua, vb)
        };
        worst = worst.min(t.dot(&u.cross(&v)).abs());
    }
    worst.is_finite().then_some(worst)
}

/// How square the path must stay to a held section: a tenth, which is about 84 degrees off.
///
/// Past this the same profile is smeared over ten times its own width and the sweep is thinner
/// than it reads anywhere, which is a shape nobody asked for rather than a shape to bound.
const COS_FLOOR: f64 = 0.1;

#[cfg(test)]
mod held_tests {
    use super::*;
    use crate::Profile;

    /// A 20 mm square in the XY plane about the origin, as a profile.
    fn square() -> Profile {
        Profile::new(vec![
            Point2::new(-10.0, -10.0),
            Point2::new(10.0, -10.0),
            Point2::new(10.0, 10.0),
            Point2::new(-10.0, 10.0),
        ])
        .expect("a closed square")
    }

    /// The profile's own basis: the XY plane, so its normal is +Z.
    fn flat() -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        (Vector3::z(), Vector3::x(), Vector3::y())
    }

    /// **A parallel sweep keeps the section in the profile's own plane the whole way.** Swept
    /// along a path that leans, the solid leans with it — but every cut across it parallel to
    /// the profile is the profile, which a square section cannot say: that one turns to stay
    /// perpendicular to the path, so its cuts are ellipses to the profile's plane.
    #[test]
    fn a_parallel_section_stays_parallel_and_a_square_one_turns() {
        let (normal, u, v) = flat();
        // A path leaning 45 degrees in XZ, 50 mm of rise.
        let spine = Spine::new(vec![Point3::origin(), Point3::new(50.0, 0.0, 50.0)]);
        let parallel = Swept::held(square(), spine.clone(), u, Held::Parallel { normal, u, v })
            .expect("a leaning path still crosses the profile's plane");

        // Half way along, the section is the profile itself, centred on the path.
        let middle = Point3::new(25.0, 0.0, 25.0);
        assert!(parallel.distance(middle) < 0.0, "the middle of the sweep is solid");
        // Nine millimetres across the section, in the profile's own plane: still inside.
        assert!(
            parallel.distance(middle + Vector3::new(9.0, 0.0, 0.0)) < 0.0,
            "a parallel section is the profile in its own plane"
        );
        // And eleven is outside it, so the section really is 20 mm across in that plane.
        assert!(
            parallel.distance(middle + Vector3::new(11.0, 0.0, 0.0)) > 0.0,
            "and no wider than the profile"
        );

        // The square section turns with the path: it is perpendicular to a 45-degree path, so
        // the same in-plane offset of 9 mm along X is only 6.4 mm across the section and still
        // inside, while 20 mm along X leaves it.
        let held_square = Swept::new(square(), spine, u).expect("a straight path sweeps");
        assert!(held_square.distance(middle) < 0.0);
        assert!(
            held_square.distance(middle + Vector3::new(20.0, 0.0, 0.0)) > 0.0,
            "a square section is 20 mm across square to the path, which is less along X"
        );
    }

    /// **A flush section starts in the profile's plane and keeps that tilt.** With a path that
    /// meets the profile at 45 degrees, the start face lies flat in the profile's plane — so a
    /// point just inside the profile and a hair above its plane is solid, where a square section
    /// would have cut that corner off.
    #[test]
    fn a_flush_section_starts_in_the_profiles_plane() {
        let (normal, u, v) = flat();
        let spine = Spine::new(vec![Point3::origin(), Point3::new(50.0, 0.0, 50.0)]);
        let flush = Swept::held(square(), spine.clone(), u, Held::Flush { normal, u, v })
            .expect("a leaning path still crosses the profile's plane");

        // Just inside the profile's near edge, a hair above its plane. A flush start keeps it:
        // the start face IS the profile's plane, so anything above that plane and inside the
        // section is solid. A square start cuts it away, because its own start face is square
        // to the path and this corner lies behind it.
        let corner = Point3::new(-9.0, 0.0, 0.5);
        assert!(
            flush.distance(corner) < 0.0,
            "the start face is the profile's own plane, so this corner is solid: {}",
            flush.distance(corner)
        );
        // The square section cuts that same corner away, because its start face is square to
        // the path and this point is behind it.
        let square_held = Swept::new(square(), spine, u).expect("sweeps");
        assert!(
            square_held.distance(corner) > 0.0,
            "a square start cuts the corner the flush one keeps: {}",
            square_held.distance(corner)
        );
    }

    /// **A path that runs nearly along its own section is refused, not bounded.** The same
    /// profile is then smeared over many times its own width, and no divisor makes that the
    /// shape anybody asked for.
    #[test]
    fn a_path_that_runs_along_its_section_is_refused() {
        let (normal, u, v) = flat();
        // Almost entirely in the profile's own plane: 3 degrees out of it.
        let spine = Spine::new(vec![
            Point3::origin(),
            Point3::new(100.0, 0.0, 100.0 * (3.0f64).to_radians().tan()),
        ]);
        let made = Swept::held(square(), spine, u, Held::Parallel { normal, u, v });
        assert!(
            matches!(made, Err(SweepError::SectionAlmostAlongThePath { .. })),
            "3 degrees off the profile's own plane should be refused"
        );
    }

    /// And every reading stays a lower bound: never further from nought than the truth. Checked
    /// against the shape a parallel sweep along a straight path actually is — a sheared box,
    /// whose exact distance is known outside it.
    #[test]
    fn a_held_section_never_reads_further_than_the_truth() {
        let (normal, u, v) = flat();
        let spine = Spine::new(vec![Point3::origin(), Point3::new(0.0, 0.0, 60.0)]);
        // Straight up: parallel and square agree here, and both are exact against a box.
        let parallel = Swept::held(square(), spine, u, Held::Parallel { normal, u, v })
            .expect("a vertical path is square to the profile");
        for p in [
            Point3::new(0.0, 0.0, 30.0),
            Point3::new(15.0, 0.0, 30.0),
            Point3::new(0.0, 0.0, 80.0),
            Point3::new(30.0, 40.0, 30.0),
        ] {
            let read = parallel.distance(p);
            // The true distance to a 20x20x60 box standing on the origin.
            let q = Vector3::new(p.x.abs() - 10.0, p.y.abs() - 10.0, (p.z - 30.0).abs() - 30.0);
            let truth = q.map(|c| c.max(0.0)).norm() + q.max().min(0.0);
            assert!(
                read.abs() <= truth.abs() + 1e-9,
                "at {p:?} the sweep read {read} against a true {truth}"
            );
            assert_eq!(
                read > 0.0,
                truth > 0.0,
                "at {p:?} the sweep and the box disagree about inside"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Profile;
    use crate::Extrusion;
    use std::f64::consts::{PI, TAU};

    fn square(half: f64) -> Profile {
        Profile::new(vec![
            Point2::new(-half, -half),
            Point2::new(half, -half),
            Point2::new(half, half),
            Point2::new(-half, half),
        ])
        .unwrap()
    }

    fn circle(r: f64, n: usize) -> Profile {
        Profile::new(
            (0..n)
                .map(|i| {
                    let a = TAU * i as f64 / n as f64;
                    Point2::new(r * a.cos(), r * a.sin())
                })
                .collect(),
        )
        .unwrap()
    }

    /// A planar S: a quarter turn up through vertical, then the mirror of it — the tangent
    /// passes through `+Z` at the first arc's middle and the curve inflects between the two.
    pub(super) fn s_bend(radius: f64, samples: usize) -> Vec<Point3<f64>> {
        let mut out = Vec::new();
        for i in 0..=samples {
            let phi = -PI / 2.0 + PI * i as f64 / samples as f64;
            out.push(Point3::new(radius * phi.cos(), 0.0, radius + radius * phi.sin()));
        }
        for i in 1..=samples {
            let phi = PI / 2.0 - PI * i as f64 / samples as f64;
            out.push(Point3::new(-2.0 * radius + radius * -phi.cos(), 0.0, 3.0 * radius + radius * phi.sin()));
        }
        out
    }

    /// A straight spine adds nothing: the sweep is the extrude of the same profile.
    ///
    /// The frame contributes nothing on a straight line and the stretch is one, so this
    /// isolates the placement — `(u, v, w)` against `(x, y, z)` — and the caps.
    #[test]
    fn a_straight_spine_reproduces_an_extrude() {
        let profile = square(6.0);
        let spine = Spine::new(vec![Point3::origin(), Point3::new(0.0, 0.0, 50.0)]);
        let swept = Swept::new(profile.clone(), spine, Vector3::x()).unwrap();
        let extrude = Extrusion { rims: Default::default(), footprint: profile.into(), start: 0.0, height: 50.0, depth: crate::Depth::One, draft_tangent: 0.0, corners: Default::default() };
        assert_eq!(swept.stretch(), 1.0);
        for x in -3..=3 {
            for y in -3..=3 {
                for z in -2..=12 {
                    let p = Point3::new(x as f64 * 4.0, y as f64 * 4.0, z as f64 * 5.0);
                    let (a, b) = (swept.distance(p), extrude.distance(p));
                    assert!((a - b).abs() < 1e-9, "at {p:?}: sweep {a}, extrude {b}");
                }
            }
        }
    }

    /// A circle carried round a ring is a torus, and a torus is analytic.
    ///
    /// **The one test that checks the frame, the mapping and the bound against something that
    /// is not this code.** Every section of a torus is congruent and lies in the half-plane
    /// through the axis, which is exactly the section plane the nearest ring point gives — so
    /// the sweep's reading should be the torus's own distance, *undivided*: the bend stretches
    /// only the arc, and a closed ring has no caps to read it (`sweep-spec.md` §4). The stretch
    /// `1 / (1 − r/R)` is still the leaf's number, for what reads the arc. The polygonal
    /// approximations to both circles are what the tolerance is for.
    #[test]
    fn a_circle_along_a_ring_is_a_torus() {
        let (big, small) = (30.0, 5.0);
        let ring: Vec<Point3<f64>> = (0..=360)
            .map(|i| {
                let a = TAU * (i % 360) as f64 / 360.0;
                Point3::new(big * a.cos(), big * a.sin(), 0.0)
            })
            .collect();
        let swept = Swept::new(circle(small, 96), Spine::new(ring), Vector3::x()).unwrap();
        assert!(swept.is_closed());
        let expected_stretch = 1.0 / (1.0 - small / big);
        assert!(
            (swept.stretch() - expected_stretch).abs() < 0.02,
            "stretch {} for r/R = {}",
            swept.stretch(),
            small / big
        );
        let torus = crate::Torus { centre: Point3::origin(), major_radius: big, minor_radius: small };
        let mut worst = 0.0f64;
        for x in -8..=8 {
            for y in -8..=8 {
                for z in -3..=3 {
                    let p = Point3::new(x as f64 * 5.0, y as f64 * 5.0, z as f64 * 3.0);
                    let got = swept.distance(p);
                    let want = torus.distance(p);
                    worst = worst.max((got - want).abs());
                    // Never overstated in *magnitude*: a lower bound reads no further from the
                    // surface than the truth on either side of it. The first draft of this
                    // line asserted `sweep <= torus`, which is the right direction outside and
                    // the wrong one inside, and it failed on an interior point reading -0.297
                    // against -0.359 -- which was the stretch the reading was divided by then.
                    assert!(got.abs() <= want.abs() + 0.05, "at {p:?}: {got} overstates {want}");
                    if want.abs() > 0.1 {
                        assert!(got.signum() == want.signum(), "at {p:?}: {got} vs {want} disagree on side");
                    }
                }
            }
        }
        assert!(worst < 0.05, "the sweep and the torus disagree by {worst} mm somewhere");
    }

    /// The frame is continuous through vertical and through an inflection.
    ///
    /// A fixed up-vector frame fails the first — the section spins a full turn where the tangent
    /// passes through `+Z` — and Frenet fails the second. The rotation-minimising frame does
    /// neither, and this is the property the design was chosen for.
    #[test]
    fn the_frame_does_not_flip_through_vertical_or_an_inflection() {
        let spine = Spine::new(s_bend(20.0, 60));
        let frame = SpineFrame::along(&spine, Vector3::y()).unwrap();
        let normals = frame.normals();
        for w in normals.windows(2) {
            assert!(
                w[0].dot(&w[1]) > 0.98,
                "the normal turned {:.1}° between two samples",
                w[0].dot(&w[1]).clamp(-1.0, 1.0).acos().to_degrees()
            );
        }
        // And it stayed perpendicular to the tangent all the way along.
        for (i, n) in normals.iter().enumerate() {
            let (t, _) = frame.at(&spine, spine.points()[i].coords.norm() * 0.0 + {
                let pts = spine.points();
                let mut arc = 0.0;
                for k in 1..=i {
                    arc += (pts[k] - pts[k - 1]).norm();
                }
                arc
            });
            assert!(n.dot(&t).abs() < 1e-6, "sample {i}: normal is {:.3} off square", n.dot(&t));
        }
    }

    /// The field never claims more room than there is, along a spine that bends both ways.
    #[test]
    fn a_sweep_never_overstates_the_room_before_its_surface() {
        let swept = Swept::new(square(4.0), Spine::new(s_bend(20.0, 40)), Vector3::y()).unwrap();
        let dirs = [Vector3::x(), Vector3::y(), Vector3::z(), -Vector3::x(), -Vector3::z()];
        for x in -6..=6 {
            for y in -3..=3 {
                for z in -2..=14 {
                    let from = Point3::new(x as f64 * 8.0, y as f64 * 4.0, z as f64 * 6.0);
                    let claimed = swept.distance(from);
                    if claimed <= 0.0 {
                        continue;
                    }
                    for d in dirs {
                        for step in 1..=16 {
                            let at = from + d * (claimed * step as f64 / 16.0);
                            assert!(
                                swept.distance(at) > -1e-9,
                                "from {from:?} along {d:?}: crossed the surface inside a claimed {claimed}",
                            );
                        }
                    }
                }
            }
        }
    }

    /// **Beside its walls a square sweep reads a true distance.** The bend stretches only the
    /// arc: a rotation-minimising frame turns about no axis of its own, so the section's two
    /// coordinates change at exactly the rate a point moves, and only the caps read the arc
    /// (`sweep-spec.md` §4). So beside the walls the reading has the unit gradient every exact
    /// source is held to, where dividing everything by the stretch made it 0.72 here.
    #[test]
    fn beside_its_walls_a_square_sweep_reads_a_true_distance() {
        // A quarter turn of radius 40 carrying a 16 mm square: a stretch of 1/(1 − 8√2/40).
        let arc: Vec<Point3<f64>> = (0..=90)
            .map(|i| {
                let a = 0.5 * PI * i as f64 / 90.0;
                Point3::new(40.0 * a.cos(), 40.0 * a.sin(), 0.0)
            })
            .collect();
        let swept = Swept::new(square(8.0), Spine::new(arc), Vector3::z()).unwrap();
        // Out from the middle of the turn along the section's two axes, both ways, inside the
        // walls and out past them -- the inside of the bend included, which is where the
        // section planes converge.
        let mid = PI / 4.0;
        let centre = Point3::new(40.0 * mid.cos(), 40.0 * mid.sin(), 0.0);
        let radial = Vector3::new(mid.cos(), mid.sin(), 0.0);
        let mut points = Vec::new();
        for d in [-6.0, -3.0, -1.0, 1.0, 3.0, 6.0, 12.0] {
            for axis in [radial, -radial, Vector3::z(), -Vector3::z()] {
                points.push(centre + axis * (8.0 + d));
            }
        }
        let deviation = crate::gradient_deviation(&swept, &points, 1e-3);
        assert!(deviation < 1e-3, "the reading's gradient is {deviation} off a unit one");
    }

    /// **Between the turns of a coil the field claims no more room than there is.** A point
    /// midway between two turns is as near the one's spine as the other's, and the section at
    /// its own nearest place can be out on an arm turned away from it while the other turn's
    /// is turned toward it. Dividing by the bend's stretch did not cover that: a 16 mm square
    /// on a 20 mm arm round a coil of radius 60 read past its surface by 8.4 mm. What stops it
    /// is the tube the section's own reading is trusted in (`sweep-spec.md` §4).
    #[test]
    fn a_sweep_never_overstates_the_room_between_the_turns_of_a_coil() {
        let coil: Vec<Point3<f64>> = (0..=120)
            .map(|i| {
                let a = 2.0 * TAU * i as f64 / 120.0;
                Point3::new(60.0 * a.cos(), 60.0 * a.sin(), 15.0 * a)
            })
            .collect();
        let on_an_arm = Profile::new(vec![
            Point2::new(12.0, -8.0),
            Point2::new(28.0, -8.0),
            Point2::new(28.0, 8.0),
            Point2::new(12.0, 8.0),
        ])
        .unwrap();
        let swept = Swept::new(on_an_arm, Spine::new(coil), Vector3::x()).unwrap();
        // Round the outside of the coil, from beside it to well clear, over the height of the
        // middle turn.
        let mut points = Vec::new();
        for k in 0..36 {
            let a = TAU * k as f64 / 36.0;
            for r in [65.0, 75.0, 85.0, 100.0] {
                for z in 0..14 {
                    points.push(Point3::new(r * a.cos(), r * a.sin(), 50.0 + 7.0 * z as f64));
                }
            }
        }
        let over = crate::overstatement(&swept, &points, &crate::probe_directions(), 0.25, 1.2);
        assert!(over < 1e-9, "claimed {over} mm more room than there was");
    }

    /// **Where nothing comes back, the tube is the tightest bend's**: a sixty degree arc of
    /// radius 50 has a tube of 50, and a straight spine has no edge to its tube at all.
    #[test]
    fn a_spine_that_never_turns_a_right_angle_has_the_tube_of_its_tightest_bend() {
        let arc: Vec<Point3<f64>> = (0..=120)
            .map(|i| {
                let a = PI / 3.0 * i as f64 / 120.0;
                Point3::new(50.0 * a.cos(), 50.0 * a.sin(), 0.0)
            })
            .collect();
        let tube = tube_radius(&arc);
        assert!((tube - 50.0).abs() < 0.01, "a 60 degree arc of radius 50 has a tube of {tube}");
        let straight = [Point3::origin(), Point3::new(0.0, 0.0, 100.0)];
        assert!(tube_radius(&straight).is_infinite());
    }

    /// **Cut unevenly, a bend still claims no more room than there is.** The frame turns within
    /// a segment at the rate its two end tangents differ, so a short segment between long ones
    /// turns it three times faster than the bend here, and leans it off the segment on the way;
    /// the section then steps across each joint's inner bisector, which the reading allows for
    /// ([`frame_tilt`]) -- read without that allowance, 0.0096 mm past the surface at 20 from the
    /// spine. Three quarters of a turn of radius 60, cut alternately into segments of 0.15 and
    /// 0.03 radians, carrying a 30 mm square that reaches 21 of the tube's 40: probed from beside
    /// the walls to well inside the bend at a tenth of a millimetre, and across the joints'
    /// bisectors at four thousandths, which is fine enough to see the step.
    #[test]
    fn an_unevenly_cut_bend_never_overstates_the_room_before_its_surface() {
        let mut arc = Vec::new();
        let (mut a, mut long) = (0.0f64, true);
        while a < 1.5 * PI {
            arc.push(Point3::new(60.0 * a.cos(), 60.0 * a.sin(), 0.0));
            a += if long { 0.15 } else { 0.03 };
            long = !long;
        }
        let swept = Swept::new(square(15.0), Spine::new(arc.clone()), Vector3::z()).unwrap();
        assert!(swept.tube.is_some(), "the section reads in its tube here, or this tests nothing");
        let mut points = Vec::new();
        for k in 0..48 {
            let a = 1.5 * PI * (k as f64 + 0.5) / 48.0;
            for r in [20.0, 30.0, 38.0, 44.0, 78.0, 85.0, 95.0] {
                for z in [-24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0] {
                    points.push(Point3::new(r * a.cos(), r * a.sin(), z));
                }
            }
        }
        let over = crate::overstatement(&swept, &points, &crate::probe_directions(), 0.1, 1.2);
        assert!(over < 1e-9, "claimed {over} mm more room than there was");
        // Across the bisectors of joints of both kinds, short leg first and long leg first.
        let mut across = Vec::new();
        for i in (1..arc.len() - 1).step_by(9) {
            let (e0, e1) = ((arc[i] - arc[i - 1]).normalize(), (arc[i + 1] - arc[i]).normalize());
            let (along, inward) = ((e0 + e1).normalize(), (e1 - e0).normalize());
            for depth in [16.0, 18.0, 20.0, 22.0] {
                for side in [inward, -inward] {
                    for off in [-0.05, 0.0, 0.05] {
                        across.push(arc[i] + side * depth + along * off);
                    }
                }
            }
        }
        let in_plane: Vec<Vector3<f64>> = (0..48)
            .map(|k| {
                let a = TAU * k as f64 / 48.0;
                Vector3::new(a.cos(), a.sin(), 0.0)
            })
            .collect();
        let over = crate::overstatement(&swept, &across, &in_plane, 0.004, 1.2);
        assert!(over < 1e-9, "across the joints, claimed {over} mm more room than there was");
    }

    /// **A spine that comes back closes its tube at half the gap.** Between the turns of a coil
    /// of pitch 94 the tube is no wider than 47, whatever the bend allows (64 here); across a U
    /// whose legs are 80 apart, no wider than 40; and a ring's is kept to 0.71 of its radius,
    /// the price of a rule that admits a quarter of a circle -- conservative, never generous.
    #[test]
    fn a_spine_that_comes_back_has_a_tube_no_wider_than_half_the_gap() {
        let coil: Vec<Point3<f64>> = (0..=240)
            .map(|i| {
                let a = 2.0 * TAU * i as f64 / 240.0;
                Point3::new(60.0 * a.cos(), 60.0 * a.sin(), 15.0 * a)
            })
            .collect();
        let tube = tube_radius(&coil);
        assert!(tube > 30.0 && tube <= 15.0 * PI, "a coil of pitch {} has a tube of {tube}", 30.0 * PI);
        let mut u: Vec<Point3<f64>> =
            (0..20).map(|i| Point3::new(-40.0, 0.0, 100.0 - 5.0 * i as f64)).collect();
        u.extend((0..=90).map(|i| {
            let a = PI + PI * i as f64 / 90.0;
            Point3::new(40.0 * a.cos(), 0.0, 40.0 * a.sin())
        }));
        u.extend((1..=20).map(|i| Point3::new(40.0, 0.0, 5.0 * i as f64)));
        let tube = tube_radius(&u);
        assert!(tube > 25.0 && tube <= 40.0, "a U 80 wide has a tube of {tube}");
        let ring: Vec<Point3<f64>> = (0..=180)
            .map(|i| {
                let a = TAU * (i % 180) as f64 / 180.0;
                Point3::new(50.0 * a.cos(), 50.0 * a.sin(), 0.0)
            })
            .collect();
        let tube = tube_radius(&ring);
        assert!(tube > 34.0 && tube <= 50.0 * std::f64::consts::FRAC_1_SQRT_2 + 0.1, "a ring of 50 has a tube of {tube}");
    }

    /// **A corner that turns a right angle or more has no tube at all**, however long its legs:
    /// the two legs meet at a gap of nothing that is square to both, and a section there has no
    /// one nearest place to be read from. Its bend alone would have allowed a hundred.
    #[test]
    fn a_corner_past_a_right_angle_has_no_tube() {
        let turn = 100f64.to_radians();
        let corner = [
            Point3::new(-100.0, 0.0, 0.0),
            Point3::origin(),
            Point3::new(100.0 * turn.cos(), 100.0 * turn.sin(), 0.0),
        ];
        assert_eq!(tube_radius(&corner), 0.0);
        // And a sweep along it falls back to dividing everything, rather than reading a section
        // it has no tube for.
        let swept = Swept::new(square(4.0), Spine::new(corner.to_vec()), Vector3::z()).unwrap();
        assert!(swept.tube.is_none());
    }

    /// A polyline in the XY plane that runs `start` along +X to the origin and then turns left
    /// by each `(degrees, length)` in turn.
    fn legs(start: f64, turns: &[(f64, f64)]) -> Vec<Point3<f64>> {
        let mut points = vec![Point3::new(-start, 0.0, 0.0), Point3::origin()];
        let mut heading = 0.0f64;
        for &(degrees, length) in turns {
            heading += degrees.to_radians();
            let last = *points.last().unwrap();
            points.push(last + Vector3::new(heading.cos(), heading.sin(), 0.0) * length);
        }
        points
    }

    /// How far a march from `from` along `dir` goes past the surface inside the room `field`
    /// claims there: nought where the claim held. `crate::overstatement`'s march kept to one
    /// direction, so that two readings of one solid can be set against each other at the same
    /// point over the same samples -- they share a zero set, so they meet the surface at the
    /// same sample, and only what they claimed can differ.
    fn claim_past(field: &Swept, from: Point3<f64>, dir: Vector3<f64>, step: f64) -> f64 {
        let claimed = field.distance(from);
        if !(claimed > 0.0) {
            return 0.0;
        }
        let mut t = 0.0;
        while t < claimed * 1.2 {
            if field.distance(from + dir * t) <= 0.0 {
                return (claimed - t).max(0.0);
            }
            t += step;
        }
        0.0
    }

    /// **Cut coarsely, in its tube a sweep claims no more room than dividing claimed.** A
    /// polyline's frame takes the bisector for its tangent at a joint and leans off each
    /// segment toward it, so the section is read across a plane tilted up to half the joint's
    /// turn: the solid bulges toward `reach / cos` there, and where a short leg turns its frame
    /// faster than a long one the section steps across the joint's inner bisector. Neither is
    /// second order in a coarse cut, and the reading in the tube has to allow for both
    /// (`sweep-spec.md` §4.1): reading the section as it was, 20 mm squares round 10 and 20 mm
    /// legs between 40° joints claimed up to 0.14 and 0.32 mm more than dividing did, and a
    /// closed hexagon, whose frame turns its whole closing corner at once, 10.8 more. Probed
    /// round every joint, both sides and out of the plane, at the same samples for both
    /// readings, so the comparison is exact rather than a tolerance.
    #[test]
    fn in_its_tube_a_coarsely_cut_polyline_claims_no_more_room_than_dividing_did() {
        let hexagon: Vec<Point3<f64>> = (0..=6)
            .map(|i| {
                let a = TAU * (i % 6) as f64 / 6.0;
                Point3::new(100.0 * a.cos(), 100.0 * a.sin(), 0.0)
            })
            .collect();
        // A coil cut into alternately long and short pieces, for a spine that leaves its plane.
        let mut coil = Vec::new();
        let (mut a, mut long) = (0.0f64, true);
        while a < 1.5 * TAU {
            coil.push(Point3::new(60.0 * a.cos(), 60.0 * a.sin(), 25.0 * a));
            a += if long { 0.35 } else { 0.07 };
            long = !long;
        }
        let cases = [
            ("a 2 mm leg between 44° joints", legs(100.0, &[(44.0, 2.0), (44.0, 100.0)]), 30.0),
            ("the same carrying a 20 mm square", legs(100.0, &[(44.0, 2.0), (44.0, 100.0)]), 10.0),
            ("a 10 mm leg between 40° joints", legs(100.0, &[(40.0, 10.0), (40.0, 100.0)]), 10.0),
            ("a 20 mm leg between 40° joints", legs(100.0, &[(40.0, 20.0), (40.0, 100.0)]), 10.0),
            ("a 10 mm leg between 30° joints", legs(100.0, &[(30.0, 10.0), (30.0, 100.0)]), 10.0),
            ("a 4 mm leg between 10° joints", legs(100.0, &[(10.0, 4.0), (10.0, 100.0)]), 10.0),
            ("a 5 mm leg from the start to a 30° joint", legs(5.0, &[(30.0, 100.0)]), 10.0),
            ("a hexagon closing where it starts", hexagon, 30.0),
            ("a coil cut unevenly", coil, 8.0),
        ];
        let mut directions: Vec<Vector3<f64>> = (0..48)
            .map(|k| {
                let a = TAU * k as f64 / 48.0;
                Vector3::new(a.cos(), a.sin(), 0.0)
            })
            .collect();
        directions.extend(crate::probe_directions().iter().map(|d| d.normalize()));
        let mut worst = Vec::new();
        for (what, points, half) in cases {
            let swept = Swept::new(square(half), Spine::new(points.clone()), Vector3::z()).unwrap();
            let mut divided = swept.clone();
            divided.tube = None;
            let closed = (points[0] - points[points.len() - 1]).norm() < 1e-9;
            let n = points.len();
            let joints: Vec<usize> = if closed { (0..n - 1).collect() } else { (1..n - 1).collect() };
            let step = half / 500.0;
            let mut excess = 0.0f64;
            for &i in joints.iter().take(8) {
                let before = if i == 0 { points[n - 2] } else { points[i - 1] };
                let (e0, e1) = ((points[i] - before).normalize(), (points[i + 1] - points[i]).normalize());
                let along = (e0 + e1).normalize();
                let inward = (e1 - e0).try_normalize(1e-12).unwrap_or_else(|| any_perpendicular(along).normalize());
                let across = along.cross(&inward);
                let leg = (points[i] - before).norm().min((points[i + 1] - points[i]).norm());
                for k in -2..=2 {
                    for depth in [0.8, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2] {
                        for side in [inward, -inward, across] {
                            let from = points[i] + along * (k as f64 * leg / 4.0) + side * (depth * half);
                            for &dir in &directions {
                                let more = claim_past(&swept, from, dir, step)
                                    - claim_past(&divided, from, dir, step);
                                excess = excess.max(more);
                            }
                        }
                    }
                }
            }
            worst.push((what, excess));
        }
        assert!(
            worst.iter().all(|(_, excess)| *excess < 1e-9),
            "claimed more past the surface than dividing did: {worst:?}"
        );
    }

    /// **Beside a short leg between two sharp joints, the field claims no room the bulge at the
    /// leg's end takes.** The frame leans 22° off the leg at each end of it, so a 60 mm square
    /// is read 30 / cos 22° = 32.4 mm deep there and 30 at the middle. A point 45 mm out from the
    /// middle read 15, undivided, where a solid point 1 mm along at 32.3 deep is 12.7 away.
    #[test]
    fn beside_a_short_leg_the_field_claims_no_room_the_bulge_at_its_end_takes() {
        let points = legs(100.0, &[(44.0, 2.0), (44.0, 100.0)]);
        let swept = Swept::new(square(30.0), Spine::new(points.clone()), Vector3::z()).unwrap();
        let leg = (points[2] - points[1]).normalize();
        let outward = Vector3::new(leg.y, -leg.x, 0.0);
        let middle = nalgebra::center(&points[1], &points[2]);
        let mut worst = (0.0f64, 0.0, 0.0);
        for end in [points[1], points[2]] {
            for k in 0..=24 {
                let solid = end + outward * (30.0 + 0.1 * k as f64);
                if swept.distance(solid) > 0.0 {
                    continue;
                }
                for depth in [35.0, 40.0, 45.0, 50.0, 60.0] {
                    let from = middle + outward * depth;
                    let (claimed, room) = (swept.distance(from), (solid - from).norm());
                    if claimed - room > worst.0 {
                        worst = (claimed - room, claimed, depth);
                    }
                }
            }
        }
        let (over, claimed, depth) = worst;
        assert!(over <= 0.0, "{depth} mm out from the leg claimed {claimed}, {over} past solid");
    }

    /// **A closed spine's tube is closed by the corner where it starts, too.** That corner is a
    /// joint like every other, and here it is the tightest: a path that runs straight out of its
    /// start, round 330° of an arc of radius 100 and straight back in, meeting itself at 30°
    /// between two 5 mm legs -- a bend of radius 9.5, where the arc allows 100 and the far side
    /// about 70. Reading only the joints between the first point and the last left it out.
    #[test]
    fn a_closed_paths_tube_is_closed_by_the_corner_where_it_starts() {
        // Out along +X, then round an arc centred 100 above, ending on the line that comes back
        // into the start heading 330°: that puts the arc's start 2 − √3 of its radius along.
        let along = 100.0 * (2.0 - 3f64.sqrt());
        let centre = Point3::new(along, 100.0, 0.0);
        let mut path = vec![Point3::origin(), Point3::new(5.0, 0.0, 0.0)];
        for k in 0..=22 {
            let a = (-90.0 + 15.0 * k as f64).to_radians();
            path.push(centre + Vector3::new(a.cos(), a.sin(), 0.0) * 100.0);
        }
        let back = Vector3::new(30f64.to_radians().cos(), -(30f64.to_radians().sin()), 0.0);
        let end = *path.last().unwrap();
        assert!(end.coords.cross(&back).norm() < 1e-9, "the arc ends on the way back in: {end:?}");
        path.push(Point3::origin() - back * 5.0);
        path.push(Point3::origin());
        let corner = 30f64.to_radians() / 5.0;
        let tube = tube_radius(&path);
        assert!(
            tube <= 1.0 / corner + 1e-9,
            "a closing corner of radius {:.2} gave a tube of {tube:.2}",
            1.0 / corner
        );
    }

    /// A profile wider than the bend is refused by name, and one just narrower is not.
    #[test]
    fn a_profile_wider_than_the_bend_is_refused() {
        // A quarter circle of radius 10: curvature 0.1 per mm.
        let arc: Vec<Point3<f64>> = (0..=30)
            .map(|i| {
                let a = 0.5 * PI * i as f64 / 30.0;
                Point3::new(10.0 * a.cos(), 10.0 * a.sin(), 0.0)
            })
            .collect();
        let wide = Swept::new(square(11.0), Spine::new(arc.clone()), Vector3::z());
        let err = wide.expect_err("a square reaching further than the bend's radius");
        assert!(
            matches!(err, SweepError::FoldsThroughItself { .. }),
            "refused as {err:?} rather than for folding"
        );
        assert!(err.to_string().contains("fold"), "{err}");
        // Reach is the corner, `11·√2`, so this one fails too; half-width 6 reaches 8.5 and fits.
        Swept::new(square(6.0), Spine::new(arc), Vector3::z()).expect("fits inside the bend");
    }
}

/// What a draft does to the section's corners -- the owner's ruling of 2026-10-03, that both
/// are right answers and the block says which.
#[cfg(test)]
mod corner_tests {
    use super::*;
    use crate::{Extrusion, Profile};
    use fieldform_scene::Corners;

    /// A 20 mm square, whose four right angles are the corners in question.
    fn square(half: f64) -> Profile {
        Profile::new(vec![
            Point2::new(-half, -half),
            Point2::new(half, -half),
            Point2::new(half, half),
            Point2::new(-half, half),
        ])
        .unwrap()
    }

    fn extrude(corners: Corners, draft: f64) -> Extrusion {
        Extrusion { rims: Default::default(),
            footprint: square(10.0).into(),
            start: 0.0,
            height: 100.0,
            depth: crate::Depth::One,
            draft_tangent: draft.to_radians().tan(),
            corners,
        }
    }

    /// **The corner is where each answer says it is, and the faces are in the same place.**
    /// Half way up a 100 mm extrude drafted 10°, the wall has moved 8.82 mm: the round corner
    /// stands that far from the drawn vertex along its bisector and the sharp one stands
    /// `8.82·√2` -- while off the middle of a face the two read the same number, because a
    /// face does not care how the corners are cut.
    #[test]
    fn a_sharp_corner_stands_where_the_two_walls_meet_and_a_round_one_on_the_arc() {
        let (round, sharp) = (extrude(Corners::Round, 10.0), extrude(Corners::Sharp, 10.0));
        let z = 50.0;
        let moved = z * 10f64.to_radians().tan();
        let vertex = Point3::new(10.0, 10.0, z);
        let bisector = Vector3::new(1.0, 1.0, 0.0).normalize();
        // Each reads nought on its own contour, to the correction the draft's lean costs.
        let on_round = round.distance(vertex + bisector * moved);
        let on_sharp = sharp.distance(vertex + bisector * moved * std::f64::consts::SQRT_2);
        assert!(on_round.abs() < 1e-9, "the round corner is an arc of the offset: {on_round}");
        assert!(on_sharp.abs() < 1e-9, "the sharp corner is the mitre: {on_sharp}");
        // And the sharp solid holds what the round one does not: the mitre's own apex.
        let apex = vertex + bisector * (moved * std::f64::consts::SQRT_2 - 0.1);
        assert!(sharp.distance(apex) < 0.0 && round.distance(apex) > 0.0, "the mitre is material");
        // Off the middle of a face, nothing differs.
        for at in [Point3::new(0.0, 20.0, z), Point3::new(0.0, -20.0, z), Point3::new(25.0, 0.0, z)] {
            let (a, b) = (round.distance(at), sharp.distance(at));
            assert!((a - b).abs() < 1e-9, "a face is a face either way, at {at:?}: {a} vs {b}");
        }
    }

    /// **The walls still lean at the draft angle**, which is what makes it a draft. Measured
    /// off the middle of a face, where both answers agree, and along the mitre, where the
    /// edge is steeper in plan and the faces either side of it are not.
    #[test]
    fn mitred_corners_do_not_change_what_the_draft_angle_means() {
        for degrees in [3.0f64, 10.0, 25.0] {
            let solid = extrude(Corners::Sharp, degrees);
            let edge_at = |z: f64, dir: Vector3<f64>| {
                let (mut a, mut b) = (0.0, 80.0);
                for _ in 0..80 {
                    let mid = 0.5 * (a + b);
                    if solid.distance(Point3::new(0.0, 0.0, z) + dir * mid) < 0.0 {
                        a = mid;
                    } else {
                        b = mid;
                    }
                }
                0.5 * (a + b)
            };
            // **Between two heights inside the solid, not from its own start cap.** At the
            // plane the slab reads nought and the solid is never negative there, so a
            // bisection started at z = 0 finds the cap rather than the wall -- which read a
            // 3 degree draft as 10.
            let face = Vector3::x();
            let lean = ((edge_at(90.0, face) - edge_at(10.0, face)) / 80.0).atan().to_degrees();
            assert!(
                (lean - degrees).abs() < 1e-6,
                "a sharp draft of {degrees} leans its face {lean}"
            );
            // The mitre edge itself rises further out per millimetre, by its own √2.
            let bisector = Vector3::new(1.0, 1.0, 0.0).normalize();
            let along = (edge_at(90.0, bisector) - edge_at(10.0, bisector)) / 80.0;
            let want = degrees.to_radians().tan() * std::f64::consts::SQRT_2;
            assert!(
                (along - want).abs() < 1e-6,
                "the mitre of a {degrees}\u{b0} draft runs out at {along}, not {want}"
            );
        }
    }

    /// **Nothing to pay without a draft.** With upright walls there is no offset, so the two
    /// are the same solid everywhere -- and a document that never asks keeps the field it had.
    #[test]
    fn with_no_draft_the_two_are_one_solid() {
        let (round, sharp) = (extrude(Corners::Round, 0.0), extrude(Corners::Sharp, 0.0));
        for i in 0..400 {
            let p = Point3::new(
                -30.0 + (i % 13) as f64 * 5.0,
                -30.0 + (i % 17) as f64 * 3.9,
                -10.0 + (i % 23) as f64 * 5.3,
            );
            assert_eq!(round.distance(p), sharp.distance(p), "at {p:?}");
        }
    }

    /// **A draft grows a band; it does not slide it.** A draft offsets the *solid*, so both
    /// faces of a wall move outward along their own normals and the band gets thicker by the
    /// offset on each side, staying centred on its curve. Moving the band's middle instead
    /// would slide a drafted tube's whole wall outward keeping its thickness — which is what
    /// the shader did, agreeing on the outer face (so every outside view looked right) and
    /// differing by twice the offset down the bore.
    ///
    /// Asserted on the kernel in both the extrude and the sweep, because the two are one
    /// construction and the shader mirrors whichever they say.
    #[test]
    fn a_draft_grows_a_wall_band_rather_than_sliding_it() {
        use fieldform_scene::{Corners, WallEnds, WallSide};
        // A 40 mm square loop given a 4 mm wall about its curve, drafted 10 degrees over
        // 100 mm: at the far end the offset is 17.63 mm, so the band should run from
        // 2 + 17.63 inside the curve to the same outside -- 39.3 mm thick, still centred.
        let loop_points = vec![
            Point2::new(-20.0, -20.0),
            Point2::new(20.0, -20.0),
            Point2::new(20.0, 20.0),
            Point2::new(-20.0, 20.0),
        ];
        let wall = crate::Wall::closed(
            Profile::new(loop_points).expect("a square"),
            4.0,
            WallSide::Centre,
        )
        .expect("a wall round a loop");
        let _ = WallEnds::Square;
        let footprint: crate::Footprint = wall.into();
        let t = 100.0 * 10f64.to_radians().tan();
        // On the curve itself the band is deepest: half the thickness plus the offset.
        let on_the_curve = footprint.offset_by(Point2::new(20.0, 0.0), t, Corners::Round);
        assert!(
            (on_the_curve + (2.0 + t)).abs() < 1e-9,
            "on the curve a drafted band is {} deep, not {}",
            -on_the_curve,
            2.0 + t
        );
        // And its two faces are that far either side of the curve, which is what "grows" means.
        for (at, what) in [(20.0 - (2.0 + t), "the inner face"), (20.0 + 2.0 + t, "the outer")] {
            let read = footprint.offset_by(Point2::new(at, 0.0), t, Corners::Round);
            assert!(read.abs() < 1e-9, "{what} should be at {at}, where the band reads {read}");
        }
        // Sliding would have put both faces 17.63 mm further out and left the band 4 mm thick,
        // so the inner face would read solid here rather than nought.
        let slid = footprint.offset_by(Point2::new(20.0 + t - 2.0, 0.0), t, Corners::Round);
        assert!(
            slid < -1e-6,
            "a slid band would have its inner face here; a grown one is solid, read {slid}"
        );
    }

    /// **A wall's corners cannot be mitred, and the block is told so rather than quietly
    /// rounded.** A band has two faces and mitring both is not built; a solid that said Sharp
    /// and drew Round is the kind of wrong nothing reports.
    #[test]
    fn a_wall_refuses_to_have_its_corners_mitred() {
        use fieldform_scene::{Corners, WallSide};
        let wall = crate::Wall::closed(
            Profile::new(vec![
                Point2::new(-10.0, -10.0),
                Point2::new(10.0, -10.0),
                Point2::new(10.0, 10.0),
                Point2::new(-10.0, 10.0),
            ])
            .expect("a square"),
            3.0,
            WallSide::Centre,
        )
        .expect("a wall");
        let law = |corners| SectionLaw {
            taper: Taper::Draft { tangent: 6f64.to_radians().tan() },
            corners,
            ..SectionLaw::default()
        };
        let built = |corners| {
            Swept::under_law(
                wall.clone(),
                Spine::new(vec![Point3::origin(), Point3::new(0.0, 0.0, 60.0)]),
                Vector3::x(),
                Held::Square,
                law(corners),
            )
        };
        assert_eq!(
            built(Corners::Sharp).expect_err("a wall cannot be mitred"),
            SweepError::SharpCornersOnAWall
        );
        built(Corners::Round).expect("and rounding one is ordinary");
    }

    /// **A drafted sweep along a straight path is the drafted extrude, corners and all.** The
    /// two were one construction before there was a choice here (`sweep-v2-spec.md` §8) and
    /// this is what keeps them one now that there is.
    #[test]
    fn a_sharp_drafted_sweep_along_a_straight_path_is_the_sharp_drafted_extrude() {
        for corners in Corners::ALL {
            let law = SectionLaw {
                taper: Taper::Draft { tangent: 7f64.to_radians().tan() },
                corners,
                ..SectionLaw::default()
            };
            let swept = Swept::under_law(
                square(10.0),
                Spine::new(vec![Point3::origin(), Point3::new(0.0, 0.0, 100.0)]),
                Vector3::x(),
                Held::Square,
                law,
            )
            .expect("a straight path");
            let solid = extrude(corners, 7.0);
            for x in -4..=4 {
                for y in -4..=4 {
                    for z in -2..=22 {
                        let p = Point3::new(x as f64 * 6.0, y as f64 * 6.0, z as f64 * 5.0);
                        let (a, b) = (swept.distance(p), solid.distance(p));
                        assert!(
                            (a - b).abs() < 1e-9,
                            "{corners:?} at {p:?}: sweep {a}, extrude {b}"
                        );
                    }
                }
            }
        }
    }

    /// **The mitre is held, so a needle cannot throw a spike across the part.** A corner of
    /// five degrees would reach 22.9 times the offset; held to four, the solid stays inside a
    /// box the extent can state.
    #[test]
    fn a_needle_sharp_corner_keeps_a_finite_mitre() {
        // A sliver: two long edges meeting at about five degrees.
        let needle = Profile::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(100.0, -4.4),
            Point2::new(100.0, 4.4),
        ])
        .expect("a triangle");
        let reach = needle
            .miters()
            .iter()
            .map(|m| m.norm())
            .fold(0.0f64, f64::max);
        assert!(
            reach <= fieldform_scene::MITRE_LIMIT + 1e-9,
            "the mitre reaches {reach} times the offset, past the limit"
        );
        // And the tip is still material after an offset, rather than a spike or a gap.
        let t = 2.0;
        let tip = Point2::new(-t * 1.5, 0.0);
        assert!(
            needle.offset_by(tip, t, Corners::Sharp) < 0.0,
            "the held mitre still covers the tip"
        );
    }
}

/// The section law -- `sweep-v2-spec.md` §8's acceptance, for the leaf.
#[cfg(test)]
mod law_tests {
    use super::*;
    use crate::{Cone, Extrusion, Profile, Torus};
    use fieldform_scene::RampShape;
    use std::f64::consts::{PI, TAU};

    fn square(half: f64) -> Profile {
        Profile::new(vec![
            Point2::new(-half, -half),
            Point2::new(half, -half),
            Point2::new(half, half),
            Point2::new(-half, half),
        ])
        .unwrap()
    }

    fn circle(r: f64, n: usize) -> Profile {
        Profile::new(
            (0..n)
                .map(|i| {
                    let a = TAU * i as f64 / n as f64;
                    Point2::new(r * a.cos(), r * a.sin())
                })
                .collect(),
        )
        .unwrap()
    }

    /// Straight up for `length` millimetres.
    fn up(length: f64) -> Spine {
        Spine::new(vec![Point3::origin(), Point3::new(0.0, 0.0, length)])
    }

    fn plain() -> SectionLaw {
        SectionLaw::default()
    }

    fn scaled(end: f64) -> SectionLaw {
        SectionLaw { taper: Taper::Scale { end }, ..plain() }
    }

    fn drafted(degrees: f64) -> SectionLaw {
        SectionLaw { taper: Taper::Draft { tangent: degrees.to_radians().tan() }, ..plain() }
    }

    fn twisted(degrees: f64) -> SectionLaw {
        SectionLaw { twist: degrees.to_radians(), ..plain() }
    }

    /// **v1 unchanged.** Under the plain law the stretch is exactly the number it was,
    /// `1 / (1 − κ·r)`, and every reading is the reading it was.
    #[test]
    fn the_plain_law_is_the_sweep_it_always_was() {
        let bend: Vec<Point3<f64>> = (0..=40)
            .map(|i| {
                let a = 0.5 * PI * i as f64 / 40.0;
                Point3::new(30.0 * a.sin(), 0.0, 30.0 * (1.0 - a.cos()))
            })
            .collect();
        let before = Swept::new(square(5.0), Spine::new(bend.clone()), Vector3::y()).unwrap();
        let under = Swept::under_law(square(5.0), Spine::new(bend), Vector3::y(), Held::Square, plain())
            .unwrap();
        assert_eq!(before.stretch(), under.stretch(), "the plain law must cost nothing");
        assert!(under.law().is_plain());
        for p in [Point3::new(10.0, 2.0, 3.0), Point3::new(30.0, -4.0, 20.0), Point3::new(0.0, 0.0, 60.0)] {
            assert_eq!(before.distance(p), under.distance(p), "at {p:?}");
        }
    }

    /// **Taper against an analytic cone.** A circle of radius 20 along a straight 100 mm spine
    /// with end scale 0.5 is the frustum of radii 20 and 10. Its surface is the cone's, and its
    /// undivided reading overstates the cone's by at most `1 / cos` of the cone's half-angle --
    /// the reading is taken square to the path and the wall leans -- which is what the
    /// stretch `√(1 + K²)` is there to cover: divided, it never overstates.
    #[test]
    fn a_scaled_circle_along_a_straight_spine_is_a_cone() {
        let swept =
            Swept::under_law(circle(20.0, 720), up(100.0), Vector3::x(), Held::Square, scaled(0.5))
                .unwrap();
        let cone = Cone {
            base: Point3::origin(),
            top: Point3::new(0.0, 0.0, 100.0),
            base_radius: 20.0,
            top_radius: 10.0,
        };
        let lean = 1.0 / (10.0f64 / 100.0).atan().cos();
        for i in 0..400 {
            let p = Point3::new(
                -30.0 + (i % 17) as f64 * 3.7,
                -30.0 + (i % 23) as f64 * 2.6,
                -10.0 + (i % 29) as f64 * 4.3,
            );
            let want = cone.distance(p);
            let raw = swept.distance(p) * swept.stretch();
            let divided = swept.distance(p);
            // Agree about inside, clear of the surface and of the polygonal circle's error.
            if want.abs() > 0.2 {
                assert_eq!(raw < 0.0, want < 0.0, "at {p:?}: sweep {raw} against cone {want}");
            }
            // The sectionwise reading is within the lean of the cone's, and the divided one
            // never reads further from the surface than the cone does.
            assert!(
                raw.abs() <= want.abs() * lean + 0.2,
                "at {p:?}: {raw} is more than the lean over {want}"
            );
            assert!(divided.abs() <= want.abs() + 0.2, "at {p:?}: {divided} overstates {want}");
        }
        // And the surface is where the cone's is: along a radial at each height.
        for z in [10.0, 50.0, 90.0] {
            let r = 20.0 - 10.0 * z / 100.0;
            assert!(swept.distance(Point3::new(r - 0.1, 0.0, z)) < 0.0, "inside at z = {z}");
            assert!(swept.distance(Point3::new(r + 0.1, 0.0, z)) > 0.0, "outside at z = {z}");
        }
    }

    /// **Draft against the extrude.** A square along a straight 100 mm spine with a 5° draft
    /// is the extrude of the same square with a 5° draft, pointwise, divisor and all: the two
    /// drafts are one construction.
    #[test]
    fn a_drafted_sweep_along_a_straight_spine_is_the_drafted_extrude() {
        let swept =
            Swept::under_law(square(10.0), up(100.0), Vector3::x(), Held::Square, drafted(5.0))
                .unwrap();
        let extrude = Extrusion { rims: Default::default(),
            footprint: square(10.0).into(),
            start: 0.0,
            height: 100.0,
            depth: crate::Depth::One,
            draft_tangent: 5f64.to_radians().tan(),
            corners: Default::default(),
        };
        for x in -4..=4 {
            for y in -4..=4 {
                for z in -2..=22 {
                    let p = Point3::new(x as f64 * 5.0, y as f64 * 5.0, z as f64 * 5.0);
                    let (a, b) = (swept.distance(p), extrude.distance(p));
                    assert!((a - b).abs() < 1e-9, "at {p:?}: sweep {a}, extrude {b}");
                }
            }
        }
    }

    /// **A scaled wall keeps its thickness.** A centre wall of 2 mm on a circle of radius 20,
    /// scaled to 0.5 along a straight path: the band is 2 mm wide at the start and 2 mm wide
    /// at the end, about circles of radius 20 and 10.
    #[test]
    fn a_scaled_wall_on_a_loop_keeps_its_thickness() {
        let wall = crate::Wall::closed(circle(20.0, 720), 2.0, fieldform_scene::WallSide::Centre)
            .expect("a wall round a circle");
        let swept = Swept::under_law(wall, up(100.0), Vector3::x(), Held::Square, scaled(0.5))
            .unwrap();
        // The band's edges along a radial, found where the reading crosses nought.
        let edges = |z: f64| {
            let mut found = Vec::new();
            let mut last = swept.distance(Point3::new(0.0, 0.0, z));
            for i in 1..=3000 {
                let r = i as f64 * 0.01;
                let d = swept.distance(Point3::new(r, 0.0, z));
                if (d < 0.0) != (last < 0.0) {
                    found.push(r);
                }
                last = d;
            }
            found
        };
        for (z, radius) in [(1.0, 19.9), (50.0, 15.0), (99.0, 10.1)] {
            let found = edges(z);
            assert_eq!(found.len(), 2, "two edges at z = {z}: {found:?}");
            let thickness = found[1] - found[0];
            assert!(
                (thickness - 2.0).abs() < 0.03,
                "at z = {z} the wall is {thickness:.3} mm thick: {found:?}"
            );
            let middle = 0.5 * (found[0] + found[1]);
            assert!((middle - radius).abs() < 0.03, "at z = {z} the wall is about {middle:.2}, not {radius}");
        }
    }

    /// **Twist, in the sense `Op::Twist` has.** A square along a straight spine twisted 60°
    /// over 100 mm: half way up the section has turned 30°, so the corner drawn at 45° stands
    /// at 75° -- a point there 13.9 mm out is inside, and one at 15° is past the middle of a
    /// side. (At 45° of turn a square's corners land on the axes whichever way it turned, which
    /// is how the first draft of this test checked nothing.) The warp op reads a body at the
    /// point turned by `−rate·z`, so the body appears turned counter-clockwise about its axis,
    /// and so does this.
    #[test]
    fn a_twisted_square_turns_the_way_the_warp_does() {
        let swept = Swept::under_law(square(10.0), up(100.0), Vector3::x(), Held::Square, twisted(60.0))
            .unwrap();
        let corner = 10.0 * std::f64::consts::SQRT_2 - 0.25;
        let at = |degrees: f64| {
            let a = degrees.to_radians();
            Point3::new(corner * a.cos(), corner * a.sin(), 50.0)
        };
        assert!(swept.distance(at(75.0)) < 0.0, "the corner has turned on to 75 degrees");
        assert!(swept.distance(at(15.0)) > 0.0, "and not to 15");
        // At the start next to nothing has turned, and at the end it has turned 60 degrees:
        // the corner drawn at 45 stands at 105.
        assert!(swept.distance(Point3::new(9.7, 9.7, 0.5)) < 0.0);
        let end = 105f64.to_radians();
        assert!(swept.distance(Point3::new(corner * end.cos(), corner * end.sin(), 99.0)) < 0.0);
        // Against the warp's own arithmetic, undivided: the section read at the point turned
        // back by the law's angle.
        let plain = Swept::new(square(10.0), up(100.0), Vector3::x()).unwrap();
        for p in [Point3::new(6.0, 11.0, 30.0), Point3::new(-12.0, 3.0, 70.0), Point3::new(9.0, -9.0, 50.0)] {
            let angle = -(PI / 3.0) * p.z / 100.0;
            let (s, c) = angle.sin_cos();
            let back = Point3::new(p.x * c - p.y * s, p.x * s + p.y * c, p.z);
            let (got, want) = (swept.distance(p) * swept.stretch(), plain.distance(back));
            assert!((got - want).abs() < 1e-9, "at {p:?}: {got} against the warp's {want}");
        }
    }

    /// **Twist is invisible on a circle.** A circle carried round three quarters of a ring,
    /// twisted two full turns on the way, is still the torus -- which catches a turn applied
    /// to the wrong coordinates, since a circle about its own centre is the one section a turn
    /// cannot move.
    #[test]
    fn a_twisted_circle_round_an_arc_is_still_the_torus() {
        let (big, small) = (30.0, 5.0);
        let arc: Vec<Point3<f64>> = (0..=270)
            .map(|i| {
                let a = TAU * i as f64 / 360.0;
                Point3::new(big * a.cos(), big * a.sin(), 0.0)
            })
            .collect();
        let swept = Swept::under_law(circle(small, 96), Spine::new(arc), Vector3::x(), Held::Square, twisted(720.0))
            .unwrap();
        let torus = Torus { centre: Point3::origin(), major_radius: big, minor_radius: small };
        // Inside the swept three quarters, clear of the caps by more than the tube's radius,
        // since within it the cap is the nearer face and the torus has none.
        for i in 0..300 {
            let a = 0.3 + (1.5 * PI - 0.6) * (i % 37) as f64 / 37.0;
            let r = big - 8.0 + 16.0 * (i % 11) as f64 / 11.0;
            let z = -7.0 + 14.0 * (i % 13) as f64 / 13.0;
            let p = Point3::new(r * a.cos(), r * a.sin(), z);
            let (got, want) = (swept.distance(p) * swept.stretch(), torus.distance(p));
            assert!((got - want).abs() < 0.06, "at {p:?}: twisted sweep {got}, torus {want}");
        }
    }

    /// **The law's shape.** A G1 or G2 law reaches both ends with no slope, so the scale a
    /// hair in from either end is the end's own; a G0 law is already moving.
    #[test]
    fn a_smooth_law_leaves_both_ends_flat() {
        for (shape, flat) in [(RampShape::Linear, false), (RampShape::Smooth, true), (RampShape::Smoother, true)] {
            let law = SectionLaw { shape, ..scaled(0.5) };
            let near_start = (law.scale_at(0.01) - law.scale_at(0.0)) / 0.01;
            let near_end = (law.scale_at(1.0) - law.scale_at(0.99)) / 0.01;
            let plain_slope = -0.5;
            if flat {
                assert!(near_start.abs() < 0.03 && near_end.abs() < 0.03, "{shape:?}: {near_start}, {near_end}");
            } else {
                assert!((near_start - plain_slope).abs() < 1e-9 && (near_end - plain_slope).abs() < 1e-9);
            }
            // And in the field: the section's half-width a millimetre in from the end.
            let swept = Swept::under_law(square(10.0), up(100.0), Vector3::x(), Held::Square, law).unwrap();
            let half_width = |z: f64| {
                (0..2000)
                    .map(|i| i as f64 * 0.01)
                    .find(|x| swept.distance(Point3::new(*x, 0.0, z)) > 0.0)
                    .expect("an edge")
            };
            let at_end = half_width(99.0);
            if flat {
                assert!((at_end - 5.0).abs() < 0.02, "{shape:?}: {at_end} a millimetre from the end");
            } else {
                assert!((at_end - 5.05).abs() < 0.02, "{shape:?}: {at_end} a millimetre from the end");
            }
        }
    }

    /// **The scale's bound is the reach, and the field holds it** (`sweep-v2-spec.md` §3, backlog
    /// item 114). The owner's wing: a lens 40 mm across with three sharp corners, narrowed to a
    /// hundredth over 90 mm under a G2 law. The bound it was divided by took the solid's reach
    /// over its least scale and came to 82.9; the proved one is `|s′|·r` and comes to 1.08. The
    /// field divided by it has no gradient over one anywhere sampled -- outside and in, beside
    /// the wide end, the narrow end and the caps -- and a claimed distance is never crossed
    /// before it is used up, which is the contract itself.
    #[test]
    fn a_tapered_lens_reads_a_unit_gradient_under_the_new_bound() {
        let lens = Profile::new(vec![
            Point2::new(-20.0, 0.0),
            Point2::new(0.0, -4.35),
            Point2::new(20.0, 0.0),
            Point2::new(10.0, 5.0),
            Point2::new(0.0, 12.3),
            Point2::new(-10.0, 5.0),
        ])
        .unwrap();
        let law = SectionLaw { shape: RampShape::Smoother, ..scaled(0.01) };
        let swept = Swept::under_law(lens, up(90.0), Vector3::x(), Held::Square, law).unwrap();
        let stretch = swept.stretch();
        assert!((stretch - 1.082).abs() < 0.01, "divided by {stretch}, where the old bound was 82.9");
        let mut points = Vec::new();
        for x in -12..=12 {
            for y in -6..=8 {
                for z in -3..=33 {
                    points.push(Point3::new(x as f64 * 2.1, y as f64 * 2.3 + 0.37, z as f64 * 3.0 - 2.0));
                }
            }
        }
        // Each axis quotient of a 1-Lipschitz reading is at most one; the norm of the three may
        // pass it only at a crease, by the quotient's own width, and 2 % is that at h = 1e-3.
        let h = 1e-3;
        for p in &points {
            let d = |o: Vector3<f64>| swept.distance(p + o);
            let grad = Vector3::new(
                d(Vector3::new(h, 0.0, 0.0)) - d(Vector3::new(-h, 0.0, 0.0)),
                d(Vector3::new(0.0, h, 0.0)) - d(Vector3::new(0.0, -h, 0.0)),
                d(Vector3::new(0.0, 0.0, h)) - d(Vector3::new(0.0, 0.0, -h)),
            ) / (2.0 * h);
            assert!(grad.norm() <= 1.02, "at {p:?} the divided reading changes at {} per mm", grad.norm());
        }
        let dirs = [Vector3::x(), Vector3::y(), Vector3::z(), -Vector3::x(), -Vector3::y(), -Vector3::z()];
        for from in points.iter().step_by(3) {
            let claimed = swept.distance(*from);
            if claimed <= 0.0 {
                continue;
            }
            for d in dirs {
                for step in 1..=16 {
                    let at = from + d * (claimed * step as f64 / 16.0);
                    assert!(
                        swept.distance(at) > -1e-9,
                        "from {from:?} along {d:?}: crossed the surface inside a claimed {claimed}",
                    );
                }
            }
        }
    }

    /// **The extent.** A window of 25% to 75% on a straight 100 mm spine is the prism between
    /// 25 and 75 mm up, and the section at the window's start is the profile carried there.
    #[test]
    fn a_window_cuts_the_solid_to_a_stretch_of_the_path() {
        let law = SectionLaw { window: (0.25, 0.75), ..plain() };
        let swept = Swept::under_law(square(10.0), up(100.0), Vector3::x(), Held::Square, law).unwrap();
        assert!(swept.distance(Point3::new(0.0, 0.0, 10.0)) > 0.0, "before the window");
        assert!(swept.distance(Point3::new(0.0, 0.0, 26.0)) < 0.0, "just inside its start");
        assert!(swept.distance(Point3::new(0.0, 0.0, 74.0)) < 0.0, "just inside its end");
        assert!(swept.distance(Point3::new(0.0, 0.0, 90.0)) > 0.0, "after the window");
        // The caps are where the window says, exactly.
        assert!((swept.distance(Point3::new(0.0, 0.0, 20.0)) - 5.0).abs() < 1e-9);
        assert!((swept.distance(Point3::new(0.0, 0.0, 80.0)) - 5.0).abs() < 1e-9);
        // And the scale runs over the window, not the whole path: scaled to 0.5, the section is
        // whole at 25 mm and half at 75.
        let law = SectionLaw { window: (0.25, 0.75), ..scaled(0.5) };
        let swept = Swept::under_law(square(10.0), up(100.0), Vector3::x(), Held::Square, law).unwrap();
        assert!(swept.distance(Point3::new(9.9, 0.0, 25.5)) < 0.0, "whole at the window's start");
        assert!(swept.distance(Point3::new(5.1, 0.0, 74.5)) > 0.0, "half at its end");
        assert!(swept.distance(Point3::new(4.9, 0.0, 74.5)) < 0.0);
    }

    /// **Never overstates**, over a bent spine with the section doubled and turned half a
    /// turn on the way: the field never claims more room than there is, in six directions.
    #[test]
    fn a_tapered_twisted_sweep_never_overstates_the_room_before_its_surface() {
        let law = SectionLaw { taper: Taper::Scale { end: 2.0 }, twist: PI, ..plain() };
        let swept = Swept::under_law(square(3.0), Spine::new(super::tests::s_bend(20.0, 40)), Vector3::y(), Held::Square, law)
            .unwrap();
        let dirs = [Vector3::x(), Vector3::y(), Vector3::z(), -Vector3::x(), -Vector3::y(), -Vector3::z()];
        for x in -6..=6 {
            for y in -3..=3 {
                for z in -2..=14 {
                    let from = Point3::new(x as f64 * 8.0, y as f64 * 4.0, z as f64 * 6.0);
                    let claimed = swept.distance(from);
                    if claimed <= 0.0 {
                        continue;
                    }
                    for d in dirs {
                        for step in 1..=16 {
                            let at = from + d * (claimed * step as f64 / 16.0);
                            assert!(
                                swept.distance(at) > -1e-9,
                                "from {from:?} along {d:?}: crossed the surface inside a claimed {claimed}",
                            );
                        }
                    }
                }
            }
        }
    }

    /// **Refusals, by name.**
    #[test]
    fn the_laws_refusals_are_by_name() {
        let build = |law: SectionLaw, held: Held| {
            Swept::under_law(square(10.0), up(100.0), Vector3::x(), held, law)
        };
        let flat = Held::Parallel { normal: Vector3::z(), u: Vector3::x(), v: Vector3::y() };
        // A scale of nought or less closes the section.
        for end in [0.0, -1.0] {
            let err = build(scaled(end), Held::Square).expect_err("closes");
            assert!(matches!(err, SweepError::TaperClosesTheSection { .. }), "{err:?}");
            assert!(err.to_string().contains("closes the section"), "{err}");
        }
        // A draft inward by more than the section is deep: a 10 mm half-width over 100 mm is
        // 5.7 degrees, so 6 closes it and 5 does not.
        let err = build(drafted(-6.0), Held::Square).expect_err("closes");
        assert!(matches!(err, SweepError::TaperClosesTheSection { .. }), "{err:?}");
        build(drafted(-5.0), Held::Square).expect("still open at the far end");
        // A twist on a section that never turns.
        let err = build(twisted(30.0), flat).expect_err("no axis");
        assert_eq!(err, SweepError::TwistOnAParallelSection);
        build(scaled(0.5), flat).expect("a parallel section can still taper");
        // A window that is nothing.
        for window in [(0.5, 0.5), (0.7, 0.3), (-0.1, 0.5), (0.0, 1.2)] {
            let err = build(SectionLaw { window, ..plain() }, Held::Square).expect_err("empty");
            assert!(matches!(err, SweepError::EmptyWindow { .. }), "{window:?}: {err:?}");
        }
        // A draft on an open wall.
        let open = crate::Wall::open(
            vec![Point2::new(-10.0, 0.0), Point2::new(10.0, 0.0)],
            2.0,
            fieldform_scene::WallSide::Centre,
            fieldform_scene::WallEnds::Square,
        )
        .unwrap();
        let err = Swept::under_law(open.clone(), up(100.0), Vector3::x(), Held::Square, drafted(5.0))
            .expect_err("no inside");
        assert_eq!(err, SweepError::DraftOnAnOpenWall);
        Swept::under_law(open, up(100.0), Vector3::x(), Held::Square, scaled(0.5))
            .expect("an open wall scales whole");
        // A law on a closed path.
        let ring: Vec<Point3<f64>> = (0..=90)
            .map(|i| {
                let a = TAU * (i % 90) as f64 / 90.0;
                Point3::new(50.0 * a.cos(), 50.0 * a.sin(), 0.0)
            })
            .collect();
        let err = Swept::under_law(square(5.0), Spine::new(ring.clone()), Vector3::x(), Held::Square, twisted(90.0))
            .expect_err("no start");
        assert_eq!(err, SweepError::LawOnAClosedPath);
        Swept::under_law(square(5.0), Spine::new(ring), Vector3::x(), Held::Square, plain())
            .expect("the plain law is fine on a ring");
        // And the fold, after tapering up: a 6 mm square fits a bend of radius 10 and its
        // double does not.
        let arc: Vec<Point3<f64>> = (0..=30)
            .map(|i| {
                let a = 0.5 * PI * i as f64 / 30.0;
                Point3::new(10.0 * a.cos(), 10.0 * a.sin(), 0.0)
            })
            .collect();
        Swept::under_law(square(6.0), Spine::new(arc.clone()), Vector3::z(), Held::Square, plain())
            .expect("fits");
        let err = Swept::under_law(square(6.0), Spine::new(arc), Vector3::z(), Held::Square, scaled(2.0))
            .expect_err("doubled, it folds");
        assert!(matches!(err, SweepError::FoldsThroughItself { .. }), "{err:?}");
    }
}
