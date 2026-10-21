// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! NURBS curves — the parametric layer's first real geometry type.
//!
//! A curve is *parametric* in the two-layer model's sense: unlike a cage, it is not edited
//! destructively but stored and re-evaluated, and downstream operations reference it by id.
//! That is what lets a pipe's radius be a parameter rather than a baked result.
//!
//! # Why NURBS rather than something simpler
//!
//! Because that is what arrives through the door. IGES entity 126 — the interchange format
//! every CAD system emits curves as — is a rational B-spline, and lines, arcs and conics all
//! degrade into it exactly. Storing anything less would mean approximating at import, which
//! is precisely the kind of hidden accuracy loss D2 exists to prevent.
//!
//! Polynomial (non-rational) curves are the same type with every weight at 1, so there is no
//! second code path for them.

use crate::id::CurveId;
use nalgebra::Point3;
use serde::{Deserialize, Serialize};

#[derive(thiserror::Error, Debug, PartialEq)]
pub enum CurveError {
    #[error("degree must be at least 1, got {0}")]
    DegreeTooLow(usize),
    #[error("a degree-{degree} curve needs at least {needed} control points, got {got}")]
    TooFewControlPoints {
        degree: usize,
        needed: usize,
        got: usize,
    },
    #[error("expected {expected} knots for {points} control points of degree {degree}, got {got}")]
    KnotCount {
        expected: usize,
        got: usize,
        points: usize,
        degree: usize,
    },
    #[error("knot vector must be non-decreasing; knot {index} goes backwards")]
    KnotsNotSorted { index: usize },
    #[error("expected {expected} weights, one per control point, got {got}")]
    WeightCount { expected: usize, got: usize },
    #[error(
        "the curve has interior knots, so it is more than one span — changing the degree of \
         one is a different problem from elevating a Bézier and is not done here"
    )]
    NotSingleSpan,
    #[error("weight {index} is {value}; weights must be positive")]
    NonPositiveWeight { index: usize, value: f64 },
    #[error("the knot vector leaves an empty parameter domain")]
    EmptyDomain,
}

/// A NURBS curve in canonical millimetres.
///
/// Serialized through [`CurveData`], so a curve read from a file has passed the same
/// validation as one built in the app.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(into = "CurveData", try_from = "CurveData")]
pub struct Curve {
    pub id: CurveId,
    pub name: String,
    degree: usize,
    control_points: Vec<Point3<f64>>,
    /// Which control points sit on a point block, if any.
    ///
    /// **A position is one thing, so it binds as one thing.** This was three independent
    /// scalars — an X variable, a Y variable, a Z variable — which let a control point be half
    /// attached to one place and half to another, a state nobody wants and every reader would
    /// have to think about. A point is the type that already means "a position", every other
    /// block in the program takes its place from one, and a curve end is no different.
    ///
    /// **The position is stored twice, deliberately, and this says which copy is the truth.**
    /// The point block is; `control_points` holds where it currently is, derived by
    /// [`Curve::apply_places`] and by nothing else.
    ///
    /// The alternative was making `control_points` a list of places, which is the shape with
    /// one description of one fact — and it would have put a graph into the signature of
    /// `tessellate`, `ends` and eighty call sites that ask a curve where it is. Those are
    /// geometry, and geometry that has to be handed the rest of the document to answer "where
    /// does this curve start" is a worse thing than a derived value with a single derivation.
    ///
    /// Empty for a curve that sits on nothing, which is almost all of them, so a document using
    /// none of this stores none of it.
    #[serde(skip)]
    on_point: Vec<Option<crate::NodeId>>,
    weights: Vec<f64>,
    knots: Vec<f64>,
    closed: bool,
}

/// How a curve is written to a file.
///
/// A mirror of [`Curve`]'s fields rather than a derive on the type itself, for two reasons:
/// the fields are private because [`Curve::new`] is what enforces that the knots, weights and
/// control points agree, and loading must go back through that check rather than around it;
/// and control points are written as flat `[x, y, z]` arrays so the format does not inherit
/// whatever shape nalgebra happens to serialize a point in.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurveData {
    pub id: CurveId,
    pub name: String,
    pub degree: usize,
    pub control_points: Vec<[f64; 3]>,
    /// Which control points sit on a block, as `[point index, block]` pairs.
    ///
    /// Pairs rather than a parallel array of options, so a curve with one placed end writes one
    /// entry instead of a null per point — and a file that places nothing writes nothing at all,
    /// which is every file written before this existed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub on_point: Vec<(usize, crate::NodeId)>,
    pub weights: Vec<f64>,
    pub knots: Vec<f64>,
    pub closed: bool,
}

impl From<Curve> for CurveData {
    fn from(c: Curve) -> Self {
        Self {
            id: c.id,
            name: c.name,
            degree: c.degree,
            control_points: c.control_points.iter().map(|p| [p.x, p.y, p.z]).collect(),
            on_point: c
                .on_point
                .iter()
                .enumerate()
                .filter_map(|(i, node)| node.map(|node| (i, node)))
                .collect(),
            weights: c.weights,
            knots: c.knots,
            closed: c.closed,
        }
    }
}

impl TryFrom<CurveData> for Curve {
    type Error = CurveError;

    fn try_from(d: CurveData) -> Result<Self, CurveError> {
        let bindings = d.on_point;
        let mut curve = Curve::new(
            d.id,
            d.name,
            d.degree,
            d.control_points
                .into_iter()
                .map(|[x, y, z]| Point3::new(x, y, z))
                .collect(),
            d.weights,
            d.knots,
            d.closed,
        )?;
        // After construction, since `new` is what fixes how many control points there are and
        // the bindings are indexed against that. An index the curve does not have is dropped
        // rather than refused: it describes a coordinate that is not there, and the geometry is
        // still perfectly good without it.
        for (i, node) in bindings {
            curve.place_on(i, Some(node));
        }
        Ok(curve)
    }
}

/// How close two curve ends must be to count as meeting, in millimetres.
///
/// **Not a sampling tolerance.** Two ends either meet or they do not, and that is a fact about
/// the geometry rather than about how finely anything is evaluated -- so a profile must not
/// close at draft quality and open at final, which is what deriving this from a sampling
/// tolerance once produced. A tenth of a micron: far below anything anybody can see, wide
/// enough to absorb float noise, and reachable only by snapping two ends together on purpose.
/// A generous figure would silently join curves somebody meant to leave apart, and the failure
/// would be a solid where they wanted two open paths.
///
/// One number, read by the loop walk, the joints, the relations and the checks on open, so
/// they can never disagree about whether two ends meet.
pub const JOINED_MM: f64 = 1e-4;

impl Curve {
    /// The same curve run the other way: points and weights reversed, the knots mirrored, so
    /// what was its end is the new copy's start and every formula written for a start applies.
    pub(crate) fn reversed(&self) -> Curve {
        let (lo, hi) = (self.knots[0], self.knots[self.knots.len() - 1]);
        Curve {
            id: self.id,
            name: self.name.clone(),
            degree: self.degree,
            control_points: self.control_points.iter().rev().copied().collect(),
            on_point: self.on_point.iter().rev().copied().collect(),
            weights: self.weights.iter().rev().copied().collect(),
            knots: self.knots.iter().rev().map(|k| lo + hi - k).collect(),
            closed: self.closed,
        }
    }

    /// The clamped start's derivative factors: `k1 = p/(u[p+1] - u[1])`,
    /// `k2 = p(p-1)/(u[p+1] - u[1])`, and the two knot spans the second derivative divides
    /// by. On a single span all of them are the Bézier values, `p`, `p(p-1)`, 1 and 1.
    pub(crate) fn start_factors(&self) -> (f64, f64, f64, f64) {
        let p = self.degree;
        let u = &self.knots;
        let span1 = u[p + 1] - u[1];
        let span2 = if p + 2 < u.len() { u[p + 2] - u[2] } else { span1 };
        (p as f64 / span1, (p * p.saturating_sub(1)) as f64 / span1, span1, span2)
    }

    /// The first and second derivatives at the start, weights included.
    ///
    /// **The rational curve's actual derivatives, not the control polygon's.** For a weighted
    /// curve the end tangent direction ignores the weights and the second derivative does not
    /// (`sketcher-spec.md` §15.4), so a G2 joint derived from the polygon alone is wrong the
    /// moment a conic is on either side. Derivatives of the homogeneous curve are exact for a
    /// clamped B-spline from its first two knot spans, and the rational ones follow from them:
    /// `R' = (Aw' - w' R)/w`, `R'' = (Aw'' - 2 w' R' - w'' R)/w`.
    pub(crate) fn start_derivatives(&self) -> (nalgebra::Vector3<f64>, nalgebra::Vector3<f64>) {
        let p = &self.control_points;
        let w = &self.weights;
        let (k1, k2, span1, span2) = self.start_factors();
        let hom = |i: usize| p[i].coords * w[i];
        let aw1 = (hom(1) - hom(0)) * k1;
        let w1 = (w[1] - w[0]) * k1;
        let r = p[0].coords;
        let r1 = (aw1 - r * w1) / w[0];
        if self.degree < 2 || p.len() < 3 {
            // A line's second derivative is nothing, and `k2` says so; said here so a
            // knot span of zero width behind it is never divided by.
            return (r1, nalgebra::Vector3::zeros());
        }
        let aw2 = ((hom(2) - hom(1)) / span2 - (hom(1) - hom(0)) / span1) * k2;
        let w2 = ((w[2] - w[1]) / span2 - (w[1] - w[0]) / span1) * k2;
        let r2 = (aw2 - r1 * (2.0 * w1) - r * w2) / w[0];
        (r1, r2)
    }

    /// The derivatives at `end`, taken in the direction that leaves the joint into the curve.
    pub(crate) fn derivatives_leaving(
        &self,
        end: crate::relation::End,
    ) -> (nalgebra::Vector3<f64>, nalgebra::Vector3<f64>) {
        match end {
            crate::relation::End::Start => self.start_derivatives(),
            crate::relation::End::End => self.reversed().start_derivatives(),
        }
    }

    /// Builds a curve, rejecting anything that isn't a well-formed NURBS.
    ///
    /// Validated at construction rather than at evaluation for the same reason [`Cage`] is:
    /// a malformed curve read from a file is expected input, and should fail once with a
    /// specific message instead of producing quiet nonsense at every later query.
    ///
    /// [`Cage`]: fieldform_cage::Cage
    pub fn new(
        id: CurveId,
        name: impl Into<String>,
        degree: usize,
        control_points: Vec<Point3<f64>>,
        weights: Vec<f64>,
        knots: Vec<f64>,
        closed: bool,
    ) -> Result<Self, CurveError> {
        if degree < 1 {
            return Err(CurveError::DegreeTooLow(degree));
        }
        if control_points.len() < degree + 1 {
            return Err(CurveError::TooFewControlPoints {
                degree,
                needed: degree + 1,
                got: control_points.len(),
            });
        }
        if weights.len() != control_points.len() {
            return Err(CurveError::WeightCount {
                expected: control_points.len(),
                got: weights.len(),
            });
        }
        for (index, &w) in weights.iter().enumerate() {
            if w <= 0.0 {
                return Err(CurveError::NonPositiveWeight { index, value: w });
            }
        }
        let expected = control_points.len() + degree + 1;
        if knots.len() != expected {
            return Err(CurveError::KnotCount {
                expected,
                got: knots.len(),
                points: control_points.len(),
                degree,
            });
        }
        for i in 1..knots.len() {
            if knots[i] < knots[i - 1] {
                return Err(CurveError::KnotsNotSorted { index: i });
            }
        }
        if knots[degree] >= knots[control_points.len()] {
            return Err(CurveError::EmptyDomain);
        }

        Ok(Self {
            id,
            name: name.into(),
            degree,
            control_points,
            // Nothing driven until something is, so a curve nobody parameterises carries no
            // extra state and writes none.
            on_point: Vec::new(),
            weights,
            knots,
            closed,
        })
    }

    /// A polyline as a degree-1 curve, which is what IGES lines and composite polylines
    /// become. Kept as a constructor rather than a separate type so everything downstream
    /// sees one curve representation.
    pub fn polyline(
        id: CurveId,
        name: impl Into<String>,
        points: Vec<Point3<f64>>,
        closed: bool,
    ) -> Result<Self, CurveError> {
        let n = points.len();
        // A clamped degree-1 knot vector is just 0, 0, 1, 2, ..., n-2, n-1, n-1.
        let mut knots = Vec::with_capacity(n + 2);
        knots.push(0.0);
        for i in 0..n {
            knots.push(i as f64);
        }
        knots.push((n.saturating_sub(1)) as f64);
        Self::new(id, name, 1, points, vec![1.0; n], knots, closed)
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn control_points(&self) -> &[Point3<f64>] {
        &self.control_points
    }

    /// The block a control point sits on, if it sits on one.
    pub fn placed_on(&self, index: usize) -> Option<crate::NodeId> {
        self.on_point.get(index).copied().flatten()
    }

    /// Puts a control point on a block, or takes it off with `None`.
    ///
    /// Out of range is ignored, for the reason `set_control_point` gives: the caller is an
    /// interactive selection, and undo can shrink a curve out from under one.
    pub fn place_on(&mut self, index: usize, node: Option<crate::NodeId>) {
        if index >= self.control_points.len() {
            return;
        }
        if self.on_point.len() != self.control_points.len() {
            self.on_point.resize(self.control_points.len(), None);
        }
        self.on_point[index] = node;
        // Nothing sits on anything any more, so the list goes rather than staying as a row of
        // `None` that every save would have to skip.
        if self.on_point.iter().all(Option::is_none) {
            self.on_point.clear();
        }
    }

    /// Every block this curve sits on, so a caller can ask what it depends on.
    pub fn sits_on(&self) -> impl Iterator<Item = crate::NodeId> + '_ {
        self.on_point.iter().flatten().copied()
    }

    /// Writes the current position of every block a control point sits on into the geometry.
    ///
    /// **The single derivation.** `control_points` is what the whole application reads and what
    /// `on_point` is the truth about, so this is the one place the two are brought into
    /// agreement — called when the graph changes and when a document is opened. A control point
    /// whose block has gone is left where it was rather than moved to the origin: the curve
    /// keeps the shape somebody drew, which is the safer of the two wrong answers.
    ///
    /// Returns whether anything moved, so a caller can skip the work that follows when nothing
    /// did.
    pub fn apply_places(&mut self, at: &dyn Fn(crate::NodeId) -> Option<Point3<f64>>) -> bool {
        let mut moved = false;
        for (i, node) in self.on_point.iter().enumerate() {
            let Some(node) = node else { continue };
            let Some(to) = at(*node) else { continue };
            if let Some(point) = self.control_points.get_mut(i)
                && (*point - to).norm() > f64::EPSILON
            {
                *point = to;
                moved = true;
            }
        }
        moved
    }

    /// Moves one control point, ignoring an index the curve does not have.
    ///
    /// The only mutation the private fields allow, and it is allowed because it cannot
    /// invalidate the curve: [`Curve::new`]'s checks are all about how many control points
    /// there are relative to the degree, the weights and the knot vector, and moving one
    /// changes none of those counts. Adding or removing a point would, which is why there is
    /// no method for it here — that has to go back through `new`.
    ///
    /// Out of range is ignored rather than panicking because the caller is an interactive
    /// selection, and undo can shrink a curve out from under one exactly as it can a cage.
    pub fn set_control_point(&mut self, index: usize, position: Point3<f64>) {
        if let Some(slot) = self.control_points.get_mut(index) {
            *slot = position;
        }
    }

    /// Sets how hard one control point pulls the curve toward itself.
    ///
    /// The weight in the rational sense — [`Curve::evaluate`] divides by the summed weights,
    /// so 1 is neutral, more pulls the curve nearer that point and less lets it drift away.
    /// It is the other thing a control point has besides a position, and the reason this curve
    /// type can hold an exact circle at all.
    ///
    /// Clamped positive rather than refused. Zero makes the point contribute nothing and can
    /// make every weight in a span underflow together, which has no curve at all as its
    /// answer; negative ones put a pole inside the domain, where the curve runs off to
    /// infinity and back. [`Curve::new`] rejects both at construction and this keeps that true
    /// afterwards, so a drag toward zero slows to a stop instead of falling off a cliff.
    pub fn set_weight(&mut self, index: usize, weight: f64) {
        if let Some(slot) = self.weights.get_mut(index) {
            *slot = weight.max(1e-6);
        }
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// True if every weight is 1, i.e. the curve is an ordinary polynomial B-spline.
    pub fn is_rational(&self) -> bool {
        self.weights.iter().any(|&w| (w - 1.0).abs() > 1e-12)
    }

    /// Whether every knot sits at one end or the other, so the curve is a single polynomial
    /// piece — a Bézier.
    ///
    /// The distinction matters because a single span is what makes raising the degree a short
    /// exact formula rather than a general B-spline problem, so it is what
    /// [`Curve::with_degree`] insists on.
    pub fn is_single_span(&self) -> bool {
        let (lo, hi) = (self.knots[0], self.knots[self.knots.len() - 1]);
        self.knots
            .iter()
            .all(|k| (k - lo).abs() < 1e-12 || (k - hi).abs() < 1e-12)
    }

    /// The same curve at a different degree.
    ///
    /// # Raising is exact; lowering is not
    ///
    /// Degree elevation of a Bézier preserves the shape *exactly* — the curve of degree `n + 1`
    /// through the blended control points is the same curve, point for point. Applied
    /// repeatedly it reaches any higher degree with nothing lost, so going 3 → 9 and shaping
    /// the result is a real workflow rather than an approximation that drifts.
    ///
    /// The blend is done in homogeneous coordinates — `(w·P, w)` rather than `P` — which is
    /// what keeps it exact for a weighted curve too, instead of correct only while every weight
    /// is 1.
    ///
    /// Lowering cannot be exact: a degree-9 curve has shapes no cubic can hold, so something
    /// must be discarded and the only question is what. This samples the existing curve at
    /// evenly spaced parameters and takes those as the new control points, which pins both
    /// endpoints exactly — a Bézier passes through its first and last — and approximates in
    /// between. Callers should say so rather than implying a round trip returns the original.
    pub fn with_degree(&self, degree: usize) -> Result<Curve, CurveError> {
        if !self.is_single_span() {
            return Err(CurveError::NotSingleSpan);
        }
        let mut points = self.control_points.clone();
        let mut weights = self.weights.clone();

        if degree > self.degree {
            for from in self.degree..degree {
                // Homogeneous, so a weighted curve elevates as exactly as an unweighted one.
                let homogeneous: Vec<(nalgebra::Vector3<f64>, f64)> = points
                    .iter()
                    .zip(&weights)
                    .map(|(p, w)| (p.coords * *w, *w))
                    .collect();
                let n = from as f64 + 1.0;
                let mut raised = Vec::with_capacity(points.len() + 1);
                raised.push(homogeneous[0]);
                for i in 1..homogeneous.len() {
                    let a = i as f64 / n;
                    let (pa, wa) = homogeneous[i - 1];
                    let (pb, wb) = homogeneous[i];
                    raised.push((pa * a + pb * (1.0 - a), wa * a + wb * (1.0 - a)));
                }
                raised.push(homogeneous[homogeneous.len() - 1]);

                points = raised
                    .iter()
                    .map(|(p, w)| Point3::from(p / w.max(1e-12)))
                    .collect();
                weights = raised.iter().map(|(_, w)| *w).collect();
            }
        } else if degree < self.degree {
            let (lo, hi) = self.domain();
            points = (0..=degree)
                .map(|i| self.evaluate(lo + (hi - lo) * i as f64 / degree as f64))
                .collect();
            // The sampled points are positions on the curve, not weighted pulls toward it, so
            // carrying the old weights across would apply them to the wrong thing entirely.
            weights = vec![1.0; points.len()];
        } else {
            return Ok(self.clone());
        }

        let knots: Vec<f64> = std::iter::repeat_n(0.0, degree + 1)
            .chain(std::iter::repeat_n(1.0, degree + 1))
            .collect();
        let mut raised =
            Curve::new(self.id, self.name.clone(), degree, points, weights, knots, self.closed)?;

        // **The ends keep what they sit on.** Changing degree builds a new curve, which starts
        // sitting on nothing -- so a start attached to a point forgot its point the moment
        // somebody turned a line into a curve, silently, and there was nothing on the row to
        // say it had happened.
        //
        // Only the ends, because only the ends survive the change: elevation keeps the first
        // and last control points exactly, and reduction samples the curve at its own domain
        // ends, so both agree about where the curve begins and stops. The points in between are
        // a different set entirely -- there may not even be the same number of them -- and a
        // placement carried onto one would be attaching a block to a point nobody chose.
        let last = self.control_points.len().saturating_sub(1);
        let new_last = raised.control_points.len().saturating_sub(1);
        if let Some(node) = self.placed_on(0) {
            raised.place_on(0, Some(node));
        }
        if last > 0
            && new_last > 0
            && let Some(node) = self.placed_on(last)
        {
            raised.place_on(new_last, Some(node));
        }
        Ok(raised)
    }

    /// The same curve under a new id and name.
    ///
    /// An id is an identity, so a copy needs its own — two curves answering to one id are one
    /// curve as far as everything downstream is concerned, and a pipe swept along "either of
    /// them" is a pipe that moves when the wrong one does. It goes back through
    /// [`Curve::new`] rather than mutating a clone, so a copy passes the same validation the
    /// original did instead of inheriting an assumption about it.
    pub fn copy_as(&self, id: CurveId, name: impl Into<String>) -> Result<Curve, CurveError> {
        Curve::new(
            id,
            name,
            self.degree,
            self.control_points.clone(),
            self.weights.clone(),
            self.knots.clone(),
            self.closed,
        )
    }

    /// Tessellates finely enough that no joint turns by more than `max_turn_degrees`.
    ///
    /// # Why an angle and not a tolerance
    ///
    /// [`Curve::tessellate`] answers "how far is the polyline from the curve", which is the
    /// right question for a mesh and the wrong one for a pipe. A tube's surface is every point
    /// at `radius` from the spine, so its normal points away from whichever segment is nearest
    /// and *turns a corner at every joint* — and shading is the normal, so the visible defect
    /// is the size of that corner, not the distance to the curve.
    ///
    /// The two are related by the curvature rather than by a constant:
    ///
    /// ```text
    /// turn per joint  ≈  2 · sqrt(2 · tolerance / radius_of_curvature)
    /// ```
    ///
    /// The square root is what makes a position tolerance the wrong control: halving it cuts
    /// the turn by only 30%. Measured on a 310 mm cubic by
    /// `fieldform-field/examples/spine_angle.rs`:
    ///
    /// | tolerance | segments | worst turn |
    /// |---|---|---|
    /// | 0.35 mm | 22 | 6.35° |
    /// | 0.10 mm | 36 | 3.96° |
    /// | 0.02 mm | 97 | 1.19° |
    /// | 0.005 mm | 193 | 0.60° |
    ///
    /// 0.35 mm is what the viewport used to ask for — a spine well within half a millimetre of
    /// the true curve, and six degrees of normal discontinuity at every joint, which is bands
    /// across the tube and, through a blend, across whatever the tube was blended into.
    ///
    /// Halving until the angle is met rather than solving for it, because the curvature varies
    /// along the curve and the worst joint is the one that matters. A handful of tessellations
    /// at rebuild time is nothing next to being visibly wrong; `floor_mm` stops a curve with a
    /// genuine cusp — where no tolerance ever satisfies the angle — from subdividing forever.
    pub fn tessellate_to_angle(&self, max_turn_degrees: f64, floor_mm: f64) -> Vec<Point3<f64>> {
        let mut tolerance = (self.length(0.5) * 1e-3).max(floor_mm);
        let mut points = self.tessellate(tolerance);
        for _ in 0..12 {
            if worst_turn_degrees(&points) <= max_turn_degrees || tolerance <= floor_mm {
                break;
            }
            tolerance = (tolerance * 0.25).max(floor_mm);
            points = self.tessellate(tolerance);
        }
        points
    }

    /// Where the curve begins and where it stops.
    ///
    /// **Evaluated, not read off the control points.** A clamped NURBS passes through its first
    /// and last control points and the two answers agree there, but an unclamped one does not —
    /// and a block built between a curve's ends should sit where the curve visibly is. Cheap
    /// either way: two evaluations, against a tessellation of the whole thing.
    ///
    /// A closed curve gives the same point twice, which is a true answer to the question and
    /// leaves the caller to decide what to do about it.
    pub fn ends(&self) -> (Point3<f64>, Point3<f64>) {
        let (start, end) = self.domain();
        (self.evaluate(start), self.evaluate(end))
    }

    /// The valid parameter range, `[knots[degree], knots[n]]`.
    pub fn domain(&self) -> (f64, f64) {
        (self.knots[self.degree], self.knots[self.control_points.len()])
    }

    /// The knot span index containing `t`.
    ///
    /// Binary search over the non-empty spans, with the domain's upper end clamped into the
    /// last one — at `t == t_max` the half-open span convention would otherwise fall off the
    /// end and evaluate nothing.
    fn span(&self, t: f64) -> usize {
        let n = self.control_points.len();
        if t >= self.knots[n] {
            // Walk back over any repeated final knots to the last span with width.
            let mut i = n - 1;
            while i > self.degree && self.knots[i] >= self.knots[n] {
                i -= 1;
            }
            return i;
        }
        let (mut lo, mut hi) = (self.degree, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if t < self.knots[mid] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo - 1
    }

    /// The `degree + 1` non-zero B-spline basis functions at `t`, by Cox-de Boor.
    fn basis(&self, span: usize, t: f64) -> Vec<f64> {
        let p = self.degree;
        let mut n = vec![0.0; p + 1];
        let mut left = vec![0.0; p + 1];
        let mut right = vec![0.0; p + 1];
        n[0] = 1.0;

        for j in 1..=p {
            left[j] = t - self.knots[span + 1 - j];
            right[j] = self.knots[span + j] - t;
            let mut saved = 0.0;
            for r in 0..j {
                let denom = right[r + 1] + left[j - r];
                // A zero denominator means a repeated knot makes this basis function vanish;
                // the standard algorithm's convention is that the term contributes nothing.
                let temp = if denom.abs() < f64::EPSILON {
                    0.0
                } else {
                    n[r] / denom
                };
                n[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            n[j] = saved;
        }
        n
    }

    /// The point on the curve at parameter `t`, clamped to the domain.
    ///
    /// Evaluated in homogeneous coordinates and projected, which is what makes this exact for
    /// rational curves (arcs, circles, conics) rather than merely close.
    pub fn evaluate(&self, t: f64) -> Point3<f64> {
        let (lo, hi) = self.domain();
        let t = t.clamp(lo, hi);
        let span = self.span(t);
        let basis = self.basis(span, t);

        let mut numerator = nalgebra::Vector3::zeros();
        let mut denominator = 0.0;
        for (i, &b) in basis.iter().enumerate() {
            let index = span - self.degree + i;
            let w = self.weights[index] * b;
            numerator += self.control_points[index].coords * w;
            denominator += w;
        }

        if denominator.abs() < f64::EPSILON {
            // Only reachable if every contributing weight underflowed; falling back to the
            // nearest control point beats emitting NaN into the geometry pipeline.
            return self.control_points[span - self.degree];
        }
        Point3::from(numerator / denominator)
    }

    /// Samples the curve as a polyline whose chordal deviation from the true curve is at most
    /// `tolerance_mm`.
    ///
    /// Adaptive rather than uniform, and the reason is the same one that governs the cage's
    /// limit surface: accuracy should be a **stated bound**, not a side effect of a sample
    /// count someone picked. A caller asking for 0.01 mm gets 0.01 mm on a tight arc and only
    /// a handful of segments on a straight run.
    pub fn tessellate(&self, tolerance_mm: f64) -> Vec<Point3<f64>> {
        let tolerance = tolerance_mm.max(1e-9);
        let (lo, hi) = self.domain();

        // Break at interior knots first: continuity can drop there, and bisection alone can
        // step straight over a corner without ever sampling near it.
        let mut breaks = vec![lo];
        for &k in &self.knots {
            if k > lo && k < hi && *breaks.last().expect("seeded with lo") < k {
                breaks.push(k);
            }
        }
        breaks.push(hi);

        // Then cut each span uniformly before refining. Pure bisection asks "is the midpoint
        // near the chord?", which an S-shaped span answers *yes* while bulging hard either
        // side of it — the midpoint sits on the chord by symmetry. Seeding past that costs a
        // few evaluations and removes a whole class of silently-too-coarse output. Degree 1
        // needs none of this: each span is already exactly a straight segment.
        let seeds = if self.degree == 1 {
            1
        } else {
            (self.degree * 2).max(4)
        };

        let mut out = vec![self.evaluate(lo)];
        for window in breaks.windows(2) {
            let (a, b) = (window[0], window[1]);
            for i in 0..seeds {
                let t0 = a + (b - a) * i as f64 / seeds as f64;
                let t1 = a + (b - a) * (i + 1) as f64 / seeds as f64;
                self.refine(t0, t1, tolerance, 0, &mut out);
                out.push(self.evaluate(t1));
            }
        }
        out
    }

    /// Bisects `[t0, t1]` until the span is flat to within `tolerance`.
    ///
    /// Flatness is judged from several interior samples rather than the midpoint alone, for
    /// the symmetry reason described in [`Self::tessellate`]. `depth` caps recursion well
    /// beyond what any sane tolerance needs, so a cusp or a degenerate span cannot spin
    /// forever.
    fn refine(&self, t0: f64, t1: f64, tolerance: f64, depth: usize, out: &mut Vec<Point3<f64>>) {
        const MAX_DEPTH: usize = 20;
        const PROBES: usize = 4;

        let p0 = self.evaluate(t0);
        let p1 = self.evaluate(t1);
        let flat = (1..PROBES).all(|i| {
            let t = t0 + (t1 - t0) * i as f64 / PROBES as f64;
            distance_to_segment(self.evaluate(t), p0, p1) <= tolerance
        });

        if depth >= MAX_DEPTH || flat {
            return;
        }
        let mid = 0.5 * (t0 + t1);
        self.refine(t0, mid, tolerance, depth + 1, out);
        out.push(self.evaluate(mid));
        self.refine(mid, t1, tolerance, depth + 1, out);
    }

    /// Total length of the curve, measured on a polyline accurate to `tolerance_mm`.
    pub fn length(&self, tolerance_mm: f64) -> f64 {
        self.tessellate(tolerance_mm)
            .windows(2)
            .map(|w| (w[1] - w[0]).norm())
            .sum()
    }
}

/// The largest turn between consecutive segments of a polyline, in degrees.
///
/// Zero for anything with fewer than three points, which has no joint to turn at.
fn worst_turn_degrees(points: &[Point3<f64>]) -> f64 {
    let mut worst: f64 = 0.0;
    for w in points.windows(3) {
        let a = (w[1] - w[0]).try_normalize(1e-12);
        let b = (w[2] - w[1]).try_normalize(1e-12);
        if let (Some(a), Some(b)) = (a, b) {
            worst = worst.max(a.dot(&b).clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    worst
}

/// Unsigned distance from `p` to the segment `a`-`b`.
fn distance_to_segment(p: Point3<f64>, a: Point3<f64>, b: Point3<f64>) -> f64 {
    let ab = b - a;
    let len2 = ab.norm_squared();
    if len2 < f64::EPSILON {
        return (p - a).norm();
    }
    let t = ((p - a).dot(&ab) / len2).clamp(0.0, 1.0);
    (p - (a + ab * t)).norm()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The curve in `fieldform_nurbs_curve_2000mm.igs`: a cubic Bezier as a NURBS, with
    /// endpoints 2000 mm apart.
    fn test_curve() -> Curve {
        Curve::new(
            CurveId::new(),
            "NURBS",
            3,
            vec![
                Point3::new(-1000.0, 0.0, 0.0),
                Point3::new(-333.333333, 500.0, 0.0),
                Point3::new(333.333333, -500.0, 0.0),
                Point3::new(1000.0, 0.0, 0.0),
            ],
            vec![1.0; 4],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            false,
        )
        .unwrap()
    }

    #[test]
    fn a_clamped_curve_interpolates_its_end_control_points() {
        let c = test_curve();
        let (lo, hi) = c.domain();
        let start = c.evaluate(lo);
        let end = c.evaluate(hi);
        assert!((start - Point3::new(-1000.0, 0.0, 0.0)).norm() < 1e-9, "{start:?}");
        assert!((end - Point3::new(1000.0, 0.0, 0.0)).norm() < 1e-9, "{end:?}");
    }

    /// With uniform weights and a single Bezier span, the midpoint has the closed-form value
    /// `(P0 + 3P1 + 3P2 + P3) / 8`. Checking against arithmetic done by hand is what makes
    /// this a test of the basis functions rather than of itself.
    #[test]
    fn the_midpoint_matches_the_bezier_closed_form() {
        let c = test_curve();
        let mid = c.evaluate(0.5);
        let cp = c.control_points();
        let expected = Point3::from(
            (cp[0].coords + cp[1].coords * 3.0 + cp[2].coords * 3.0 + cp[3].coords) / 8.0,
        );
        assert!((mid - expected).norm() < 1e-9, "{mid:?} vs {expected:?}");
        // This particular curve is antisymmetric, so the midpoint is the origin.
        assert!(mid.coords.norm() < 1e-6, "expected the origin, got {mid:?}");
    }

    #[test]
    fn the_basis_functions_form_a_partition_of_unity() {
        let c = test_curve();
        let (lo, hi) = c.domain();
        for i in 0..=20 {
            let t = lo + (hi - lo) * i as f64 / 20.0;
            let span = c.span(t);
            let sum: f64 = c.basis(span, t).iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "at t={t} the basis summed to {sum}");
        }
    }

    /// A NURBS with the right weights is a circle *exactly*, which is the property that makes
    /// rational curves worth carrying. A quarter circle of radius r uses weights
    /// `1, 1/sqrt(2), 1` — if the projection were wrong, every sample but the endpoints would
    /// drift off the radius.
    #[test]
    fn a_rational_quarter_circle_stays_exactly_on_its_radius() {
        let r = 25.0;
        let w = 1.0 / 2.0_f64.sqrt();
        let c = Curve::new(
            CurveId::new(),
            "arc",
            2,
            vec![
                Point3::new(r, 0.0, 0.0),
                Point3::new(r, r, 0.0),
                Point3::new(0.0, r, 0.0),
            ],
            vec![1.0, w, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            false,
        )
        .unwrap();

        assert!(c.is_rational());
        for i in 0..=32 {
            let t = i as f64 / 32.0;
            let p = c.evaluate(t);
            assert!(
                (p.coords.norm() - r).abs() < 1e-12,
                "at t={t} radius was {}, expected {r}",
                p.coords.norm()
            );
        }
    }

    /// A Bézier and a Bézier of the same curve at one higher degree are the *same curve*, not
    /// a curve and a good approximation of it. That is the whole reason raising the degree is a
    /// safe thing to offer: a person going 3 → 5 → 9 to get more control points must not be
    /// paying for each step with a shape that drifts.
    #[test]
    fn raising_the_degree_does_not_move_the_curve() {
        let id = CurveId::new();
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 20.0, 0.0),
            Point3::new(30.0, -5.0, 4.0),
            Point3::new(50.0, 0.0, 0.0),
        ];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let cubic =
            Curve::new(id, "c", 3, points, vec![1.0; 4], knots, false).unwrap();

        let mut raised = cubic.clone();
        for degree in [5usize, 7, 9] {
            raised = raised.with_degree(degree).unwrap();
            assert_eq!(raised.degree(), degree);
            assert_eq!(raised.control_points().len(), degree + 1);
            for i in 0..=40 {
                let t = i as f64 / 40.0;
                let moved = (raised.evaluate(t) - cubic.evaluate(t)).norm();
                assert!(moved < 1e-9, "degree {degree} moved the curve by {moved} at t = {t}");
            }
        }
    }

    /// The same, for a weighted curve. Elevation is done in homogeneous coordinates precisely
    /// so this holds — blending the positions alone is exact only while every weight is 1, and
    /// would be silently wrong on the one kind of curve that most needs its shape preserved.
    #[test]
    fn raising_the_degree_of_a_weighted_curve_does_not_move_it_either() {
        let id = CurveId::new();
        let points = vec![
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
        ];
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        // The classic rational quarter circle.
        let arc = Curve::new(id, "arc", 2, points, vec![1.0, w, 1.0], knots, false).unwrap();

        let raised = arc.with_degree(5).unwrap();
        for i in 0..=40 {
            let t = i as f64 / 40.0;
            let moved = (raised.evaluate(t) - arc.evaluate(t)).norm();
            assert!(moved < 1e-9, "the arc moved by {moved} at t = {t}");
            // And it is still a circle, which is the property the weights were carrying.
            assert!((raised.evaluate(t).coords.norm() - 10.0).abs() < 1e-9);
        }
    }

    /// Lowering discards shape, and says so by keeping the ends rather than pretending the rest
    /// survived. A test asserting a round trip returned the original would be asserting
    /// something false.
    #[test]
    fn lowering_the_degree_keeps_the_ends_and_approximates_between_them() {
        let id = CurveId::new();
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 30.0, 0.0),
            Point3::new(20.0, -30.0, 0.0),
            Point3::new(30.0, 30.0, 0.0),
            Point3::new(40.0, 0.0, 0.0),
        ];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let quartic = Curve::new(id, "q", 4, points, vec![1.0; 5], knots, false).unwrap();

        let cubic = quartic.with_degree(3).unwrap();
        assert_eq!(cubic.degree(), 3);
        assert_eq!(cubic.control_points().len(), 4);
        assert!((cubic.evaluate(0.0) - quartic.evaluate(0.0)).norm() < 1e-9);
        assert!((cubic.evaluate(1.0) - quartic.evaluate(1.0)).norm() < 1e-9);
    }

    /// A curve with interior knots is more than one span, and elevating one is a different
    /// problem. Refused by name rather than quietly producing a curve of the right degree and
    /// the wrong shape.
    #[test]
    fn a_multi_span_curve_refuses_to_change_degree() {
        let id = CurveId::new();
        let line = Curve::polyline(
            id,
            "poly",
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(10.0, 0.0, 0.0),
                Point3::new(20.0, 0.0, 0.0),
            ],
            false,
        )
        .unwrap();
        assert!(!line.is_single_span());
        assert!(matches!(line.with_degree(3), Err(CurveError::NotSingleSpan)));
    }

    /// The tessellation's contract is a bound, so test the bound: every sampled midpoint
    /// between consecutive output points must lie within tolerance of the chord.
    #[test]
    fn tessellation_honours_its_stated_tolerance() {
        let c = test_curve();
        for &tol in &[10.0, 1.0, 0.05] {
            let poly = c.tessellate(tol);
            assert!(poly.len() >= 2);

            // Walk the curve densely and check each sample against the nearest chord.
            let (lo, hi) = c.domain();
            let mut worst: f64 = 0.0;
            for i in 0..=2000 {
                let t = lo + (hi - lo) * i as f64 / 2000.0;
                let p = c.evaluate(t);
                let d = poly
                    .windows(2)
                    .map(|w| distance_to_segment(p, w[0], w[1]))
                    .fold(f64::INFINITY, f64::min);
                worst = worst.max(d);
            }
            assert!(
                worst <= tol,
                "tolerance {tol} mm was violated: worst deviation {worst} mm over {} segments",
                poly.len() - 1
            );
        }
    }

    #[test]
    fn a_tighter_tolerance_costs_more_segments() {
        let c = test_curve();
        let coarse = c.tessellate(100.0).len();
        let fine = c.tessellate(0.01).len();
        assert!(fine > coarse, "coarse {coarse}, fine {fine}");
    }

    /// A straight line needs no refinement however tight the tolerance — the adaptive step
    /// must key off deviation, not off parameter range.
    #[test]
    fn a_straight_line_stays_two_points() {
        let c = Curve::polyline(
            CurveId::new(),
            "line",
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(100.0, 0.0, 0.0)],
            false,
        )
        .unwrap();
        assert_eq!(c.tessellate(1e-6).len(), 2);
        assert!((c.length(1e-6) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn malformed_curves_are_rejected_with_a_specific_error() {
        let id = CurveId::new();
        let pts = vec![Point3::origin(); 4];

        assert_eq!(
            Curve::new(id, "x", 3, pts.clone(), vec![1.0; 4], vec![0.0; 7], false),
            Err(CurveError::KnotCount {
                expected: 8,
                got: 7,
                points: 4,
                degree: 3
            })
        );
        assert_eq!(
            Curve::new(id, "x", 3, vec![Point3::origin(); 2], vec![1.0; 2], vec![0.0; 6], false),
            Err(CurveError::TooFewControlPoints {
                degree: 3,
                needed: 4,
                got: 2
            })
        );
        assert_eq!(
            Curve::new(
                id,
                "x",
                3,
                pts.clone(),
                vec![1.0, 1.0, -1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                false
            ),
            Err(CurveError::NonPositiveWeight {
                index: 2,
                value: -1.0
            })
        );
        // knots[degree] == knots[n] leaves nothing to evaluate over, which is a malformed
        // curve rather than a zero-length one.
        assert_eq!(
            Curve::new(
                id,
                "x",
                3,
                pts,
                vec![1.0; 4],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                false
            )
            .map(|c| c.domain()),
            Err(CurveError::EmptyDomain)
        );
    }
}


#[cfg(test)]
mod ends_are_exact {
    use super::*;

    /// A curve's tessellated ends are its first and last control points, to the bit.
    ///
    /// **What snapping rests on.** Bringing two curves together copies one endpoint onto the
    /// other exactly, and `profile_loop` then walks from one to the next by comparing the ends of
    /// their *tessellations* — which are evaluated, not copied. If evaluation drifted, two ends
    /// that had been made identical would come back merely close, and a profile would close or
    /// not depending on arithmetic nobody can see.
    ///
    /// Measured rather than assumed, and the measurement is zero at every offered degree: a
    /// clamped knot vector puts the whole of the basis on the end control point, so the sum is
    /// that point and nothing is added to it. `JOINED_MM` is therefore pure margin rather than a
    /// working tolerance — worth knowing, because it means the join is an equality test wearing
    /// a tolerance's clothes, and a future change here would quietly promote it to a real one.
    #[test]
    fn a_curves_tessellated_ends_are_its_end_control_points() {
        for degree in [1usize, 3, 5, 7, 9] {
            let points: Vec<Point3<f64>> = (0..=degree)
                .map(|i| {
                    let t = i as f64 / degree as f64;
                    // Deliberately not round numbers: exactness on 1.0 and 0.0 would prove
                    // nothing about a curve somebody actually drew.
                    Point3::new(37.219 * t, 12.5 * (t * 3.1).sin(), -4.75 * t * t)
                })
                .collect();
            let knots: Vec<f64> = std::iter::repeat_n(0.0, degree + 1)
                .chain(std::iter::repeat_n(1.0, degree + 1))
                .collect();
            let curve = Curve::new(
                CurveId::new(),
                "probe".to_string(),
                degree,
                points.clone(),
                vec![1.0; points.len()],
                knots,
                false,
            )
            .expect("a clamped single span");

            let drawn = curve.tessellate(1e-3);
            assert_eq!(
                drawn.first(),
                points.first(),
                "degree {degree} did not start where its first control point is"
            );
            assert_eq!(
                drawn.last(),
                points.last(),
                "degree {degree} did not end where its last control point is"
            );
        }
    }
}

#[cfg(test)]
mod placed_control_points {
    use super::*;

    fn line() -> Curve {
        Curve::polyline(
            CurveId::new(),
            "Line".to_string(),
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 0.0, 0.0)],
            false,
        )
        .expect("two points make a line")
    }

    /// **A control point follows the block it sits on**, as one position rather than three.
    ///
    /// It bound as three independent scalars first — an X variable, a Y, a Z — which let an end
    /// be half attached to one place and half to another. A position is one thing, and a point
    /// is the type that already means one.
    #[test]
    fn a_placed_control_point_follows_its_block() {
        let node = crate::NodeId::new();
        let mut curve = line();
        curve.place_on(1, Some(node));
        assert_eq!(curve.placed_on(1), Some(node));
        assert_eq!(curve.sits_on().collect::<Vec<_>>(), vec![node]);

        let moved = curve.apply_places(&|asked| {
            (asked == node).then_some(Point3::new(4.0, 5.0, 6.0))
        });
        assert!(moved, "it moved, and says so");
        assert_eq!(curve.control_points()[1], Point3::new(4.0, 5.0, 6.0));
        // The other end is untouched: one control point is placed, not the curve.
        assert_eq!(curve.control_points()[0], Point3::new(0.0, 0.0, 0.0));

        // Applying the same position again is not a move, so a caller can skip what follows.
        assert!(!curve.apply_places(&|_| Some(Point3::new(4.0, 5.0, 6.0))));
    }

    /// A block that has gone leaves the curve where it was, rather than at the origin.
    ///
    /// The safer of the two wrong answers: the shape somebody drew survives, where collapsing
    /// the end to zero would silently redraw their curve.
    #[test]
    fn a_missing_block_leaves_the_shape_alone() {
        let mut curve = line();
        curve.place_on(1, Some(crate::NodeId::new()));
        curve.apply_places(&|_| None);
        assert_eq!(curve.control_points()[1], Point3::new(10.0, 0.0, 0.0));
    }

    /// Taking a point off a block keeps where it got to, and stores nothing.
    #[test]
    fn unplacing_keeps_the_position_and_stores_nothing() {
        let node = crate::NodeId::new();
        let mut curve = line();
        curve.place_on(1, Some(node));
        curve.apply_places(&|_| Some(Point3::new(-3.0, 0.0, 0.0)));
        curve.place_on(1, None);
        assert_eq!(curve.placed_on(1), None);
        assert_eq!(curve.control_points()[1].x, -3.0, "it keeps where it got to");

        // Sitting on nothing, so nothing is written — which is every curve in every file that
        // existed before this.
        let data: CurveData = curve.into();
        assert!(data.on_point.is_empty());
    }

    /// **Changing degree keeps what the ends sit on.**
    ///
    /// Reported: turning a line into a degree-2 curve forgot that its start was on a point.
    /// `with_degree` builds a new curve, and a new curve sits on nothing — so the attachment
    /// went silently, with nothing on the row to say it had.
    ///
    /// Only the ends carry across, and that is not a shortcut: elevation keeps the first and
    /// last control points exactly, reduction samples the curve at its own domain ends, and the
    /// points in between are a different set — often a different *number* — so a placement moved
    /// onto one would attach a block to a point nobody chose.
    #[test]
    fn changing_degree_keeps_what_the_ends_sit_on() {
        let start = crate::NodeId::new();
        let end = crate::NodeId::new();
        let mut curve = line();
        curve.place_on(0, Some(start));
        curve.place_on(1, Some(end));

        let raised = curve.with_degree(2).expect("a line elevates");
        assert_eq!(raised.control_points().len(), 3, "degree 2 has three points");
        assert_eq!(raised.placed_on(0), Some(start));
        assert_eq!(raised.placed_on(2), Some(end), "the *new* last point, not the old index");
        assert_eq!(raised.placed_on(1), None, "and the one in between sits on nothing");

        // And back down again, without losing them on the way.
        let lowered = raised.with_degree(1).expect("and reduces");
        assert_eq!(lowered.placed_on(0), Some(start));
        assert_eq!(lowered.placed_on(1), Some(end));
    }

    /// A placement survives a round trip, and one naming a point the curve lacks is dropped.
    #[test]
    fn a_placement_round_trips_and_a_stale_index_is_dropped() {
        let node = crate::NodeId::new();
        let mut curve = line();
        curve.place_on(0, Some(node));
        let data: CurveData = curve.into();
        assert_eq!(data.on_point, vec![(0, node)]);
        let back = Curve::try_from(data).expect("reads back");
        assert_eq!(back.placed_on(0), Some(node));

        // A file naming a control point this curve does not have describes something that is
        // not there; the geometry is still good, so the placement goes rather than the curve.
        let mut stale: CurveData = line().into();
        stale.on_point = vec![(9, node)];
        let read = Curve::try_from(stale).expect("still a good curve");
        assert_eq!(read.sits_on().count(), 0);
    }
}
