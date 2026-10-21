// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A curve given a thickness, as a signed distance in its own plane.
//!
//! # The other half of what a profile-based block can read
//!
//! [`Profile`] answers for a closed loop, which has an inside. An open curve has none, and
//! `profile.rs` says what it wants instead: *a thickness, and becomes a rib rather than a solid.*
//! This is that operation, written once, so an extrude, a revolve and a sweep can each read
//! either kind and change nothing but the type of the thing they ask
//! (the outline-wall-spec design note).
//!
//! # A wall is the union of its fibres
//!
//! Stand at a point of the curve and reach across it by the wall's thickness: that short
//! straight piece is a *fibre*, and the wall is every fibre of every point. On a polyline the
//! fibres of one segment make a rectangle, and at a joint they fan out, on the side the turn
//! opens away from, into a sector — which is all a wall is, and it needs no coordinates to say.
//!
//! So a query asks one thing of every segment: *is the foot of my perpendicular on you, and
//! which side of you am I?* The nearest foot on the wall's side says how deep in the wall's
//! band the point is; the nearest foot on the other side says how far it is from the curve
//! itself, which is the wall's back face. The lesser of those two readings is the distance.
//!
//! # The faces that are not a fibre's tip
//!
//! That covers every face but one kind: **the fibre itself, wherever it is the last one.** The
//! two at the curve's ends are the obvious case, and a square end is exactly that. The others
//! are inside the curve, wherever a run's end fibre is not buried in its neighbour — past a
//! corner sharper than a right angle on the wall's side, beside a run too short to cover it,
//! and all round a bend tighter than the wall is thick. Those are found once, when the wall is
//! built, by cutting every run's two end fibres against everything else the wall is made of
//! and keeping what is left bare.
//!
//! The long edges are cut the same way, for a query *inside* the material. There the feet say
//! only how deep a point is in the nearest run's band, and that band may stop a hair away with
//! a neighbour's edge the real surface beyond it — so the depth is measured to the boundary
//! itself, which is what is left of every edge once the rest of the wall has been cut from it.
//!
//! **The first build had none of this, and was wrong wherever it was needed**: it read up to a
//! whole thickness too far across a bare end fibre and jumped there, and it read which side a
//! joint's fan was on from the run after it, which holds only up to a right angle. Neither was
//! caught by a table whose every corner was 90° or less and whose every bend was wider than its
//! wall. Both were found by a review that fuzzed against the pieces written out the long way,
//! and that check, once kept, found the depth.
//! `a_wall_is_the_union_of_its_fibres_whatever_the_chain` is it.
//!
//! # The inside of a corner sharper than a right angle is cut back to a mitre
//!
//! Up to a right angle the inside of a corner looks after itself: each run's band ends buried
//! in the other's, and where their inner edges cross is a plain mitre. Past it, a band left to
//! carry on to its own run's end would cross the other run and come out on the far side of the
//! curve — material on the side nobody asked for. So at such a joint each of the two bands is
//! cut by the line of the run it meets, and keeps only what is on that run's wall side. Ruled
//! by the owner 2026-10-02, after the first build left the triangle in.
//!
//! It is a one-sided wall's rule and no other's. A wall about its curve, on the inside of a
//! sharp corner, is everything within half a thickness of the curve, and nothing of it is on
//! a side it should not be. And it is cut by the *line* of the next run, not by the run: a
//! corner into a run too short to reach across the band chamfers that band rather than
//! notching it, which is what mitring two lines means.
//!
//! # Why not tube coordinates, which is what a sweep does
//!
//! The spec asked for the sweep's construction one dimension down: find the *nearest* point of
//! the curve, read a lateral coordinate and an arc length there, and treat the wall as a
//! rectangle in those two. `examples/wall_construction.rs` measures both over the same chains,
//! and its table is beside §3 of the spec. Three things separate them, all of them about
//! one-sided and square-ended walls:
//!
//! - **Arc length is the only coordinate that stretches.** The lateral one is a distance and
//!   moves at unit rate everywhere; only the arc length runs fast on the inside of a bend.
//!   Dividing the whole field by the worst stretch anywhere on the curve makes every wall read
//!   short — and refusing past `κ·r = 1` refuses every sharp corner on the wall's side, where
//!   the wall itself is an ordinary mitre.
//! - **The nearest strand does not know whose wall a point is in.** Where a curve's left faces
//!   its own right — a spiral, a hook — the side read off the nearest strand flips across the
//!   line midway between them, and the field jumps there. A jump is a claim of room that is
//!   not there, which is the one thing the contract forbids.
//! - **Fibres need neither.** Each foot is judged on its own segment, so nothing is divided and
//!   nothing is refused, and outside its own material the reading is the Euclidean distance.
//!
//! # A closed loop is asked through the distance it already answers
//!
//! A loop has a signed distance, so a wall on it is that distance moved and folded:
//! `|d − m| − t/2`. No fibres, no ends, and the fill rule is the profile's own.

use fieldform_scene::{WallEnds, WallSide, JOINED_MM};
use nalgebra::{Point2, Vector2};

use crate::{Normalization, Profile};

/// Why a curve could not become a wall. Each names the number to change.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum WallError {
    #[error(
        "a wall needs a curve with some length to run along, and this one has {0} distinct \
         point(s) after tessellation"
    )]
    TooFewPoints(usize),
    #[error("a wall needs a thickness above zero, and this one is {0} mm")]
    NoThickness(f64),
    #[error("a wall's curve has to be made of numbers, and point {0} of this one is not")]
    PointNotFinite(usize),
    #[error(
        "a wall's curve must be flat, and this one strays {0:.3} mm from the plane it is being \
         read in. A thickness is measured across the curve, in its plane, and a curve that \
         leaves the plane has no one direction across it."
    )]
    NotPlanar(f64),
}

/// One straight piece of an open chain: which way it runs and how far.
#[derive(Debug, Clone, Copy)]
struct Run {
    along: Vector2<f64>,
    length: f64,
    /// Which side the joint at this run's start fans out on: `+1` left, `-1` right.
    ///
    /// **The side the turn opens away from, read from the turn itself** — the cross of the two
    /// runs — and never from where a query sits beside one of them. Past a right angle the fan
    /// reaches round behind the next run's own line, so a query in it is on the *other* side of
    /// that run from the side the fan is on. Unread on the first run, which has no joint.
    fan: f64,
    /// How far that fan reaches: the band's extent on the side it is on, and nought for a wall
    /// that lies wholly on the other side, whose joint is then a bare corner of the curve.
    reach: f64,
    /// Whether this run's band is cut back, at its start, by the line of the run before it.
    ///
    /// Set where the curve turns *toward* a one-sided wall by more than a right angle. There a
    /// band left to run to its own end would cross the other run and come out on the far side
    /// of the curve, so each of the two is cut by the other's line and keeps what is on that
    /// run's wall side: a mitre.
    mitre_start: bool,
    /// The same at its end, by the line of the run after it.
    mitre_end: bool,
}

/// A straight stretch of the wall's own boundary, as its two ends.
type Face = (Point2<f64>, Point2<f64>);

#[derive(Debug, Clone)]
enum Shape {
    /// A chain with two ends, without its first point repeated.
    ///
    /// `faces` and `sides` are the wall's flat boundary, found once: the bare stretches of
    /// every run's two end fibres, and of its two long edges. See [`bare_boundary`].
    Open {
        points: Vec<Point2<f64>>,
        runs: Vec<Run>,
        ends: WallEnds,
        faces: Vec<Face>,
        sides: Vec<Face>,
    },
    /// A loop, read through its own signed distance.
    ///
    /// `left_is_inside` is the loop's winding, taken from its signed area when the wall is
    /// built: walking a counter-clockwise loop, the inside is on the left.
    Closed { profile: Profile, left_is_inside: bool },
}

/// A curve given a thickness and a side: the band of material along it.
#[derive(Debug, Clone)]
pub struct Wall {
    shape: Shape,
    thickness: f64,
    side: WallSide,
    /// The wall's own distance on a grid, when one has been asked for. See [`Wall::baked`].
    baked: Option<crate::BakedProfile>,
}

impl Wall {
    /// A wall along an open chain of points, already in the plane's own coordinates.
    pub fn open(
        points: Vec<Point2<f64>>,
        thickness: f64,
        side: WallSide,
        ends: WallEnds,
    ) -> Result<Self, WallError> {
        let thickness = a_thickness(thickness)?;
        if let Some(i) = points.iter().position(|p| !(p.x.is_finite() && p.y.is_finite())) {
            return Err(WallError::PointNotFinite(i));
        }
        // **Two points closer than two curve ends may be and still count as joined are one
        // point.** A step that small has no direction worth reading, and reading one anyway is
        // not harmless here: a point a millionth of a millimetre *behind* its predecessor is a
        // reversal, and a reversal fans out a half-round of the wall's whole thickness. So the
        // floor is the one the document already judges "the same place" by, not arithmetic's.
        let mut kept: Vec<Point2<f64>> = Vec::with_capacity(points.len());
        for p in points {
            if kept.last().is_none_or(|q| (p - q).norm() > JOINED_MM) {
                kept.push(p);
            }
        }
        if kept.len() < 2 {
            return Err(WallError::TooFewPoints(kept.len()));
        }
        let (lo, hi) = open_band(thickness, side);
        let facing = facing(side);
        let mut runs: Vec<Run> = Vec::with_capacity(kept.len() - 1);
        for w in kept.windows(2) {
            let d = w[1] - w[0];
            let length = d.norm();
            let mut run = Run {
                along: d / length,
                length,
                fan: -1.0,
                reach: 0.0,
                mitre_start: false,
                mitre_end: false,
            };
            if let Some(before) = runs.last_mut() {
                // A left turn fans out on the right. A dead reversal has no turn to read, and
                // is taken as a left one so that nothing hangs on the sign of a rounding.
                let leftward = cross(before.along, run.along) >= 0.0;
                (run.fan, run.reach) =
                    if leftward { (-1.0, (-lo).max(0.0)) } else { (1.0, hi.max(0.0)) };
                // Toward the wall, and past a right angle: the two bands are mitred.
                let toward = if leftward { facing > 0.0 } else { facing < 0.0 };
                if toward && before.along.dot(&run.along) < 0.0 {
                    before.mitre_end = true;
                    run.mitre_start = true;
                }
            }
            runs.push(run);
        }
        // A round-ended wall about its curve is a chain of capsules and has no flat face.
        let (faces, sides) = if side == WallSide::Centre && ends == WallEnds::Round {
            (Vec::new(), Vec::new())
        } else {
            bare_boundary(&kept, &runs, lo, hi, facing, ends == WallEnds::Round)
        };
        Ok(Self {
            shape: Shape::Open { points: kept, runs, ends, faces, sides },
            thickness,
            side,
            baked: None,
        })
    }

    /// A wall along a closed loop. It has no ends, so nothing is asked about them.
    pub(crate) fn closed(profile: Profile, thickness: f64, side: WallSide) -> Result<Self, WallError> {
        let thickness = a_thickness(thickness)?;
        // **A loop whose lobes cancel has no winding, and is read as clockwise.** A figure of
        // eight with equal lobes has a signed area of nought, so its sign is rounding: the
        // same bow-tie came out counter-clockwise or clockwise depending on where in the
        // document it had been moved to, and a one-sided wall on it changed sides with it. An
        // area that is nothing beside the loop's own size is treated as none, which does not
        // make the choice right — on such a loop "left" is inside one lobe and outside the
        // other whichever is chosen — but does make it the same choice every time.
        let (lo, hi) = profile.bounds();
        let left_is_inside = signed_area(profile.points()) > 1e-9 * (hi - lo).norm_squared();
        Ok(Self { shape: Shape::Closed { profile, left_is_inside }, thickness, side, baked: None })
    }

    /// The side the wall stands on, as it was asked for. Only the tests here read it back off a
    /// built wall, so only the tests compile this (decision 1.4).
    #[cfg(test)]
    fn side(&self) -> WallSide {
        self.side
    }

    /// How an open wall stops, or `None` for a loop, which does not.
    pub fn ends(&self) -> Option<WallEnds> {
        match &self.shape {
            Shape::Open { ends, .. } => Some(*ends),
            Shape::Closed { .. } => None,
        }
    }

    pub fn is_closed(&self) -> bool {
        matches!(self.shape, Shape::Closed { .. })
    }

    /// Which of a loop's two sides is its inside, in the words a wall's side is stored in —
    /// or `None` for an open chain, which has none.
    ///
    /// **For a panel that offers Inside and Outside where the document holds left and right.**
    /// Which is which is the loop's winding, and that is settled here, where the wall is
    /// built: asking here is what stops a panel from settling it a second way and showing
    /// "Inside" over a wall that is drawn outside.
    pub fn inside(&self) -> Option<WallSide> {
        match &self.shape {
            Shape::Open { .. } => None,
            Shape::Closed { left_is_inside, .. } => {
                Some(if *left_is_inside { WallSide::Left } else { WallSide::Right })
            }
        }
    }

    /// The curve the wall runs along, for a caller that has to hand it to something else. A
    /// loop's closing segment is implied, as a profile's is.
    pub(crate) fn points(&self) -> &[Point2<f64>] {
        match &self.shape {
            Shape::Open { points, .. } => points,
            Shape::Closed { profile, .. } => profile.points(),
        }
    }

    /// Where the material lies across the curve, as the two lateral offsets it runs between.
    ///
    /// Positive is to the left of an open chain and **outward** from a loop — the sign a
    /// loop's own distance already has — so the answer for a loop has its winding folded in
    /// and a caller never has to ask which way it was drawn.
    pub fn band(&self) -> (f64, f64) {
        match &self.shape {
            Shape::Open { .. } => open_band(self.thickness, self.side),
            Shape::Closed { left_is_inside, .. } => match self.side {
                WallSide::Centre => (-0.5 * self.thickness, 0.5 * self.thickness),
                WallSide::Left if *left_is_inside => (-self.thickness, 0.0),
                WallSide::Right if !*left_is_inside => (-self.thickness, 0.0),
                _ => (0.0, self.thickness),
            },
        }
    }

    /// How far the wall reaches, as a box in its own plane.
    pub(crate) fn bounds(&self) -> (Point2<f64>, Point2<f64>) {
        let (lo_band, hi_band) = self.band();
        // An open wall's sides are not the plane's axes, so it is padded by its furthest reach
        // either way. A loop's positive side is outward, so only that half grows the box.
        let reach = match &self.shape {
            Shape::Open { .. } => lo_band.abs().max(hi_band.abs()),
            Shape::Closed { .. } => hi_band.max(0.0),
        };
        let mut lo = Point2::new(f64::INFINITY, f64::INFINITY);
        let mut hi = Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in self.points() {
            lo = Point2::new(lo.x.min(p.x - reach), lo.y.min(p.y - reach));
            hi = Point2::new(hi.x.max(p.x + reach), hi.y.max(p.y + reach));
        }
        (lo, hi)
    }

    /// The same wall offset outward by `t`, with its corners cut as `corners` says.
    ///
    /// A loop's wall is its curve's distance folded about a band, so offsetting the curve and
    /// keeping the band is the whole of it and the band stays as thick as it was. An open
    /// wall has no inside to push out from -- a draft on one is refused by name -- so it is
    /// given the reading it always had.
    pub(crate) fn offset_by(&self, q: Point2<f64>, t: f64, corners: fieldform_scene::Corners) -> f64 {
        // **Rounding is this wall's own reading less the offset, grid and all.** It has to go
        // through `distance` rather than through the loop it runs along: a wall may carry a
        // grid of its own, and that grid holds the band *already folded*, where the loop
        // beneath it holds a curve that still has to be. Folding a reading that was folded
        // once put a baked wall 0.69 mm out, which `gpu_matches_cpu` said at once.
        // **And sharp corners are the same offset**, which is the whole of why this is one
        // line. Mitring moves a *corner* further than its walls; it does not change what the
        // offset does to the walls, so a band still grows by `t` on each face either way. An
        // earlier version mitred the loop and folded an unchanged band about it, which slid the
        // band outward and kept its thickness — so the same wall drafted the same amount was a
        // different thickness depending on a corner toggle, which is absurd and is the thing
        // this must not do. `Corners::Sharp` on a wall is refused upstream for now
        // (`CompileError::SharpCornersOnAWall`) because mitring *both* faces of a band is two
        // displaced walks per query and wants its own spec; until then a wall rounds.
        let _ = corners;
        self.distance(q) - t
    }

    /// The wall of this curve **scaled by `scale` about the origin and pushed out by
    /// `offset`**, with its thickness kept -- what a tapered sweep reads at a station
    /// (`sweep-v2-spec.md` §2).
    ///
    /// **Taper acts on the curve; the thickness is the wall's own and is kept.** A wall round a
    /// loop is a band folded about a contour of the loop's distance, so scaling the loop and
    /// moving the contour are both one line here, exact, and the band stays as thick as it was:
    /// a tapered tube keeps its wall. An open wall is not a function of its curve's distance
    /// alone -- its mitres and end faces are built for one thickness on one curve, and the wall
    /// of the scaled curve at the same thickness is a different wall at every station -- so an
    /// open wall is **scaled whole**, thickness and all, which is exact and is said beside the
    /// field; and an offset of an open wall is refused before this is reached, since a band has
    /// no inside to be pushed out from. `offset` must be nought for an open wall, and is.
    ///
    /// Walked rather than read from a grid, since a grid holds the band at one scale and one
    /// contour.
    pub(crate) fn distance_tapered(&self, q: Point2<f64>, scale: f64, offset: f64) -> f64 {
        let half = self.thickness * 0.5;
        let (lo, hi) = self.band();
        match &self.shape {
            Shape::Closed { profile, .. } => {
                (scale * profile.walk(q / scale) - 0.5 * (lo + hi) - offset).abs() - half
            }
            Shape::Open { .. } => {
                debug_assert!(offset == 0.0, "an open wall takes no offset; refuse it upstream");
                scale * self.walk(q / scale)
            }
        }
    }

    /// The signed distance from `q` to the wall: negative in its material. From the grid if
    /// the wall carries one, and otherwise worked out from the curve.
    pub fn distance(&self, q: Point2<f64>) -> f64 {
        match &self.baked {
            Some(grid) => grid.distance(q),
            None => self.walk(q),
        }
    }

    /// The same wall, answering from a grid instead of from its curve.
    ///
    /// **A display convenience, by the rule a profile is baked by** ([`Profile::baked`]) and
    /// for the reason one is: [`walk`](Self::walk) reads every run of the chain at every
    /// query, so a wall along a curve that turns most of the way round cost five times the
    /// frame of the same solid drawn as a profile, which reads a grid —
    /// `fieldform-gpu/examples/wall_cost.rs`. Ruled for by the owner, 2026-10-02.
    ///
    /// **The grid holds the wall's distance, not its curve's.** That is the whole difference
    /// from the bake a loop's own profile may carry, which a wall never reads: a reading short
    /// by a margin, folded about the middle of the band, comes out long on the far side. Here
    /// the fold is done first, exactly, and what is sampled is the result — so the margin is
    /// taken off the wall's own distance and can only ever understate it.
    ///
    /// A wall on a grid is a bound and says so ([`normalization`](Self::normalization)), as a
    /// baked profile does.
    pub(crate) fn baked(mut self, cells: usize, reads: (f64, f64)) -> Self {
        let grid = crate::BakedProfile::of(self.bounds(), |q| self.walk(q), cells, reads);
        self.baked = Some(grid);
        self
    }

    /// The same wall, answering from a grid somebody else already built for it. For a cache,
    /// which keys a grid by [`fingerprint`](Self::fingerprint) and by how it was laid out.
    pub(crate) fn with_bake(mut self, bake: crate::BakedProfile) -> Self {
        self.baked = Some(bake);
        self
    }

    /// The grid, for a caller that has to hand it to a shader.
    pub fn bake(&self) -> Option<&crate::BakedProfile> {
        self.baked.as_ref()
    }

    /// Whether this wall answers from a grid.
    pub fn is_baked(&self) -> bool {
        self.baked.is_some()
    }

    /// A hash of everything the wall's distance is a function of: the curve's points in
    /// order, whether they close, the thickness, the side, and how an open one ends.
    ///
    /// **The identity of a bake**, as [`Profile::fingerprint`] is of a profile's. Begun with a
    /// word of its own, so a wall and a profile over the same points are never one key: they
    /// share a cache, and their grids hold different fields.
    pub(crate) fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        "wall".hash(&mut hasher);
        self.is_closed().hash(&mut hasher);
        self.thickness.to_bits().hash(&mut hasher);
        (self.side as u8).hash(&mut hasher);
        self.ends().map(|ends| ends as u8).hash(&mut hasher);
        let points = self.points();
        points.len().hash(&mut hasher);
        for p in points {
            p.x.to_bits().hash(&mut hasher);
            p.y.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// The signed distance from `q` to the wall, worked out from its curve: negative in its
    /// material. What [`distance`](Self::distance) answers when nothing has been baked, and
    /// what a bake is sampled *from*.
    ///
    /// **Outside the material this is the Euclidean distance**, for every side and both kinds
    /// of end. Inside, it is the depth below the nearer face and never more than that — the
    /// ordinary reading of a union, which every boolean here already gives.
    pub(crate) fn walk(&self, q: Point2<f64>) -> f64 {
        let half = self.thickness * 0.5;
        let (lo, hi) = self.band();
        match &self.shape {
            Shape::Closed { profile, .. } => {
                // A distance moved by a constant is the distance to the moved curve, and folded
                // about the middle of the band it is the distance to the band. Every step keeps
                // the rate at one, and the nearest face is reached straight along the gradient,
                // so nothing is lost on the way.
                //
                // **Walked, even when the profile carries a bake.** A baked reading is the true
                // one less a margin, so it never overstates — as a *signed* number. The fold
                // is what breaks that: on the far side of the band's middle, less becomes more,
                // and a wall asked from inside its own hole read up to five millimetres too far
                // and was marched straight through. A fold needs a bound on both sides and a
                // bake has one, so the boundary is what is asked.
                (profile.walk(q) - 0.5 * (lo + hi)).abs() - half
            }
            Shape::Open { points, runs, ends, faces, sides } => {
                // Nought when the wall is about the curve — where every foot is on the wall's
                // side, because both sides are.
                let facing = facing(self.side);
                // A round-ended wall about its curve is the curve's own distance less half the
                // thickness, ends and all: a chain of capsules, and exact.
                let capsules = facing == 0.0 && *ends == WallEnds::Round;

                // The nearest foot with `q` on the wall's side of it, and the nearest with `q`
                // on the other. Kept apart because they are distances to two different faces:
                // one to the middle of somebody's band, one to the curve as the wall's back.
                let mut with = f64::INFINITY;
                let mut against = f64::INFINITY;
                let mut take = |distance: f64, side: f64| {
                    if side * facing >= 0.0 {
                        with = with.min(distance);
                    } else {
                        against = against.min(distance);
                    }
                };
                let mut past_previous = false;
                for (i, run) in runs.iter().enumerate() {
                    let rel = q - points[i];
                    let along = rel.dot(&run.along);
                    if along < 0.0 {
                        // Before this run. If it is also past the one before, it is in the
                        // sector the joint between them fans out into, and the joint is the
                        // foot — on the side the fan is on, whichever side of either run the
                        // query happens to be. Before the *first* run there is no joint: a
                        // square end has its own face, and only a capsule's end is round
                        // about the point itself.
                        if i > 0 && past_previous {
                            take(rel.norm(), run.fan);
                        } else if i == 0 && capsules {
                            take(rel.norm(), 0.0);
                        }
                    } else if along <= run.length {
                        // Positive to the left of the run, which is what "left" means.
                        let lateral = cross(run.along, rel);
                        // **A mitre cuts a fibre short, and what matters is where.** The query
                        // is on some fibre of this run, which starts on the curve — always on
                        // the kept side of the cut — and runs out through the query. If the cut
                        // crosses it beyond the band, the band on this fibre is whole and the
                        // foot stands as it would with no mitre at all, though the query may
                        // itself be past the cut. If it crosses within the band, this run has
                        // nothing for a query beyond the crossing: the cut is its face there,
                        // and that is answered by the run that did the cutting, as the curve
                        // it is, or by the cut's own bare stretch.
                        //
                        // `on_curve` and `at_query` are how far the two ends of the fibre are
                        // to the kept side of the cut, so the crossing is the share of the
                        // way along at which that reaches nought.
                        let cut_short = |on_curve: f64, at_query: f64| {
                            at_query < 0.0
                                && lateral.abs() * on_curve / (on_curve - at_query)
                                    < self.thickness
                        };
                        let cut_away = lateral * facing > 0.0
                            && ((run.mitre_start && {
                                let before = runs[i - 1].along;
                                cut_short(
                                    facing * cross(before, run.along) * along,
                                    facing * cross(before, rel),
                                )
                            }) || (run.mitre_end && {
                                let after = runs[i + 1].along;
                                cut_short(
                                    facing * cross(run.along, after) * (run.length - along),
                                    facing * cross(after, q - points[i + 1]),
                                )
                            }));
                        if !cut_away {
                            take(lateral.abs(), lateral);
                        }
                    }
                    past_previous = along > run.length;
                }
                let last = points[points.len() - 1];
                if past_previous && capsules {
                    with = with.min((q - last).norm());
                }

                // Deep in a band is `|d − t/2| − t/2`; about the curve the band is centred on
                // the curve, so it is the plain `d − t/2`.
                let fibres = if facing == 0.0 {
                    with - half
                } else {
                    ((with - half).abs() - half).min(against)
                };
                if capsules {
                    return fibres;
                }

                let nearest = |stretches: &[Face]| {
                    stretches
                        .iter()
                        .map(|(a, b)| to_segment(q, *a, *b))
                        .fold(f64::INFINITY, f64::min)
                };

                // A round end on a one-sided wall: a half-round of the wall's own thickness
                // beyond the square face, which is that wall and a disc on the middle of each
                // end fibre. Where the end run is at least half the thickness long, the half of
                // the disc's rim behind the end is buried in that run's own band, so from
                // behind the end the rim is no nearer than its two tips. A shorter run does
                // not reach back far enough to bury it, and a mitred one may have had the band
                // cut from under it, so for those the whole rim is measured to.
                let (mut disc, mut rounds) = (f64::INFINITY, f64::INFINITY);
                if facing != 0.0 && *ends == WallEnds::Round {
                    let (first, final_run) = (runs[0], runs[runs.len() - 1]);
                    for (at, run, beyond) in
                        [(points[0], first, -first.along), (last, final_run, final_run.along)]
                    {
                        let across = left_of(run.along);
                        let from_centre = q - (at + across * (0.5 * (lo + hi)));
                        let d = from_centre.norm() - half;
                        disc = disc.min(d);
                        let buried_behind =
                            run.length >= half && !run.mitre_start && !run.mitre_end;
                        rounds = rounds.min(if from_centre.dot(&beyond) >= 0.0 || !buried_behind {
                            d.abs()
                        } else {
                            let tip = |way: f64| (from_centre - across * (way * half)).norm();
                            tip(1.0).min(tip(-1.0))
                        });
                    }
                }

                // **Outside, the feet are the whole answer but for the bare end fibres**, which
                // are the only faces that are not the tip of a fibre. A foot on a stretch of
                // edge that is buried in another run does no harm here: what is buried is
                // further off than the surface that buries it.
                if fibres > 0.0 && disc > 0.0 {
                    return fibres.min(nearest(faces)).min(disc);
                }
                // **Inside, the depth is to the boundary itself**, and the feet cannot say
                // where that is: the nearest run's band may stop a hair away, with a
                // neighbour's edge the real surface beyond it. So it is measured to what was
                // found bare when the wall was built, and to every round.
                //
                // **Every fan's round, not only the one the query is in.** Where a bend is
                // tighter than the wall two fans overlap, and the surface nearest a point in
                // one may be the other's arc. The rounds are not cut the way the flat edges
                // are, so a stretch of arc buried in another piece is still measured to —
                // which reads a depth too small there, and that is the direction allowed.
                for j in 1..runs.len() {
                    let run = runs[j];
                    if run.reach <= 0.0 {
                        continue;
                    }
                    let rel = q - points[j];
                    let before = runs[j - 1].along;
                    rounds = rounds.min(if rel.dot(&before) >= 0.0 && rel.dot(&run.along) <= 0.0 {
                        (rel.norm() - run.reach).abs()
                    } else {
                        // Not in its sector, so no nearer to its arc than the arc's two ends.
                        let tip = |along: Vector2<f64>| {
                            (rel - left_of(along) * (run.fan * run.reach)).norm()
                        };
                        tip(before).min(tip(run.along))
                    });
                }
                -nearest(faces).min(nearest(sides)).min(rounds)
            }
        }
    }

    /// What a solid built on this wall can promise about its distance.
    ///
    /// **A distance, for every side, both kinds of end, and a loop as much as a chain** —
    /// which is not what the spec expected of a one-sided or square-ended wall. It expected a
    /// bound, because the construction it described divides by a stretch, and this one has
    /// nothing to divide by. `a_wall_moves_at_unit_rate_outside_its_own_material` is the check
    /// D8 asks for, and `a_wall_is_the_union_of_its_fibres_whatever_the_chain` is the one that
    /// says the distance is to the right shape.
    ///
    /// A loop whose *profile* carries a bake is no exception, because that bake is not what is
    /// asked: see where the loop is read. **A wall that carries its own is a bound**, for the
    /// reason a baked profile is — the grid's reading has a margin taken off it, and a field
    /// answering from one while claiming a distance is the silent wrong shape D8 is about.
    pub fn normalization(&self) -> Normalization {
        if self.is_baked() {
            Normalization::LowerBound
        } else {
            Normalization::Distance
        }
    }
}

/// Which of the three polylines a shader is handed for a wall.
///
/// Three, because that is what the field reads: the curve itself, and the two lists of bare
/// edges that were worked out from it. They go up as separate bindings so that each is a span
/// the shader can walk from end to end, with nothing inside one to say where another begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WallPart {
    /// The curve the wall runs along. See [`Wall::packed`] for what rides in each point's `z`.
    Chain,
    /// The bare end fibres, mitre cuts and fan edges, as pairs of points: the flat faces an
    /// open wall has besides its long ones. Empty for a loop and for a round-ended wall about
    /// its curve, neither of which has any.
    Faces,
    /// The bare long edges, as pairs of points. Read only from inside the material, where they
    /// are what the depth is measured to.
    Sides,
}

impl Wall {
    /// One of the polylines a shader needs for this wall, as the points to upload.
    ///
    /// # What a point of the chain carries
    ///
    /// `x` and `y` are the point. **`z` is what was decided about the run that starts there**:
    /// one if its band is mitred at its start, two if at its end, and four if the joint at its
    /// start fans out to the left rather than the right. Those three are decisions, made here
    /// from the turn between two runs, and a shader that made them again from the same points
    /// in single precision could make them differently — a turn of exactly a right angle is a
    /// mitre or it is not, and the two sides of a wall would then be two different shapes.
    /// So they are made once and sent.
    ///
    /// A loop carries nothing in `z`: it is read through its signed distance, which has no
    /// runs to decide anything about. Its faces and sides are empty for the same reason.
    pub fn packed(&self, part: WallPart) -> Vec<nalgebra::Point3<f64>> {
        let flat = |p: &Point2<f64>| nalgebra::Point3::new(p.x, p.y, 0.0);
        let pairs = |edges: &[Face]| edges.iter().flat_map(|(a, b)| [flat(a), flat(b)]).collect();
        match (&self.shape, part) {
            (Shape::Closed { profile, .. }, WallPart::Chain) => {
                profile.points().iter().map(flat).collect()
            }
            (Shape::Closed { .. }, _) => Vec::new(),
            (Shape::Open { points, runs, .. }, WallPart::Chain) => points
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let code = runs.get(i).map_or(0.0, |run| {
                        f64::from(u8::from(run.mitre_start))
                            + 2.0 * f64::from(u8::from(run.mitre_end))
                            + if run.fan > 0.0 { 4.0 } else { 0.0 }
                    });
                    nalgebra::Point3::new(p.x, p.y, code)
                })
                .collect(),
            (Shape::Open { faces, .. }, WallPart::Faces) => pairs(faces),
            (Shape::Open { sides, .. }, WallPart::Sides) => pairs(sides),
        }
    }
}

/// What a profile-based solid reads in its own plane: a closed profile, or a wall.
///
/// The kernel's side of `fieldform_scene::Outline`, with the curves already walked and laid
/// flat. An extrusion holds one of these and asks it for a distance, and that is the whole of
/// what it knows about it — which is how one solid serves a thick extrude and a thin one.
#[derive(Debug, Clone)]
pub enum Footprint {
    Closed(Profile),
    Wall(Wall),
}

impl Footprint {
    /// The signed distance in the plane: negative in the material.
    pub fn distance(&self, q: Point2<f64>) -> f64 {
        match self {
            Footprint::Closed(profile) => profile.distance(q),
            Footprint::Wall(wall) => wall.distance(q),
        }
    }

    /// The section offset outward by `t`, with its corners cut as `corners` says: what a
    /// drafted solid reads at a height.
    ///
    /// **Round is the offset of a distance field and sharp is the mitre**, and both are exact
    /// -- see [`Profile::offset_by`], which is where the arithmetic is. A wall round a loop
    /// offsets its own curve and keeps its band, so the band stays as thick as it was whichever
    /// way the corners are cut; an open wall has no inside to be pushed out from, and a draft
    /// on one is refused before this is reached, so it takes the reading it always took.
    pub(crate) fn offset_by(&self, q: Point2<f64>, t: f64, corners: fieldform_scene::Corners) -> f64 {
        match self {
            Footprint::Closed(profile) => profile.offset_by(q, t, corners),
            Footprint::Wall(wall) => wall.offset_by(q, t, corners),
        }
    }

    /// The section scaled by `scale` about the origin and pushed out by `offset`: what a
    /// tapered sweep reads at a station. `scale · d(q / scale)` is the exact distance to the
    /// scaled section, and taking `offset` off it is the exact distance to its outward contour
    /// -- so a tapered section is as exact in its own plane as the section was. A wall keeps
    /// its thickness where it can; see [`Wall::distance_tapered`].
    pub(crate) fn distance_tapered(&self, q: Point2<f64>, scale: f64, offset: f64) -> f64 {
        match self {
            Footprint::Closed(profile) => scale * profile.distance(q / scale) - offset,
            Footprint::Wall(wall) => wall.distance_tapered(q, scale, offset),
        }
    }

    /// How deep the material goes, in millimetres: the most negative reading over a sampling
    /// of the section's box, which is a floor under the inradius -- a sample can miss the
    /// deepest point and never invent one. What a draft that narrows the section is held to
    /// (`sweep-v2-spec.md` §2, by way of `round-chamfer-spec.md` §4's proxy): past this the
    /// section has closed before the end of the path.
    pub(crate) fn depth(&self) -> f64 {
        let (lo, hi) = self.bounds();
        let n = 48;
        let mut deepest = 0.0f64;
        for i in 0..=n {
            for j in 0..=n {
                let q = Point2::new(
                    lo.x + (hi.x - lo.x) * i as f64 / n as f64,
                    lo.y + (hi.y - lo.y) * j as f64 / n as f64,
                );
                deepest = deepest.min(self.distance(q));
            }
        }
        -deepest
    }

    /// Whether this section is a curve given a thickness rather than a closed profile — the
    /// one kind whose corners a draft cannot mitre yet, because a band has two faces.
    pub(crate) fn is_wall(&self) -> bool {
        matches!(self, Footprint::Wall(_))
    }

    /// A wall along an open chain: the one kind of section a draft cannot push out, having no
    /// inside to push it from, and the one a taper scales whole.
    pub(crate) fn is_open_wall(&self) -> bool {
        matches!(self, Footprint::Wall(wall) if !wall.is_closed())
    }

    /// How far it reaches, as a box in its own plane.
    pub(crate) fn bounds(&self) -> (Point2<f64>, Point2<f64>) {
        match self {
            Footprint::Closed(profile) => profile.bounds(),
            Footprint::Wall(wall) => wall.bounds(),
        }
    }

    /// The distance at radius `r` and height `z`, for a section turned about an axis.
    ///
    /// A profile is read where it was laid, at `(r, z)`. **A wall is read on both sides of the
    /// axis where its band crosses it.** A curve is laid in the half-plane by its points, and
    /// its band then goes a thickness to one side or both — which near the axis is across it.
    /// That part of the section is material too, and turned about the axis it sweeps the
    /// radii it would have had on this side: so the reading is the nearer of the wall at
    /// `(r, z)` and the wall at `(−r, z)`. **Exact outside the material**, since the two
    /// together are a section that mirrors in the axis, and the nearest point of such a
    /// section to a point on this side is on this side. Inside, where the wall and its mirror
    /// overlap, it is the deeper of two depths each taken to its own wall's face — never more
    /// than the true depth, and less where the nearer face is one the other wall covers:
    /// the ordinary reading of a union.
    ///
    /// Only where the band does cross, which is a fact about the wall and is asked once by
    /// [`crosses_the_axis`](Self::crosses_the_axis): a wall that keeps to one side is nearer
    /// from its own side everywhere, and asking twice would be a second walk for nothing.
    pub(crate) fn about_axis(&self, r: f64, z: f64) -> f64 {
        let here = self.distance(Point2::new(r, z));
        if self.crosses_the_axis() {
            here.min(self.distance(Point2::new(-r, z)))
        } else {
            here
        }
    }

    /// The section's distance read in the **whole** plane through the axis, `u` signed across
    /// it: what a partial revolve's cap is, seen from its own half-plane and from the other
    /// side of the axis alike. A band that crosses the axis is material on both sides of it
    /// there too, so it is read mirrored as [`about_axis`](Self::about_axis) reads it.
    pub(crate) fn across_axis(&self, u: f64, z: f64) -> f64 {
        let here = self.distance(Point2::new(u, z));
        if self.crosses_the_axis() {
            here.min(self.distance(Point2::new(-u, z)))
        } else {
            here
        }
    }

    /// Whether a wall's band reaches across the axis it is turned about, so that
    /// [`about_axis`](Self::about_axis) reads it from both sides. Never, for a profile.
    pub fn crosses_the_axis(&self) -> bool {
        match self {
            Footprint::Closed(_) => false,
            Footprint::Wall(wall) => wall.bounds().0.x < 0.0,
        }
    }

    /// How far the section reaches from its own origin, which is what a sweep bends about:
    /// the furthest of a profile's points, and for a wall the furthest its band can be, a
    /// curve point and the band's reach beyond it.
    pub(crate) fn reach_from_origin(&self) -> f64 {
        match self {
            Footprint::Closed(profile) => {
                profile.points().iter().map(|q| q.coords.norm()).fold(0.0f64, f64::max)
            }
            Footprint::Wall(wall) => {
                let (lo, hi) = wall.band();
                let curve = wall.points().iter().map(|q| q.coords.norm()).fold(0.0f64, f64::max);
                curve + lo.abs().max(hi.abs())
            }
        }
    }

    /// What a solid built on it can promise: a bound where it answers from a grid, and a
    /// distance where it is worked out from its curve. The same for a profile and for a wall.
    pub fn normalization(&self) -> Normalization {
        match self {
            Footprint::Closed(profile) => crate::profile::baked_normalization(profile),
            Footprint::Wall(wall) => wall.normalization(),
        }
    }
}

impl From<Profile> for Footprint {
    fn from(profile: Profile) -> Self {
        Footprint::Closed(profile)
    }
}

impl From<Wall> for Footprint {
    fn from(wall: Wall) -> Self {
        Footprint::Wall(wall)
    }
}

fn a_thickness(thickness: f64) -> Result<f64, WallError> {
    if !thickness.is_finite() || thickness <= 0.0 {
        return Err(WallError::NoThickness(thickness));
    }
    Ok(thickness)
}

/// Which side of an open curve its wall is on: `+1` left, `-1` right, nought about the curve.
fn facing(side: WallSide) -> f64 {
    match side {
        WallSide::Centre => 0.0,
        WallSide::Left => 1.0,
        WallSide::Right => -1.0,
    }
}

/// Where an open wall's material lies across its curve: positive to the left of travel.
fn open_band(thickness: f64, side: WallSide) -> (f64, f64) {
    match side {
        WallSide::Centre => (-0.5 * thickness, 0.5 * thickness),
        WallSide::Left => (0.0, thickness),
        WallSide::Right => (-thickness, 0.0),
    }
}

/// The z of `a × b`: positive when `b` is to the left of `a`.
fn cross(a: Vector2<f64>, b: Vector2<f64>) -> f64 {
    a.x * b.y - a.y * b.x
}

/// The direction a quarter turn to the left of `along`.
fn left_of(along: Vector2<f64>) -> Vector2<f64> {
    Vector2::new(-along.y, along.x)
}

/// The shoelace sum: positive for a loop that runs counter-clockwise.
fn signed_area(points: &[Point2<f64>]) -> f64 {
    let n = points.len();
    (0..n)
        .map(|i| {
            let (a, b) = (points[i], points[(i + 1) % n]);
            a.x * b.y - b.x * a.y
        })
        .sum::<f64>()
        * 0.5
}

/// Distance from `q` to the segment `a`-`b`, which is never a point: a face has length.
fn to_segment(q: Point2<f64>, a: Point2<f64>, b: Point2<f64>) -> f64 {
    let ab = b - a;
    let t = ((q - a).dot(&ab) / ab.norm_squared()).clamp(0.0, 1.0);
    (q - (a + ab * t)).norm()
}

/// Narrows `span` to the offsets `s` where `c + s·d` is not negative.
fn keep_where(span: &mut (f64, f64), c: f64, d: f64) {
    if d.abs() < 1e-14 {
        if c < 0.0 {
            span.1 = span.0 - 1.0;
        }
    } else if d > 0.0 {
        span.0 = span.0.max(-c / d);
    } else {
        span.1 = span.1.min(-c / d);
    }
}

/// Narrows `span` to the offsets `s` where `rel + s·across` is within `radius` of the origin.
fn keep_within(span: &mut (f64, f64), rel: Vector2<f64>, across: Vector2<f64>, radius: f64) {
    let b = rel.dot(&across);
    let inside = b * b - (rel.norm_squared() - radius * radius);
    if inside < 0.0 {
        span.1 = span.0 - 1.0;
        return;
    }
    span.0 = span.0.max(-b - inside.sqrt());
    span.1 = span.1.min(-b + inside.sqrt());
}

/// One straight edge of a piece of the wall: a point on its line, and the way into the piece.
#[derive(Debug, Clone, Copy)]
struct Edge {
    on: Point2<f64>,
    inward: Vector2<f64>,
    /// Whether a query outside the wall has to be told of it. A long edge is found by the foot
    /// of a perpendicular; an end fibre and a mitre's cut are not.
    flat: bool,
}

/// The wall's flat boundary: the bare stretches of every piece's edges.
///
/// # What is being found
///
/// A run's fibres make a rectangle: two long edges, and two short ones that are its first
/// fibre and its last. Where the run is mitred, a corner of that rectangle is cut off along
/// the line of the run it meets. Either way the piece is convex, and most of its outline is
/// not the wall's surface at all. Along an ordinary curve each end fibre is buried — in the
/// next run's piece on the side the curve turns toward, in the joint's fan on the side it
/// turns away from — and each long edge is bare for its whole length but for the little the
/// next run laps over it on the inside of the bend.
///
/// What is left bare is the boundary. An end fibre comes out into the open at the curve's own
/// two ends, and wherever the next run is too short to reach across it — which on a smooth
/// curve is a bend tighter than the wall is thick. That is the fold a sweep refuses. Here it
/// is a shape like any other, with flat faces where fibres stop.
///
/// The end fibres and the mitres' cuts come back first and the long edges second, because they
/// are read apart: a query outside the wall needs only the first, and one inside it needs both.
///
/// # How
///
/// Each edge is a straight line, and every other piece of the wall — a run's piece, a fan, an
/// end's disc — is convex, so it covers one stretch of that line or none. The stretches are
/// gathered, and what is left between them is bare.
///
/// **Tested a hair to the open side of the edge, not on it.** An edge lies on the rim of its
/// own piece, and what matters is what is on the far side of that rim: another piece there
/// buries it, and nothing there leaves it bare. Asking on the edge itself cannot tell those
/// apart where two pieces share a line — a return leg at a right angle has its long edge along
/// the first run's end fibre, both with material on the same side, and a test with any give in
/// it calls each buried by the other and loses the surface between them.
///
/// Every edge against every piece, so the square of the run count, once per build and never
/// per sample. A piece or a fan too far away to reach is passed over by its distance alone,
/// which makes the square a cheap one and leaves it a square: 3.9 ms at 256 runs and 30 at a
/// thousand, for a wall about its curve (`fieldform-gpu/examples/wall_cost.rs`). A chain much
/// longer than that wants its neighbours found by where they are, not tried in turn.
fn bare_boundary(
    points: &[Point2<f64>],
    runs: &[Run],
    lo: f64,
    hi: f64,
    facing: f64,
    round: bool,
) -> (Vec<Face>, Vec<Face>) {
    /// How far to the open side an edge is tested, in millimetres.
    const OPEN_SIDE: f64 = 1e-8;
    /// The shortest bare stretch worth keeping. Above what the offset leaves at a joint,
    /// where the test line cuts the corner a real edge only touches.
    const SLIVER: f64 = 1e-6;
    /// Further than any edge runs, for a line not yet cut to its piece.
    const FAR: f64 = 1e30;
    let reach = lo.abs().max(hi.abs());
    let (middle, half) = (0.5 * (lo + hi), 0.5 * (hi - lo));
    let discs: Vec<Point2<f64>> = if round {
        let (first, final_run) = (runs[0], runs[runs.len() - 1]);
        vec![
            points[0] + left_of(first.along) * middle,
            points[points.len() - 1] + left_of(final_run.along) * middle,
        ]
    } else {
        Vec::new()
    };

    // Each run's piece, as the lines that bound it: its rectangle, and a cut for each mitre.
    let pieces: Vec<Vec<Edge>> = runs
        .iter()
        .enumerate()
        .map(|(i, run)| {
            let across = left_of(run.along);
            let mut edges = vec![
                Edge { on: points[i], inward: run.along, flat: true },
                Edge { on: points[i + 1], inward: -run.along, flat: true },
                Edge { on: points[i] + across * lo, inward: across, flat: false },
                Edge { on: points[i] + across * hi, inward: -across, flat: false },
            ];
            if run.mitre_start {
                let wall_side = left_of(runs[i - 1].along) * facing;
                edges.push(Edge { on: points[i], inward: wall_side, flat: true });
            }
            if run.mitre_end {
                let wall_side = left_of(runs[i + 1].along) * facing;
                edges.push(Edge { on: points[i + 1], inward: wall_side, flat: true });
            }
            edges
        })
        .collect();

    let mut covered: Vec<(f64, f64)> = Vec::new();
    // The bare stretches of `from + s·direction`, for `s` from nought to `length`: an edge of
    // piece `own`, whose open side is `outward`.
    let mut bare = |own: usize,
                    from: Point2<f64>,
                    direction: Vector2<f64>,
                    length: f64,
                    outward: Vector2<f64>,
                    out: &mut Vec<Face>| {
        let origin = from + outward * OPEN_SIDE;
        covered.clear();
        for (k, piece) in pieces.iter().enumerate() {
            if k == own {
                continue;
            }
            // A piece is within its run's rectangle, which is what it is too far away by.
            let other = runs[k];
            let between = (origin + direction * (0.5 * length))
                - (points[k] + other.along * (0.5 * other.length));
            if between.norm() > 0.5 * (length + other.length) + reach + SLIVER {
                continue;
            }
            let mut span = (0.0, length);
            for edge in piece {
                let (at, rate) = (edge.inward.dot(&(origin - edge.on)), edge.inward.dot(&direction));
                keep_where(&mut span, at, rate);
            }
            if span.1 > span.0 {
                covered.push(span);
            }
        }
        for j in 1..runs.len() {
            if runs[j].reach <= 0.0 {
                continue;
            }
            // A fan is within its own round of its joint, and too far away by the same test a
            // piece is. Without this a wall about its curve, which has a fan at every joint,
            // cut every edge against every fan: 42 ms to build a thousand runs, and 30 with
            // it. The one-sided wall on the same chain has no fans to try and is the control,
            // 15 ms either way (`fieldform-gpu/examples/wall_cost.rs`, the first table).
            let from_middle = (origin + direction * (0.5 * length)) - points[j];
            if from_middle.norm() > 0.5 * length + runs[j].reach + SLIVER {
                continue;
            }
            let rel = origin - points[j];
            let (before, after) = (runs[j - 1].along, runs[j].along);
            let mut span = (0.0, length);
            keep_where(&mut span, rel.dot(&before), direction.dot(&before));
            keep_where(&mut span, -rel.dot(&after), -direction.dot(&after));
            keep_within(&mut span, rel, direction, runs[j].reach);
            if span.1 > span.0 {
                covered.push(span);
            }
        }
        for centre in &discs {
            let mut span = (0.0, length);
            keep_within(&mut span, origin - centre, direction, half);
            if span.1 > span.0 {
                covered.push(span);
            }
        }

        covered.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut reached = 0.0;
        for &(start, end) in &covered {
            if start - reached > SLIVER {
                out.push((from + direction * reached, from + direction * start));
            }
            reached = reached.max(end);
        }
        if length - reached > SLIVER {
            out.push((from + direction * reached, from + direction * length));
        }
    };

    let (mut faces, mut sides) = (Vec::new(), Vec::new());
    for (k, piece) in pieces.iter().enumerate() {
        for (e, edge) in piece.iter().enumerate() {
            // The edge itself: its line, cut to what the piece's other lines leave of it. A
            // line a mitre has cut clean away leaves nothing, and is no edge.
            let direction = left_of(edge.inward);
            let mut span = (-FAR, FAR);
            for (o, other) in piece.iter().enumerate() {
                if o != e {
                    let (at, rate) =
                        (other.inward.dot(&(edge.on - other.on)), other.inward.dot(&direction));
                    keep_where(&mut span, at, rate);
                }
            }
            if span.1 - span.0 > SLIVER {
                let out = if edge.flat { &mut faces } else { &mut sides };
                let from = edge.on + direction * span.0;
                bare(k, from, direction, span.1 - span.0, -edge.inward, out);
            }
        }
    }
    // **A fan has two straight edges of its own**, along the end fibres of the two runs it
    // sits between, and each is buried in that run's piece — until a mitre at the run's other
    // end cuts the piece away from under it. Then the fan's edge is the wall's surface, and
    // it belongs to no run.
    for j in 1..runs.len() {
        let run = runs[j];
        if run.reach <= 0.0 {
            continue;
        }
        let before = runs[j - 1].along;
        bare(usize::MAX, points[j], left_of(before) * run.fan, run.reach, -before, &mut faces);
        bare(usize::MAX, points[j], left_of(run.along) * run.fan, run.reach, run.along, &mut faces);
    }
    (faces, sides)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScalarField;
    use nalgebra::{Point3, Vector3};

    /// A wall read as a solid of endless height, which is what `gradient_deviation` and
    /// `overstatement` know how to ask. A prism's distance is its section's, so nothing is
    /// added by the third axis and nothing is hidden by it.
    struct Upright<'a>(&'a Wall);

    impl ScalarField for Upright<'_> {
        fn distance(&self, p: Point3<f64>) -> f64 {
            self.0.distance(Point2::new(p.x, p.y))
        }
    }

    const SIDES: [WallSide; 3] = [WallSide::Centre, WallSide::Left, WallSide::Right];
    const ENDS: [WallEnds; 2] = [WallEnds::Square, WallEnds::Round];

    /// A circular arc about the origin, counter-clockwise, with a joint every degree — what a
    /// tessellated curve looks like by the time it arrives here.
    fn arc(radius: f64, from_degrees: f64, to_degrees: f64) -> Vec<Point2<f64>> {
        let steps = (to_degrees - from_degrees).abs().ceil() as usize;
        (0..=steps)
            .map(|i| {
                let a = (from_degrees + (to_degrees - from_degrees) * i as f64 / steps as f64)
                    .to_radians();
                Point2::new(radius * a.cos(), radius * a.sin())
            })
            .collect()
    }

    /// Two and a half turns winding outward, 12 mm from one turn to the next. Its left is
    /// toward the middle, so every turn's left faces the right of the turn inside it.
    fn spiral() -> Vec<Point2<f64>> {
        (0..=300)
            .map(|i| {
                let a = (i as f64 * 3.0).to_radians();
                let r = 20.0 + 12.0 * a / std::f64::consts::TAU;
                Point2::new(r * a.cos(), r * a.sin())
            })
            .collect()
    }

    /// A grid over a box, leaning off its axes so no sample sits exactly on a face.
    fn grid(lo: (f64, f64), hi: (f64, f64), step: f64) -> Vec<Point2<f64>> {
        let mut out = Vec::new();
        let mut y = lo.1;
        while y <= hi.1 {
            let mut x = lo.0;
            while x <= hi.0 {
                out.push(Point2::new(x + 0.013 * y, y + 0.007 * x));
                x += step;
            }
            y += step;
        }
        out
    }

    /// The exact distance to an axis-aligned box, inside and out.
    fn box_distance(q: Point2<f64>, lo: (f64, f64), hi: (f64, f64)) -> f64 {
        let dx = (q.x - 0.5 * (lo.0 + hi.0)).abs() - 0.5 * (hi.0 - lo.0);
        let dy = (q.y - 0.5 * (lo.1 + hi.1)).abs() - 0.5 * (hi.1 - lo.1);
        if dx > 0.0 || dy > 0.0 { dx.max(0.0).hypot(dy.max(0.0)) } else { dx.max(dy) }
    }

    /// Cuts a convex polygon back to the side of a line that `inward` points to.
    fn cut(polygon: &[Point2<f64>], on: Point2<f64>, inward: Vector2<f64>) -> Vec<Point2<f64>> {
        let mut kept = Vec::new();
        for i in 0..polygon.len() {
            let (a, b) = (polygon[i], polygon[(i + 1) % polygon.len()]);
            let (da, db) = (inward.dot(&(a - on)), inward.dot(&(b - on)));
            if da >= 0.0 {
                kept.push(a);
            }
            if (da >= 0.0) != (db >= 0.0) {
                kept.push(a + (b - a) * (da / (da - db)));
            }
        }
        kept
    }

    /// The exact distance to a convex polygon given counter-clockwise: negative inside it.
    fn polygon_distance(polygon: &[Point2<f64>], q: Point2<f64>) -> f64 {
        if polygon.len() < 3 {
            return f64::INFINITY;
        }
        let (mut least, mut inside) = (f64::INFINITY, true);
        for i in 0..polygon.len() {
            let (a, b) = (polygon[i], polygon[(i + 1) % polygon.len()]);
            if (b - a).norm() < 1e-12 {
                continue;
            }
            least = least.min(to_segment(q, a, b));
            inside &= cross(b - a, q - a) >= 0.0;
        }
        if inside { -least } else { least }
    }

    /// **The wall written the long way**: every run's rectangle, cut back where it is mitred,
    /// every joint's fan and each round end's disc, each measured on its own and the least
    /// taken.
    ///
    /// Outside the wall that is the Euclidean distance to it, exactly, by the plainest route
    /// there is. Inside it is negative and means nothing more than that — two rectangles
    /// sharing a face both read nought on it — which is why the kernel is not built this way,
    /// and why only the sign of an inside reading is ever compared.
    ///
    /// Shares nothing with `Wall::distance` but the definition of a fibre and of a mitre: no
    /// feet, no sides kept apart, no boundary found in advance.
    fn pieces(
        points: &[Point2<f64>],
        thickness: f64,
        side: WallSide,
        ends: WallEnds,
        q: Point2<f64>,
    ) -> f64 {
        let (lo, hi) = open_band(thickness, side);
        let (middle, half) = (0.5 * (lo + hi), 0.5 * (hi - lo));
        let runs: Vec<(Vector2<f64>, f64)> = points
            .windows(2)
            .map(|w| ((w[1] - w[0]).normalize(), (w[1] - w[0]).norm()))
            .collect();
        let facing = facing(side);
        // Whether the joint at the start of run `j` is mitred: a turn toward a one-sided wall,
        // past a right angle.
        let mitred = |j: usize| {
            let (before, after) = (runs[j - 1].0, runs[j].0);
            let toward = if cross(before, after) >= 0.0 { facing > 0.0 } else { facing < 0.0 };
            toward && before.dot(&after) < 0.0
        };
        let mut least = f64::INFINITY;
        for (i, (along, _)) in runs.iter().enumerate() {
            let across = left_of(*along);
            let (a, b) = (points[i], points[i + 1]);
            let mut piece = vec![a + across * lo, b + across * lo, b + across * hi, a + across * hi];
            if i > 0 && mitred(i) {
                piece = cut(&piece, a, left_of(runs[i - 1].0) * facing);
            }
            if i + 1 < runs.len() && mitred(i + 1) {
                piece = cut(&piece, b, left_of(runs[i + 1].0) * facing);
            }
            least = least.min(polygon_distance(&piece, q));
        }
        for j in 1..runs.len() {
            let (before, after) = (runs[j - 1].0, runs[j].0);
            // A left turn fans out on the right, and so does a dead reversal.
            let on_left = cross(before, after) < 0.0;
            let radius = if on_left { hi } else { -lo };
            if radius <= 0.0 {
                continue;
            }
            let rel = q - points[j];
            let outward = if on_left { 1.0 } else { -1.0 };
            least = least.min(if rel.dot(&before) >= 0.0 && rel.dot(&after) <= 0.0 {
                rel.norm() - radius
            } else {
                let edge = |along: Vector2<f64>| {
                    to_segment(q, points[j], points[j] + left_of(along) * (outward * radius))
                };
                edge(before).min(edge(after))
            });
        }
        if ends == WallEnds::Round {
            let last = runs.len() - 1;
            for (at, along) in [(points[0], runs[0].0), (points[last + 1], runs[last].0)] {
                least = least.min((q - (at + left_of(along) * middle)).norm() - half);
            }
        }
        least
    }

    /// Numbers that are the same every run, since a test that fails one time in ten is worse
    /// than none.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn a_centre_wall_with_round_ends_on_a_straight_run_is_a_capsule() {
        let (a, b) = (Point2::new(0.0, 0.0), Point2::new(60.0, 0.0));
        let wall = Wall::open(vec![a, b], 8.0, WallSide::Centre, WallEnds::Round).unwrap();
        for q in grid((-20.0, -15.0), (80.0, 15.0), 2.3) {
            let capsule = to_segment(q, a, b) - 4.0;
            assert!((wall.distance(q) - capsule).abs() < 1e-12, "at {q:?}");
        }
        assert_eq!(wall.normalization(), Normalization::Distance);
    }

    /// The acceptance case the extrude will inherit: a square-ended wall on a straight run is
    /// a box, so extruded it is a cuboid — to the last digit, corners included, and for a wall
    /// to one side as much as for one about the curve.
    #[test]
    fn a_square_ended_wall_on_a_straight_run_is_a_box() {
        let chain = vec![Point2::new(0.0, 0.0), Point2::new(60.0, 0.0)];
        for (side, lo_y, hi_y) in [
            (WallSide::Centre, -4.0, 4.0),
            (WallSide::Left, 0.0, 8.0),
            (WallSide::Right, -8.0, 0.0),
        ] {
            let wall = Wall::open(chain.clone(), 8.0, side, WallEnds::Square).unwrap();
            for q in grid((-20.0, -20.0), (80.0, 20.0), 2.3) {
                let exact = box_distance(q, (0.0, lo_y), (60.0, hi_y));
                assert!(
                    (wall.distance(q) - exact).abs() < 1e-12,
                    "{side:?} at {q:?}: {} against {exact}",
                    wall.distance(q)
                );
            }
        }
    }

    /// **The check the first build did not have, and would have failed.** Chains that wander
    /// as they like — corners up to 170° either way, runs from a twentieth of the thickness to
    /// six times it — against the wall written the long way. Three things, at every probe:
    /// the same side of the surface; outside, the same distance; and inside, a depth that a
    /// ball of that radius really does fit in.
    ///
    /// Its controls are the tests above and below it, where the answer is known in closed
    /// form and both this and the reference have to give it.
    #[test]
    fn a_wall_is_the_union_of_its_fibres_whatever_the_chain() {
        let mut rng = Lcg(0x5eed_1234_abcd_0001);
        let around: Vec<Vector2<f64>> = (0..12)
            .map(|i| {
                let a = (i as f64 * 30.0 + 7.0).to_radians();
                Vector2::new(a.cos(), a.sin())
            })
            .collect();
        let (mut outside, mut inside, mut sharp, mut short) = (0usize, 0usize, 0usize, 0usize);
        for side in SIDES {
            for ends in ENDS {
                for _ in 0..40 {
                    let thickness = 2.0 + 7.0 * rng.next();
                    let mut chain = vec![Point2::new(0.0, 0.0)];
                    let mut heading = 0.0f64;
                    for _ in 0..5 {
                        let length = thickness * (0.05 + 5.95 * rng.next() * rng.next());
                        let turn = (rng.next() * 2.0 - 1.0) * (170.0f64).to_radians();
                        heading += turn;
                        sharp += (turn.abs() > std::f64::consts::FRAC_PI_2) as usize;
                        short += (length < thickness) as usize;
                        let from = chain[chain.len() - 1];
                        chain.push(from + Vector2::new(heading.cos(), heading.sin()) * length);
                    }
                    let wall = Wall::open(chain.clone(), thickness, side, ends).unwrap();
                    let (lo, hi) = wall.bounds();
                    let pad = 0.5 * thickness;
                    for j in 0..36 {
                        for i in 0..36 {
                            let q = Point2::new(
                                lo.x - pad + (hi.x - lo.x + 2.0 * pad) * (i as f64 + 0.37) / 36.0,
                                lo.y - pad + (hi.y - lo.y + 2.0 * pad) * (j as f64 + 0.61) / 36.0,
                            );
                            let truth = pieces(&chain, thickness, side, ends, q);
                            let f = wall.distance(q);
                            let said = || {
                                format!(
                                    "{side:?} {ends:?}, {thickness} mm on {chain:?} at {q:?}: \
                                     the wall reads {f} and its pieces {truth}"
                                )
                            };
                            if truth > 1e-6 {
                                outside += 1;
                                assert!((f - truth).abs() < 1e-6, "{}", said());
                            } else if truth < -1e-6 {
                                inside += 1;
                                assert!(f < 0.0, "{}", said());
                                // A depth is a ball that fits. Shrunk by a hair, so a ball
                                // exactly touching a face is not failed for rounding.
                                let fits = f.abs() * (1.0 - 1e-6);
                                for d in &around {
                                    let rim = pieces(&chain, thickness, side, ends, q + d * fits);
                                    assert!(rim < 1e-6, "too deep by a ball: {}", said());
                                }
                            }
                        }
                    }
                }
            }
        }
        // The cases the first build got wrong are the ones this has to have crossed.
        assert!(sharp > 200 && short > 200, "{sharp} sharp corners and {short} short runs");
        assert!(outside > 100_000 && inside > 20_000, "{outside} outside, {inside} inside");
    }

    /// A wall on a circle is an annulus, wherever the band is put — and it moves at unit rate
    /// in the hole as well as outside, which is what lets all three claim a distance.
    #[test]
    fn a_wall_on_a_circle_is_an_annulus_whichever_side_it_is_on() {
        let (radius, thickness) = (30.0, 6.0);
        let circle = || Profile::new(arc(radius, 0.0, 360.0)).expect("a circle closes");
        // The polygon sits inside the circle by its sagitta, which is all the two may differ by.
        let sagitta = radius * (1.0 - (0.5f64).to_radians().cos()) + 1e-9;

        // Counter-clockwise, so left is inside.
        for (side, middle) in [
            (WallSide::Centre, radius),
            (WallSide::Left, radius - 0.5 * thickness),
            (WallSide::Right, radius + 0.5 * thickness),
        ] {
            let wall = Wall::closed(circle(), thickness, side).unwrap();
            let mut outside = Vec::new();
            for i in 0..90 {
                let (a, rho) = ((i as f64 * 37.3).to_radians(), 8.0 + i as f64 * 0.47);
                let q = Point2::new(rho * a.cos(), rho * a.sin());
                let annulus = (rho - middle).abs() - 0.5 * thickness;
                assert!(
                    (wall.distance(q) - annulus).abs() < sagitta,
                    "{side:?} at radius {rho}: {} against {annulus}",
                    wall.distance(q)
                );
                if annulus > 0.5 {
                    outside.push(Point3::new(q.x, q.y, 0.0));
                }
            }
            assert!(outside.len() > 40, "the probes must cover the hole and the outside");
            let off = crate::gradient_deviation(&Upright(&wall), &outside, 1e-5);
            assert!(off < 1e-4, "{side:?} is {off} off unit rate");
            assert_eq!(wall.normalization(), Normalization::Distance);
        }
    }

    /// Inside and outside are read from the loop, not from how it happened to be drawn: the
    /// same square traced the other way round keeps its inside, and "left" changes sides.
    #[test]
    fn a_wall_on_a_loop_reads_its_side_from_the_loops_own_winding() {
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(40.0, 0.0),
            Point2::new(40.0, 40.0),
            Point2::new(0.0, 40.0),
        ];
        let backwards: Vec<Point2<f64>> = square.iter().rev().copied().collect();
        // One millimetre inside the bottom edge, and one outside it.
        let (just_in, just_out) = (Point2::new(20.0, 1.0), Point2::new(20.0, -1.0));

        for (points, inner, outer) in [
            (square, WallSide::Left, WallSide::Right),
            (backwards, WallSide::Right, WallSide::Left),
        ] {
            let on = |side| Wall::closed(Profile::new(points.clone()).unwrap(), 2.0, side).unwrap();
            assert!(on(inner).distance(just_in) < 0.0 && on(inner).distance(just_out) > 0.0);
            assert!(on(outer).distance(just_out) < 0.0 && on(outer).distance(just_in) > 0.0);
            assert_eq!(on(inner).band(), (-2.0, 0.0), "a loop's inside is its negative side");
            assert_eq!(on(outer).band(), (0.0, 2.0));
            // And the wall says which that is, whatever side it is itself on — which is what
            // lets a panel offer Inside and Outside over a document that holds left and right.
            for side in [WallSide::Centre, inner, outer] {
                assert_eq!(on(side).inside(), Some(inner));
            }
        }
        // An open chain has no inside to name.
        let open = Wall::open(
            vec![Point2::new(0.0, 0.0), Point2::new(40.0, 0.0)],
            2.0,
            WallSide::Left,
            WallEnds::Square,
        )
        .unwrap();
        assert_eq!(open.inside(), None);
    }

    /// A bow-tie's two lobes wind opposite ways and cancel, so its signed area is rounding —
    /// and it came out with either sign depending on where the loop had been moved to. Both
    /// placements here are ones a review found on opposite sides of nought.
    #[test]
    fn a_loop_with_no_winding_does_not_pick_its_side_by_where_it_sits() {
        let bow = [(-10.0, -5.0), (10.0, 5.0), (10.0, -5.0), (-10.0, 5.0)];
        let placed = |turn: f64, shift: (f64, f64)| {
            let points = bow
                .iter()
                .map(|(x, y)| {
                    Point2::new(
                        x * turn.cos() - y * turn.sin() + shift.0,
                        x * turn.sin() + y * turn.cos() + shift.1,
                    )
                })
                .collect();
            Wall::closed(Profile::new(points).unwrap(), 1.0, WallSide::Left).unwrap().band()
        };
        let at_rest = placed(0.0, (0.0, 0.0));
        for (turn, shift) in [
            (2.034701269983068, (-69.83016521509961, 30.186894607970743)),
            (2.7246761881093495, (-86.02891528507621, -81.85739733122699)),
        ] {
            assert_eq!(placed(turn, shift), at_rest, "turned {turn} and moved {shift:?}");
        }
    }

    /// A bake reads short by its margin, which is safe until it is folded: this wall, asked
    /// from the middle of its own hole, said 23.3 mm where there are 18. It walks the boundary
    /// now, so a bake on the loop changes nothing — neither the reading nor what it claims.
    #[test]
    fn a_wall_on_a_baked_loop_reads_the_loops_true_distance() {
        let square = || {
            Profile::new(vec![
                Point2::new(-20.0, -20.0),
                Point2::new(20.0, -20.0),
                Point2::new(20.0, 20.0),
                Point2::new(-20.0, 20.0),
            ])
            .unwrap()
        };
        for side in SIDES {
            let exact = Wall::closed(square(), 4.0, side).unwrap();
            let baked = Wall::closed(square().baked(64, (0.0, 0.0)), 4.0, side).unwrap();
            for q in grid((-36.0, -36.0), (36.0, 36.0), 1.7) {
                assert_eq!(baked.distance(q), exact.distance(q), "{side:?} at {q:?}");
            }
            assert_eq!(baked.normalization(), Normalization::Distance);
        }
        let about = Wall::closed(square().baked(64, (0.0, 0.0)), 4.0, WallSide::Centre).unwrap();
        assert!((about.distance(Point2::new(0.0, 0.0)) - 18.0).abs() < 1e-12);
    }

    /// Sign only, pointwise: the two one-sided walls share nothing but the curve, and between
    /// them they are the wall about it at twice the thickness.
    #[test]
    fn left_and_right_on_an_arc_are_disjoint_and_together_are_the_centre_wall_of_twice_the_thickness() {
        let chain = arc(40.0, 0.0, 120.0);
        let on = |t, side| Wall::open(chain.clone(), t, side, WallEnds::Square).unwrap();
        let (left, right, both) =
            (on(5.0, WallSide::Left), on(5.0, WallSide::Right), on(10.0, WallSide::Centre));
        let mut seen = [0usize; 3];
        for q in grid((-40.0, -20.0), (60.0, 60.0), 1.3) {
            let (l, r, c) = (left.distance(q), right.distance(q), both.distance(q));
            // On a face the sign is rounding, and a face is not what this is about.
            if l.abs().min(r.abs()).min(c.abs()) < 1e-6 {
                continue;
            }
            assert!(!(l < 0.0 && r < 0.0), "both walls claim {q:?}");
            assert_eq!(l < 0.0 || r < 0.0, c < 0.0, "at {q:?}: left {l}, right {r}, centre {c}");
            seen[0] += (l < 0.0) as usize;
            seen[1] += (r < 0.0) as usize;
            seen[2] += (c > 0.0) as usize;
        }
        assert!(seen.iter().all(|n| *n > 100), "every region must be sampled: {seen:?}");
    }

    /// D8's test, on the walls the spec expected to be bounds. Probes on both sides of a bend
    /// and beyond an end, outside the material and clear of where two faces are equally near.
    #[test]
    fn a_wall_moves_at_unit_rate_outside_its_own_material() {
        let chain = arc(40.0, 0.0, 120.0);
        let mut probes: Vec<Point2<f64>> = Vec::new();
        for i in 0..60 {
            let a = (8.0 + i as f64 * 1.73).to_radians();
            for rho in [29.0 + (i % 5) as f64, 47.0 + (i % 7) as f64] {
                probes.push(Point2::new(rho * a.cos(), rho * a.sin()));
            }
        }
        // And past the start, which is at (40, 0) heading up the page.
        probes.extend(grid((33.0, -9.0), (47.0, -2.0), 1.7));

        for side in SIDES {
            for ends in ENDS {
                let wall = Wall::open(chain.clone(), 5.0, side, ends).unwrap();
                let outside: Vec<Point3<f64>> = probes
                    .iter()
                    .filter(|q| wall.distance(**q) > 0.5)
                    .map(|q| Point3::new(q.x, q.y, 0.0))
                    .collect();
                assert!(outside.len() > 80, "{side:?} {ends:?} left {} probes", outside.len());
                let off = crate::gradient_deviation(&Upright(&wall), &outside, 1e-5);
                assert!(off < 1e-4, "{side:?} {ends:?} is {off} off unit rate");
                assert_eq!(wall.normalization(), Normalization::Distance);
            }
        }
    }

    /// The case tube coordinates get wrong, and the reason this is built from fibres. Between
    /// two turns of a spiral a point is on the right of the inner turn and the left of the
    /// outer, so the side read off whichever is nearer flips half way across.
    /// `examples/wall_construction.rs` prints what that costs; this holds the answer in place,
    /// with a wall thin enough to leave a gap and one thick enough to cross the half-way line.
    #[test]
    fn a_one_sided_wall_on_a_spiral_never_claims_more_room_than_there_is() {
        let compass: Vec<Vector3<f64>> = (0..8)
            .map(|i| {
                let a = (i as f64 * 45.0 + 11.0).to_radians();
                Vector3::new(a.cos(), a.sin(), 0.0)
            })
            .collect();
        let probes: Vec<Point3<f64>> = grid((-62.0, -62.0), (62.0, 62.0), 4.3)
            .into_iter()
            .map(|q| Point3::new(q.x, q.y, 0.0))
            .collect();
        for thickness in [4.0, 8.0] {
            for side in [WallSide::Left, WallSide::Right] {
                let wall = Wall::open(spiral(), thickness, side, WallEnds::Square).unwrap();
                let over = crate::overstatement(&Upright(&wall), &probes, &compass, 0.05, 1.2);
                assert!(over < 1e-9, "{side:?} at {thickness} mm claims {over} mm too much");
            }
        }
    }

    /// What a sharp corner becomes, said as a test because it is a choice: round on the side
    /// the corner turns away from, where the fibres fan out, and a plain mitre on the side it
    /// turns toward, where they overlap. An exact outer mitre is deferred by the spec.
    #[test]
    fn a_sharp_corner_is_round_outside_and_mitred_inside() {
        // East, then north: a left turn, so the left is the inside of the corner.
        let chain =
            vec![Point2::new(0.0, 0.0), Point2::new(50.0, 0.0), Point2::new(50.0, 50.0)];
        let on = |side| Wall::open(chain.clone(), 4.0, side, WallEnds::Square).unwrap();

        // Outside the corner the wall is a quarter-round about the corner point: a point
        // 3.5 mm from it is in, and one a mitre would have reached, 4.9 mm away, is not.
        let outer = on(WallSide::Right);
        assert!(outer.distance(Point2::new(52.5, -2.5)) < 0.0);
        let beyond = outer.distance(Point2::new(53.5, -3.5));
        assert!((beyond - ((3.5f64).hypot(3.5) - 4.0)).abs() < 1e-12, "got {beyond}");

        // Inside it the two bands meet at (46, 4), square, with nothing rounded off: the meeting
        // point is on the surface, and a point in the notch reads the nearer of the two flat
        // faces — 1.5 mm above one, 2 mm beside the other.
        let inner = on(WallSide::Left);
        assert!(inner.distance(Point2::new(46.2, 3.8)) < 0.0);
        assert!(inner.distance(Point2::new(46.0, 4.0)).abs() < 1e-12);
        let clear = inner.distance(Point2::new(44.0, 5.5));
        assert!((clear - 1.5).abs() < 1e-12, "got {clear}");
        // And the corner costs the wall nothing: no divisor, so a point 3 mm off a face well
        // away from the corner reads 3 mm, on either side.
        assert!((inner.distance(Point2::new(20.0, 7.0)) - 3.0).abs() < 1e-12);
        assert!((inner.distance(Point2::new(20.0, -3.0)) - 3.0).abs() < 1e-12);
        // And the depth is to the surface, not to the nearer run's band. In the square the two
        // bands share, a point near the curve is as deep as it is from the curve; one further
        // in is nearest the notch's own corner at (46, 4), which is on neither run's band edge
        // as that run alone would have it.
        assert!((inner.distance(Point2::new(49.0, 1.0)) + 1.0).abs() < 1e-12);
        let deep = inner.distance(Point2::new(47.5, 2.5));
        assert!((deep + (1.5f64).hypot(1.5)).abs() < 1e-12, "got {deep}");
    }

    /// A 135° corner, which is past where a corner looks after itself. Outside it the fan is
    /// whole, with no slot cut down to the corner point — the first build's fault. Inside it
    /// the two bands are mitred: neither comes through the other run onto the far side of the
    /// curve, and where they meet is a corner on the bisector, the way a draughtsman offsets
    /// two lines.
    #[test]
    fn a_corner_sharper_than_a_right_angle_is_round_outside_and_a_mitre_inside() {
        // East, then back up to the north-west: a left turn of 135°.
        let chain =
            vec![Point2::new(0.0, 0.0), Point2::new(10.0, 0.0), Point2::new(0.0, 10.0)];
        let root2 = (2.0f64).sqrt();
        for ends in ENDS {
            let on = |side| Wall::open(chain.clone(), 2.0, side, ends).unwrap();

            // 1.118 mm from the corner, in the fan: 0.882 mm deep, where it read as outside.
            let outer = on(WallSide::Right);
            let deep = outer.distance(Point2::new(10.5, -1.0));
            assert!((deep - ((1.25f64).sqrt() - 2.0)).abs() < 1e-12, "got {deep}");
            // And across the first run's end line the fan carries straight on.
            for x in [10.0, 10.000001, 10.2] {
                assert!(outer.distance(Point2::new(x, -1.0)) < -0.9, "a slot at x = {x}");
            }

            let inner = on(WallSide::Left);
            // To a tenth of a micron: the hair the boundary's own corners are found to.
            let reads = |x: f64, y: f64, expected: f64, what: &str| {
                let f = inner.distance(Point2::new(x, y));
                assert!((f - expected).abs() < 1e-7, "{ends:?}, {what}: {f}, not {expected}");
            };
            // Nothing beyond the corner: the nearest material is the corner point itself.
            reads(10.3, -1.4, (0.3f64).hypot(1.4), "beyond the corner");
            // Nothing through the second run: this was in the first run's band, where that
            // band carried on past the line of the second. It is now that far from the curve.
            reads(9.8, 1.5, 1.3 / root2, "through the second run");
            // And nothing through the first: the second run's band, the same way.
            reads(9.0, -0.5, 0.5, "through the first run");
            // Between the two runs it is solid, and as deep as it is from the nearer of them.
            reads(8.5, 1.0, -0.5 / root2, "inside the corner");
            // The bands' inner edges meet on the bisector, 2 tan 67.5° back from the corner.
            reads(10.0 - 2.0 * (1.0 + root2), 2.0, 0.0, "the mitre's own corner");
            // And beyond that corner the hollow of the V reads to the nearer inner edge.
            reads(3.0, 3.0, 4.0 / root2 - 2.0, "in the hollow");
        }
    }

    /// A return leg shorter than the wall is thick does not reach across the long run's end,
    /// so most of that end is a face. It read 3 mm clear from half a millimetre away.
    #[test]
    fn a_run_too_short_to_cover_its_neighbours_end_leaves_that_end_a_face() {
        let chain =
            vec![Point2::new(0.0, 0.0), Point2::new(50.0, 0.0), Point2::new(50.0, 3.0)];
        let left = Wall::open(chain.clone(), 8.0, WallSide::Left, WallEnds::Square).unwrap();
        assert!((left.distance(Point2::new(50.5, 6.0)) - 0.5).abs() < 1e-12);
        assert!((left.distance(Point2::new(49.9, 6.0)) + 0.1).abs() < 1e-12);
        let about = Wall::open(chain, 16.0, WallSide::Centre, WallEnds::Square).unwrap();
        assert!((about.distance(Point2::new(50.5, 6.0)) - 0.5).abs() < 1e-12);
    }

    /// The regime tube coordinates refuse — `κ·r` past one — is a shape like any other here.
    /// A 3 mm arc carrying an 8 mm wall on its inside, against the wall written the long way,
    /// and at unit rate round it.
    #[test]
    fn a_wall_thicker_than_its_bend_is_built_rather_than_refused() {
        let chain = arc(3.0, 0.0, 120.0);
        for (side, thickness) in [(WallSide::Left, 8.0), (WallSide::Centre, 12.0)] {
            let wall = Wall::open(chain.clone(), thickness, side, WallEnds::Square).unwrap();
            let (mut outside, mut inside) = (Vec::new(), 0usize);
            for q in grid((-14.0, -14.0), (14.0, 14.0), 0.23) {
                let truth = pieces(&chain, thickness, side, WallEnds::Square, q);
                let f = wall.distance(q);
                if truth > 1e-6 {
                    assert!((f - truth).abs() < 1e-6, "{side:?} at {q:?}: {f} against {truth}");
                    outside.push(Point3::new(q.x, q.y, 0.0));
                } else if truth < -1e-6 {
                    assert!(f < 0.0, "{side:?} at {q:?}: {f} against {truth}");
                    inside += 1;
                }
            }
            assert!(outside.len() > 1000 && inside > 500, "{} outside, {inside} inside", outside.len());
            let compass: Vec<Vector3<f64>> = (0..8)
                .map(|i| {
                    let a = (i as f64 * 45.0 + 11.0).to_radians();
                    Vector3::new(a.cos(), a.sin(), 0.0)
                })
                .collect();
            let sparse: Vec<Point3<f64>> = outside.iter().step_by(23).copied().collect();
            let over = crate::overstatement(&Upright(&wall), &sparse, &compass, 0.02, 1.2);
            assert!(over < 1e-9, "{side:?} claims {over} mm too much");
        }
    }

    /// A dead reversal is the limit of a sharp corner and behaves like one: nothing beyond the
    /// turn on the side the wall is not on, whichever side of the line a query falls.
    #[test]
    fn a_curve_that_doubles_back_on_itself_has_no_side_decided_by_rounding() {
        let chain =
            vec![Point2::new(0.0, 0.0), Point2::new(50.0, 0.0), Point2::new(20.0, 0.0)];
        let wall = Wall::open(chain, 4.0, WallSide::Left, WallEnds::Square).unwrap();
        for y in [1e-9, -1e-9] {
            let beyond = wall.distance(Point2::new(51.0, y));
            assert!((beyond - 1.0).abs() < 1e-6, "at y = {y}: {beyond}");
        }
    }

    /// A point a millionth of a millimetre *behind* the one before it is a reversal by the
    /// arithmetic and nothing at all by the eye, and it grew a lobe of the wall's whole
    /// thickness on the wrong side of a straight wall. Below the distance two ends may be
    /// apart and still be joined, a step is not a step.
    #[test]
    fn a_step_too_small_to_be_a_gap_is_no_corner() {
        let straight = vec![Point2::new(0.0, 0.0), Point2::new(100.0, 0.0)];
        let stray = vec![
            Point2::new(0.0, 0.0),
            Point2::new(50.0, 0.0),
            Point2::new(50.0 - 1e-9, 0.0),
            Point2::new(100.0, 0.0),
        ];
        for side in SIDES {
            let clean = Wall::open(straight.clone(), 4.0, side, WallEnds::Square).unwrap();
            let noisy = Wall::open(stray.clone(), 4.0, side, WallEnds::Square).unwrap();
            for q in grid((40.0, -8.0), (60.0, 8.0), 0.7) {
                assert!(
                    (clean.distance(q) - noisy.distance(q)).abs() < 1e-8,
                    "{side:?} at {q:?}: {} against {}",
                    noisy.distance(q),
                    clean.distance(q)
                );
            }
        }
    }

    /// The box a wall reports has to hold all of it, since an extent that is too small is
    /// geometry pruned away that is there.
    #[test]
    fn a_walls_bounds_contain_its_material() {
        let chain = arc(40.0, 0.0, 120.0);
        let circle = Profile::new(arc(30.0, 0.0, 360.0)).unwrap();
        let mut walls = Vec::new();
        for side in SIDES {
            for ends in ENDS {
                walls.push(Wall::open(chain.clone(), 5.0, side, ends).unwrap());
            }
            walls.push(Wall::closed(circle.clone(), 5.0, side).unwrap());
            // A bake's margin used to push a loop's wall out past its own box.
            walls.push(Wall::closed(circle.clone().baked(64, (0.0, 0.0)), 5.0, side).unwrap());
        }
        for wall in walls {
            let (lo, hi) = wall.bounds();
            let mut inside = 0;
            for q in grid((-60.0, -60.0), (60.0, 60.0), 0.9) {
                if wall.distance(q) < 0.0 {
                    inside += 1;
                    assert!(
                        q.x >= lo.x && q.x <= hi.x && q.y >= lo.y && q.y <= hi.y,
                        "{:?} {:?} has material at {q:?}, outside {lo:?}..{hi:?}",
                        wall.side(),
                        wall.ends()
                    );
                }
            }
            assert!(inside > 100, "the wall must have been sampled");
        }
    }

    #[test]
    fn a_wall_with_no_thickness_or_no_length_is_refused_by_the_number_to_change() {
        let chain = vec![Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)];
        for bad in [0.0, -1.0, f64::NAN] {
            let refused = Wall::open(chain.clone(), bad, WallSide::Centre, WallEnds::Square);
            assert!(matches!(refused, Err(WallError::NoThickness(_))), "{bad} was accepted");
        }
        let said = WallError::NoThickness(-1.0).to_string();
        assert!(said.contains("-1") && said.contains("thickness"), "{said}");

        // A point repeated is one point, and one point has no direction to be a wall along.
        let dot = vec![Point2::new(3.0, 3.0), Point2::new(3.0, 3.0)];
        assert_eq!(
            Wall::open(dot, 2.0, WallSide::Centre, WallEnds::Round).err(),
            Some(WallError::TooFewPoints(1))
        );

        // And a point that is not a number is named, rather than turned into a wall that
        // answers infinity to an ordinary question.
        let broken = vec![Point2::new(0.0, 0.0), Point2::new(f64::INFINITY, 0.0)];
        assert_eq!(
            Wall::open(broken, 2.0, WallSide::Centre, WallEnds::Square).err(),
            Some(WallError::PointNotFinite(1))
        );
    }
}

/// A wall answering from a grid, and what the grid owes: the same things a baked profile's
/// does, asked of the second thing a grid has ever sampled.
#[cfg(test)]
mod bake_tests {
    use super::*;

    /// Numbers that are the same every run.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Walls of every kind a grid might be asked to hold: wandering chains with sharp corners
    /// and short runs in them, to each side and with both ends, and a loop with a corner that
    /// turns the other way from the rest. Thick, so that there is material deeper than the
    /// fine band reaches and the coarse level has an inside to answer for.
    fn walls() -> Vec<(String, Wall)> {
        let mut rng = Lcg(0x5eed_ba4e_0000_0001);
        let mut out = Vec::new();
        for side in [WallSide::Centre, WallSide::Left, WallSide::Right] {
            for ends in [WallEnds::Square, WallEnds::Round] {
                for _ in 0..3 {
                    let thickness = 8.0 + 8.0 * rng.next();
                    let mut chain = vec![Point2::new(0.0, 0.0)];
                    let mut heading = 0.0f64;
                    for _ in 0..5 {
                        let length = thickness * (0.3 + 2.7 * rng.next());
                        heading += (rng.next() * 2.0 - 1.0) * (150.0f64).to_radians();
                        let from = chain[chain.len() - 1];
                        chain.push(from + Vector2::new(heading.cos(), heading.sin()) * length);
                    }
                    let wall = Wall::open(chain.clone(), thickness, side, ends).unwrap();
                    out.push((format!("{side:?} {ends:?}, {thickness:.2} mm on {chain:?}"), wall));
                }
            }
            let ell = [(-45.0, -35.0), (40.0, -35.0), (40.0, -5.0), (-10.0, -5.0), (-10.0, 40.0), (-45.0, 40.0)];
            let profile = Profile::new(ell.iter().map(|&(x, y)| Point2::new(x, y)).collect())
                .expect("an L closes");
            out.push((format!("{side:?} about an L"), Wall::closed(profile, 12.0, side).unwrap()));
        }
        out
    }

    /// **A baked wall never says a point is further from its surface than it is — from outside
    /// the wall, or from inside it.** Outside is a ray reaching the wall; inside is a ray
    /// crossing a slot the wall was cut out of, which is marched by the depth. The grid is
    /// sampled from [`Wall::walk`], so that is the truth it is held to, over the box and well
    /// beyond it, where the reading is worked out from the grid's edge.
    #[test]
    fn a_baked_wall_never_overstates_the_distance_from_either_side() {
        let (mut outside, mut inside, mut deep) = (0usize, 0usize, 0usize);
        for (what, wall) in walls() {
            assert_eq!(wall.normalization(), Normalization::Distance, "{what}");
            let baked = wall.clone().baked(1024, (0.0, 0.0));
            assert_eq!(baked.normalization(), Normalization::LowerBound, "{what}");
            let margin = baked.bake().expect("it was just baked").margin();
            // Deeper than the fine band can reach, as the profile's own test reckons it.
            let band = margin * 2.0 * 8.0 * 2.0;

            let (lo, hi) = wall.bounds();
            let pad = 0.6 * (hi - lo).max();
            for j in 0..110 {
                for i in 0..110 {
                    let q = Point2::new(
                        lo.x - pad + (hi.x - lo.x + 2.0 * pad) * (i as f64 + 0.37) / 110.0,
                        lo.y - pad + (hi.y - lo.y + 2.0 * pad) * (j as f64 + 0.61) / 110.0,
                    );
                    let (told, truth) = (baked.distance(q), wall.walk(q));
                    let said = || format!("{what} at {q:?}: the grid says {told}, the wall {truth}");
                    if truth >= 0.0 {
                        outside += 1;
                        assert!(told <= truth + 1e-9, "claims room it has not got: {}", said());
                    } else {
                        inside += 1;
                        // In the band the margin is one-sided, so that the wall has one surface.
                        assert!(told.abs() <= truth.abs() + 2.0 * margin, "{}", said());
                        if truth < -band {
                            deep += 1;
                            assert!(told.abs() <= truth.abs() + 1e-9, "too deep: {}", said());
                        }
                    }
                    // And it is the wall that was baked: near the surface, short by little.
                    if truth.abs() < margin {
                        assert!(truth - told < 3.0 * margin, "{}", said());
                    }
                }
            }
        }
        assert!(outside > 100_000 && inside > 5_000, "{outside} outside, {inside} inside");
        assert!(deep > 300, "only {deep} probes were past the band, in the coarse level's care");
    }

    /// A grid is kept under the wall's fingerprint, so the fingerprint has to move with
    /// everything the distance is made of and with nothing else — and must not be a profile's.
    #[test]
    fn a_walls_fingerprint_is_everything_its_distance_is_made_of() {
        let chain = || vec![Point2::new(0.0, 0.0), Point2::new(40.0, 0.0), Point2::new(55.0, 30.0)];
        let wall = |points: Vec<Point2<f64>>, thickness, side, ends| {
            Wall::open(points, thickness, side, ends).unwrap().fingerprint()
        };
        let base = wall(chain(), 4.0, WallSide::Left, WallEnds::Square);
        assert_eq!(base, wall(chain(), 4.0, WallSide::Left, WallEnds::Square), "the same wall");

        let mut moved = chain();
        moved[1].y += 1e-9;
        for (what, other) in [
            ("a thickness", wall(chain(), 4.5, WallSide::Left, WallEnds::Square)),
            ("a side", wall(chain(), 4.0, WallSide::Right, WallEnds::Square)),
            ("its ends", wall(chain(), 4.0, WallSide::Left, WallEnds::Round)),
            ("a point", wall(moved, 4.0, WallSide::Left, WallEnds::Square)),
        ] {
            assert_ne!(base, other, "{what} changed and the fingerprint did not");
        }

        // The same three points as a loop are a different wall, and neither is the profile
        // whose points they are — which shares the cache and holds a different field.
        let profile = Profile::new(chain()).expect("three points close");
        let around = Wall::closed(profile.clone(), 4.0, WallSide::Left).unwrap().fingerprint();
        assert_ne!(base, around, "an open chain and a loop over the same points");
        assert_ne!(around, profile.fingerprint(), "a wall and the profile it runs along");
    }
}
