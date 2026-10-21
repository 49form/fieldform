// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where a pattern's copies are: the one answer the field, the window and the shader all read.
//!
//! A pattern is a body repeated, and "repeated" used to mean one thing — `n` copies a step apart,
//! or `n` copies `360/n` apart, the first of them the body itself. Options let a row's step be its
//! whole run, centre the copies on the body, stop a ring short of a full turn, repeat a row along a
//! second step to make a grid, and leave some copies out. Each of those changes *where copy `i` is*, and a quantity read from three places is
//! three chances to read it differently (the failure `pattern_reach` was made public to end), so
//! the arithmetic is here, once, and everything else asks.
//!
//! # What a copy's position is
//!
//! Copy `i`, counted from 0, stands at `(i + bias)` steps from the body: along the step for a row,
//! round the axis for a ring. `bias` is nought unless the pattern is **symmetric**, when it is
//! `−⌊(n − 1)/2⌋` so the copies run either side of the body and one of them stands where the body
//! does. A grid is two of these at right
//! angles to nothing in particular: copy `i` is column `i mod n` of row `i div n`, and each of the
//! two has its own step and its own bias. Everything else follows from
//! that one affine map — the fold that finds a point's copy counts from the body's position plus
//! `bias` steps, the placements are built from it, and the shader is handed the same two numbers.
//!
//! # Which copies exist
//!
//! A copy is **left out** when the skip list names it, and the copy standing where the body does
//! is left out too when the block is not to hold the body (`join: false`) — which is "instance
//! zero" for a pattern that is not symmetric, and `⌊(n − 1)/2⌋` for one that is.
//!
//! # Why the count is the total, even when that costs exact symmetry
//!
//! **Ruled 2026-10-09, on what every CAD tool means by a count.** A quantity is the number of
//! bodies you end up with, the seed among them, in the mainstream CAD tools alike. Three
//! things cannot all hold at an even count: that the count is the total, that the run is centred
//! on the body, and that the body does not move. Mainstream tools give up the second — the
//! instances are biased half a run to one side — because a seed that moves when you tick
//! "symmetric" is the one surprise nobody accepts.
//!
//! So `bias` is `−⌊(n − 1)/2⌋`, an integer. An odd count is exactly centred and the middle copy
//! stands on the body. An even count puts one more copy above the body than below, and the count
//! is still the number typed.
//!
//! It used to be `−(n − 1)/2` exactly, half a step at an even count, so four centred copies stood
//! at ±0.5 and ±1.5 steps with the body between them: perfectly symmetric, and **five bodies from
//! a count of four**. The count meant the total at an odd number and one less than the total at
//! an even one, which is the kind of difference nobody reads off a picture.

use crate::graph_field::{resolve_frame, resolve_point};
use fieldform_scene::{
    pattern_count, CountError, Document, Graph, Node, NodeId, PatternKind, Spacing,
    FULL_TURN_DEGREES,
};
use nalgebra::{Isometry3, Point3, Rotation3, Unit, Vector3};

/// How far from a full turn a ring's angle has to be before it is an arc rather than a turn.
///
/// Angles are typed in degrees, so this is far below anything a person means and well above the
/// rounding of `360.0 / n * n`.
const FULL_TURN_SLACK_DEGREES: f64 = 1e-9;

/// The copy standing where the body does: the first for a pattern that starts at the body, and
/// `⌊(n − 1)/2⌋` for one centred on it.
///
/// Always one, since the bias is a whole number of steps — see the module's note on why the count
/// is the total. It returns an `Option` because the panel's caller reads it that way and a pattern
/// of nought copies has none.
///
/// **A function of its own because the panel draws a dot for every copy and has to know which one
/// cannot be skipped**, and a second statement of this rule beside the layout's is the kind that
/// drifts.
pub fn pattern_body_copy(total: u32, symmetric: bool) -> Option<u32> {
    if total == 0 {
        return None;
    }
    Some(if symmetric { (total - 1) / 2 } else { 0 })
}

/// The same for a grid of `columns × rows`, where copy `i` is column `i mod columns` of row
/// `i div columns`: each direction's own middle, by the same rule.
pub fn pattern_body_copy_grid(columns: u32, rows: u32, symmetric: bool) -> Option<u32> {
    if columns == 0 || rows == 0 {
        return None;
    }
    Some(if symmetric {
        (columns - 1) / 2 + columns * ((rows - 1) / 2)
    } else {
        0
    })
}

/// The two vectors that read a point's coordinates off a grid: `d1·step = 1`, `d1·step2 = 0`, and
/// `d2` the other way about, so for a point `p` (with the body at the origin) `p·d1` is how many
/// steps along the first it is and `p·d2` how many along the second — whatever angle the two make
/// and however far off their plane `p` is. `None` when the two run along one line, where there is no
/// grid to read.
pub fn grid_dual(step: &Vector3<f64>, step2: &Vector3<f64>) -> Option<[Vector3<f64>; 2]> {
    let normal = step.cross(step2);
    if !(normal.norm() > 1e-9 * step.norm() * step2.norm()) {
        return None;
    }
    let d1 = step2.cross(&normal) / step.dot(&step2.cross(&normal));
    let d2 = normal.cross(step) / step2.dot(&normal.cross(step));
    Some([d1, d2])
}

/// The geometry of one pattern's copies, in the units the field works in.
#[derive(Clone, Debug)]
pub enum PatternShape {
    /// Copies along a line.
    Row {
        /// The gap between neighbours, after [`Spacing::Extent`] has divided the run.
        step: Vector3<f64>,
        count: u32,
        /// Steps from the body to copy 0: nought, or `−(count − 1)/2` when symmetric.
        bias: f64,
    },
    /// A row repeated along a second step: `count` columns by `count2` rows.
    Grid {
        step: Vector3<f64>,
        count: u32,
        bias: f64,
        step2: Vector3<f64>,
        count2: u32,
        bias2: f64,
    },
    /// Copies about an axis.
    Ring {
        /// A point on the axis.
        at: Point3<f64>,
        /// The rotation that takes +Z onto the axis, as the frame resolves it.
        axis: Rotation3<f64>,
        count: u32,
        /// The angle between neighbours, in radians.
        step: f64,
        /// Steps from the body to copy 0, as for a row.
        bias: f64,
        /// Whether the copies stop short of a full turn. A turn wraps round onto itself and an
        /// arc does not, which changes how a point finds its nearest copy.
        arc: bool,
    },
}

/// Why a pattern has no layout.
#[derive(Clone, Debug, PartialEq)]
pub enum PatternRefusal {
    /// The count did not work out. See [`CountError`].
    Count(CountError),
    /// A number the layout needs would not resolve — the step, the axis, the angle.
    Unresolved(&'static str),
    /// A row whose step is its whole run, with one copy: there is no gap to divide into.
    ExtentOfOne,
    /// A ring that stops short of a full turn, with one copy: `angle/(n − 1)` is a division by
    /// nought and the copy could be anywhere in the arc.
    ArcOfOne,
    /// An angle that is not a positive number of degrees.
    Angle(f64),
    /// The skip list names the copy that stands where the body does. Leaving it out would delete
    /// the body from a pattern that is meant to include it.
    SkipsBody(u32),
    /// Nothing is left: every copy is skipped or is the body the block is not holding.
    NothingLeft,
    /// A second step that runs along the first: there is no grid in a line.
    ParallelSteps,
    /// A grid whose copies are rounded into each other. The fold a row or ring rounds by sums over a
    /// window of copies along one axis; over two it would sum a window of windows, with its own
    /// bound to prove, and is not built — refused rather than drawn hard.
    RoundedGrid,
    /// More copies than a pattern builds: the columns times the rows.
    TooManyCopies(u64),
}

impl PatternRefusal {
    /// What to say, naming the thing that was typed.
    pub fn say(&self) -> String {
        match self {
            PatternRefusal::Count(e) => e.say(),
            PatternRefusal::Unresolved(what) => {
                format!("a pattern's {what} did not work out — check the expression driving it")
            }
            PatternRefusal::ExtentOfOne => {
                "a pattern spaced by its whole run needs at least two copies — with one there is \
                 no gap to divide the run into"
                    .to_string()
            }
            PatternRefusal::ArcOfOne => {
                "a pattern spread over part of a turn needs at least two copies — with one there \
                 is nothing to spread"
                    .to_string()
            }
            PatternRefusal::Angle(v) => {
                format!("a pattern's angle is {v}°, and it has to be a positive number of degrees")
            }
            PatternRefusal::SkipsBody(i) => format!(
                "copy {i} stands where the body does and cannot be skipped — leave the body out of \
                 the block instead, or move the pattern so the body is not one of its copies"
            ),
            PatternRefusal::NothingLeft => {
                "every copy of this pattern is skipped, so it would hold nothing".to_string()
            }
            PatternRefusal::ParallelSteps => {
                "a pattern's two steps run along the same line, so they make no grid — turn the \
                 second one away from the first"
                    .to_string()
            }
            PatternRefusal::RoundedGrid => {
                "a pattern over two directions cannot round its copies into each other yet — set \
                 the fillet to nought, or use one direction"
                    .to_string()
            }
            PatternRefusal::TooManyCopies(n) => format!(
                "a pattern of {n} copies is over the {} a pattern builds",
                fieldform_scene::PATTERN_COUNT_MAX
            ),
        }
    }
}

/// Where a pattern's copies are and which of them exist. See the module's account.
#[derive(Clone, Debug)]
pub struct PatternLayout {
    pub shape: PatternShape,
    /// How many copies there are before any is left out.
    pub total: u32,
    /// For each copy, whether the block leaves it out: skipped by name, or the body's own copy in
    /// a block that does not hold the body. Always `total` long.
    pub excluded: Vec<bool>,
    /// The copy standing where the body does, if one does.
    pub body: Option<u32>,
}

impl PatternLayout {
    /// The layout of the pattern at `node`, resolved through the document.
    ///
    /// Panics on a node that is not a pattern: asking for one is a programming error, and a quiet
    /// answer would be a pattern of nothing.
    pub fn of(document: &Document, graph: &Graph, node: NodeId) -> Result<Self, PatternRefusal> {
        let Some(Node::Pattern { count, kind, join, options, fillet, .. }) = graph.node(node) else {
            panic!("a pattern's layout was asked of something that is not a pattern");
        };
        let numbers = |r: &fieldform_scene::ParamRef| {
            r.resolve(&|id| document.parameter_value(id))
        };
        let (columns, _) = pattern_count(document, count).map_err(PatternRefusal::Count)?;
        // A whole number of steps, so a copy always stands where the body does and the count is
        // always the total — see the module's note. Integer division, deliberately.
        let centre = |n: u32| if options.symmetric { -f64::from((n.max(1) - 1) / 2) } else { 0.0 };
        // A step read as a gap, or as the whole run divided by one less than its count.
        let spaced = |step: Vector3<f64>, n: u32| match options.spacing {
            Spacing::Step => Ok(step),
            Spacing::Extent if n > 1 => Ok(step / (f64::from(n) - 1.0)),
            Spacing::Extent => Err(PatternRefusal::ExtentOfOne),
        };
        let mut total = columns;
        let shape = match kind {
            PatternKind::Linear { step } => {
                let step = resolve_point(step, &numbers)
                    .map_err(|_| PatternRefusal::Unresolved("step"))?
                    .coords;
                let step = spaced(step, columns)?;
                match &options.second {
                    None => PatternShape::Row { step, count: columns, bias: centre(columns) },
                    Some(second) => {
                        let (rows, _) = pattern_count(document, &second.count)
                            .map_err(PatternRefusal::Count)?;
                        let step2 = resolve_point(&second.step, &numbers)
                            .map_err(|_| PatternRefusal::Unresolved("second step"))?
                            .coords;
                        let step2 = spaced(step2, rows)?;
                        if grid_dual(&step, &step2).is_none() {
                            return Err(PatternRefusal::ParallelSteps);
                        }
                        let product = u64::from(columns) * u64::from(rows);
                        if product > u64::from(fieldform_scene::PATTERN_COUNT_MAX) {
                            return Err(PatternRefusal::TooManyCopies(product));
                        }
                        if numbers(&fillet.radius).is_some_and(|k| k > 0.0) {
                            return Err(PatternRefusal::RoundedGrid);
                        }
                        total = columns * rows;
                        PatternShape::Grid {
                            step,
                            count: columns,
                            bias: centre(columns),
                            step2,
                            count2: rows,
                            bias2: centre(rows),
                        }
                    }
                }
            }
            PatternKind::Radial { frame } => {
                let (at, axis) = resolve_frame(document, graph, frame, &numbers)
                    .map_err(|_| PatternRefusal::Unresolved("axis"))?;
                let degrees = numbers(&options.angle).ok_or(PatternRefusal::Unresolved("angle"))?;
                if !degrees.is_finite() || degrees <= 0.0 {
                    return Err(PatternRefusal::Angle(degrees));
                }
                let arc = degrees < FULL_TURN_DEGREES - FULL_TURN_SLACK_DEGREES;
                let step = if !arc {
                    std::f64::consts::TAU / f64::from(columns)
                } else if columns > 1 {
                    degrees.to_radians() / (f64::from(columns) - 1.0)
                } else {
                    return Err(PatternRefusal::ArcOfOne);
                };
                PatternShape::Ring {
                    at: Point3::from(at),
                    axis,
                    count: columns,
                    step,
                    bias: centre(columns),
                    arc,
                }
            }
        };

        let body = match &shape {
            PatternShape::Grid { count, count2, .. } => {
                pattern_body_copy_grid(*count, *count2, options.symmetric)
            }
            _ => pattern_body_copy(total, options.symmetric),
        };
        if let Some(body) = body
            && options.skip.contains(&body)
        {
            return Err(PatternRefusal::SkipsBody(body));
        }
        let mut excluded = vec![false; total as usize];
        for &i in &options.skip {
            // A number past the last copy is kept in the document and means nothing here, so
            // shrinking the count does not erase what was chosen.
            if let Some(slot) = excluded.get_mut(i as usize) {
                *slot = true;
            }
        }
        if !*join
            && total > 1
            && let Some(body) = body
        {
            excluded[body as usize] = true;
        }
        if excluded.iter().all(|e| *e) {
            return Err(PatternRefusal::NothingLeft);
        }
        Ok(Self { shape, total, excluded, body })
    }

    /// The layout of the pattern at `node`, or a harmless one-copy stand-in when it has none.
    ///
    /// **For the shader generator, which must always emit *something*.** A pattern the layout
    /// refuses has already stopped the field from compiling, so nothing generated for it is ever
    /// marched; the generator only has to reach that refusal without a NaN or a loop of no length.
    /// The stand-in is one copy at the body with a step of nothing, which is what the generator did
    /// for an unresolved step before options existed.
    pub fn of_or_inert(document: &Document, graph: &Graph, node: NodeId) -> Self {
        Self::of(document, graph, node).unwrap_or_else(|_| {
            let kind = match graph.node(node) {
                Some(Node::Pattern { kind, .. }) => *kind,
                _ => panic!("a pattern's layout was asked of something that is not a pattern"),
            };
            let shape = match kind {
                PatternKind::Linear { .. } => PatternShape::Row {
                    step: Vector3::zeros(),
                    count: 1,
                    bias: 0.0,
                },
                PatternKind::Radial { .. } => PatternShape::Ring {
                    at: Point3::origin(),
                    axis: Rotation3::identity(),
                    count: 1,
                    step: std::f64::consts::TAU,
                    bias: 0.0,
                    arc: false,
                },
            };
            Self { shape, total: 1, excluded: vec![false], body: Some(0) }
        })
    }

    /// The layout of a pattern with none of the options, from the kind and count a caller already
    /// has. For the call sites that predate options and ask the fold's questions of a bare kind.
    ///
    /// `join` is taken as `true`: nothing is left out.
    pub(crate) fn plain(
        document: &Document,
        graph: &Graph,
        kind: &PatternKind,
        count: u32,
    ) -> Result<Self, PatternRefusal> {
        let numbers = |r: &fieldform_scene::ParamRef| {
            r.resolve(&|id| document.parameter_value(id))
        };
        let count = count.max(1);
        let shape = match kind {
            PatternKind::Linear { step } => PatternShape::Row {
                step: resolve_point(step, &numbers)
                    .map_err(|_| PatternRefusal::Unresolved("step"))?
                    .coords,
                count,
                bias: 0.0,
            },
            PatternKind::Radial { frame } => {
                let (at, axis) = resolve_frame(document, graph, frame, &numbers)
                    .map_err(|_| PatternRefusal::Unresolved("axis"))?;
                PatternShape::Ring {
                    at: Point3::from(at),
                    axis,
                    count,
                    step: std::f64::consts::TAU / f64::from(count),
                    bias: 0.0,
                    arc: false,
                }
            }
        };
        Ok(Self { shape, total: count, excluded: vec![false; count as usize], body: Some(0) })
    }

    /// Where copy `i` stands, as the motion that takes the body there.
    ///
    /// The field stores the inverse of this, since it asks the body about the point carried the
    /// other way.
    pub(crate) fn placement(&self, i: u32) -> Isometry3<f64> {
        match &self.shape {
            PatternShape::Row { step, bias, .. } => {
                let k = f64::from(i) + bias;
                Isometry3::translation(step.x * k, step.y * k, step.z * k)
            }
            PatternShape::Grid { step, count, bias, step2, bias2, .. } => {
                // Column `i mod count` of row `i div count`.
                let (a, b) = (f64::from(i % count) + bias, f64::from(i / count) + bias2);
                let at = step * a + step2 * b;
                Isometry3::translation(at.x, at.y, at.z)
            }
            PatternShape::Ring { at, axis, step, bias, .. } => {
                // The frame's own axis, which is what its rotation carries +Z onto. A ring could
                // once only spread about X, Y or Z; a frame says any direction, and says it the
                // same way every other block does.
                let unit = Unit::new_normalize(axis * Vector3::z());
                let turn = Rotation3::from_axis_angle(&unit, step * (f64::from(i) + bias));
                // About the stated point, not the world origin: rotate the offset from it and put
                // it back.
                Isometry3::from_parts((at.coords - turn * at.coords).into(), turn.into())
            }
        }
    }

    /// How many copies the block builds.
    pub(crate) fn held(&self) -> usize {
        self.excluded.iter().filter(|e| !**e).count()
    }

    /// How many it leaves out.
    pub fn left_out(&self) -> usize {
        self.total as usize - self.held()
    }

    /// Steps from the body to copy 0.
    pub fn bias(&self) -> f64 {
        match &self.shape {
            PatternShape::Row { bias, .. }
            | PatternShape::Grid { bias, .. }
            | PatternShape::Ring { bias, .. } => *bias,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldform_scene::{
        Fillet, Frame, ParamRef, PatternOptions, PatternSecond, Place,
    };

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    /// One pattern of a unit sphere, in a graph of its own, and the node that holds it.
    fn pattern(
        kind: PatternKind,
        count: u32,
        join: bool,
        options: PatternOptions,
    ) -> (Document, Graph, NodeId) {
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(1.0) });
        let id = graph.add(Node::Pattern {
            input: ball.into(),
            count: lit(f64::from(count)),
            kind,
            join,
            fillet: Fillet::hard(),
            options,
        });
        (Document::new(), graph, id)
    }

    fn row(step: f64) -> PatternKind {
        PatternKind::Linear { step: [lit(step), lit(0.0), lit(0.0)] }
    }

    fn ring() -> PatternKind {
        PatternKind::Radial { frame: Frame::default() }
    }

    /// Where each copy's origin ends up, as `(x, y)`, to read a layout off the numbers.
    fn centres(layout: &PatternLayout) -> Vec<(f64, f64)> {
        (0..layout.total)
            .map(|i| {
                let p = layout.placement(i) * Point3::new(1.0, 0.0, 0.0);
                (p.x, p.y)
            })
            .collect()
    }

    fn degrees(layout: &PatternLayout) -> Vec<f64> {
        centres(layout).iter().map(|(x, y)| y.atan2(*x).to_degrees().rem_euclid(360.0)).collect()
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// A ring of 4 over a full turn puts its copies at 0, 90, 180 and 270 degrees.
    #[test]
    fn four_copies_round_a_full_turn_are_a_quarter_turn_apart() {
        let (doc, graph, id) = pattern(ring(), 4, true, PatternOptions::default());
        let layout = PatternLayout::of(&doc, &graph, id).unwrap();
        let got = degrees(&layout);
        for (g, want) in got.iter().zip([0.0, 90.0, 180.0, 270.0]) {
            assert!(near(*g, want), "{got:?}");
        }
    }

    /// **Over less than a turn the copies are `angle/(n − 1)` apart**, so the last stands at the far
    /// end of the angle — 0, 60, 120, 180 for four over half a turn. The full-turn rule would put
    /// them at 0, 45, 90, 135 and leave the arc's far end empty, which is what this is for.
    #[test]
    fn four_copies_over_half_a_turn_end_at_the_far_end_of_it() {
        let options = PatternOptions { angle: lit(180.0), ..PatternOptions::default() };
        let (doc, graph, id) = pattern(ring(), 4, true, options);
        let layout = PatternLayout::of(&doc, &graph, id).unwrap();
        let got = degrees(&layout);
        for (g, want) in got.iter().zip([0.0, 60.0, 120.0, 180.0]) {
            assert!(near(*g, want), "{got:?}");
        }
    }

    /// An angle of a full turn or more is a full turn: `n` copies `360/n` apart, never `n − 1`
    /// gaps over 360 degrees, which would stack the last copy on the first.
    #[test]
    fn an_angle_of_a_full_turn_or_more_is_a_full_turn() {
        for angle in [360.0, 540.0] {
            let options = PatternOptions { angle: lit(angle), ..PatternOptions::default() };
            let (doc, graph, id) = pattern(ring(), 4, true, options);
            let layout = PatternLayout::of(&doc, &graph, id).unwrap();
            let got = degrees(&layout);
            for (g, want) in got.iter().zip([0.0, 90.0, 180.0, 270.0]) {
                assert!(near(*g, want), "{angle}: {got:?}");
            }
        }
    }

    /// A partial angle with one copy has nothing to spread and is refused by name.
    #[test]
    fn a_partial_angle_with_one_copy_is_refused() {
        let options = PatternOptions { angle: lit(90.0), ..PatternOptions::default() };
        let (doc, graph, id) = pattern(ring(), 1, true, options);
        assert_eq!(PatternLayout::of(&doc, &graph, id).unwrap_err(), PatternRefusal::ArcOfOne);
    }

    /// A row of 5 with the step as the whole run puts the last copy exactly one run away; with
    /// the step as a gap it is four steps away.
    #[test]
    fn a_whole_run_ends_one_run_away_and_a_gap_ends_four_gaps_away() {
        let extent = PatternOptions { spacing: Spacing::Extent, ..PatternOptions::default() };
        let (doc, graph, id) = pattern(row(100.0), 5, true, extent);
        let layout = PatternLayout::of(&doc, &graph, id).unwrap();
        assert!(near(centres(&layout)[4].0 - 1.0, 100.0), "{:?}", centres(&layout));
        assert!(near(centres(&layout)[1].0 - 1.0, 25.0));

        let (doc, graph, id) = pattern(row(100.0), 5, true, PatternOptions::default());
        let layout = PatternLayout::of(&doc, &graph, id).unwrap();
        assert!(near(centres(&layout)[4].0 - 1.0, 400.0), "{:?}", centres(&layout));
    }

    /// A whole run of one copy has no gap to divide into.
    #[test]
    fn a_whole_run_with_one_copy_is_refused() {
        let extent = PatternOptions { spacing: Spacing::Extent, ..PatternOptions::default() };
        let (doc, graph, id) = pattern(row(100.0), 1, true, extent);
        assert_eq!(PatternLayout::of(&doc, &graph, id).unwrap_err(), PatternRefusal::ExtentOfOne);
    }

    /// Symmetric puts three copies at −1, 0 and +1 steps, and four at −1.5, −0.5, +0.5, +1.5.
    #[test]
    fn symmetric_copies_run_evenly_either_side_of_the_body() {
        let symmetric = PatternOptions { symmetric: true, ..PatternOptions::default() };
        let (doc, graph, id) = pattern(row(10.0), 3, true, symmetric.clone());
        let layout = PatternLayout::of(&doc, &graph, id).unwrap();
        let xs: Vec<f64> = centres(&layout).iter().map(|c| c.0 - 1.0).collect();
        for (x, want) in xs.iter().zip([-10.0, 0.0, 10.0]) {
            assert!(near(*x, want), "{xs:?}");
        }
        assert_eq!(layout.body, Some(1), "the middle copy is the body");

        // An even count cannot be both centred on the body and a count of the bodies it yields,
        // and the count wins: four copies at −10, 0, +10, +20, one of them on the body, so four
        // bodies are drawn. It used to be ±5 and ±15 with the body between them, which was five
        // bodies from a count of four.
        let (doc, graph, id) = pattern(row(10.0), 4, true, symmetric);
        let layout = PatternLayout::of(&doc, &graph, id).unwrap();
        let xs: Vec<f64> = centres(&layout).iter().map(|c| c.0 - 1.0).collect();
        for (x, want) in xs.iter().zip([-10.0, 0.0, 10.0, 20.0]) {
            assert!(near(*x, want), "{xs:?}");
        }
        assert_eq!(layout.body, Some(1), "and one copy stands on the body, as at every count");
    }

    /// A symmetric arc spreads about the body: three over 120 degrees at −60, 0 and +60.
    #[test]
    fn a_symmetric_arc_is_spread_about_the_body() {
        let options =
            PatternOptions { symmetric: true, angle: lit(120.0), ..PatternOptions::default() };
        let (doc, graph, id) = pattern(ring(), 3, true, options);
        let layout = PatternLayout::of(&doc, &graph, id).unwrap();
        let got: Vec<f64> = centres(&layout)
            .iter()
            .map(|(x, y)| y.atan2(*x).to_degrees())
            .collect();
        for (g, want) in got.iter().zip([-60.0, 0.0, 60.0]) {
            assert!(near(*g, want), "{got:?}");
        }
    }

    /// Skip leaves out exactly the numbers named, and numbers past the end mean nothing.
    #[test]
    fn skip_leaves_out_exactly_the_copies_named() {
        let options = PatternOptions { skip: vec![2, 5, 99], ..PatternOptions::default() };
        let (doc, graph, id) = pattern(row(10.0), 6, true, options);
        let layout = PatternLayout::of(&doc, &graph, id).unwrap();
        let out: Vec<usize> =
            layout.excluded.iter().enumerate().filter(|(_, e)| **e).map(|(i, _)| i).collect();
        assert_eq!(out, vec![2, 5], "and 99, past the end, left nothing out");
        assert_eq!(layout.held(), 4);
    }

    /// The copy standing on the body cannot be skipped: the first, or the middle when centred.
    #[test]
    fn the_copy_on_the_body_cannot_be_skipped() {
        let options = PatternOptions { skip: vec![0], ..PatternOptions::default() };
        let (doc, graph, id) = pattern(row(10.0), 4, true, options);
        assert_eq!(PatternLayout::of(&doc, &graph, id).unwrap_err(), PatternRefusal::SkipsBody(0));

        let options =
            PatternOptions { symmetric: true, skip: vec![1], ..PatternOptions::default() };
        let (doc, graph, id) = pattern(row(10.0), 3, true, options);
        assert_eq!(PatternLayout::of(&doc, &graph, id).unwrap_err(), PatternRefusal::SkipsBody(1));

        // At an even centred count a copy stands on the body as well, and it is copy 1 there.
        let options =
            PatternOptions { symmetric: true, skip: vec![1], ..PatternOptions::default() };
        let (doc, graph, id) = pattern(row(10.0), 4, true, options);
        assert_eq!(PatternLayout::of(&doc, &graph, id).unwrap_err(), PatternRefusal::SkipsBody(1));

        // And one further out may go, which is what the refusal above is distinguishing.
        let options =
            PatternOptions { symmetric: true, skip: vec![3], ..PatternOptions::default() };
        let (doc, graph, id) = pattern(row(10.0), 4, true, options);
        assert!(PatternLayout::of(&doc, &graph, id).is_ok());
    }

    /// A block that does not hold the body leaves out the copy standing on it — instance zero for
    /// a pattern that starts at the body, and `⌊(n − 1)/2⌋` for one centred on it, at every count.
    #[test]
    fn a_block_not_holding_the_body_leaves_out_the_copy_on_it() {
        let (doc, graph, id) = pattern(row(10.0), 4, false, PatternOptions::default());
        let layout = PatternLayout::of(&doc, &graph, id).unwrap();
        assert_eq!(layout.excluded, vec![true, false, false, false]);

        let symmetric = PatternOptions { symmetric: true, ..PatternOptions::default() };
        let (doc, graph, id) = pattern(row(10.0), 5, false, symmetric.clone());
        let layout = PatternLayout::of(&doc, &graph, id).unwrap();
        assert_eq!(layout.excluded, vec![false, false, true, false, false]);

        // At an even centred count too, since the bias is a whole number of steps.
        let (doc, graph, id) = pattern(row(10.0), 4, false, symmetric);
        let layout = PatternLayout::of(&doc, &graph, id).unwrap();
        assert_eq!(layout.excluded, vec![false, true, false, false]);
    }

    /// A pattern that is left with nothing is refused rather than drawn as an empty body.
    #[test]
    fn a_pattern_left_with_nothing_is_refused() {
        let options = PatternOptions { skip: vec![1, 2], ..PatternOptions::default() };
        let (doc, graph, id) = pattern(row(10.0), 3, false, options);
        assert_eq!(PatternLayout::of(&doc, &graph, id).unwrap_err(), PatternRefusal::NothingLeft);
    }

    /// **A grid is `count × count2` copies, column `i mod count` of row `i div count`**, so three
    /// by four is twelve and the cap counts twelve.
    #[test]
    fn a_grid_of_three_by_four_has_twelve_copies_in_row_order() {
        let options = PatternOptions {
            second: Some(PatternSecond { step: [lit(0.0), lit(5.0), lit(0.0)], count: lit(4.0) }),
            ..PatternOptions::default()
        };
        let (doc, graph, id) = pattern(row(10.0), 3, true, options);
        let layout = PatternLayout::of(&doc, &graph, id).unwrap();
        assert_eq!(layout.total, 12);
        let at = |i: u32| {
            let p = layout.placement(i) * Point3::origin();
            (p.x, p.y)
        };
        assert_eq!(at(0), (0.0, 0.0));
        assert_eq!(at(2), (20.0, 0.0), "the first row runs out along the step");
        assert_eq!(at(3), (0.0, 5.0), "the next row starts above the first copy");
        assert_eq!(at(11), (20.0, 15.0));
    }

    /// Centred, a grid is centred in both directions and the body is a copy only when both counts
    /// have a middle.
    #[test]
    fn a_centred_grid_stands_on_the_body_at_each_directions_own_middle() {
        let second = |n: f64| {
            Some(PatternSecond { step: [lit(0.0), lit(5.0), lit(0.0)], count: lit(n) })
        };
        // Column ⌊(c−1)/2⌋ of row ⌊(r−1)/2⌋, whether either count is odd or even: 3x5 is column 1
        // of row 2 (copy 7), 3x4 column 1 of row 1 (copy 4), 4x5 column 1 of row 2 (copy 9).
        for (columns, rows, middle) in [(3u32, 5.0, Some(7u32)), (3, 4.0, Some(4)), (4, 5.0, Some(9))] {
            let options =
                PatternOptions { symmetric: true, second: second(rows), ..PatternOptions::default() };
            let (doc, graph, id) = pattern(row(10.0), columns, true, options);
            let layout = PatternLayout::of(&doc, &graph, id).unwrap();
            assert_eq!(layout.body, middle, "{columns} columns by {rows} rows");
            if let Some(m) = middle {
                let p = layout.placement(m) * Point3::origin();
                assert!(near(p.x, 0.0) && near(p.y, 0.0), "copy {m} is at {p:?}, not on the body");
            }
        }
    }

    /// Two steps along one line are not a grid, and a grid cannot round its copies into each other
    /// or hold more than a pattern builds: each is refused by name.
    #[test]
    fn what_a_grid_cannot_be_is_refused_by_name() {
        let second = |step: [f64; 3], n: f64| PatternOptions {
            second: Some(PatternSecond { step: step.map(lit), count: lit(n) }),
            ..PatternOptions::default()
        };
        let (doc, graph, id) = pattern(row(10.0), 3, true, second([20.0, 0.0, 0.0], 3.0));
        assert_eq!(PatternLayout::of(&doc, &graph, id).unwrap_err(), PatternRefusal::ParallelSteps);

        let (doc, graph, id) = pattern(row(10.0), 30, true, second([0.0, 5.0, 0.0], 30.0));
        assert_eq!(
            PatternLayout::of(&doc, &graph, id).unwrap_err(),
            PatternRefusal::TooManyCopies(900)
        );

        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(1.0) });
        let id = graph.add(Node::Pattern {
            input: ball.into(),
            count: lit(3.0),
            kind: row(10.0),
            join: true,
            fillet: Fillet::of(lit(2.0)),
            options: second([0.0, 5.0, 0.0], 3.0),
        });
        assert_eq!(
            PatternLayout::of(&Document::new(), &graph, id).unwrap_err(),
            PatternRefusal::RoundedGrid
        );
    }

    /// The dual vectors read a point's steps off a grid at any angle and any height: a point that
    /// is `a` steps along the first, `b` along the second and some way off their plane reads `(a, b)`.
    #[test]
    fn the_dual_vectors_read_two_coordinates_whatever_the_angle_and_height() {
        let (e1, e2) = (Vector3::new(10.0, 2.0, 0.0), Vector3::new(3.0, 8.0, 1.0));
        let [d1, d2] = grid_dual(&e1, &e2).expect("not parallel");
        let up = e1.cross(&e2).normalize() * 37.0;
        let p = e1 * 2.25 + e2 * -1.5 + up;
        assert!(near(p.dot(&d1), 2.25), "{}", p.dot(&d1));
        assert!(near(p.dot(&d2), -1.5), "{}", p.dot(&d2));
        assert!(grid_dual(&e1, &(e1 * -3.0)).is_none());
    }
}
