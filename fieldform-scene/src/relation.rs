// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Relations between curves: one curve follows another, one way, in closed form.
//!
//! `profile-group-spec.md` §5. A relation names a **master** and a **slave** and *claims*
//! specific control points of the slave, which the re-derivation pass overwrites from the
//! master's current geometry. Nothing is solved: each claimed point has one closed-form
//! expression in terms of one upstream curve, evaluated once, masters first.
//!
//! **Derived points are stored, not computed on demand** (`sketcher-spec.md` §8.1). The slave's
//! control points are ordinary control points that happen to be overwritten, so the geometry is
//! fully described by the points at all times, independently of the relation -- which is what
//! makes breaking one free and lossless, and what lets an older build that never reads
//! `relations.json` draw exactly the right shape.
//!
//! **Its own collection, not on `Curve`.** A relation names two curves; storing it on either would
//! make a curve reference another curve, which `sketcher-spec.md` §6.3 refuses: the dependency
//! runs from nodes to curves and never back. Relations sit beside curves in the document, the way
//! `node_meta` sits beside nodes.

use std::collections::{BTreeMap, HashMap, HashSet};

use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};

use crate::{Curve, CurveId, Document, RelationId, JOINED_MM};

/// Which end of a curve: the first control point's, or the last's.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum End {
    Start,
    End,
}

/// One end of a curve: whole-curve identity plus which of its two ends. The reference
/// `sketcher-spec.md` §8 called defensible, because it names nothing derived.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CurveEnd {
    pub curve: CurveId,
    pub end: End,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JointContinuity {
    G1,
    G2,
}

/// Every relation drives `slave` from `master`, one way, in closed form.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Relation {
    /// The two ends must already meet (a joint, or within [`JOINED_MM`]).
    Joint { master: CurveEnd, slave: CurveEnd, continuity: JointContinuity },
    /// Lines only: the slave's direction becomes the master's.
    Parallel { master: CurveId, slave: CurveId },
    /// Lines only, and planar: the slave's direction is the master's turned 90° in the plane
    /// both lie on. Refused when they are not on one plane.
    Perpendicular { master: CurveId, slave: CurveId },
    /// Lines only: the slave's length becomes the master's.
    EqualLength { master: CurveId, slave: CurveId },
    /// The slave moved as one piece so its centre is the master's: two circles, or two arcs,
    /// about one point. A curve's centre is the centre of the circle through three points of
    /// it a third of its run apart ([`centre_of`]) -- exactly the centre for a circle or an
    /// arc, and refused for a straight curve. Claims every control point of the slave.
    Concentric { master: CurveId, slave: CurveId },
    /// The slave is the master reflected across `axis`, a line, in the plane the master and
    /// the line lie on: point for point, so the two have the same degree, knots, weights and
    /// number of control points. Claims every control point of the slave, and follows the
    /// master and the axis both.
    Mirror { master: CurveId, slave: CurveId, axis: CurveId },
}

/// A relation as the document holds it: the relation, and why it does not currently hold, if
/// it was found not to on open (§5.4). A broken relation is kept and reported, never applied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelationRecord {
    pub relation: Relation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub broken: Option<String>,
}

impl Relation {
    pub fn master(&self) -> CurveId {
        match self {
            Relation::Joint { master, .. } => master.curve,
            Relation::Parallel { master, .. }
            | Relation::Perpendicular { master, .. }
            | Relation::EqualLength { master, .. }
            | Relation::Concentric { master, .. }
            | Relation::Mirror { master, .. } => *master,
        }
    }

    pub fn slave(&self) -> CurveId {
        match self {
            Relation::Joint { slave, .. } => slave.curve,
            Relation::Parallel { slave, .. }
            | Relation::Perpendicular { slave, .. }
            | Relation::EqualLength { slave, .. }
            | Relation::Concentric { slave, .. }
            | Relation::Mirror { slave, .. } => *slave,
        }
    }

    /// Every curve this relation follows: the master, and a mirror's axis as well. What the
    /// cycle rule and the order of re-derivation walk, since a mirror has two things upstream.
    pub fn masters(&self) -> Vec<CurveId> {
        match self {
            Relation::Mirror { master, axis, .. } => vec![*master, *axis],
            _ => vec![self.master()],
        }
    }

    /// Whether `curve` is any side of this relation.
    pub fn names(&self, curve: CurveId) -> bool {
        self.masters().contains(&curve) || self.slave() == curve
    }

    /// The relation's kind as a person would say it.
    pub fn kind(&self) -> &'static str {
        match self {
            Relation::Joint { continuity: JointContinuity::G1, .. } => "G1",
            Relation::Joint { continuity: JointContinuity::G2, .. } => "G2",
            Relation::Parallel { .. } => "parallel",
            Relation::Perpendicular { .. } => "perpendicular",
            Relation::EqualLength { .. } => "equal length",
            Relation::Concentric { .. } => "concentric",
            Relation::Mirror { .. } => "mirror",
        }
    }

    /// One line saying what follows what, by the curves' names.
    pub fn describe(&self, doc: &Document) -> String {
        let name = |id: CurveId| {
            doc.curve(id).map(|c| c.name.clone()).unwrap_or_else(|| "a deleted curve".into())
        };
        match self {
            Relation::Joint { master, slave, continuity } => format!(
                "{}: {} {} follows {} {}",
                match continuity {
                    JointContinuity::G1 => "G1",
                    JointContinuity::G2 => "G2",
                },
                name(slave.curve),
                slave.end.word(),
                name(master.curve),
                master.end.word(),
            ),
            Relation::Parallel { master, slave } => {
                format!("parallel: {} follows {}", name(*slave), name(*master))
            }
            Relation::Perpendicular { master, slave } => {
                format!("perpendicular: {} follows {}", name(*slave), name(*master))
            }
            Relation::EqualLength { master, slave } => {
                format!("equal length: {} follows {}", name(*slave), name(*master))
            }
            Relation::Concentric { master, slave } => {
                format!("concentric: {} follows {}", name(*slave), name(*master))
            }
            Relation::Mirror { master, slave, axis } => format!(
                "mirror: {} mirrors {} across {}",
                name(*slave),
                name(*master),
                name(*axis)
            ),
        }
    }
}

impl End {
    pub fn word(self) -> &'static str {
        match self {
            End::Start => "start",
            End::End => "end",
        }
    }

    /// The control-point index of this end on a curve of `count` points.
    pub fn index(self, count: usize) -> usize {
        match self {
            End::Start => 0,
            End::End => count.saturating_sub(1),
        }
    }
}

/// Why a relation was refused, or cannot be derived.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RelationError {
    #[error("{0} is not in the document")]
    MissingCurve(String),
    #[error("a curve cannot follow itself")]
    SelfReference,
    #[error(
        "the ends do not meet: they are {gap:.3} mm apart. Snap them together with Ctrl held, \
         or join the loop, before aligning them"
    )]
    EndsApart { gap: f64 },
    #[error(
        "{kind} needs {needed} interior control point{} to claim, and {name} has {has}",
        plural(.needed)
    )]
    TooFewPoints { kind: &'static str, name: String, needed: usize, has: usize },
    #[error("{kind} is between lines, and {name} is degree {degree}")]
    NotALine { kind: &'static str, name: String, degree: usize },
    #[error("{name} has both ends on points, so nothing is free to move \u{2014} ungroup one end")]
    NothingFree { name: String },
    #[error("point {index} of {name} is already owned by {owner}")]
    Owned { name: String, index: usize, owner: String },
    #[error("{slave} cannot follow {master}: {closes} already makes {master} follow {slave}")]
    Cycle { slave: String, master: String, closes: String },
    #[error("{a} and {b} are not on one plane, so there is no plane to turn 90\u{b0} in")]
    NotPlanar { a: String, b: String },
    #[error("{name} has no centre: three points along it lie on one line")]
    NoCentre { name: String },
    #[error("{slave} cannot mirror {master} point for point: {why}")]
    NotAMirrorPair { master: String, slave: String, why: String },
    #[error("{master} and {axis} are not on one plane, so there is no line in it to mirror across")]
    NoMirrorPlane { master: String, axis: String },
}

/// "s" unless the count is one: a message that says "1 points" reads as a template.
fn plural(n: &usize) -> &'static str {
    if *n == 1 { "" } else { "s" }
}

/// The slave's control points a relation claims, by index (§5.2).
///
/// A joint claims the first interior point next to the slave end for G1, the first two for
/// G2. A relation between lines claims the line's free end: the one not sitting on a point,
/// preferring the line's end over its start when both are free. Whether another relation
/// already owns one of these is [`refuse_reason`]'s question, not this one's.
pub(crate) fn claims(relation: &Relation, doc: &Document) -> Result<Vec<usize>, RelationError> {
    let curve = |id: CurveId| {
        doc.curve(id).ok_or_else(|| RelationError::MissingCurve(format!("{id:?}")))
    };
    match relation {
        Relation::Joint { slave, continuity, .. } => {
            let c = curve(slave.curve)?;
            let n = c.control_points().len();
            let (needed, kind) = match continuity {
                JointContinuity::G1 => (1, "G1"),
                JointContinuity::G2 => (2, "G2"),
            };
            // Interior points only: the far end is not a joint's to claim.
            let interior = n.saturating_sub(2);
            if interior < needed {
                return Err(RelationError::TooFewPoints {
                    kind,
                    name: c.name.clone(),
                    needed,
                    has: interior,
                });
            }
            Ok(match slave.end {
                End::Start => (1..=needed).collect(),
                End::End => (1..=needed).map(|k| n - 1 - k).collect(),
            })
        }
        Relation::Parallel { slave, .. }
        | Relation::Perpendicular { slave, .. }
        | Relation::EqualLength { slave, .. } => {
            let c = curve(*slave)?;
            line_check(relation.kind(), c)?;
            for index in [1usize, 0] {
                if c.placed_on(index).is_none() {
                    return Ok(vec![index]);
                }
            }
            Err(RelationError::NothingFree { name: c.name.clone() })
        }
        // The whole slave: a concentric one moves as one piece, a mirror is rebuilt point for
        // point.
        Relation::Concentric { slave, .. } | Relation::Mirror { slave, .. } => {
            Ok((0..curve(*slave)?.control_points().len()).collect())
        }
    }
}

fn line_check(kind: &'static str, c: &Curve) -> Result<(), RelationError> {
    if c.degree() != 1 || c.control_points().len() != 2 {
        return Err(RelationError::NotALine { kind, name: c.name.clone(), degree: c.degree() });
    }
    Ok(())
}

/// Whether a new relation may be made, and the points it would claim if so (§5.2).
///
/// Checked when the relation is created and refused, naming the conflict, when any rule fails:
/// the curves exist and differ; a joint's ends meet; the slave has the points to claim; no
/// claimed point sits on a block or is claimed by another relation; a perpendicular's lines lie
/// on one plane; and the relation closes no cycle through the ones already there.
pub fn refuse_reason(doc: &Document, relation: &Relation) -> Result<Vec<usize>, RelationError> {
    refuse_reason_replacing(doc, relation, None)
}

/// [`refuse_reason`] for a relation that would take the place of `except`, which owns nothing
/// and closes nothing here because it leaves in the same entry the new one arrives in. How a
/// corner's continuity is changed from G1 to G2 in place, and how a line's "parallel to" is
/// pointed at a different line, without the old relation refusing its own successor.
pub fn refuse_reason_replacing(
    doc: &Document,
    relation: &Relation,
    except: Option<RelationId>,
) -> Result<Vec<usize>, RelationError> {
    let master = doc
        .curve(relation.master())
        .ok_or_else(|| RelationError::MissingCurve(format!("{:?}", relation.master())))?;
    let slave = doc
        .curve(relation.slave())
        .ok_or_else(|| RelationError::MissingCurve(format!("{:?}", relation.slave())))?;
    if master.id == slave.id {
        return Err(RelationError::SelfReference);
    }
    match relation {
        Relation::Joint { master: m, slave: s, .. } => {
            let a = master.control_points()[m.end.index(master.control_points().len())];
            let b = slave.control_points()[s.end.index(slave.control_points().len())];
            let gap = (a - b).norm();
            if gap > JOINED_MM {
                return Err(RelationError::EndsApart { gap });
            }
        }
        Relation::Parallel { .. } | Relation::EqualLength { .. } => {
            line_check(relation.kind(), master)?;
        }
        Relation::Perpendicular { .. } => {
            line_check(relation.kind(), master)?;
            line_check(relation.kind(), slave)?;
            if plane_normal(master, slave).is_none() {
                return Err(RelationError::NotPlanar {
                    a: master.name.clone(),
                    b: slave.name.clone(),
                });
            }
        }
        Relation::Concentric { .. } => {
            for c in [master, slave] {
                if centre_of(c).is_none() {
                    return Err(RelationError::NoCentre { name: c.name.clone() });
                }
            }
        }
        Relation::Mirror { axis, .. } => {
            let axis = doc
                .curve(*axis)
                .ok_or_else(|| RelationError::MissingCurve(format!("{axis:?}")))?;
            if axis.id == slave.id {
                return Err(RelationError::SelfReference);
            }
            line_check("mirror", axis)?;
            if let Some(why) = mirror_mismatch(master, slave) {
                return Err(RelationError::NotAMirrorPair {
                    master: master.name.clone(),
                    slave: slave.name.clone(),
                    why,
                });
            }
            if mirror_plane(master, axis).is_none() {
                return Err(RelationError::NoMirrorPlane {
                    master: master.name.clone(),
                    axis: axis.name.clone(),
                });
            }
        }
    }
    let claimed = claims(relation, doc)?;
    // One owner per control point (rule 1).
    for &index in &claimed {
        if slave.placed_on(index).is_some() {
            return Err(RelationError::Owned {
                name: slave.name.clone(),
                index,
                owner: "the point block it sits on".to_string(),
            });
        }
        for (id, record) in &doc.relations {
            if Some(*id) == except || record.relation.slave() != slave.id {
                continue;
            }
            let theirs = claims(&record.relation, doc).unwrap_or_default();
            if theirs.contains(&index) {
                return Err(RelationError::Owned {
                    name: slave.name.clone(),
                    index,
                    owner: format!(
                        "the {} relation to {}",
                        record.relation.kind(),
                        doc.curve(record.relation.master())
                            .map(|c| c.name.clone())
                            .unwrap_or_default()
                    ),
                });
            }
        }
    }
    // Acyclic (rule 4): an edge from each slave to its master; the new one closes a cycle
    // when its master already follows, however far up, its slave.
    let mut edges: HashMap<CurveId, Vec<(CurveId, &Relation)>> = HashMap::new();
    for (id, record) in &doc.relations {
        if Some(*id) == except {
            continue;
        }
        for master in record.relation.masters() {
            edges.entry(record.relation.slave()).or_default().push((master, &record.relation));
        }
    }
    let mut stack = relation.masters();
    let mut seen = HashSet::new();
    while let Some(here) = stack.pop() {
        if !seen.insert(here) {
            continue;
        }
        for (up, via) in edges.get(&here).map(|v| v.as_slice()).unwrap_or(&[]) {
            if *up == relation.slave() {
                return Err(RelationError::Cycle {
                    slave: slave.name.clone(),
                    master: master.name.clone(),
                    closes: via.describe(doc),
                });
            }
            stack.push(*up);
        }
    }
    Ok(claimed)
}

/// The relations that claim `index` of `curve`, other than `except`.
pub fn owners_of(
    doc: &Document,
    curve: CurveId,
    index: usize,
    except: Option<RelationId>,
) -> Vec<RelationId> {
    doc.relations
        .iter()
        .filter(|(id, record)| {
            Some(**id) != except
                && record.relation.slave() == curve
                && claims(&record.relation, doc).unwrap_or_default().contains(&index)
        })
        .map(|(id, _)| *id)
        .collect()
}

/// What a relation would set its claimed points to, from the master's current geometry.
///
/// `None` when it cannot be derived: a curve is gone, or the geometry is degenerate -- a
/// master of no length, a perpendicular with no plane. The joint mathematics is
/// `sketcher-spec.md` §8.2 with the rational correction §15.4 asks for: the master's end
/// derivatives are its actual first and second derivatives, weights included, and the slave's
/// second claimed point is solved from the same β-continuity equations in homogeneous
/// coordinates, so a weighted curve on either side is exact.
pub fn derive(doc: &Document, relation: &Relation) -> Option<Vec<(usize, Point3<f64>)>> {
    let master = doc.curve(relation.master())?;
    let slave = doc.curve(relation.slave())?;
    match relation {
        Relation::Joint { master: m, slave: s, continuity } => {
            derive_joint(master, m.end, slave, s.end, *continuity)
        }
        Relation::Concentric { .. } => {
            let shift = centre_of(master)? - centre_of(slave)?;
            // The slave's centre is read off the slave, so a second pass would find a shift of
            // a rounding error and move every point by it -- forever, since the settle after
            // each change re-derives. Already there is already there.
            if shift.norm() <= 1e-9 {
                return Some(slave.control_points().iter().copied().enumerate().collect());
            }
            Some(slave.control_points().iter().map(|p| p + shift).enumerate().collect())
        }
        Relation::Mirror { axis, .. } => {
            let axis = doc.curve(*axis)?;
            if mirror_mismatch(master, slave).is_some() {
                return None;
            }
            let (origin, normal) = mirror_plane(master, axis)?;
            Some(
                master
                    .control_points()
                    .iter()
                    .map(|p| p - normal * (2.0 * (p - origin).dot(&normal)))
                    .enumerate()
                    .collect(),
            )
        }
        Relation::Parallel { .. } | Relation::Perpendicular { .. } | Relation::EqualLength { .. } => {
            let free = *claims(relation, doc).ok()?.first()?;
            let fixed = 1 - free;
            let (m0, m1) = (master.control_points()[0], master.control_points()[1]);
            let (s_fixed, s_free) = (slave.control_points()[fixed], slave.control_points()[free]);
            let d = (m1 - m0).try_normalize(1e-12)?;
            let current = s_free - s_fixed;
            let length = current.norm();
            let along = |dir: Vector3<f64>| {
                // Keep the line pointing the way it already does.
                let dir = if current.dot(&dir) < 0.0 { -dir } else { dir };
                Some(vec![(free, s_fixed + dir * length)])
            };
            match relation {
                Relation::Parallel { .. } => along(d),
                Relation::Perpendicular { .. } => {
                    let n = plane_normal(master, slave)?;
                    along(n.cross(&d).try_normalize(1e-12)?)
                }
                Relation::EqualLength { .. } => {
                    let dir = current.try_normalize(1e-12)?;
                    Some(vec![(free, s_fixed + dir * (m1 - m0).norm())])
                }
                Relation::Joint { .. } | Relation::Concentric { .. } | Relation::Mirror { .. } => {
                    unreachable!("matched above")
                }
            }
        }
    }
}

/// The normal of the plane two lines lie on, or `None` when they are skew or one has no length.
fn plane_normal(a: &Curve, b: &Curve) -> Option<Vector3<f64>> {
    let (a0, a1) = (a.control_points()[0], a.control_points()[1]);
    let (b0, b1) = (b.control_points()[0], b.control_points()[1]);
    let d = (a1 - a0).try_normalize(1e-12)?;
    // The plane through the master containing either end of the slave; then the other end
    // has to lie on it too.
    let n = [b0, b1]
        .into_iter()
        .filter_map(|p| d.cross(&(p - a0)).try_normalize(1e-9))
        .next()?;
    let off = |p: Point3<f64>| (p - a0).dot(&n).abs();
    (off(b0) <= JOINED_MM && off(b1) <= JOINED_MM).then_some(n)
}

/// The centre of the circle through three points of a curve, a third of its run apart --
/// exactly the centre for a circle or an arc, whichever of its points they land on. `None`
/// for a curve whose three points lie on one line, which is a straight one.
pub fn centre_of(c: &Curve) -> Option<Point3<f64>> {
    let (lo, hi) = c.domain();
    let at = |k: f64| c.evaluate(lo + (hi - lo) * k);
    circumcentre(at(0.0), at(1.0 / 3.0), at(2.0 / 3.0))
}

/// The centre of the circle through three points, or `None` when they lie on one line.
fn circumcentre(a: Point3<f64>, b: Point3<f64>, c: Point3<f64>) -> Option<Point3<f64>> {
    let (u, v) = (b - a, c - a);
    let w = u.cross(&v);
    // Relative, so a small arc and a large one are judged alike: the sine of the angle at `a`.
    if w.norm() <= 1e-9 * u.norm() * v.norm() {
        return None;
    }
    Some(a + (v.cross(&w) * u.norm_squared() + w.cross(&u) * v.norm_squared()) / (2.0 * w.norm_squared()))
}

/// A mirror's plane, as a point on it and its normal: through the axis, square to the plane
/// the master and the axis lie on. `None` when the axis has no length, or the master does not
/// lie on one plane with it.
fn mirror_plane(master: &Curve, axis: &Curve) -> Option<(Point3<f64>, Vector3<f64>)> {
    let (a0, a1) = (axis.control_points()[0], axis.control_points()[1]);
    let d = (a1 - a0).try_normalize(1e-12)?;
    let off_axis = |p: &Point3<f64>| d.cross(&(p - a0)).norm();
    // The plane the two lie on, through the axis and the master's point farthest from it.
    let far = master.control_points().iter().max_by(|p, q| off_axis(p).total_cmp(&off_axis(q)))?;
    let n = d.cross(&(far - a0)).try_normalize(1e-9)?;
    if master.control_points().iter().any(|p| (p - a0).dot(&n).abs() > JOINED_MM) {
        return None;
    }
    Some((a0, d.cross(&n).try_normalize(1e-12)?))
}

/// Why `slave` cannot take `master`'s points one for one, if it cannot.
fn mirror_mismatch(master: &Curve, slave: &Curve) -> Option<String> {
    if master.degree() != slave.degree() {
        return Some(format!(
            "{} is degree {} and {} is degree {}",
            master.name,
            master.degree(),
            slave.name,
            slave.degree()
        ));
    }
    if master.control_points().len() != slave.control_points().len() {
        return Some(format!(
            "{} has {} control points and {} has {}",
            master.name,
            master.control_points().len(),
            slave.name,
            slave.control_points().len()
        ));
    }
    if master.knots() != slave.knots() || master.weights() != slave.weights() {
        return Some("their knots or weights differ".into());
    }
    None
}

/// The joint mathematics, on the slave as seen leaving the joint.
///
/// A slave joined by its end is handled on its reversed copy, so the two cases are one
/// formula with the indices mapped back, rather than two formulas that could disagree.
fn derive_joint(
    master: &Curve,
    master_end: End,
    slave: &Curve,
    slave_end: End,
    continuity: JointContinuity,
) -> Option<Vec<(usize, Point3<f64>)>> {
    let n = slave.control_points().len();
    let reversed;
    let (s, map): (&Curve, Box<dyn Fn(usize) -> usize>) = match slave_end {
        End::Start => (slave, Box::new(|i| i)),
        End::End => {
            reversed = slave.reversed();
            (&reversed, Box::new(move |i| n - 1 - i))
        }
    };
    // The master's derivatives *through* the joint: the direction of travel into the slave is
    // the direction it arrives from, and reversing a parameter negates a first derivative and
    // leaves a second alone.
    let (leave1, leave2) = master.derivatives_leaving(master_end);
    let arrive1 = -leave1;
    let arrive2 = leave2;
    let dir = arrive1.try_normalize(1e-12)?;

    let pts = s.control_points();
    let k = pts[0];
    // G1: direction from the master, distance from the joint kept from wherever the user last
    // put the point (§8.1), so a derived point still slides along the tangent when dragged.
    let b1 = k + dir * (pts[1] - k).norm();
    let mut out = vec![(map(1), b1)];
    if continuity == JointContinuity::G1 {
        return Some(out);
    }

    // G2, in homogeneous coordinates. With the slave's clamped start factors
    //   Aw'(0)  = k1 (w1 b1 - w0 b0)                  w'(0)  = k1 (w1 - w0)
    //   Aw''(0) = k2 [(w2 b2 - w1 b1)/span2 - (w1 b1 - w0 b0)/span1]
    //   w''(0)  = k2 [(w2 - w1)/span2 - (w1 - w0)/span1]
    // and the rational derivatives R' = (Aw' - w' R)/w0, R'' = (Aw'' - 2 w' R' - w'' R)/w0,
    // the β-continuity conditions R'(0) = β1 A'(1), R''(0) = β1² A''(1) + β2 A'(1) give b2 as
    // an affine function of the free β2; the user's current b2 projected onto that line is
    // the minimum-displacement choice (§8.2).
    let (k1, k2, span1, span2) = s.start_factors();
    let w = s.weights();
    let (w0, w1, w2) = (w[0], w[1], w[2]);
    let r1 = (b1 - k) * (k1 * w1 / w0);
    let beta1 = r1.norm() / arrive1.norm();
    let wp = k1 * (w1 - w0);
    let wpp = k2 * ((w2 - w1) / span2 - (w1 - w0) / span1);
    let b2_for = |r2: Vector3<f64>| -> Point3<f64> {
        let aw = r2 * w0 + r1 * (2.0 * wp) + k.coords * wpp;
        let hom = b1.coords * w1 + (aw / k2 + (b1.coords * w1 - k.coords * w0) / span1) * span2;
        Point3::from(hom / w2)
    };
    let p = b2_for(arrive2 * (beta1 * beta1));
    let t = b2_for(arrive2 * (beta1 * beta1) + arrive1) - p;
    let current = pts[2];
    let beta2 = if t.norm_squared() > 1e-24 { (current - p).dot(&t) / t.dot(&t) } else { 0.0 };
    out.push((map(2), p + t * beta2));
    Some(out)
}

/// Recomputes every claimed point of every relation that holds, masters first (§5.3).
///
/// Returns what moved: curve, index, and where it went. A point already where the relation
/// puts it is not reported, which is what lets "breaking moves nothing" be checked as a bit
/// pattern rather than a tolerance. Broken relations are left alone: on open they are checked,
/// not enforced, and only `realign` clears them.
pub fn rederive(doc: &mut Document) -> Vec<(CurveId, usize, Point3<f64>)> {
    let ordered = topological(doc);
    let mut moved = Vec::new();
    for id in ordered {
        let Some(record) = doc.relations.get(&id) else { continue };
        if record.broken.is_some() {
            continue;
        }
        let relation = record.relation.clone();
        let Some(targets) = derive(doc, &relation) else { continue };
        let Some(curve) = doc.curve_mut(relation.slave()) else { continue };
        for (index, to) in targets {
            let Some(was) = curve.control_points().get(index).copied() else { continue };
            if was == to {
                continue;
            }
            curve.set_control_point(index, to);
            moved.push((relation.slave(), index, to));
        }
    }
    moved
}

/// Which relations no longer hold, and why -- what an open asks, before enforcing nothing (§5.4).
///
/// Each relation is tested against the geometry at [`JOINED_MM`], the tolerance the kernel
/// already judges ends by. A relation naming a curve that is gone is broken too, and says so.
pub fn check(doc: &Document) -> Vec<(RelationId, String)> {
    let mut out = Vec::new();
    for (id, record) in &doc.relations {
        let relation = &record.relation;
        let Some(slave) = doc.curve(relation.slave()) else {
            out.push((*id, format!("{}: the curve that follows is gone", relation.kind())));
            continue;
        };
        if relation.masters().iter().any(|m| doc.curve(*m).is_none()) {
            out.push((*id, format!("{}: the curve it follows is gone", relation.kind())));
            continue;
        }
        let Some(targets) = derive(doc, relation) else {
            out.push((*id, format!("{}: cannot be derived from the geometry", relation.kind())));
            continue;
        };
        for (index, to) in targets {
            let Some(at) = slave.control_points().get(index) else { continue };
            let off = (at - to).norm();
            if off > JOINED_MM {
                out.push((
                    *id,
                    format!(
                        "{}: point {index} of {} is {off:.3} mm from where it should be",
                        relation.describe(doc),
                        slave.name
                    ),
                ));
                break;
            }
        }
    }
    out
}

/// The relations in an order every master comes before what follows it.
///
/// Kahn's algorithm over the curves, since [`refuse_reason`] keeps the graph acyclic; a cycle
/// that somehow arrived in a file leaves its relations out of the order, which is the same as
/// being broken, and `check` will have said so.
fn topological(doc: &Document) -> Vec<RelationId> {
    let mut rank: HashMap<CurveId, usize> = HashMap::new();
    let mut incoming: HashMap<CurveId, usize> = HashMap::new();
    let mut downstream: HashMap<CurveId, Vec<CurveId>> = HashMap::new();
    let mut curves: BTreeMap<CurveId, ()> = BTreeMap::new();
    for record in doc.relations.values() {
        let s = record.relation.slave();
        curves.insert(s, ());
        incoming.entry(s).or_default();
        for m in record.relation.masters() {
            curves.insert(m, ());
            *incoming.entry(s).or_default() += 1;
            incoming.entry(m).or_default();
            downstream.entry(m).or_default().push(s);
        }
    }
    let mut ready: Vec<CurveId> = curves.keys().copied().filter(|c| incoming[c] == 0).collect();
    let mut next = 0;
    while let Some(c) = ready.pop() {
        rank.insert(c, next);
        next += 1;
        for d in downstream.get(&c).cloned().unwrap_or_default() {
            let left = incoming.get_mut(&d).expect("counted");
            *left -= 1;
            if *left == 0 {
                ready.push(d);
            }
        }
    }
    let mut ids: Vec<(usize, RelationId)> = doc
        .relations
        .iter()
        .filter_map(|(id, r)| rank.get(&r.relation.slave()).map(|k| (*k, *id)))
        .collect();
    ids.sort();
    ids.into_iter().map(|(_, id)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Curve;

    fn doc_with(curves: Vec<Curve>) -> Document {
        let mut doc = Document::new();
        for c in curves {
            doc.curves.insert(c.id, c);
        }
        doc
    }

    fn line(name: &str, a: [f64; 3], b: [f64; 3]) -> Curve {
        Curve::polyline(CurveId::new(), name, vec![Point3::from(a), Point3::from(b)], false)
            .expect("a line")
    }

    fn bezier(name: &str, points: Vec<[f64; 3]>, weights: Option<Vec<f64>>) -> Curve {
        let degree = points.len() - 1;
        let knots: Vec<f64> = std::iter::repeat_n(0.0, degree + 1)
            .chain(std::iter::repeat_n(1.0, degree + 1))
            .collect();
        let weights = weights.unwrap_or_else(|| vec![1.0; points.len()]);
        Curve::new(
            CurveId::new(),
            name,
            degree,
            points.into_iter().map(Point3::from).collect(),
            weights,
            knots,
            false,
        )
        .expect("a Bézier")
    }

    fn add(doc: &mut Document, relation: Relation) -> RelationId {
        refuse_reason(doc, &relation).expect("allowed");
        let id = RelationId::new();
        doc.relations.insert(id, RelationRecord { relation, broken: None });
        id
    }

    /// A concentric slave moves as one piece so its centre is the master's, keeps it when the
    /// master moves, and asking again moves nothing.
    #[test]
    fn a_concentric_slave_keeps_the_masters_centre_and_asking_again_moves_nothing() {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let quarter = |name: &str, c: [f64; 2], r: f64| {
            bezier(
                name,
                vec![[c[0] + r, c[1], 0.0], [c[0] + r, c[1] + r, 0.0], [c[0], c[1] + r, 0.0]],
                Some(vec![1.0, w, 1.0]),
            )
        };
        let (master, slave) = (quarter("Arc", [0.0, 0.0], 10.0), quarter("Arc 2", [5.0, 5.0], 20.0));
        assert!((centre_of(&master).unwrap() - Point3::origin()).norm() < 1e-9, "an arc's centre is its circle's");
        let (m, s) = (master.id, slave.id);
        let mut doc = doc_with(vec![master, slave]);
        add(&mut doc, Relation::Concentric { master: m, slave: s });
        rederive(&mut doc);
        let centre = |doc: &Document, id| centre_of(doc.curve(id).unwrap()).unwrap();
        assert!((centre(&doc, s) - centre(&doc, m)).norm() < 1e-9);
        let start = doc.curve(s).unwrap().control_points()[0];
        assert!(((start - centre(&doc, s)).norm() - 20.0).abs() < 1e-9, "moved as one piece, radius kept");
        for i in 0..3 {
            let p = doc.curve(m).unwrap().control_points()[i];
            doc.curve_mut(m).unwrap().set_control_point(i, p + Vector3::new(3.0, -2.0, 0.0));
        }
        rederive(&mut doc);
        assert!((centre(&doc, s) - Point3::new(3.0, -2.0, 0.0)).norm() < 1e-9);
        assert!(rederive(&mut doc).is_empty(), "asking again moves nothing");
    }

    /// A mirrored slave is the master reflected across the axis, point for point, and follows
    /// the master and the axis both.
    #[test]
    fn a_mirror_follows_its_master_and_its_axis() {
        let axis = line("Axis", [0.0, -10.0, 0.0], [0.0, 10.0, 0.0]);
        let master = bezier("Curve", vec![[2.0, 0.0, 0.0], [4.0, 3.0, 0.0], [6.0, 1.0, 0.0], [8.0, 5.0, 0.0]], None);
        let slave = bezier("Curve 2", vec![[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], None);
        let (a, m, s) = (axis.id, master.id, slave.id);
        let mut doc = doc_with(vec![axis, master, slave]);
        add(&mut doc, Relation::Mirror { master: m, slave: s, axis: a });
        rederive(&mut doc);
        let pts = |doc: &Document, id| doc.curve(id).unwrap().control_points().to_vec();
        for (p, q) in pts(&doc, m).iter().zip(&pts(&doc, s)) {
            assert!((q - Point3::new(-p.x, p.y, p.z)).norm() < 1e-12);
        }
        doc.curve_mut(m).unwrap().set_control_point(2, Point3::new(7.0, -4.0, 0.0));
        rederive(&mut doc);
        assert!((pts(&doc, s)[2] - Point3::new(-7.0, -4.0, 0.0)).norm() < 1e-12);
        // Turn the axis to lie along X, and the mirror flips Y instead.
        doc.curve_mut(a).unwrap().set_control_point(0, Point3::new(-10.0, 0.0, 0.0));
        doc.curve_mut(a).unwrap().set_control_point(1, Point3::new(10.0, 0.0, 0.0));
        rederive(&mut doc);
        for (p, q) in pts(&doc, m).iter().zip(&pts(&doc, s)) {
            assert!((q - Point3::new(p.x, -p.y, p.z)).norm() < 1e-12);
        }
        assert!(rederive(&mut doc).is_empty(), "asking again moves nothing");
    }

    /// A mirror wants a slave of the master's own shape, a straight curve has no centre, and a
    /// cycle closed through a mirror's axis is refused like any other.
    #[test]
    fn a_mirror_wants_the_masters_shape_a_line_has_no_centre_and_an_axis_cannot_follow_back() {
        let axis = line("Axis", [0.0, -10.0, 0.0], [0.0, 10.0, 0.0]);
        let master = bezier("Curve", vec![[2.0, 0.0, 0.0], [4.0, 3.0, 0.0], [6.0, 1.0, 0.0], [8.0, 5.0, 0.0]], None);
        let quad = bezier("Quad", vec![[0.0; 3], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]], None);
        let straight = line("Line", [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]);
        let (a, m, q, l) = (axis.id, master.id, quad.id, straight.id);
        let mut doc = doc_with(vec![axis, master, quad, straight]);
        assert!(matches!(
            refuse_reason(&doc, &Relation::Mirror { master: m, slave: q, axis: a }),
            Err(RelationError::NotAMirrorPair { .. })
        ));
        assert!(matches!(
            refuse_reason(&doc, &Relation::Concentric { master: m, slave: l }),
            Err(RelationError::NoCentre { .. })
        ));
        // Two lines, one mirroring the other across the axis; the axis may not then follow
        // the mirrored one, since that one follows the axis.
        let (l3, l4) = (line("L3", [2.0, 0.0, 0.0], [5.0, 4.0, 0.0]), line("L4", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
        let (i3, i4) = (l3.id, l4.id);
        doc.curves.insert(i3, l3);
        doc.curves.insert(i4, l4);
        add(&mut doc, Relation::Mirror { master: i3, slave: i4, axis: a });
        assert!(matches!(
            refuse_reason(&doc, &Relation::Parallel { master: i4, slave: a }),
            Err(RelationError::Cycle { .. })
        ));
    }

    /// A refusal counts its points in the singular when there is one.
    #[test]
    fn a_refusal_counts_its_points_in_the_singular_when_there_is_one() {
        let master = line("Line", [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        let slave = line("Line 2", [10.0, 0.0, 0.0], [20.0, 5.0, 0.0]);
        let (m, s) = (master.id, slave.id);
        let doc = doc_with(vec![master, slave]);
        let corner = |continuity| Relation::Joint {
            master: CurveEnd { curve: m, end: End::End },
            slave: CurveEnd { curve: s, end: End::Start },
            continuity,
        };
        let one = refuse_reason(&doc, &corner(JointContinuity::G1)).expect_err("a line has none");
        assert_eq!(
            one.to_string(),
            "G1 needs 1 interior control point to claim, and Line 2 has 0"
        );
        let two = refuse_reason(&doc, &corner(JointContinuity::G2)).expect_err("a line has none");
        assert_eq!(
            two.to_string(),
            "G2 needs 2 interior control points to claim, and Line 2 has 0"
        );
    }

    /// Replacing a relation is not refused by the relation being replaced.
    #[test]
    fn replacing_a_corners_relation_is_not_refused_by_the_one_it_replaces() {
        let master = line("Line", [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        let slave = bezier(
            "Curve",
            vec![[10.0, 0.0, 0.0], [14.0, 3.0, 0.0], [20.0, 6.0, 0.0], [25.0, 0.0, 0.0]],
            None,
        );
        let (m, s) = (master.id, slave.id);
        let mut doc = doc_with(vec![master, slave]);
        let corner = |continuity| Relation::Joint {
            master: CurveEnd { curve: m, end: End::End },
            slave: CurveEnd { curve: s, end: End::Start },
            continuity,
        };
        let g1 = add(&mut doc, corner(JointContinuity::G1));
        // The G1 owns the curve's first interior point, so a G2 claiming it too is refused --
        // unless it is the G1's replacement, in which case it claims the first two.
        let g2 = corner(JointContinuity::G2);
        assert!(matches!(refuse_reason(&doc, &g2), Err(RelationError::Owned { .. })));
        let claimed = refuse_reason_replacing(&doc, &g2, Some(g1)).expect("allowed in its place");
        assert!(claimed.contains(&1) && claimed.contains(&2));
        // The cycle rule looks past the replaced relation too: B parallel to A, then A
        // parallel to B closes a cycle -- unless it is the first one's replacement.
        let (a, b) = (line("A", [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]), line("B", [0.0, 5.0, 0.0], [10.0, 5.0, 0.0]));
        let (ia, ib) = (a.id, b.id);
        doc.curves.insert(ia, a);
        doc.curves.insert(ib, b);
        let first = add(&mut doc, Relation::Parallel { master: ia, slave: ib });
        let turned = Relation::Parallel { master: ib, slave: ia };
        assert!(matches!(refuse_reason(&doc, &turned), Err(RelationError::Cycle { .. })));
        assert!(refuse_reason_replacing(&doc, &turned, Some(first)).is_ok());
    }

    /// Tangent direction and curvature at a parameter, by finite differences of `evaluate`
    /// alone -- the polynomial formulas are what is being checked, so they are not used here.
    fn tangent_and_curvature(c: &Curve, t: f64) -> (Vector3<f64>, f64) {
        let h = 1e-4;
        let (lo, hi) = c.domain();
        let t = t.clamp(lo + 2.0 * h, hi - 2.0 * h);
        let d1 = (c.evaluate(t + h) - c.evaluate(t - h)) / (2.0 * h);
        let d2 = (c.evaluate(t + h).coords - 2.0 * c.evaluate(t).coords + c.evaluate(t - h).coords)
            / (h * h);
        let kappa = d1.cross(&d2).norm() / d1.norm().powi(3);
        (d1.normalize(), kappa)
    }

    /// G1 holds after the master is dragged.
    #[test]
    fn g1_holds_after_the_master_is_dragged() {
        let master = line("Line", [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        let slave = bezier(
            "Curve",
            vec![[10.0, 0.0, 0.0], [14.0, 3.0, 0.0], [20.0, 6.0, 0.0], [25.0, 0.0, 0.0]],
            None,
        );
        let (m, s) = (master.id, slave.id);
        let mut doc = doc_with(vec![master, slave]);
        add(
            &mut doc,
            Relation::Joint {
                master: CurveEnd { curve: m, end: End::End },
                slave: CurveEnd { curve: s, end: End::Start },
                continuity: JointContinuity::G1,
            },
        );
        let kept = (doc.curve(s).unwrap().control_points()[1] - Point3::new(10.0, 0.0, 0.0)).norm();
        rederive(&mut doc);
        // Along the line's direction, at the distance the user had.
        let b1 = doc.curve(s).unwrap().control_points()[1];
        assert!((b1 - Point3::new(10.0 + kept, 0.0, 0.0)).norm() < 1e-9, "{b1:?}");

        // Drag the master's start so the line tips up; the slave's tangent follows.
        doc.curve_mut(m).unwrap().set_control_point(0, Point3::new(0.0, -5.0, 0.0));
        let moved = rederive(&mut doc);
        assert_eq!(moved.len(), 1, "one derived point moved");
        let (tm, _) = tangent_and_curvature(doc.curve(m).unwrap(), 1.0);
        let (ts, _) = tangent_and_curvature(doc.curve(s).unwrap(), 0.0);
        assert!((tm - ts).norm() < 1e-4, "tangents differ: {tm:?} vs {ts:?}");
    }

    /// G2 holds after the master is dragged, curvature measured numerically on both sides.
    #[test]
    fn g2_holds_after_the_master_is_dragged() {
        let master = bezier(
            "Master",
            vec![[0.0, 0.0, 0.0], [5.0, 4.0, 0.0], [10.0, 4.0, 0.0], [15.0, 0.0, 0.0]],
            None,
        );
        let slave = bezier(
            "Slave",
            vec![[15.0, 0.0, 0.0], [18.0, -2.0, 0.0], [22.0, -6.0, 0.0], [30.0, 0.0, 0.0]],
            None,
        );
        let (m, s) = (master.id, slave.id);
        let mut doc = doc_with(vec![master, slave]);
        add(
            &mut doc,
            Relation::Joint {
                master: CurveEnd { curve: m, end: End::End },
                slave: CurveEnd { curve: s, end: End::Start },
                continuity: JointContinuity::G2,
            },
        );
        for drag in [None, Some(Point3::new(9.0, 7.0, 1.0))] {
            if let Some(to) = drag {
                doc.curve_mut(m).unwrap().set_control_point(2, to);
            }
            rederive(&mut doc);
            let (tm, km) = tangent_and_curvature(doc.curve(m).unwrap(), 1.0);
            let (ts, ks) = tangent_and_curvature(doc.curve(s).unwrap(), 0.0);
            assert!((tm - ts).norm() < 1e-3, "tangents differ: {tm:?} vs {ts:?}");
            assert!(
                (km - ks).abs() < 1e-3 * km.max(1.0),
                "curvatures differ: {km} vs {ks} (drag {drag:?})"
            );
        }
    }

    /// G2 against a weighted master matches curvature numerically.
    ///
    /// The polynomial formula reads the control points and is wrong for a rational curve; the
    /// derivation here reads the actual derivatives. Measured by finite differences of
    /// `evaluate` on both sides of the joint, which knows nothing of either formula.
    #[test]
    fn g2_against_a_weighted_master_matches_curvature_numerically() {
        // A weighted conic: the middle weight pulls the curve toward its control point, which
        // changes the curvature at the end without changing the tangent.
        let master = bezier(
            "Conic",
            vec![[0.0, 0.0, 0.0], [10.0, 8.0, 0.0], [20.0, 0.0, 0.0]],
            Some(vec![1.0, 0.6, 1.0]),
        );
        let slave = bezier(
            "Slave",
            vec![[20.0, 0.0, 0.0], [24.0, -3.0, 0.0], [30.0, -5.0, 0.0], [40.0, 0.0, 0.0]],
            Some(vec![1.0, 1.4, 0.8, 1.0]),
        );
        let (m, s) = (master.id, slave.id);
        let mut doc = doc_with(vec![master, slave]);
        add(
            &mut doc,
            Relation::Joint {
                master: CurveEnd { curve: m, end: End::End },
                slave: CurveEnd { curve: s, end: End::Start },
                continuity: JointContinuity::G2,
            },
        );
        rederive(&mut doc);
        let (tm, km) = tangent_and_curvature(doc.curve(m).unwrap(), 1.0);
        let (ts, ks) = tangent_and_curvature(doc.curve(s).unwrap(), 0.0);
        assert!((tm - ts).norm() < 1e-3, "tangents differ: {tm:?} vs {ts:?}");
        assert!(km > 1e-3, "the conic has curvature to match");
        assert!((km - ks).abs() < 2e-3 * km, "curvatures differ: {km} vs {ks}");
    }

    /// A slave joined by its *end* follows the same way, on its reversed copy.
    #[test]
    fn a_slave_joined_by_its_end_follows_the_master_too() {
        let master = line("Line", [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        let slave = bezier(
            "Curve",
            vec![[25.0, 0.0, 0.0], [20.0, 6.0, 0.0], [14.0, 3.0, 0.0], [10.0, 0.0, 0.0]],
            None,
        );
        let (m, s) = (master.id, slave.id);
        let mut doc = doc_with(vec![master, slave]);
        add(
            &mut doc,
            Relation::Joint {
                master: CurveEnd { curve: m, end: End::End },
                slave: CurveEnd { curve: s, end: End::End },
                continuity: JointContinuity::G2,
            },
        );
        doc.curve_mut(m).unwrap().set_control_point(0, Point3::new(0.0, -5.0, 0.0));
        rederive(&mut doc);
        let (tm, km) = tangent_and_curvature(doc.curve(m).unwrap(), 1.0);
        let (ts, ks) = tangent_and_curvature(doc.curve(s).unwrap(), 1.0);
        // Arriving at the joint from the slave's side is leaving it the master's way.
        assert!((tm + ts).norm() < 1e-3, "tangents: {tm:?} vs {ts:?}");
        assert!((km - ks).abs() < 1e-3, "a line has no curvature and neither may the joint: {ks}");
    }

    /// Parallel, perpendicular and equal length hold after the master is dragged.
    #[test]
    fn parallel_perpendicular_and_equal_length_hold_after_the_master_is_dragged() {
        for kind in ["parallel", "perpendicular", "equal length"] {
            let master = line("Master", [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
            let slave = line("Slave", [0.0, 5.0, 0.0], [3.0, 9.0, 0.0]);
            let (m, s) = (master.id, slave.id);
            let mut doc = doc_with(vec![master, slave]);
            let relation = match kind {
                "parallel" => Relation::Parallel { master: m, slave: s },
                "perpendicular" => Relation::Perpendicular { master: m, slave: s },
                _ => Relation::EqualLength { master: m, slave: s },
            };
            add(&mut doc, relation);
            rederive(&mut doc);
            doc.curve_mut(m).unwrap().set_control_point(1, Point3::new(8.0, 6.0, 0.0));
            rederive(&mut doc);
            let mp = doc.curve(m).unwrap().control_points().to_vec();
            let sp = doc.curve(s).unwrap().control_points().to_vec();
            let dm = (mp[1] - mp[0]).normalize();
            let ds = sp[1] - sp[0];
            assert_eq!(sp[0], Point3::new(0.0, 5.0, 0.0), "{kind}: the fixed end stays");
            match kind {
                "parallel" => assert!(ds.normalize().cross(&dm).norm() < 1e-9, "{kind}: {ds:?}"),
                "perpendicular" => assert!(ds.dot(&dm).abs() < 1e-9, "{kind}: {ds:?}"),
                _ => assert!(
                    (ds.norm() - (mp[1] - mp[0]).norm()).abs() < 1e-9,
                    "{kind}: {}",
                    ds.norm()
                ),
            }
            if kind != "equal length" {
                assert!((ds.norm() - 5.0).abs() < 1e-9, "{kind}: the length is kept");
            }
        }
    }

    /// A relation claiming an owned point is refused, naming the owner.
    #[test]
    fn a_relation_claiming_an_owned_point_is_refused_naming_the_owner() {
        let master = line("Line", [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        let other = line("Other", [10.0, 9.0, 0.0], [10.0, 0.0, 0.0]);
        let mut slave = bezier(
            "Curve",
            vec![[10.0, 0.0, 0.0], [14.0, 3.0, 0.0], [20.0, 6.0, 0.0], [25.0, 0.0, 0.0]],
            None,
        );
        let (m, s, o) = (master.id, slave.id, other.id);
        let block = crate::NodeId::new();
        slave.place_on(1, Some(block));
        let mut doc = doc_with(vec![master, other, slave]);
        let g1 = |from: CurveId| Relation::Joint {
            master: CurveEnd { curve: from, end: End::End },
            slave: CurveEnd { curve: s, end: End::Start },
            continuity: JointContinuity::G1,
        };
        let refused = refuse_reason(&doc, &g1(m)).expect_err("point 1 sits on a block");
        assert!(
            matches!(&refused, RelationError::Owned { index: 1, owner, .. } if owner.contains("point block")),
            "{refused}"
        );
        // Free the point; the first relation is allowed, and a second on the same point names it.
        doc.curve_mut(s).unwrap().place_on(1, None);
        add(&mut doc, g1(m));
        let refused = refuse_reason(&doc, &g1(o)).expect_err("point 1 is now the G1's");
        assert!(
            matches!(&refused, RelationError::Owned { index: 1, owner, .. } if owner.contains("G1 relation to Line")),
            "{refused}"
        );
    }

    /// A relation that closes a cycle is refused, naming the relation that closes it.
    #[test]
    fn a_relation_that_closes_a_cycle_is_refused_naming_it() {
        let a = line("A", [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        let b = line("B", [0.0, 5.0, 0.0], [3.0, 9.0, 0.0]);
        let c = line("C", [0.0, 10.0, 0.0], [2.0, 15.0, 0.0]);
        let (ia, ib, ic) = (a.id, b.id, c.id);
        let mut doc = doc_with(vec![a, b, c]);
        add(&mut doc, Relation::Parallel { master: ia, slave: ib });
        add(&mut doc, Relation::EqualLength { master: ib, slave: ic });
        let refused =
            refuse_reason(&doc, &Relation::Parallel { master: ic, slave: ia }).expect_err("a cycle");
        // Named from the slave's side: the relation through which A is already followed.
        assert!(
            matches!(&refused, RelationError::Cycle { closes, .. } if closes.contains("parallel: B follows A")),
            "{refused}"
        );
    }

    /// Breaking a relation moves nothing: the control points are bit-identical.
    #[test]
    fn breaking_a_relation_moves_nothing() {
        let master = line("Line", [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        let slave = bezier(
            "Curve",
            vec![[10.0, 0.0, 0.0], [14.0, 3.0, 0.0], [20.0, 6.0, 0.0], [25.0, 0.0, 0.0]],
            None,
        );
        let (m, s) = (master.id, slave.id);
        let mut doc = doc_with(vec![master, slave]);
        let id = add(
            &mut doc,
            Relation::Joint {
                master: CurveEnd { curve: m, end: End::End },
                slave: CurveEnd { curve: s, end: End::Start },
                continuity: JointContinuity::G2,
            },
        );
        rederive(&mut doc);
        let bits = |doc: &Document| -> Vec<u64> {
            doc.curve(s)
                .unwrap()
                .control_points()
                .iter()
                .flat_map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])
                .collect()
        };
        let before = bits(&doc);
        doc.relations.remove(&id);
        assert!(rederive(&mut doc).is_empty());
        assert_eq!(bits(&doc), before);
    }

    /// A relation that no longer holds on open is reported and not applied.
    #[test]
    fn a_relation_that_no_longer_holds_on_open_is_reported_and_not_applied() {
        let master = line("Line", [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        let slave = bezier(
            "Curve",
            vec![[10.0, 0.0, 0.0], [14.0, 3.0, 0.0], [20.0, 6.0, 0.0], [25.0, 0.0, 0.0]],
            None,
        );
        let (m, s) = (master.id, slave.id);
        let mut doc = doc_with(vec![master, slave]);
        let id = add(
            &mut doc,
            Relation::Joint {
                master: CurveEnd { curve: m, end: End::End },
                slave: CurveEnd { curve: s, end: End::Start },
                continuity: JointContinuity::G1,
            },
        );
        rederive(&mut doc);
        assert!(check(&doc).is_empty(), "it holds as derived");
        // An older build moved the derived point off the tangent and saved.
        let off = Point3::new(14.0, 3.0, 0.0);
        doc.curve_mut(s).unwrap().set_control_point(1, off);
        let broken = check(&doc);
        assert_eq!(broken.len(), 1);
        assert_eq!(broken[0].0, id);
        assert!(broken[0].1.contains("point 1 of Curve"), "{}", broken[0].1);
        doc.relations.get_mut(&id).unwrap().broken = Some(broken[0].1.clone());
        assert!(rederive(&mut doc).is_empty(), "a broken relation is not applied");
        assert_eq!(doc.curve(s).unwrap().control_points()[1], off);
    }

    /// The order every master comes before what follows it: a chain re-derives in one pass.
    #[test]
    fn a_chain_of_relations_rederives_masters_first_in_one_pass() {
        let a = line("A", [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        let b = line("B", [0.0, 5.0, 0.0], [3.0, 9.0, 0.0]);
        let c = line("C", [0.0, 10.0, 0.0], [2.0, 15.0, 0.0]);
        let (ia, ib, ic) = (a.id, b.id, c.id);
        let mut doc = doc_with(vec![a, b, c]);
        // Inserted downstream first, so an order that followed the map would get C wrong.
        add(&mut doc, Relation::Parallel { master: ib, slave: ic });
        add(&mut doc, Relation::Parallel { master: ia, slave: ib });
        doc.curve_mut(ia).unwrap().set_control_point(1, Point3::new(6.0, 8.0, 0.0));
        rederive(&mut doc);
        let dir = |id: CurveId| {
            let p = doc.curve(id).unwrap().control_points();
            (p[1] - p[0]).normalize()
        };
        assert!(dir(ia).cross(&dir(ic)).norm() < 1e-9, "C followed B followed A in one pass");
    }
}
