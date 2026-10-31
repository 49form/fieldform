// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A hole block read into the field it is, once for every reader: the compile, the extent and
//! the shader's slots all ask this, so the section is worked out and refused in one place.

use fieldform_scene::{Document, Ends, Graph, HoleDepth, HoleKind, HoleTip, ParamRef};
use nalgebra::Point3;

use super::{resolve_ends, CompileError};
use crate::{Hole, HoleSection, Thread, ThreadProfile};

/// The tip's own depth below the shoulder for a drill of included angle `angle` degrees at
/// radius `radius`.
pub fn drill_point_depth(radius: f64, angle: f64) -> f64 {
    radius / (angle.to_radians() * 0.5).tan()
}

/// A hole's section from its numbers, refused by name where it is not a hole. `thread` is the
/// tapped hole's thread, whose minor radius is the drill's.
pub fn hole_section(
    kind: &HoleKind,
    depth: &HoleDepth,
    tip: &HoleTip,
    resolve: &dyn Fn(&ParamRef) -> Option<f64>,
) -> Result<(HoleSection, Option<Thread>), CompileError> {
    let number = |r: &ParamRef| resolve(r).ok_or(CompileError::UnresolvedRadius);
    let positive = |what: &'static str, v: f64| {
        if v > 0.0 { Ok(v) } else { Err(CompileError::HoleSizeNotPositive(what, v)) }
    };
    let blind = match depth {
        HoleDepth::Blind(d) => Some(positive("depth", number(d)?)?),
        HoleDepth::Through => None,
    };
    let mut thread = None;
    // The entry's steps, `(r, z)` from the mouth down, and the radius the hole carries on at.
    let (mut wall, radius) = match kind {
        HoleKind::Simple { diameter } => {
            let r = positive("diameter", number(diameter)?)? * 0.5;
            (vec![(r, 0.0)], r)
        }
        HoleKind::Counterbore { diameter, bore_diameter, bore_depth } => {
            let r = positive("diameter", number(diameter)?)? * 0.5;
            let rb = number(bore_diameter)? * 0.5;
            if !(rb > r) {
                return Err(CompileError::HoleStepNotWider("counterbore", rb * 2.0, r * 2.0));
            }
            let db = positive("counterbore depth", number(bore_depth)?)?;
            if blind.is_some_and(|d| db >= d) {
                return Err(CompileError::HoleStepTooDeep("counterbore", db, blind.unwrap_or(0.0)));
            }
            (vec![(rb, 0.0), (rb, db), (r, db)], r)
        }
        HoleKind::Countersink { diameter, sink_diameter, sink_angle } => {
            let r = positive("diameter", number(diameter)?)? * 0.5;
            let rs = number(sink_diameter)? * 0.5;
            if !(rs > r) {
                return Err(CompileError::HoleStepNotWider("countersink", rs * 2.0, r * 2.0));
            }
            let angle = number(sink_angle)?;
            if !(angle > 0.0 && angle < 180.0) {
                return Err(CompileError::HoleAngle("countersink", angle));
            }
            let ds = drill_point_depth(rs - r, angle);
            if blind.is_some_and(|d| ds >= d) {
                return Err(CompileError::HoleStepTooDeep("countersink", ds, blind.unwrap_or(0.0)));
            }
            (vec![(rs, 0.0), (r, ds)], r)
        }
        HoleKind::Tapped { designation, thread_depth, clearance } => {
            let (d, p) = fieldform_scene::thread_designation(designation)
                .ok_or_else(|| CompileError::UnknownThreadDesignation(designation.clone()))?;
            let length = positive("thread depth", number(thread_depth)?)?;
            if blind.is_some_and(|depth| length > depth) {
                return Err(CompileError::ThreadDeeperThanHole(length, blind.unwrap_or(0.0)));
            }
            let t = Thread {
                profile: ThreadProfile { major_radius: d * 0.5, pitch: p },
                length,
                clearance: number(clearance)?,
                hand: 1.0,
            };
            // Drilled to the thread's own minor radius, so the two meet at the root.
            let r = t.inner_radius();
            if !(r > 0.0) {
                return Err(CompileError::HelixSectionCrossesTheAxis("thread", r));
            }
            thread = Some(t);
            (vec![(r, 0.0)], r)
        }
    };
    // Straight up from the mouth by the lead-in, so the cutter's top is not in the body's face
    // (`HoleSection`), and the thread with it.
    let lead = HoleSection::lead_in(wall[0].0);
    wall.insert(0, (wall[0].0, -lead));
    if let Some(t) = thread.as_mut() {
        t.length += lead;
    }
    let tip = match blind {
        None => None,
        Some(d) => {
            wall.push((radius, d));
            Some(match tip {
                HoleTip::Flat => d,
                HoleTip::Drill(angle) => {
                    let angle = number(angle)?;
                    if !(angle > 0.0 && angle < 180.0) {
                        return Err(CompileError::HoleAngle("drill point", angle));
                    }
                    d + drill_point_depth(radius, angle)
                }
            })
        }
    };
    Ok((HoleSection { wall, tip }, thread))
}

/// The hole a [`fieldform_scene::Node::Hole`] is, placed, or the refusal by name.
pub fn hole_of(
    document: &Document,
    graph: &Graph,
    ends: &Ends,
    kind: &HoleKind,
    depth: &HoleDepth,
    tip: &HoleTip,
    resolve: &dyn Fn(&ParamRef) -> Option<f64>,
) -> Result<Hole, CompileError> {
    let (entry, toward): (Point3<f64>, Point3<f64>) = resolve_ends(ends, document, graph, resolve)?;
    let axis = (toward - entry).try_normalize(1e-12).ok_or(CompileError::HoleNeedsADirection)?;
    let (section, thread) = hole_section(kind, depth, tip, resolve)?;
    Ok(Hole { entry, axis, section, thread })
}
