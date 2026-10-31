// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! A coil block and a thread block read into the fields they are, once for every reader.
//!
//! The compile, the step budget, the extent and the shader generator all need the same
//! numbers -- the pitch, the radius the bound is taken at, the turns -- and each of them working
//! the designation table and the size pair out for itself is four places for one arithmetic to
//! disagree. So each block is read here, refusals and all, and the readers ask.

use fieldform_scene::{CoilSection, CoilSize, Document, Hand, Node, ParamRef};

use super::CompileError;
use crate::{Coil, Thread, ThreadProfile, WireSection};

/// The coil a [`Node::Coil`] is, in its own frame, or the refusal by name.
pub fn coil_of(
    diameter: &ParamRef,
    size: &CoilSize,
    section: &CoilSection,
    hand: Hand,
    resolve: &dyn Fn(&ParamRef) -> Option<f64>,
) -> Result<Coil, CompileError> {
    let mean_radius = resolve(diameter).ok_or(CompileError::UnresolvedRadius)? * 0.5;
    let (turns, pitch) = size.turns_and_pitch(resolve).ok_or(CompileError::UnresolvedRadius)?;
    // `!(> 0)` so a NaN is refused with the rest.
    if !(pitch > 0.0) {
        return Err(CompileError::HelixNeedsAPositivePitch("coil", pitch));
    }
    if !(turns > 0.0) {
        return Err(CompileError::CoilNeedsTurns(turns));
    }
    let wire = resolve(section.wire()).ok_or(CompileError::UnresolvedRadius)?;
    if !(wire > 0.0) {
        return Err(CompileError::CoilNeedsAWire(wire));
    }
    let section = match section {
        CoilSection::Circle { .. } => WireSection::Circle { radius: wire * 0.5 },
        CoilSection::Square { .. } => WireSection::Square { half: wire * 0.5 },
    };
    // Taller along the axis than the pitch, the turns would fuse: the three copies the field
    // reads would then not be enough, and the solid would not be a coil anyway.
    if wire > pitch {
        return Err(CompileError::CoilTurnsWouldFuse(wire, pitch));
    }
    let coil = Coil { mean_radius, pitch, turns, section, hand: hand.sign() };
    if !(coil.inner_radius() > 0.0) {
        return Err(CompileError::HelixSectionCrossesTheAxis("coil", coil.inner_radius()));
    }
    Ok(coil)
}

/// The major diameter and pitch a thread reads: the table's, where a designation names one, and
/// the typed pair otherwise.
pub fn thread_major_and_pitch(
    major: &ParamRef,
    pitch: &ParamRef,
    designation: Option<&str>,
    resolve: &dyn Fn(&ParamRef) -> Option<f64>,
) -> Result<(f64, f64), CompileError> {
    match designation {
        Some(name) => fieldform_scene::thread_designation(name)
            .ok_or_else(|| CompileError::UnknownThreadDesignation(name.to_string())),
        None => Ok((
            resolve(major).ok_or(CompileError::UnresolvedRadius)?,
            resolve(pitch).ok_or(CompileError::UnresolvedRadius)?,
        )),
    }
}

/// The thread a [`Node::Thread`] is, in its own frame, or the refusal by name.
pub fn thread_of(
    major: &ParamRef,
    pitch: &ParamRef,
    length: &ParamRef,
    hand: Hand,
    clearance: &ParamRef,
    designation: Option<&str>,
    resolve: &dyn Fn(&ParamRef) -> Option<f64>,
) -> Result<Thread, CompileError> {
    let (d, p) = thread_major_and_pitch(major, pitch, designation, resolve)?;
    if !(p > 0.0) {
        return Err(CompileError::HelixNeedsAPositivePitch("thread", p));
    }
    let length = resolve(length).ok_or(CompileError::UnresolvedRadius)?;
    if !(length > 0.0) {
        return Err(CompileError::ThreadNeedsALength(length));
    }
    let clearance = resolve(clearance).ok_or(CompileError::UnresolvedRadius)?;
    let thread = Thread {
        profile: ThreadProfile { major_radius: d * 0.5, pitch: p },
        length,
        clearance,
        hand: hand.sign(),
    };
    if !(thread.inner_radius() > 0.0) {
        return Err(CompileError::HelixSectionCrossesTheAxis("thread", thread.inner_radius()));
    }
    Ok(thread)
}

/// The helical solid a node is, if it is one, as the pitch and the radius its bound is taken at
/// -- what the step budget divides by -- or `None` for any other node and for one that is
/// refused.
pub fn helical_bound(document: &Document, node: &Node) -> Option<(f64, f64)> {
    let resolve = |r: &ParamRef| document.resolve(r);
    match node {
        Node::Coil { diameter, size, section, hand, .. } => {
            coil_of(diameter, size, section, *hand, &resolve).ok().map(|c| (c.pitch, c.inner_radius()))
        }
        Node::Thread { major, pitch, length, hand, clearance, designation, .. } => {
            thread_of(major, pitch, length, *hand, clearance, designation.as_deref(), &resolve)
                .ok()
                .map(|t| (t.profile.pitch, t.inner_radius()))
        }
        _ => None,
    }
}

/// What a helical node's value is divided by, one for anything else: the global stretch of the
/// map at the radius its bound is taken at.
pub fn helical_divisor(document: &Document, node: &Node) -> f64 {
    helical_bound(document, node).map_or(1.0, |(p, r)| crate::helix_stretch(p, r))
}
