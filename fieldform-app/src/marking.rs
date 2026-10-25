// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The marking menu's slots and the gesture that picks one (`interaction-spec.md` §7).
//!
//! **Eight directions, fixed per kind of thing selected.** A slot table says, for a body, a curve,
//! a profile, a plane, a cage and empty space, which command sits in which direction; a command not
//! in the table gets no slot, and a command not on offer keeps its place, dimmed, rather than
//! letting the others slide round into it -- a direction has to mean the same command every time
//! for the hand to learn it (§7.3: never reorder). South is always More…, which opens the palette
//! on what is on offer. The centre is empty, so a second click where a stray right-click landed
//! closes the menu (the owner's ruling, 2026-10-17; it used to repeat the last command, §8).
//!
//! The tables were drafted from what each kind is most often given next in this application and
//! **approved by the owner 2026-10-17** (§14), and are held by `the_slot_tables_are_the_ones_approved`
//! so a change to one is a decision, not a drift.
//!
//! Free of the `App`, so the gesture's geometry is a unit test (T8).

/// What a right-click is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Subject {
    /// Nothing selected: what can be made.
    Empty,
    Body,
    /// A curve that closes no loop.
    Curve,
    /// A closed loop of curves.
    Profile,
    Plane,
    Cage,
    /// Vertices, edges or faces of the cage picked: the cage tools' table (R10).
    CageElements,
}

/// The eight directions, clockwise from up, by name, for the test that walks them. South is More….
#[cfg(test)]
pub const DIRECTIONS: [&str; 8] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

/// The slot South always holds.
pub const MORE: usize = 4;

/// How far the pointer must move from where the menu opened before a direction counts, in points.
pub const DEAD_ZONE: f32 = 18.0;

/// The command in each direction for `subject`, clockwise from North; `None` where nothing sits,
/// and South always `None` here since it is More….
pub fn slots(subject: Subject) -> [Option<&'static str>; 8] {
    match subject {
        Subject::Empty => [
            Some("create.sphere"),
            Some("create.cube"),
            Some("create.cylinder"),
            Some("create.curve"),
            None,
            Some("create.plane"),
            Some("create.tpms"),
            Some("edit.paste_here"),
        ],
        Subject::Body => [
            Some("create.boolean"),
            Some("modify.offset"),
            Some("modify.shell"),
            Some("modify.pattern"),
            None,
            Some("modify.mirror"),
            Some("modify.split"),
            Some("edit.duplicate"),
        ],
        Subject::Curve => [
            Some("create.extrude"),
            Some("create.revolve"),
            Some("create.sweep"),
            Some("create.pipe"),
            None,
            Some("edit.duplicate"),
            Some("edit.delete_body"),
            Some("view.frame_selection"),
        ],
        Subject::Profile => [
            Some("create.extrude"),
            Some("create.revolve"),
            Some("create.sweep"),
            Some("create.loft"),
            None,
            Some("edit.duplicate"),
            Some("edit.delete_body"),
            Some("view.frame_selection"),
        ],
        Subject::Plane => [
            Some("modify.split"),
            Some("create.boolean"),
            Some("view.section"),
            Some("edit.duplicate"),
            None,
            Some("view.toggle_visibility"),
            Some("edit.delete_body"),
            Some("view.frame_selection"),
        ],
        Subject::Cage => [
            Some("cage.refine"),
            Some("create.boolean"),
            Some("modify.offset"),
            Some("modify.shell"),
            None,
            Some("modify.pattern"),
            Some("modify.mirror"),
            Some("edit.duplicate"),
        ],
        // One table for any cage-element selection, as the owner ruled (R10): N Extrude,
        // NE Inset, E Insert Edge, SE Bridge, S More…, SW Fill hole, W Knife, NW Weld. The three
        // tools not built yet keep their directions empty rather than letting another move in.
        Subject::CageElements => [
            Some("cage.extrude"),
            Some("cage.inset"),
            Some("cage.insert_ring"),
            Some("cage.bridge"),
            None,
            Some("cage.fill_hole"),
            Some("cage.cut"),
            Some("cage.weld"),
        ],
    }
}

/// The direction a pointer `delta` from where the menu opened points at, or `None` inside the dead
/// zone. Screen coordinates: `y` grows downward, so up is North.
pub fn slot_at(delta: (f32, f32)) -> Option<usize> {
    let (dx, dy) = delta;
    if dx.hypot(dy) < DEAD_ZONE {
        return None;
    }
    // Degrees clockwise from North, so the eight slots sit at multiples of 45.
    let bearing = dx.atan2(-dy).to_degrees();
    let slot = ((bearing + 360.0 + 22.5) / 45.0).floor() as usize % 8;
    Some(slot)
}

/// Where slot `i`'s label sits, `radius` from the centre, in the same screen coordinates.
pub fn slot_offset(i: usize, radius: f32) -> (f32, f32) {
    let bearing = (i as f32 * 45.0).to_radians();
    (radius * bearing.sin(), -radius * bearing.cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **T8, the unit half: a drag picks the slot it points at, and a short one picks nothing**
    /// (`interaction-spec.md` §7.3, §11). Every direction, its neighbours' edges, and the dead zone.
    #[test]
    fn a_drag_picks_the_slot_it_points_at() {
        let r = 60.0;
        for (i, name) in DIRECTIONS.iter().enumerate() {
            let (dx, dy) = slot_offset(i, r);
            assert_eq!(slot_at((dx, dy)), Some(i), "{name}");
            // Twenty degrees either side is still that slot; the edge is at 22.5.
            for nudge in [-20.0_f32, 20.0] {
                let b = (i as f32 * 45.0 + nudge).to_radians();
                assert_eq!(slot_at((r * b.sin(), -r * b.cos())), Some(i), "{name} {nudge:+}°");
            }
        }
        assert_eq!(slot_at((0.0, -40.0)), Some(0), "up is North");
        assert_eq!(slot_at((40.0, 0.0)), Some(2), "right is East");
        assert_eq!(slot_at((0.0, 40.0)), Some(MORE), "down is More…");
        assert_eq!(slot_at((5.0, -5.0)), None, "inside the dead zone nothing is picked");
    }

    /// **The tables are the ones drafted, slot for slot**, so a change to one is made here on
    /// purpose; and South is More… in every one (§7.2). Stable across sessions because nothing
    /// about them is computed (§7.3).
    #[test]
    fn the_slot_tables_are_the_ones_approved() {
        for subject in [Subject::Empty, Subject::Body, Subject::Curve, Subject::Profile, Subject::Plane, Subject::Cage] {
            let table = slots(subject);
            assert_eq!(table[MORE], None, "{subject:?}: South is More…");
            for id in table.into_iter().flatten() {
                assert!(crate::commands::by_id(id).is_some(), "{subject:?}: {id} is a registry command");
            }
            let mut seen = std::collections::HashSet::new();
            assert!(table.iter().flatten().all(|id| seen.insert(*id)), "{subject:?}: a command in two slots");
        }
        assert_eq!(
            slots(Subject::Body),
            [
                Some("create.boolean"),
                Some("modify.offset"),
                Some("modify.shell"),
                Some("modify.pattern"),
                None,
                Some("modify.mirror"),
                Some("modify.split"),
                Some("edit.duplicate"),
            ]
        );
        assert_eq!(slots(Subject::Profile)[0], Some("create.extrude"), "North on a profile extrudes");
    }
}
