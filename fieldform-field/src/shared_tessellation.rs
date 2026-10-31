// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! One tessellation of a cage, shared by everything that wants it at the same time.
//!
//! The viewport's field and the self-intersection check are meant to look at the *same* surface:
//! a check run on a coarser proxy can miss a fold the field then draws with the wrong sign, which
//! is the failure the check exists to prevent. They used to guarantee that by each tessellating at
//! the same level, so a large cage paid for its limit surface twice -- 5 s of the 14 s measured at
//! 24 576 faces, `fieldform-field/examples/self_intersection_cost.rs` -- and for its memory twice
//! while both were alive. Now whoever asks first builds it, anyone who asks for the same geometry
//! at the same level while it is alive waits for that one build and gets that one surface, and it
//! is freed when the last of them lets go.
//!
//! **Weak on purpose.** Nothing here keeps a tessellation alive: at 24 576 faces it is hundreds of
//! megabytes, and a cache that held the last one would hold it for ever. The cost is that two
//! consumers that never overlap each build their own, which is correct and merely not shared.

use fieldform_cage::Cage;
use fieldform_subd::{LimitSurface, Tessellation};
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Arc, Mutex, OnceLock, Weak};

struct Built {
    tessellation: OnceLock<Tessellation>,
}

/// A cage's tessellation at one level, shared. Derefs to the [`Tessellation`].
#[derive(Clone)]
pub struct SharedTessellation(Arc<Built>);

impl Deref for SharedTessellation {
    type Target = Tessellation;

    fn deref(&self) -> &Tessellation {
        self.0.tessellation.get().expect("a shared tessellation is built before it is handed out")
    }
}

type Key = (u64, usize);

static LIVE: Mutex<Option<HashMap<Key, Weak<Built>>>> = Mutex::new(None);

/// The cage's limit surface tessellated at `levels`, built once for however many callers want it
/// at once.
///
/// The per-corner parameter gradients are dropped: a distance field does not read them and the
/// check does not either, they are the largest of the per-triangle tables, and the one consumer
/// that does (the conformal cell map) tessellates for itself.
pub fn shared_tessellation(cage: &Cage, levels: usize) -> SharedTessellation {
    let key = (cage.geometry_fingerprint(), levels);
    let built = {
        let mut guard = LIVE.lock().unwrap_or_else(|e| e.into_inner());
        let live = guard.get_or_insert_with(HashMap::new);
        live.retain(|_, w| w.strong_count() > 0);
        match live.get(&key).and_then(Weak::upgrade) {
            Some(built) => built,
            None => {
                let built = Arc::new(Built { tessellation: OnceLock::new() });
                live.insert(key, Arc::downgrade(&built));
                built
            }
        }
    };
    // The lock is not held while building: a second caller finds the same `Built` and waits here,
    // on the `OnceLock`, for the first to finish -- and a caller wanting a different cage is not
    // kept waiting behind it.
    built.tessellation.get_or_init(|| {
        let mut t = LimitSurface::new(cage).tessellate(levels);
        t.triangle_gradients = Vec::new();
        t
    });
    SharedTessellation(built)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Two callers that overlap get one surface, not two** -- the point of sharing it. The
    /// second asks while the first still holds its tessellation, and both read the same memory.
    #[test]
    fn callers_that_overlap_share_one_tessellation() {
        let cage = Cage::cube_of(100.0);
        let first = shared_tessellation(&cage, 3);
        let second = shared_tessellation(&cage, 3);
        assert!(std::ptr::eq(&*first, &*second), "the same tessellation, not an equal one");
        let other_level = shared_tessellation(&cage, 4);
        assert!(!std::ptr::eq(&*first, &*other_level), "a different level is a different surface");
        assert!(other_level.triangles.len() > first.triangles.len());
    }

    /// **Nothing is kept alive for a caller who has not arrived yet**: once every holder lets go
    /// the next asker builds afresh, so a large cage's hundreds of megabytes are freed.
    #[test]
    fn a_tessellation_nobody_holds_is_freed() {
        // A cage of its own: the map is global, and the test beside this one holds a 100 mm cube
        // at the same level while it runs, which this one would see alive under a parallel run.
        let cage = Cage::cube_of(73.0);
        let weak = {
            let held = shared_tessellation(&cage, 3);
            Arc::downgrade(&held.0)
        };
        assert!(weak.upgrade().is_none(), "no holder, no tessellation");
    }

    /// **Callers on different threads asking at once share the one build** -- the case that
    /// matters, the check and the field starting together.
    #[test]
    fn simultaneous_callers_on_threads_share_a_build() {
        let cage = Cage::cube_of(100.0);
        let pointers: Vec<usize> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    s.spawn(|| {
                        let t = shared_tessellation(&cage, 4);
                        // Hold it a moment so the others find it alive.
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        std::ptr::from_ref(&*t) as usize
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(pointers.windows(2).all(|w| w[0] == w[1]), "{pointers:?}");
    }
}
