// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The traced view's tests: how a band and a draft are sized, and which compile or bake is kept.

    /// The property the staleness reasoning rests on: however wrong the cost estimate is, what
    /// runs is bounded by a band this view actually submitted.
    ///
    /// This is what replaced a field that recorded which view each cost came from. That field
    /// was written and never read, and this is why there was nothing for it to decide — a cost
    /// cannot say which view it measured, so the cap is taken from the submission side instead,
    /// where it cannot be stale.
    #[test]
    fn a_band_never_grows_past_the_growth_cap_however_cheap_the_estimate_claims_a_row_is() {
        // A cost of almost nothing asks for the whole frame.
        let greedy = band_rows(BAND_BUDGET_MS, Some(1e-9), Some(40), 1000, 100_000);
        assert_eq!(
            greedy,
            ((40.0 * BAND_GROWTH) as u32).max(MIN_BAND_PIXELS.div_ceil(1000)),
            "a stale, far too cheap row cost should still be held to the cap",
        );
        // And the cap compounds only by a band that really went up: ten rounds from one row
        // cannot reach a tall frame in one step.
        let mut rows = 1;
        for _ in 0..3 {
            rows = band_rows(BAND_BUDGET_MS, Some(1e-9), Some(rows), 1, 100_000);
        }
        assert!(rows < 100_000, "the cap should take several rounds, not one: {rows}");
    }

    /// With nothing measured yet, a band is the smallest that fills the card — never the frame.
    #[test]
    fn a_band_with_no_estimate_yet_is_the_floor_rather_than_the_whole_frame() {
        let height = 2000;
        let floor = MIN_BAND_PIXELS.div_ceil(640).clamp(1, height);
        assert_eq!(band_rows(BAND_BUDGET_MS, None, None, 640, height), floor);
        assert_eq!(band_rows(BAND_BUDGET_MS, Some(0.5), None, 640, height), floor);
        assert_eq!(band_rows(BAND_BUDGET_MS, None, Some(120), 640, height), floor);
        assert!(floor < height, "a floor that is the frame would measure nothing");
    }

    /// A dear row shrinks the band, which is the direction that keeps a band inside its budget.
    #[test]
    fn a_dearer_row_asks_for_fewer_rows() {
        let cheap = band_rows(BAND_BUDGET_MS, Some(0.01), Some(1000), 640, 4000);
        let dear = band_rows(BAND_BUDGET_MS, Some(10.0), Some(1000), 640, 4000);
        assert!(dear < cheap, "dear {dear} should be under cheap {cheap}");
        // Never past the frame, and never under the floor, whatever the arithmetic says.
        assert!(band_rows(BAND_BUDGET_MS, Some(1e-9), Some(u32::MAX), 640, 4000) <= 4000);
        assert!(band_rows(BAND_BUDGET_MS, Some(1e9), Some(1000), 640, 4000) >= 1);
    }
    use super::*;

    /// Every still band after the first starts on a tile row, walking a whole frame of them.
    ///
    /// **The sequence, not one call**, because the property is about where bands *start* and a
    /// band starts where the last one ended. A frame height that is not a multiple of 8 is in the
    /// list, since the last band ends there and must still leave every earlier start aligned; so is
    /// a budget of one row, which the floor would normally prevent and this must survive anyway.
    /// **A draft bands only at the ladder's last rung, and only over the draft budget.**
    #[test]
    fn a_draft_bands_only_when_the_ladder_has_run_out_and_the_frame_still_misses_its_budget() {
        let floor = DRAFT_LADDER.len() - 1;
        // Nothing measured: whole, which is what the ladder measures.
        assert!(!draft_bands(floor, None, 240, 240));
        // The owner's heavy case at the ladder's floor: 55 ms over 240 x 240, over the 50 ms
        // a draft has. This is the case P10 exists for, and the 3.3x it measured.
        assert!(draft_bands(floor, Some(55.0 / (240.0 * 240.0)), 240, 240));
        // A frame inside the draft budget stays one submission however many pixels it has.
        let inside = DRAFT_BUDGET_MS * 0.5 / (240.0 * 240.0);
        assert!(!draft_bands(floor, Some(inside), 240, 240));
        // The question is about the frame, so the same cost per pixel can go either way on
        // size alone.
        let dear = 203.0 / (720.0 * 720.0);
        assert!(draft_bands(floor, Some(dear), 720, 720));
        assert!(!draft_bands(floor, Some(dear), 64, 64));
    }

    /// **Item 104, as the property that would have caught it: a document the ladder can still
    /// shrink is never banded, however dear it is.**
    ///
    /// The four properties written with P10 covered this function's arithmetic and never asked
    /// the question that mattered, which is why a regression with a one-line cause reached the
    /// owner. Shrinking the picture costs resolution; banding costs the picture its coherence,
    /// and a sheared image is the dearer of the two. So every rung is tried first.
    #[test]
    fn a_draft_the_ladder_can_still_shrink_is_never_banded() {
        // The heaviest thing measured on this machine: 203 ms at 720 px, 20x its budget.
        let hopeless = 203.0 / (720.0 * 720.0);
        for rung in 0..DRAFT_LADDER.len() - 1 {
            assert!(
                !draft_bands(rung, Some(hopeless), 720, 720),
                "rung {rung} of {} banded instead of dropping a rung",
                DRAFT_LADDER.len()
            );
        }
        // Only the last rung, where there is no smaller picture left, may band.
        assert!(draft_bands(DRAFT_LADDER.len() - 1, Some(hopeless), 720, 720));
    }

    /// And the regression itself, stated as a number: a draft inside its own budget but over the
    /// interface's is exactly the case that used to band, and must not.
    #[test]
    fn a_draft_between_the_two_budgets_does_not_band() {
        let floor = DRAFT_LADDER.len() - 1;
        // 20 ms over 517 x 424: inside the 50 ms a draft has, over the 11 ms a band is sized
        // to. The owner's "banding and skipping frames" was this frame, cut into seven.
        let twenty = 20.0 / (517.0 * 424.0);
        assert!(twenty * 517.0 * 424.0 > INTERFACE_BUDGET_MS, "the case is the one meant");
        assert!(!draft_bands(floor, Some(twenty), 517, 424));
    }

    #[test]
    fn the_ladder_is_told_what_a_frame_cost_not_what_a_band_did() {
        // Nineteen bands of a 720-row frame, as `interface_latency` measures them: each one
        // cheap, the frame dear. Summed, the ladder sees the frame.
        let mut sweep = None;
        let mut fed = None;
        for band in 0..19u32 {
            let rows = if band == 18 { 720 - 18 * 38 } else { 38 };
            let (ms, covered) = sweep_with(sweep, 720, 720, 10.7, rows);
            if covered >= 720 {
                fed = Some(ms);
                sweep = None;
            } else {
                sweep = Some((720, 720, ms, covered));
            }
        }
        let whole = fed.expect("the sweep completed");
        assert!((whole - 19.0 * 10.7).abs() < 1e-3, "{whole}");
        // Which is over the draft budget, so the ladder goes down a rung — where a band's own
        // 10.7 ms would have climbed it.
        assert!(whole > DRAFT_BUDGET_MS);
        assert_eq!(next_rung(0, 0, whole), (1, 0));
        assert_eq!(next_rung(0, 0, 10.7), (0, 1));
    }

    #[test]
    fn a_resized_draft_sweep_measures_one_size_or_neither() {
        let part = sweep_with(None, 720, 720, 40.0, 300);
        assert_eq!(part, (40.0, 300));
        // The window changed between bands: what was gathered describes a frame that is no
        // longer being drawn, so it is dropped rather than added to.
        let after = sweep_with(Some((720, 720, part.0, part.1)), 480, 480, 12.0, 100);
        assert_eq!(after, (12.0, 100));
    }

    #[test]
    fn a_band_is_sized_by_the_budget_it_is_given() {
        // The same measured row cost, two budgets: the band scales with the budget, and the
        // interface's is the smaller, so its band is the smaller.
        let per_row = Some(0.2);
        let last = Some(10_000);
        let still = band_rows(BAND_BUDGET_MS, per_row, last, 640, 4000);
        let draft = band_rows(INTERFACE_BUDGET_MS, per_row, last, 640, 4000);
        assert!(draft < still, "draft {draft} still {still}");
        assert_eq!(still, (BAND_BUDGET_MS / 0.2) as u32);
        assert_eq!(draft, (INTERFACE_BUDGET_MS / 0.2) as u32);
        // But never below the floor that fills the card, whatever the budget asks — which is
        // what bounds the whole of P10's prize.
        let floor = MIN_BAND_PIXELS.div_ceil(640);
        assert_eq!(band_rows(0.001, per_row, last, 640, 4000), floor);
    }

    #[test]
    fn every_still_band_after_the_first_starts_on_a_tile_row() {
        for height in [405u32, 720, 732, 1080, 2560] {
            for wanted in [1u32, 7, 13, 18, 57, 100, 2000] {
                let mut first = 0u32;
                while first < height {
                    assert_eq!(
                        first % TILE_ROWS,
                        0,
                        "a band of {height} rows asked for {wanted} starts at row {first}"
                    );
                    let rows = tile_aligned_rows(wanted.clamp(1, height), height);
                    assert!(rows >= 1 && rows <= height);
                    first = (first + rows).min(height);
                }
            }
        }
    }

    fn key(geometry: u64, resolution: usize) -> BakeKey {
        BakeKey {
            leaf: fieldform_gpu::BakedLeaf::Cage(fieldform_scene::CageId::from_raw(7)),
            resolution,
            box_bits: [1, 2, 3, 4, 5, 6],
            band_bits: 9,
            geometry,
            levels: 4,
        }
    }

    /// **A cage's band is filed under the level its field was built at.** The level is read off
    /// the scene's tolerance, which nothing else in the key describes: Cage detail and the size of
    /// whatever else is on screen move it while the cage, its box and its band stay put. At the
    /// floor that cannot show, since the viewport's tolerance never asks past it on a cage that
    /// fills the view; a cage cut down by an intersection is framed on what is left of it, and
    /// there turning Cage detail moves the level between 4, 5 and 6 — and the view went on
    /// drawing the band of the old level until the cage itself was edited.
    ///
    /// The level and not the tolerance, because a tolerance that asks for the same level builds
    /// the same field, and the tolerance moves a little with every frame of a drag of another
    /// body: keyed on it, every cage would re-bake per frame, the miss each leaf's own band was
    /// made to stop.
    #[test]
    fn a_tolerance_that_asks_for_another_cage_level_files_the_bake_under_another_key() {
        let mut document = fieldform_scene::Document::new();
        let cube = Cage::cube_of(1000.0);
        let leaf = fieldform_gpu::BakedLeaf::Cage(document.add_cage(cube.clone()));
        let bounds = (Point3::new(-600.0, -600.0, -600.0), Point3::new(600.0, 600.0, 600.0));
        let filed =
            |tolerance: f64| rebuild::bake_key(&document, leaf, tolerance, 128, bounds, 300.0);

        // The starting view's tolerance, a finer one at the same level, and two past the floor.
        let (view, finer, past, further) = (7.8, 2.0, 1.0, 0.25);
        let level = |tolerance: f64| rebuild::cage_levels(&cube, tolerance);
        assert_eq!(level(view), level(finer), "the premise: two tolerances at one level");
        assert!(level(view) < level(past) && level(past) < level(further), "and two above it");

        assert!(
            filed(view) == filed(finer),
            "a tolerance that asks for the same level builds the same field, so keeps its band"
        );
        assert!(
            filed(view) != filed(past),
            "level {} and level {} are two surfaces, and one key served both",
            level(view),
            level(past)
        );
        assert!(
            filed(past) != filed(further),
            "level {} and level {} are two surfaces, and one key served both",
            level(past),
            level(further)
        );
    }

    /// **A shader that comes back late is dropped, not drawn.** It takes seconds, and the
    /// commonest change during one is the sharpest: hiding the last body leaves nothing to draw,
    /// and installing a compile started before that put the hidden body back on screen beside a
    /// panel correctly saying the part was empty.
    #[test]
    fn a_shader_compiled_for_a_shape_that_has_changed_is_not_installed() {
        assert!(compile_is_current(7, 7), "the newest request is what is wanted");
        assert!(!compile_is_current(6, 7), "a rebuild has happened since");
        assert!(!compile_is_current(0, 1), "and the first one can be overtaken too");

    }

    /// **A shader slower to compile than the gap between two rebuilds must still be able to
    /// land.** Every rebuild takes a new serial, so a compile overtaken by one is discarded —
    /// and on the M4000 under DX12 the patterned pipe compiles in five to nine seconds while
    /// an orbit asks for a rebuild about twice a second, so every compile was abandoned for an
    /// identical one and the march never ran at all. A rebuild that wants the source already
    /// being compiled adopts that worker instead of starting a second.
    #[test]
    fn a_rebuild_adopts_the_compile_already_running_for_its_own_source() {
        let wanted = "fn march() { /* the pipe */ }";
        assert!(compile_already_wanted(wanted, wanted), "the same source is the same shader");
        assert!(
            !compile_already_wanted("fn march() { /* the pipe, filleted */ }", wanted),
            "a changed graph is a changed shader, and that compile is no longer the one wanted"
        );
        // The two rules together are what the livelock needed: adopting keeps one worker, and
        // carrying the current serial over is what stops `compile_is_current` discarding it.
        let (asked, adopted) = (9u64, 9u64);
        assert!(
            compile_is_current(adopted, asked),
            "an adopted compile carries the newest serial, so it is not thrown away on arrival"
        );
    }

    /// **A band in flight is stopped only when its geometry has moved.** Two rounds of this cost
    /// the owner an afternoon: first the settled bake blocked the interface, then stopping it on
    /// every invalidation meant a scene touched every few seconds never sharpened at all. The
    /// resolution is deliberately not part of the comparison — a drag asks for draft bands while
    /// a settled one is being made, and that is not a reason to throw the settled one away.
    #[test]
    fn a_settled_bake_is_stopped_only_when_its_shape_has_moved() {
        let pending = vec![key(0xabc, 128)];

        // The same shape, asked for at the draft resolution mid-drag: keep going.
        let same: std::collections::HashSet<BakeKey> =
            [shape_of(&key(0xabc, DRAFT_BAKE_RESOLUTION))].into_iter().collect();
        assert!(!bake_is_obsolete(&pending, &same));

        // Edited: a different fingerprint is a different shape, so it is worth nothing now.
        let moved: std::collections::HashSet<BakeKey> =
            [shape_of(&key(0xdef, 128))].into_iter().collect();
        assert!(bake_is_obsolete(&pending, &moved));

        // Hidden, or otherwise not asked for at all: also nothing to finish.
        assert!(bake_is_obsolete(&pending, &std::collections::HashSet::new()));

        // And nothing pending is never obsolete, whatever is on screen.
        assert!(!bake_is_obsolete(&[], &std::collections::HashSet::new()));
    }

    /// **A change of display quality asks for a settled rebuild whenever it changes how the
    /// model is sampled, and only then.** The top two rungs both bake 192 cells and differ in how
    /// finely spines and profiles are walked; with the bake's resolution the only thing compared,
    /// moving between them asked for nothing, and the old walk stayed on screen until an edit
    /// happened to rebuild. Every pair of rungs both ways, so the ones that already rebuilt are
    /// held to it too; and the setting is told every frame, so staying put must ask for nothing.
    #[test]
    fn a_change_of_display_quality_asks_for_a_settled_rebuild_exactly_when_it_samples_the_model_differently()
    {
        use crate::DisplayQuality;
        for from in DisplayQuality::ALL {
            for to in DisplayQuality::ALL {
                let mut view = TracedView::default();
                view.set_fineness(from.fineness());
                view.stale = None;
                view.set_fineness(to.fineness());
                // Spelled out from the rungs rather than read off the key, so this says what a
                // rebuild reads and not merely that two keys compare the way they compare.
                let samples_differently = from.cells() != to.cells()
                    || from.spine_degrees() != to.spine_degrees()
                    || from.profile_degrees() != to.profile_degrees();
                assert_eq!(
                    view.stale,
                    samples_differently.then_some(Quality::Final),
                    "{} to {}",
                    from.label(),
                    to.label()
                );
            }
        }
        // And the spine on its own, which the app walks and the view never reads: the top two
        // rungs differ in profile turn as well, so a key that left the spine out would still
        // pass the ladder above.
        let full = DisplayQuality::Full.fineness();
        let mut view = TracedView::default();
        view.set_fineness(full);
        view.stale = None;
        view.set_fineness(Fineness { spine_degrees: full.spine_degrees / 2.0, ..full });
        assert_eq!(view.stale, Some(Quality::Final), "a finer spine over the same bake");
    }

    /// One slow draft drops a rung at once; climbing back takes a run of cheap ones, and a
    /// draft between the two thresholds resets that run — so the size cannot oscillate.
    #[test]
    fn the_draft_ladder_drops_at_once_and_climbs_only_after_a_run_of_cheap_frames() {
        assert_eq!(next_rung(0, 0, 80.0), (1, 0));
        assert_eq!(next_rung(1, 3, 80.0), (2, 0));
        // Never below the bottom rung.
        assert_eq!(next_rung(DRAFT_LADDER.len() - 1, 0, 500.0), (DRAFT_LADDER.len() - 1, 0));
        // Four cheap frames are a streak, not a climb; the fifth climbs.
        let mut state = (2, 0);
        for _ in 0..DRAFT_CLIMB_STREAK - 1 {
            state = next_rung(state.0, state.1, 5.0);
            assert_eq!(state.0, 2);
        }
        assert_eq!(next_rung(state.0, state.1, 5.0), (1, 0));
        // A frame in between resets the streak.
        assert_eq!(next_rung(1, 4, 30.0), (1, 0));
        // Cheap at the top rung stays at the top.
        assert_eq!(next_rung(0, 9, 5.0), (0, 10));
    }

    /// The tiers are what the plan's table says, and the bottom rung exists.
    #[test]
    fn the_ladder_runs_from_the_drag_size_down_to_a_rung_under_the_z840() {
        assert_eq!(DRAFT_LADDER, [720, 480, 360, 240]);
        assert_eq!(DRAFT_LADDER[0], INTERACTIVE_PIXELS);
    }

    /// **A spine is tessellated again only when its curve, its angle or its floor has moved, and
    /// every spine handed over is the one a fresh tessellation gives.** A rebuild for a number, a
    /// settle or a band arriving moves none of the three, and used to walk every curve anyway.
    #[test]
    fn a_spine_is_tessellated_again_only_when_its_curve_or_the_walk_asked_of_it_moves() {
        use fieldform_scene::Curve;
        let cubic = |id: CurveId, reach: f64| {
            Curve::new(
                id,
                "spine",
                3,
                vec![
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(reach, 40.0, 0.0),
                    Point3::new(80.0, -40.0, 10.0),
                    Point3::new(120.0, 0.0, 0.0),
                ],
                vec![1.0; 4],
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                false,
            )
            .expect("a cubic")
        };
        let (a, b) = (CurveId::new(), CurveId::new());
        let mut curves = std::collections::HashMap::new();
        curves.insert(a, cubic(a, 40.0));
        curves.insert(b, cubic(b, 10.0));
        let fresh = |curves: &std::collections::HashMap<CurveId, Curve>,
                     ids: &[CurveId],
                     degrees: f64,
                     floor: f64| {
            ids.iter()
                .map(|id| (*id, curves[id].tessellate_to_angle(degrees, floor)))
                .collect::<Vec<_>>()
        };

        let mut kept = KeptSpines::default();
        let mut step = |curves: &std::collections::HashMap<CurveId, Curve>,
                        ids: &[CurveId],
                        degrees: f64,
                        floor: f64,
                        made: usize,
                        what: &str| {
            let (spines, tessellated) = kept.spines(curves, ids.iter().copied(), degrees, floor);
            assert_eq!(tessellated, made, "{what}: how many were tessellated");
            assert_eq!(spines, fresh(curves, ids, degrees, floor), "{what}: the spines");
        };

        step(&curves, &[a, b], 1.0, 0.01, 2, "the first rebuild");
        step(&curves, &[a, b], 1.0, 0.01, 0, "a rebuild that moved no curve");
        curves.get_mut(&a).expect("a").set_control_point(1, Point3::new(45.0, 40.0, 0.0));
        step(&curves, &[a, b], 1.0, 0.01, 1, "one control point of one curve");
        curves.get_mut(&b).expect("b").set_weight(2, 2.0);
        step(&curves, &[a, b], 1.0, 0.01, 1, "one weight of the other");
        step(&curves, &[a, b], 0.5, 0.01, 2, "a finer angle");
        step(&curves, &[a, b], 0.5, 0.02, 2, "another floor");
        // A curve nothing reads any more is let go, so the kept set is only ever what was last
        // asked for -- and is tessellated afresh when something reads it again.
        step(&curves, &[b], 0.5, 0.02, 0, "one curve no longer read");
        step(&curves, &[a, b], 0.5, 0.02, 1, "and read again");
        // A curve the document no longer holds is left out, as a missing curve always was.
        curves.remove(&a);
        step(&curves, &[b], 0.5, 0.02, 0, "a curve gone from the document");
    }

    /// A spine with enough turn in it that walks at different angles are different walks.
    fn bent_spine(id: CurveId) -> fieldform_scene::Curve {
        fieldform_scene::Curve::new(
            id,
            "spine",
            3,
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 120.0, 0.0),
                Point3::new(120.0, 200.0, 0.0),
                Point3::new(200.0, 200.0, 0.0),
            ],
            vec![1.0; 4],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            false,
        )
        .expect("a cubic")
    }

    /// **A draft walks a spine at two degrees of turn on every rung, and the settle after it walks
    /// it at the rung's own** (loop spec P9). A draft is at most 720 px across, where a joint of
    /// one degree and one of two cannot be told apart; only a still can show the difference, and
    /// every frame of a gesture used to walk the spine as finely as the still it would settle
    /// into. Never finer than the rung asks, either: a rung already at two degrees walks there.
    #[test]
    fn a_draft_walks_a_spine_at_two_degrees_and_its_settle_at_the_rungs_own_turn() {
        use crate::DisplayQuality;
        let id = CurveId::new();
        let mut curves = std::collections::HashMap::new();
        curves.insert(id, bent_spine(id));
        let floor = 1e-4;
        let walked = |degrees: f64| vec![(id, curves[&id].tessellate_to_angle(degrees, floor))];
        // Points, for a message that can be read; the comparison is of the whole walk.
        let count = |spines: &Spines| spines[0].1.len();
        for rung in DisplayQuality::ALL {
            let still = rung.spine_degrees();
            if still < 2.0 {
                assert_ne!(walked(2.0), walked(still), "{}: two angles, one walk", rung.label());
            }
            let mut view = TracedView::default();
            view.set_fineness(rung.fineness());
            view.stale = None;
            view.invalidate(Quality::Draft);
            let draft = view.spines(&curves, [id], still, floor);
            let wanted = walked(still.max(2.0));
            assert!(
                draft == wanted,
                "{}: a draft walked {} points, where {}° walks {}",
                rung.label(),
                count(&draft),
                still.max(2.0),
                count(&wanted)
            );
            view.invalidate(Quality::Final);
            let settled = view.spines(&curves, [id], still, floor);
            let wanted = walked(still);
            assert!(
                settled == wanted,
                "{}: the settle after it walked {} points, where {still}° walks {}",
                rung.label(),
                count(&settled),
                count(&wanted)
            );
        }
    }

    /// **A gesture and its settle each keep what they walked, so going from one to the other walks
    /// no spine again until its curve moves.** The two walk at different angles, and one kept set
    /// would be walked afresh at each change of tier: every spine in the part twice a gesture,
    /// moved or not, which is the walking keeping them was for. A curve that moves is walked once
    /// by each.
    #[test]
    fn a_gesture_and_its_settle_walk_no_spine_again_until_its_curve_moves() {
        use crate::DisplayQuality;
        let (a, b) = (CurveId::new(), CurveId::new());
        let mut curves = std::collections::HashMap::new();
        curves.insert(a, bent_spine(a));
        curves.insert(b, bent_spine(b));
        let rung = DisplayQuality::Balanced;
        let mut view = TracedView::default();
        view.set_fineness(rung.fineness());
        let mut walk = |curves: &std::collections::HashMap<CurveId, fieldform_scene::Curve>,
                        quality: Quality,
                        made: usize,
                        what: &str| {
            view.stale = Some(quality);
            let (_, walked) = view.walk_spines(curves, [a, b], rung.spine_degrees(), 1e-4);
            assert_eq!(walked, made, "{what}: how many were walked");
        };
        walk(&curves, Quality::Draft, 2, "the first frame of a gesture");
        walk(&curves, Quality::Draft, 0, "its next frame");
        walk(&curves, Quality::Final, 2, "the settle");
        walk(&curves, Quality::Draft, 0, "the next gesture");
        walk(&curves, Quality::Final, 0, "and its settle");
        curves.get_mut(&a).expect("a").set_control_point(1, Point3::new(10.0, 120.0, 0.0));
        walk(&curves, Quality::Draft, 1, "a gesture that moves one curve");
        walk(&curves, Quality::Draft, 0, "a frame of it that moves nothing more");
        walk(&curves, Quality::Final, 1, "and its settle");
    }

    /// The scene a rebuild is handed, with nothing in it but what generating reads.
    fn scene_of<'a>(
        document: &'a fieldform_scene::Document,
        graph: &'a Graph,
        morph_divisors: &'a [(fieldform_scene::NodeId, f64)],
    ) -> Scene<'a> {
        Scene {
            document,
            graph,
            morph_divisors,
            tolerance_mm: 0.1,
            reads_mm: 1.0,
            profile_fineness: 1.0,
            profile_degrees: 1.0,
            curves: Vec::new(),
            meshes: Vec::new(),
            cage_bounds: (Point3::origin(), Point3::origin()),
        }
    }

    /// **A rebuild nothing has invalidated keeps the program generated last, and that program is
    /// the one a fresh generate makes, to the byte and the slot; after any invalidation the
    /// program is generated afresh.** A settle and a band arriving are rebuilds nothing
    /// invalidated, and each generated the shader again.
    ///
    /// **A value takes the full path too, and this shows why rather than asserting it.** The
    /// source and which slots are named both stand when a named value moves, but slots worked
    /// out from the value at generation -- here the two bodies' cull extents -- move with it, so
    /// a program kept across the edit would draw the old extents, which is missing material.
    #[test]
    fn a_rebuild_nothing_invalidated_keeps_the_program_a_fresh_generate_would_make() {
        use fieldform_scene::{Document, Node, ParamRef, ParamValue, Place, Turn};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let r = doc.add_parameter("r", 20.0);
        let mut graph = Graph::empty();
        graph.add(Node::Sphere {
            centre: Place::At([lit(-40.0), lit(0.0), lit(0.0)]),
            radius: ParamRef::Named(r),
        });
        let other = graph.add(Node::Sphere {
            centre: Place::At([lit(40.0), lit(0.0), lit(0.0)]),
            radius: lit(15.0),
        });
        let fresh = |doc: &Document, graph: &Graph, divisors: &[(fieldform_scene::NodeId, f64)]| {
            fieldform_gpu::generate_with(doc, graph, divisors).expect("generates")
        };
        let mut view = TracedView::default();

        let (first, generated) = view.program_for(&scene_of(&doc, &graph, &[])).expect("generates");
        assert!(generated, "the first rebuild has nothing to keep");
        assert_eq!(first, fresh(&doc, &graph, &[]));

        let (kept, generated) = view.program_for(&scene_of(&doc, &graph, &[])).expect("kept");
        assert!(!generated, "a rebuild nothing invalidated generates nothing");
        assert_eq!(kept, fresh(&doc, &graph, &[]), "and keeps what a fresh generate makes");

        // A named value. Its own slot is read every frame; the cull extents are not.
        doc.parameters.get_mut(&r).expect("r").value = ParamValue::Literal(30.0);
        view.invalidate(Quality::Draft);
        let (named, generated) = view.program_for(&scene_of(&doc, &graph, &[])).expect("ok");
        assert!(generated, "a named value invalidates, and generates afresh");
        assert_eq!(named, fresh(&doc, &graph, &[]));
        assert_eq!(named.source, kept.source, "the source stands across a named value");
        assert_ne!(named.params, kept.params, "but slots worked out from it have moved");

        // An unnamed number, which is its own slot's value as well as the extents'.
        if let Some(Node::Sphere { radius, .. }) = graph.node_mut(other) {
            *radius = lit(16.0);
        }
        view.invalidate(Quality::Draft);
        let (unnamed, generated) = view.program_for(&scene_of(&doc, &graph, &[])).expect("ok");
        assert!(generated, "an unnamed number invalidates, and generates afresh");
        assert_eq!(unnamed, fresh(&doc, &graph, &[]));
        assert_eq!(unnamed.source, named.source, "the source stands across an unnamed number");
        assert_ne!(unnamed.params, named.params);

        // Structure: a body added, then a body of another kind.
        let third = graph.add(Node::Sphere {
            centre: Place::At([lit(0.0), lit(60.0), lit(0.0)]),
            radius: lit(10.0),
        });
        view.invalidate(Quality::Final);
        let (added, generated) = view.program_for(&scene_of(&doc, &graph, &[])).expect("ok");
        assert!(generated, "a body added");
        assert_eq!(added, fresh(&doc, &graph, &[]));
        assert_ne!(added.source, unnamed.source);
        if let Some(node) = graph.node_mut(third) {
            *node = Node::Cuboid { edge: Default::default(),
                centre: Place::At([lit(0.0), lit(60.0), lit(0.0)]),
                size: [lit(10.0), lit(10.0), lit(10.0)],
                turn: Turn::default(),
            };
        }
        view.invalidate(Quality::Final);
        let (changed, generated) = view.program_for(&scene_of(&doc, &graph, &[])).expect("ok");
        assert!(generated, "a body of another kind");
        assert_eq!(changed, fresh(&doc, &graph, &[]));
        assert_ne!(changed.source, added.source);

        // What a rebuild is handed besides the document is held to the kept program's too, so
        // a graph or divisors that moved with nothing said still generate afresh.
        let mut moved = graph.clone();
        if let Some(Node::Sphere { radius, .. }) = moved.node_mut(other) {
            *radius = lit(17.0);
        }
        let (by_graph, generated) = view.program_for(&scene_of(&doc, &moved, &[])).expect("ok");
        assert!(generated, "a graph that is not the one kept");
        assert_eq!(by_graph, fresh(&doc, &moved, &[]));
        let divisors = [(other, 2.0)];
        let (by_divisors, generated) =
            view.program_for(&scene_of(&doc, &moved, &divisors)).expect("ok");
        assert!(generated, "divisors that are not the ones kept");
        assert_eq!(by_divisors, fresh(&doc, &moved, &divisors));
        let (_, generated) = view.program_for(&scene_of(&doc, &moved, &divisors)).expect("ok");
        assert!(!generated, "and kept again once nothing has moved");

        // A graph that cannot be generated is refused, and the program kept is still the one
        // its own graph makes.
        assert!(view.program_for(&scene_of(&doc, &Graph::empty(), &[])).is_err());
        let (after, generated) = view.program_for(&scene_of(&doc, &moved, &divisors)).expect("ok");
        assert!(!generated, "a refusal of another graph moves nothing kept");
        assert_eq!(after, fresh(&doc, &moved, &divisors));
    }

    /// A frame's view with every field set to something a test can move away from.
    fn a_view<'a>(lookup: &'a dyn Fn(ParamId) -> Option<f64>) -> View<'a> {
        View {
            eye: Point3::new(300.0, -200.0, 150.0),
            target: Point3::new(0.0, 0.0, 10.0),
            up: Vector3::z(),
            scene: (Point3::new(0.0, 0.0, 10.0), 80.0),
            fov_deg: 45.0,
            orthographic: false,
            section: None,
            lookup,
            selected: None,
            material: Material::default(),
            occlusion: true,
            exact_finish: false,
            environment: EnvironmentChoice::Studio,
            draft: None,
            max_pixels: 0,
            interactive_pixels: INTERACTIVE_PIXELS,
            still_frames: STILL_FRAMES,
        }
    }

    /// **Every field of a view, the frame's size and every value the shader reads reach the key,
    /// so a change to any one of them starts the still over.** A field left out is a stale image
    /// that stays on screen: a slider drag that did not restart the average would blend eight
    /// frames of eight different radii.
    ///
    /// Built from views rather than from keys, and [`View`] is named whole beside the pairs, so a
    /// field added to it does not compile here until it is given a pair or set aside by name.
    /// [`ViewKey::of`] takes a view apart whole for the same reason, so the field cannot be added
    /// without a decision about the key either. The lookup is the one set aside in both: it
    /// reaches the key as the values it resolves, which have lines of their own.
    #[test]
    fn a_change_to_any_one_thing_the_image_depends_on_changes_the_key() {
        let none = |_: ParamId| -> Option<f64> { None };
        let values = || vec![2.0, 30.0, 0.5];
        let key = |view: &View<'_>| ViewKey::of(view, values(), 1280, 720);
        let base = || a_view(&none);
        let framed = |middle_y, reach| View {
            scene: (Point3::new(0.0, middle_y, 10.0), reach),
            ..base()
        };
        let cut = |normal, offset| View { section: Some((normal, offset)), ..base() };
        let picked = |body| View { selected: Some(body), ..base() };
        let finish = |roughness| View {
            material: Material { roughness, ..Material::default() },
            ..base()
        };
        let pulled = |angle| View { draft: Some(Draft::new(Vector3::z(), angle, 0.5)), ..base() };
        assert!(key(&base()) == key(&base()), "the same frame twice is one key");

        // Every field, by name: one added to `View` stops this compiling until it is named here,
        // which is where to give it a pair below. Struct update fills it in silently everywhere
        // else in this test.
        let View {
            eye: _,
            target: _,
            up: _,
            scene: _,
            fov_deg: _,
            orthographic: _,
            section: _,
            lookup: _,
            selected: _,
            material: _,
            occlusion: _,
            exact_finish: _,
            environment: _,
            draft: _,
            max_pixels: _,
            interactive_pixels: _,
            still_frames: _,
        } = base();
        let (x, flipped) = (Vector3::x(), -Vector3::x());
        let pairs: Vec<(&str, View<'_>, View<'_>)> = vec![
            ("the eye", base(), View { eye: Point3::new(300.0, -200.0, 151.0), ..base() }),
            ("the target", base(), View { target: Point3::new(0.0, 0.0, 11.0), ..base() }),
            ("up", base(), View { up: Vector3::y(), ..base() }),
            ("the scene's middle", framed(0.0, 80.0), framed(1.0, 80.0)),
            ("the scene's reach", framed(0.0, 80.0), framed(0.0, 81.0)),
            ("the field of view", base(), View { fov_deg: 30.0, ..base() }),
            ("orthographic", base(), View { orthographic: true, ..base() }),
            ("a section", base(), cut(x, 0.0)),
            ("where the section cuts", cut(x, 0.0), cut(x, 2.0)),
            ("which way the section faces", cut(x, 0.0), cut(flipped, 0.0)),
            ("a body selected", base(), picked(0)),
            ("which body is selected", picked(0), picked(1)),
            ("the material", finish(0.65), finish(0.3)),
            ("occlusion", base(), View { occlusion: false, ..base() }),
            ("the environment", base(), View { environment: EnvironmentChoice::Zebra, ..base() }),
            ("a draft read", base(), pulled(1.0)),
            ("the draft angle", pulled(1.0), pulled(2.0)),
            ("the still's size", base(), View { max_pixels: 1440, ..base() }),
            ("the drag frame's size", base(), View { interactive_pixels: 480, ..base() }),
            ("the still's frame count", base(), View { still_frames: 4, ..base() }),
        ];
        for (what, before, after) in &pairs {
            assert!(key(before) == key(before), "{what}: the same frame twice is one key");
            assert!(key(before) != key(after), "{what}");
        }

        let base = base();
        let other = |values: Vec<f32>, width, height| ViewKey::of(&base, values, width, height);
        assert!(other(vec![2.0, 31.0, 0.5], 1280, 720) != key(&base), "a value");
        assert!(other(vec![2.0, 30.0], 1280, 720) != key(&base), "a value fewer");
        assert!(other(vec![2.0, 30.0, 0.5, 0.0], 1280, 720) != key(&base), "a value more");
        assert!(other(values(), 1281, 720) != key(&base), "the width");
        assert!(other(values(), 1280, 721) != key(&base), "the height");
    }

    /// The system allocator, with every allocation counted on the thread that made it.
    ///
    /// **For the whole of this binary**, since a binary has one allocator and the application is
    /// one binary: its other tests pay a thread-local add per allocation and nothing else. The
    /// count is read per thread, so tests running side by side cannot charge one with another's.
    struct Counted;

    thread_local! {
        static MADE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// One more allocation on this thread.
    fn count_one() {
        // `try_with`, because a thread being torn down can still allocate on its way out.
        let _ = MADE.try_with(|made| made.set(made.get() + 1));
    }

    unsafe impl std::alloc::GlobalAlloc for Counted {
        unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
            count_one();
            unsafe { std::alloc::GlobalAlloc::alloc(&std::alloc::System, layout) }
        }

        // These two are forwarded rather than left to the trait's own, which build both from
        // `alloc`: every growth would be a copy where the system can grow a block in place, and
        // every zeroed block written where the system can hand one back zeroed. That is a cost
        // on every other test in the binary, which the add alone is not. Each still counts once,
        // as the trait's own did, since a buffer that grows is a buffer the frame had no room in.
        unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
            count_one();
            unsafe { std::alloc::GlobalAlloc::alloc_zeroed(&std::alloc::System, layout) }
        }

        unsafe fn realloc(
            &self,
            ptr: *mut u8,
            layout: std::alloc::Layout,
            new_size: usize,
        ) -> *mut u8 {
            count_one();
            unsafe { std::alloc::GlobalAlloc::realloc(&std::alloc::System, ptr, layout, new_size) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
            unsafe { std::alloc::GlobalAlloc::dealloc(&std::alloc::System, ptr, layout) }
        }
    }

    #[global_allocator]
    static COUNTED: Counted = Counted;

    /// How many allocations `work` made on this thread.
    fn allocations_during(work: impl FnOnce()) -> usize {
        let before = MADE.with(std::cell::Cell::get);
        work();
        MADE.with(std::cell::Cell::get) - before
    }

    /// **Keying a frame allocates nothing once its two buffers exist, whether the frame moved or
    /// not.** The values were collected afresh every frame the viewport drew, so every frame made
    /// one buffer and dropped another: the key built for it and the key it replaced, or failed to
    /// replace, each held its own (performance plan §0 G).
    ///
    /// The material is cloned into the key every frame, and that is counted here too: it is plain
    /// numbers, so its clone is a copy, and one that came to allocate would fail this.
    #[test]
    fn a_frame_is_keyed_without_allocating_whether_or_not_anything_moved() {
        use fieldform_scene::ParamRef;
        let (named, unknown) = (ParamId::new(), ParamId::new());
        let radius = std::cell::Cell::new(20.0);
        let lookup = |id: ParamId| (id == named).then(|| radius.get());
        let params = [ParamRef::Literal(2.0), ParamRef::Named(named), ParamRef::Named(unknown)];
        let mut last = None;
        let mut spare = Vec::new();
        let mut frame = |view: &View<'_>| {
            let values = resolve_values(&mut spare, &params, view.lookup);
            note_view(&mut last, &mut spare, ViewKey::of(view, values, 1280, 720))
        };
        let still = a_view(&lookup);
        let turned = View { eye: Point3::new(300.0, -190.0, 150.0), ..a_view(&lookup) };
        let polished = View {
            material: Material { roughness: 0.02, ..Material::default() },
            ..a_view(&lookup)
        };
        // The first two frames give the two buffers their room.
        assert!(frame(&still), "the first frame");
        assert!(frame(&turned), "the second frame");

        let mut step = |what: &str, view: &View<'_>, moved: bool| {
            let mut said = None;
            let made = allocations_during(|| said = Some(frame(view)));
            assert_eq!(said, Some(moved), "{what}: whether the frame moved");
            assert_eq!(made, 0, "{what}: allocations");
        };
        step("a frame that moved nothing", &turned, false);
        step("the camera turned back", &still, true);
        step("nothing moved again", &still, false);
        radius.set(21.0);
        step("a named value moved", &still, true);
        step("nothing moved after the value", &still, false);
        step("the material", &polished, true);
        step("nothing moved after the material", &polished, false);
    }

    /// The rule [`note_view`] replaced, kept as the control arm of the timing below: the values
    /// collected afresh into a key of their own, which is kept only when it differs from the last.
    fn keyed_afresh(
        last: &mut Option<ViewKey>,
        params: &[fieldform_scene::ParamRef],
        view: &View<'_>,
    ) -> bool {
        let values = params.iter().map(|p| p.resolve(view.lookup).unwrap_or(0.0) as f32).collect();
        let key = ViewKey::of(view, values, 1280, 720);
        let moved = last.as_ref() != Some(&key);
        if moved {
            *last = Some(key);
        }
        moved
    }

    /// Nanoseconds a frame, and allocations in all, over `frames` frames of `frame`, which is
    /// handed each frame's number.
    fn per_frame(frames: u32, mut frame: impl FnMut(u32) -> bool) -> (f64, usize) {
        // Two frames first, as the viewport's first two make the new rule's two buffers.
        frame(0);
        frame(1);
        let mut took = std::time::Duration::ZERO;
        let made = allocations_during(|| {
            let began = std::time::Instant::now();
            for f in 0..frames {
                std::hint::black_box(frame(f));
            }
            took = began.elapsed();
        });
        (took.as_nanos() as f64 / f64::from(frames), made)
    }

    /// **What keying a frame costs, against the rule it replaced** (performance plan §0 G): over
    /// 0, 8, 64 and 512 slots, with the view still or turning every frame, the two rules take
    /// turns three times each in one process and print nanoseconds a frame. The slots alternate
    /// named and literal, and a named one is read from a table as the viewport's lookup reads it.
    /// No slots is the control: neither rule has a buffer to make, so what separates the arms
    /// there is how each is compiled, and that is the floor under every other line's difference.
    ///
    /// A timing to read rather than a property, so ignored. Read it in release, where the
    /// application is built as it ships: `cargo test -p fieldform-app --release --bin fieldform
    /// -- --ignored --nocapture keying_a_frame`. What it does assert is that each arm is the rule
    /// it is named for, by the allocations each line ends with: one a frame for the old wherever
    /// there is a slot, none for the new. The binary's counting allocator charges the old arm a
    /// thread-local add on each of those, which a shipped build does not pay.
    #[test]
    #[ignore = "a timing to read rather than a property: run in release with --ignored --nocapture"]
    fn keying_a_frame_timed_against_the_rule_it_replaced() {
        use fieldform_scene::ParamRef;
        const FRAMES: u32 = 200_000;
        let ids: Vec<ParamId> = (0..512).map(|_| ParamId::new()).collect();
        let table: std::collections::HashMap<ParamId, f64> =
            ids.iter().enumerate().map(|(i, &id)| (id, i as f64)).collect();
        let lookup = |id: ParamId| table.get(&id).copied();
        let still = a_view(&lookup);
        let turned = View { eye: Point3::new(300.0, -190.0, 150.0), ..a_view(&lookup) };
        let rounds = |times: &[(f64, usize)]| {
            times.iter().map(|(ns, _)| format!("{ns:.1}")).collect::<Vec<_>>().join(" / ")
        };
        for slots in [0, 8, 64, 512] {
            let params: Vec<ParamRef> = ids[..slots]
                .iter()
                .enumerate()
                .map(|(i, &id)| match i % 2 {
                    0 => ParamRef::Named(id),
                    _ => ParamRef::Literal(i as f64),
                })
                .collect();
            for (motion, turning) in [("still", false), ("turning", true)] {
                let view = |f: u32| if turning && f % 2 == 1 { &turned } else { &still };
                let (mut old, mut new) = (Vec::new(), Vec::new());
                for _ in 0..3 {
                    let mut last = None;
                    old.push(per_frame(FRAMES, |f| keyed_afresh(&mut last, &params, view(f))));
                    let (mut last, mut spare) = (None, Vec::new());
                    new.push(per_frame(FRAMES, |f| {
                        let values = resolve_values(&mut spare, &params, view(f).lookup);
                        note_view(&mut last, &mut spare, ViewKey::of(view(f), values, 1280, 720))
                    }));
                }
                let old_made = if slots == 0 { 0 } else { 1 };
                for (&(_, made_old), &(_, made_new)) in old.iter().zip(&new) {
                    assert_eq!(made_old, old_made * FRAMES as usize, "{slots} {motion}: the old");
                    assert_eq!(made_new, 0, "{slots} {motion}: the new");
                }
                println!(
                    "keying a frame | {slots:>3} slots | {motion:<7} | old {} ns | new {} ns \
                     | allocations a frame, old / new: {old_made} / 0",
                    rounds(&old),
                    rounds(&new),
                );
            }
        }
    }

    /// **What a frame must do about a device it cannot draw with** (loop spec S4). The decision
    /// is the one part of surviving a lost device that can be held to a property here: a real
    /// loss is a driver reset or a suspend (backlog 35), and neither can be provoked.
    ///
    /// Exhaustive over the six states, because the one that was wrong is the one nobody would
    /// write a single case for: a device reported lost with everything still held, where the
    /// frame carried on and asked it for a march.
    #[test]
    fn a_frame_lets_go_of_what_a_device_it_cannot_draw_with_made_and_asks_a_lost_one_for_nothing() {
        for (lost, made_on, must) in [
            (false, MadeOn::Nothing, MustDo::CarryOn),
            (false, MadeOn::ThisDevice, MustDo::CarryOn),
            (false, MadeOn::AnotherDevice, MustDo::LetGo),
            (true, MadeOn::Nothing, MustDo::AskNothing),
            (true, MadeOn::ThisDevice, MustDo::LetGo),
            (true, MadeOn::AnotherDevice, MustDo::LetGo),
        ] {
            assert_eq!(
                what_a_frame_must_do(lost, made_on),
                must,
                "lost {lost}, held by {made_on:?}",
            );
        }
        // One pass of the rule a frame is enough, because letting go lands in a state the rule
        // already answers: with nothing held, a live device builds afresh and a lost one is
        // asked for nothing at all.
        assert_eq!(what_a_frame_must_do(false, MadeOn::Nothing), MustDo::CarryOn);
        assert_eq!(what_a_frame_must_do(true, MadeOn::Nothing), MustDo::AskNothing);
    }

    /// One device and a queue, as a frame is handed them.
    struct Gpu {
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    /// What asking this machine for `how_many` devices on one adapter found.
    ///
    /// **Devices on one adapter rather than one from each of two**, because what makes a resource
    /// unusable is the device handle that made it and not which card it was: a second device on
    /// the one card is as foreign to the first as a second card would be, and it is what a
    /// recreated device after a loss would most likely be.
    ///
    /// **Four outcomes rather than an `Option`, because they do not mean the same thing**, and
    /// one of them is a fault. See [`devices_or_skip`].
    enum Devices {
        /// As many as were asked for, and the adapter that gave them.
        All(Vec<Gpu>, wgpu::Adapter),
        /// No adapter at all.
        NoAdapter,
        /// An adapter that would not give even one device on the application's own limits and
        /// features.
        NoDevice(wgpu::RequestDeviceError),
        /// An adapter that gave this many and refused the next.
        Fewer(usize, wgpu::RequestDeviceError),
    }

    fn devices(how_many: usize) -> Devices {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let Ok(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
            else {
                return Devices::NoAdapter;
            };
            let mut gpus = Vec::new();
            for _ in 0..how_many {
                // The application's own limits and features, so what builds here is what builds
                // there.
                let descriptor = wgpu::DeviceDescriptor {
                    required_limits: fieldform_gpu::required_limits(&adapter),
                    required_features: fieldform_gpu::timing_features(&adapter),
                    ..Default::default()
                };
                match adapter.request_device(&descriptor).await {
                    Ok((device, queue)) => gpus.push(Gpu { device, queue }),
                    Err(e) if gpus.is_empty() => return Devices::NoDevice(e),
                    Err(e) => return Devices::Fewer(gpus.len(), e),
                }
            }
            Devices::All(gpus, adapter)
        })
    }

    /// `how_many` devices, or `None` where this machine cannot give them for a reason that says
    /// nothing about the code under test — said out loud when it happens, naming `unexercised` as
    /// what the run therefore did not try.
    ///
    /// **Only a machine that cannot give them skips, and a refusal of the application's own
    /// limits and features is not that**, which is the rule
    /// `the_shaded_pass_survives_every_lighting_path` states and the reason it allows only
    /// `NoAdapter`: an adapter that will not give one device on the descriptor the application
    /// uses is the application failing to start, which is the class of fault a test exists to
    /// catch. A *later* device refused by an adapter that gave the first is this machine's own
    /// limit, and only the test that needs two ever meets it.
    fn devices_or_skip(how_many: usize, unexercised: &str) -> Option<(Vec<Gpu>, wgpu::Adapter)> {
        use std::io::Write;

        // Straight to the stream rather than through `eprintln!`, which the harness captures and
        // shows only for a failure: a skip that passes silently reads as a pass. Measured on a
        // scratch crate with one passing test -- neither `eprintln!` nor `println!` reached the
        // terminal, and a `writeln!` to the stream beside them did.
        let skipped = |why: String| {
            let _ = writeln!(std::io::stderr(), "SKIPPED: {why}, so {unexercised} was not tried");
            None
        };
        match devices(how_many) {
            Devices::All(gpus, adapter) => Some((gpus, adapter)),
            Devices::NoAdapter => skipped("no GPU adapter on this machine".to_string()),
            Devices::Fewer(given, e) => skipped(format!(
                "this adapter gave {given} of the {how_many} devices asked of it: {e}"
            )),
            Devices::NoDevice(e) => panic!(
                "the adapter refused the limits and features the application asks of it, which \
                 is the application failing to start rather than a machine without a GPU: {e}"
            ),
        }
    }

    /// An eframe render state on `gpu`, as the application is handed one.
    ///
    /// Built here because letting go of a device frees egui's own texture handles through it,
    /// and nothing in these tests has a window to get one from.
    fn render_state_on(gpu: &Gpu, adapter: &wgpu::Adapter) -> egui_wgpu::RenderState {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        egui_wgpu::RenderState {
            adapter: adapter.clone(),
            available_adapters: Vec::new(),
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
            target_format: format,
            renderer: std::sync::Arc::new(egui::mutex::RwLock::new(egui_wgpu::Renderer::new(
                &gpu.device,
                format,
                egui_wgpu::RendererOptions::default(),
            ))),
        }
    }

    /// **Everything the view holds on the GPU goes with the device that made it, and a device
    /// it has built nothing on is built on afresh** (loop spec S4).
    ///
    /// The lighting pass is what this is really about: `Renderer::device` answered for the
    /// march's pipeline alone, so a frame on another device kept the pass, the handles egui
    /// samples its textures through and the still averaged on the device before — which is a
    /// stale picture at best and a texture of the wrong device handed to the painter at worst.
    ///
    /// Skipped aloud where the machine cannot give two devices, and nowhere else
    /// ([`devices_or_skip`]). What a real loss does to the first device cannot be tested here;
    /// what a second device must find in hand can.
    #[test]
    fn every_gpu_resource_the_view_holds_goes_with_its_device_and_the_next_one_builds_afresh() {
        let Some((gpus, adapter)) =
            devices_or_skip(2, "the view letting go of what a device that has gone made")
        else {
            return;
        };
        let (first, second) = (&gpus[0], &gpus[1]);
        let maps = EnvironmentChoice::Studio.bake();
        let pass_on = |gpu: &Gpu| {
            ShadingResources::new(&gpu.device, &gpu.queue, &maps)
                .expect("the lighting pass builds on a device the adapter has just given")
        };
        let doc = fieldform_scene::Document::with_cage(Cage::cube_of(10.0));
        let cage = *doc.cages.keys().next().expect("with_cage inserts one");

        // The lighting pass through the same constructor a frame calls, and the rest of what a
        // frame leaves behind it: a handle egui would sample, the key of the picture on screen,
        // a filled skip grid, a still part way through its average, and the band sizes measured
        // of this card. Plus the CPU side, which a device has no part in.
        let mut view = TracedView {
            shading: Some(pass_on(first)),
            built_on: Some(first.device.clone()),
            display: vec![(7, egui::TextureId::User(7))],
            last_view: Some(ViewKey::of(&a_view(&|_| None), vec![1.0], 1280, 720)),
            skip_key: Some((1, vec![1.0], [0; 4], SKIP_CELLS)),
            accumulated: 3,
            still_shown: true,
            still_ms_per_row: Some((1280, 720, 4.0)),
            draft_ms_per_pixel: Some(1e-4),
            environments: std::collections::HashMap::from([(
                EnvironmentChoice::Studio,
                maps.clone(),
            )]),
            bakes: std::collections::HashMap::from([(
                BakeKey {
                    leaf: BakedLeaf::Cage(cage),
                    resolution: 32,
                    box_bits: [0; 6],
                    band_bits: 0,
                    geometry: 0,
                    levels: 4,
                },
                std::sync::Arc::new(fieldform_gpu::BakedField {
                    dims: [1, 1, 1],
                    lo: Point3::origin(),
                    cell: Vector3::repeat(1.0),
                    band: 1.0,
                    data: vec![0.0],
                    normals: vec![[0.0, 0.0, 1.0]],
                }),
            )]),
            ..Default::default()
        };

        // A device the view has built nothing on. Everything it holds is of the other one.
        let state = render_state_on(second, &adapter);
        assert!(view.ready_to_draw_on(&state, false), "a device in hand may be drawn on");
        assert!(view.shading.is_none(), "the lighting pass belongs to the device that made it");
        assert!(view.built_on.is_none(), "and nothing is held of it");
        assert!(view.display.is_empty(), "egui's handle is for a texture that has gone");
        assert!(view.last_view.is_none(), "the key of a picture that has gone");
        assert!(view.skip_key.is_none(), "the grid went with the renderer that held it");
        assert_eq!(view.accumulated, 0, "a still averaged on another device is not this one's");
        assert!(!view.still_shown, "and none of it may be shown");
        assert!(
            view.still_ms_per_row.is_none() && view.draft_ms_per_pixel.is_none(),
            "a band sized from a measurement of the device a submission reset is the fault S2 \
             exists to prevent",
        );
        assert_eq!(view.stale, Some(Quality::Final), "a rebuild is due, at the settled quality");
        // Kept, and that is what makes the rebuild a rebuild: a bake is a pure function of one
        // leaf's geometry and an environment of its own settings, so neither is of a device.
        assert_eq!(view.environments.len(), 1, "a baked environment is CPU data");
        assert_eq!(view.bakes.len(), 1, "and so is a baked band");

        // Built afresh on the device in hand, through the constructor a frame calls: carried on
        // with rather than let go of again.
        view.shading = Some(pass_on(second));
        view.built_on = Some(second.device.clone());
        assert!(view.ready_to_draw_on(&state, false), "the device that made it is in hand");
        assert!(view.shading.is_some(), "so what it made is drawn with");

        // And a device reported lost is asked for nothing, whatever is held.
        assert!(!view.ready_to_draw_on(&state, true), "nothing is asked of a lost device");
        assert!(view.shading.is_none() && view.built_on.is_none(), "which takes what it made");
        assert!(view.error().is_some(), "and the reason the viewport is empty is there to say");
        assert!(!view.ready_to_draw_on(&state, true), "and it stays that way while it is lost");
        assert!(view.error().is_some(), "with the reason still said, frame after frame");
    }

    /// **A loss arriving before the view has built anything still says why the viewport is
    /// empty** (loop spec S4, which asks for the reason in the console).
    ///
    /// The state nobody would write a case for, and the one where the frame has nothing to let
    /// go of: the device goes during the first shader compile, on a worker, so the view holds
    /// no resource of it and `MadeOn::Nothing` is what a frame reads. An empty viewport is
    /// indistinguishable from an empty document, and this is the only state in which the view
    /// used to leave it unexplained — `App::keep_the_document_if_the_gpu_was_lost` named the
    /// loss even then, so the line that was missing is the view's own.
    #[test]
    fn a_loss_before_the_view_has_built_anything_still_says_why_the_viewport_is_empty() {
        // One device, since what stands in for the loss here is the flag rather than another
        // device.
        let Some((gpus, adapter)) =
            devices_or_skip(1, "a device lost before the view had built anything")
        else {
            return;
        };
        let state = render_state_on(&gpus[0], &adapter);
        let mut view = TracedView::default();
        assert!(view.built_on.is_none(), "nothing has been built on any device");
        assert!(!view.ready_to_draw_on(&state, true), "so a lost device is asked for nothing");
        assert!(view.error().is_some(), "and the viewport says why it is empty");
    
    }

    /// **A rebuild with the same shader keeps what the ladder learned; a different one starts
    /// over** (item 105).
    ///
    /// Before this, every rebuild sent the draft ladder to its top rung. Measured over eight view
    /// changes on 48 gyroid-filled shells, that left all sixteen drafts at 517 px: each view
    /// change rebuilt, each rebuild was a 0.5 ms no-op that adopted the pipeline it already had,
    /// and each one forgot the rung the last draft had earned.
    #[test]
    fn a_rebuild_with_the_same_shader_keeps_what_the_ladder_learned() {
        let learned = |source: &str| LearnedCosts {
            source: source_identity(source),
            draft_rung: DRAFT_LADDER.len() - 1,
            draft_fast_streak: 2,
            draft_ms_per_pixel: Some(55.0 / (240.0 * 240.0)),
            draft_ms_per_row: Some((240, 240, 0.9)),
            last_draft_band_rows: Some(137),
            still_ms_per_row: Some((517, 424, 3.1)),
        };
        let a = "fn march() { /* the shape on screen */ }";
        let b = "fn march() { /* a different shape */ }";
        // The same source: every number comes back as it was, the floor rung included.
        assert_eq!(costs_to_keep(Some(learned(a)), a), Some(learned(a)));
        // A different source costs differently, so nothing carries over — the reset's own rule.
        assert_eq!(costs_to_keep(Some(learned(a)), b), None);
        // Nothing learned yet: nothing to keep.
        assert_eq!(costs_to_keep(None, a), None);
        // One character is a different program.
        assert_eq!(costs_to_keep(Some(learned(a)), &format!("{a} ")), None);
    }
