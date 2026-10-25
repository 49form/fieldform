// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The application's tests: what its tools, console, picking and view do, without a window.

    /// **Ending the curve tool never says nothing was drawn once something was** (item 111).
    ///
    /// The tool stays on between curves now, so the old message — every exit said "nothing was
    /// drawn" — became false at exactly the moment someone reads it to check their four lines
    /// survived. Every combination of a curve half-placed and curves already kept, through the one
    /// function all three exits (Escape, another tool, a new placement) say.
    #[test]
    fn ending_the_curve_tool_never_denies_a_curve_it_placed() {
        use super::{curve_tool_ended, PendingCurve};
        let pending = |placed: u32, mid: bool| PendingCurve {
            degree: 1,
            first: mid.then(nalgebra::Point3::origin),
            preview: None,
            axis: None,
            restanding: None,
            placed,
        };
        for placed in 0..4u32 {
            for mid in [false, true] {
                let said = curve_tool_ended(&pending(placed, mid), "Curve tool off");
                assert!(said.starts_with("Curve tool off"), "{said}");
                if placed > 0 {
                    assert!(!said.contains("nothing was drawn"), "placed {placed}: {said}");
                    assert!(said.contains("kept"), "placed {placed}: {said}");
                } else {
                    assert!(said.contains("nothing was drawn"), "{said}");
                }
                // A half-placed curve is said to be lost only when there is one.
                assert_eq!(said.contains("in progress"), mid && placed > 0, "{said}");
            }
        }
        assert!(curve_tool_ended(&pending(3, false), "x").contains("the 3 curves placed"));
        assert!(curve_tool_ended(&pending(1, false), "x").contains("the curve placed"));
    }

    /// **A session offers other sessions' autosaves and never takes its own for one** (2026-10-15).
    ///
    /// The failure this holds: a killed instance left its autosave, and the next instance — a
    /// scripted run — exited cleanly and removed it, because one name served every session. Now a
    /// session's own file is excluded from what it offers, every other session's is offered, newest
    /// first, and the older builds' single shared name still counts.
    #[test]
    fn a_session_offers_other_sessions_autosaves_and_never_its_own() {
        use super::other_autosaves;
        use std::time::{Duration, SystemTime};
        let t = |minutes_ago: u64| SystemTime::now() - Duration::from_secs(minutes_ago * 60);
        let ext = super::DOCUMENT_EXTENSION;
        let name = |middle: &str| format!("fieldform-autosave{middle}.{ext}");
        let listing = vec![
            (name("-100"), t(30)),
            (name("-200"), t(5)),
            (name(""), t(90)),
            (name("-300"), t(1)),
            (format!("fieldform-autosave backup.{ext}"), t(2)),
            (format!("fieldform-autosave-12x.{ext}"), t(2)),
            ("fieldform-console.log".to_string(), t(0)),
        ];
        let own = name("-300");
        let offered: Vec<String> =
            other_autosaves(&listing, &own).into_iter().map(|(n, _)| n).collect();
        assert_eq!(
            offered,
            vec![name("-200"), name("-100"), name("")],
            "every other session's, newest first, the older shared name included — never its own",
        );
        assert!(!offered.contains(&own), "a session never offers, so never deletes, its own as foreign");
        assert!(other_autosaves(&[(own.clone(), t(0))], &own).is_empty());
    }

    /// **A recovery offer is worded in the roundest words that are still true**, since what the
    /// reader wants is whether it is their work, not a duration to parse.
    #[test]
    fn how_long_ago_is_said_in_round_words() {
        use super::how_long_ago;
        use std::time::Duration;
        assert_eq!(how_long_ago(Duration::from_secs(0)), "less than a minute ago");
        assert_eq!(how_long_ago(Duration::from_secs(59)), "less than a minute ago");
        assert_eq!(how_long_ago(Duration::from_secs(60)), "a minute ago");
        assert_eq!(how_long_ago(Duration::from_secs(60 * 7)), "7 minutes ago");
        assert_eq!(how_long_ago(Duration::from_secs(60 * 59)), "59 minutes ago");
        assert_eq!(how_long_ago(Duration::from_secs(60 * 60)), "about an hour ago");
        assert_eq!(how_long_ago(Duration::from_secs(60 * 119)), "about an hour ago");
        assert_eq!(how_long_ago(Duration::from_secs(60 * 120)), "about 2 hours ago");
        assert_eq!(how_long_ago(Duration::from_secs(3600 * 24)), "1 day ago");
        assert_eq!(how_long_ago(Duration::from_secs(3600 * 24 * 3)), "3 days ago");
    }

    /// **The autosave is owed once a minute, and once per change rather than once per frame.**
    ///
    /// The third case is the one that is easy to get wrong: an autosave is not a save, so the
    /// document stays dirty afterwards — and without the revision test every frame after the
    /// first minute would write the whole document again.
    #[test]
    fn an_autosave_is_owed_once_a_minute_and_only_for_a_change() {
        use super::{autosave_due, AUTOSAVE_EVERY};
        let minute = AUTOSAVE_EVERY;
        let moment = std::time::Duration::from_secs(1);
        // Owed: changed since the last write, unsaved, and long enough ago.
        assert!(autosave_due(7, 3, true, minute));
        assert!(autosave_due(7, 3, true, minute * 10));
        // Not owed: nothing unsaved, so the document's own file is the record.
        assert!(!autosave_due(7, 3, false, minute));
        // Not owed: the autosave already is this document, though it is still unsaved.
        assert!(!autosave_due(7, 7, true, minute * 10));
        // Not owed: too soon, which is what keeps the write off every frame.
        assert!(!autosave_due(7, 3, true, moment));
        assert!(!autosave_due(7, 3, true, minute - moment));
    }

    /// The autosave is beside the logs and is not the document's own file, so that a crash
    /// cannot destroy the last state that was deliberately saved — and it is this process's own,
    /// so that no other session's exit can remove it either.
    #[test]
    fn the_autosave_is_not_the_document_s_own_file() {
        let path = super::autosave_path();
        assert_eq!(path.parent(), Some(std::env::temp_dir().as_path()));
        let own = format!("fieldform-autosave-{}.", std::process::id());
        assert!(
            path.file_name().is_some_and(|n| n.to_string_lossy().starts_with(&own)),
            "named by this process, so it is never another session's file: {path:?}",
        );
        assert_eq!(
            path.extension().map(|e| e.to_string_lossy().into_owned()),
            Some(super::DOCUMENT_EXTENSION.to_string())
        );
    }

    /// **A lost GPU keeps the document beside its own file, under its own name**, and one never
    /// saved beside the crash log that reports the loss (loop spec S3) — never over the document's
    /// file, nor over the autosave, which still holds the minute before.
    ///
    /// The current extension whatever the file was opened as, since that is what this build
    /// writes; and a file that already is a kept one is not written over by the next loss.
    #[test]
    fn a_lost_gpu_keeps_the_document_beside_its_file_or_beside_the_logs_when_it_has_none() {
        use super::{autosave_path, recovered_path, DOCUMENT_EXTENSION};
        let folder = std::path::Path::new("C:/work/brackets");
        let kept_name = format!("bracket.recovered.{DOCUMENT_EXTENSION}");
        for opened in ["bracket.fform", "bracket.fieldform"] {
            let file = folder.join(opened);
            let kept = recovered_path(Some(&file));
            assert_eq!(kept, folder.join(&kept_name), "kept beside {opened}");
            assert_ne!(kept, file);
        }
        let already_kept = folder.join(&kept_name);
        assert_ne!(recovered_path(Some(&already_kept)), already_kept);

        let untitled = recovered_path(None);
        assert_eq!(untitled.parent(), Some(std::env::temp_dir().as_path()));
        assert_eq!(
            untitled.file_name().map(|n| n.to_string_lossy().into_owned()),
            Some(format!("fieldform-untitled.recovered.{DOCUMENT_EXTENSION}"))
        );
        assert_ne!(untitled, autosave_path());
    }

    /// A folder of the temp directory's own for one test, empty, named for the test and this
    /// process so two runs at once do not share it.
    fn scratch_folder(what: &str) -> std::path::PathBuf {
        let folder = std::env::temp_dir()
            .join(format!("fieldform-test-{what}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).expect("the temp directory takes a folder");
        folder
    }

    /// **What the loss keeps is the document as it stood, edits since the last save included,
    /// and the file it was saved to is left as it was.** The edit here is the one that matters:
    /// it is newer than anything on disk, which is the whole of what the autosave cannot hold.
    #[test]
    fn the_kept_document_opens_as_it_stood_and_the_saved_file_is_left_alone() {
        use super::{recovered_path, write_recovered, Cage, Document, DEFAULT_BODY_MM};
        let folder = scratch_folder("kept");
        let file = folder.join("bracket.fform");
        let saved = Document::with_cage(Cage::cube_of(DEFAULT_BODY_MM));
        fieldform_io::to_archive(&saved).unwrap().write(&file).unwrap();

        let (mut edited, archive) = fieldform_io::open(&file).unwrap();
        let since = edited.add_parameter("since_the_last_save", 42.0);
        let kept = write_recovered(&edited, Some(&(file.clone(), archive))).unwrap();
        assert_eq!(kept, recovered_path(Some(&file)));

        let (reopened, _) = fieldform_io::open(&kept).expect("the kept file opens");
        assert_eq!(reopened.parameter_value(since), Some(42.0));
        let (on_disk, _) = fieldform_io::open(&file).unwrap();
        assert_eq!(on_disk.parameter_value(since), None, "the saved file is not touched");
        let _ = std::fs::remove_dir_all(&folder);
    }

    /// **A document whose folder has gone is kept beside the logs instead**, under the same
    /// name: a share that dropped or a drive pulled out is no reason to lose the work, and the
    /// temp directory is the one place this application always writes.
    #[test]
    fn a_kept_document_whose_folder_has_gone_lands_beside_the_logs() {
        use super::{write_recovered, Cage, Document, DEFAULT_BODY_MM, DOCUMENT_EXTENSION};
        let folder = scratch_folder("gone");
        let stem = format!("fieldform-test-gone-{}", std::process::id());
        let file = folder.join(format!("{stem}.{DOCUMENT_EXTENSION}"));
        let doc = Document::with_cage(Cage::cube_of(DEFAULT_BODY_MM));
        fieldform_io::to_archive(&doc).unwrap().write(&file).unwrap();
        let (doc, archive) = fieldform_io::open(&file).unwrap();
        std::fs::remove_dir_all(&folder).unwrap();

        let kept = write_recovered(&doc, Some(&(file, archive))).expect("kept somewhere");
        let beside_the_logs =
            std::env::temp_dir().join(format!("{stem}.recovered.{DOCUMENT_EXTENSION}"));
        assert_eq!(kept, beside_the_logs);
        assert!(fieldform_io::open(&kept).is_ok(), "the kept file opens");
        let _ = std::fs::remove_file(&kept);
    }

    /// A verify is dated, and the date is worked out here rather than by a library: a leap
    /// day, a new year and the epoch itself are the three places that arithmetic goes wrong.
    #[test]
    fn a_verification_is_dated_by_the_calendar() {
        assert_eq!(super::utc_date(0), "1970-01-01");
        assert_eq!(super::utc_date(951_782_400), "2000-02-29");
        assert_eq!(super::utc_date(951_782_400 + 86_400), "2000-03-01");
        assert_eq!(super::utc_date(1_767_225_600 - 1), "2025-12-31");
        assert_eq!(super::utc_date(1_767_225_600), "2026-01-01");
    }


    /// **A block asked for is drawn even when the block that consumes it is hidden.**
    ///
    /// Reported as a sequence: create a sphere, pattern it, include the original, hide the
    /// pattern, show the sphere — and nothing appeared. Hiding a body takes everything only
    /// *it* used away with it, which is right; but the copy that draws the sphere was made
    /// afterwards, out of a node that had already gone with its consumer.
    ///
    /// **Asked of every kind of consumer, because "globally" is the requirement.** Nothing in
    /// `drawn_graph` looks at what consumed anything, and a test over one kind would not say
    /// so — it would pass just as well on a version that special-cased patterns.
    #[test]
    fn a_shown_block_is_drawn_whatever_consumes_it_and_however_that_is_hidden() {
        use fieldform_scene::{
            Axis, BooleanOp, Fillet, Frame, Graph, Node, ParamRef, PatternKind, Place,
        };

        let lit = ParamRef::Literal;
        let ball = |graph: &mut Graph| {
            graph.add(Node::Sphere {
                centre: Place::At([lit(0.0); 3]),
                radius: lit(10.0),
            })
        };

        // Each is "one body, consumed by one block", spelled a different way.
        let consumers: Vec<(&str, fn(&mut Graph, fieldform_scene::NodeId) -> fieldform_scene::NodeId)> = vec![
            ("pattern", |g, input| {
                g.add(Node::Pattern {
                    options: Default::default(),
                    input: input.into(),
                    count: ParamRef::Literal(4.0),
                    kind: PatternKind::Radial {
                        frame: Frame::at_along(
                            Axis::Z,
                            [ParamRef::Literal(40.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)],
                        ),
                    },
                    join: true,
                    fillet: Fillet::hard(),
                })
            }),
            ("mirror", |g, input| {
                g.add(Node::Mirror {
                    input: input.into(),
                    frame: Frame::at_along(Axis::X, [ParamRef::Literal(0.0); 3]),
                    join: false,
                })
            }),
            ("offset", |g, input| {
                g.add(Node::Offset { input: input.into(), distance: ParamRef::Literal(3.0) })
            }),
            ("shell", |g, input| {
                g.add(Node::Shell { input: input.into(), thickness: ParamRef::Literal(2.0) })
            }),
            ("transform", |g, input| {
                g.add(Node::Transform {
                    input: input.into(),
                    translation: [ParamRef::Literal(20.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)],
                    rotation: [ParamRef::Literal(0.0); 3],
                    scale: ParamRef::Literal(1.0),
                    about: Place::origin(),
                })
            }),
            ("boolean", |g, input| {
                let other = g.add(Node::Sphere {
                    centre: Place::At([ParamRef::Literal(12.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)]),
                    radius: ParamRef::Literal(10.0),
                });
                g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![input.into(), other.into()] })
            }),
        ];

        for (what, consume) in consumers {
            let mut graph = Graph::empty();
            let body = ball(&mut graph);
            let consumer = consume(&mut graph, body);
            assert!(
                !graph.roots().contains(&body),
                "{what}: the body should be consumed, or the test is not testing anything"
            );

            // Shown while its consumer is hidden, which is the reported sequence.
            let (drawn, stand_ins) =
                super::drawn_graph(&graph, &[body], &|id| id == consumer);

            let roots = drawn.roots();
            assert_eq!(
                roots.len(),
                1,
                "{what}: expected the shown block alone to be drawn, got {roots:?}"
            );
            assert_eq!(
                stand_ins.get(&roots[0]),
                Some(&body),
                "{what}: the one drawn body is not the block that was asked for"
            );
            // And it is whole: every operand it names is still in the graph, or it would
            // compile to nothing.
            for input in drawn.node(roots[0]).expect("the stand-in").inputs() {
                assert!(
                    drawn.node(input).is_some(),
                    "{what}: the drawn block names an operand that was taken away"
                );
            }
        }
    }

    /// And hiding still takes the subtree with it when nothing has asked for any of it — the
    /// property the ordering above must not have broken.
    #[test]
    fn hiding_a_body_still_takes_what_only_it_used() {
        use fieldform_scene::{Graph, Node, ParamRef, Place};

        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let body = graph.add(Node::Sphere {
            centre: Place::At([lit(0.0); 3]),
            radius: lit(10.0),
        });
        let shell = graph.add(Node::Shell { input: body.into(), thickness: lit(2.0) });

        let (drawn, stand_ins) = super::drawn_graph(&graph, &[], &|id| id == shell);
        assert!(stand_ins.is_empty());
        assert!(
            drawn.roots().is_empty(),
            "hiding a shelled body drew the solid it was made from: {:?}",
            drawn.roots()
        );
    }

    /// **No two things in one write share a stamp, and the order given is the order kept.**
    ///
    /// The requirement the sort rests on rather than a detail of it. This used to read the clock
    /// once and write that one reading to every fresh node, and bodies that tie fall back to their
    /// ids — which is the random order the whole arrangement replaces, at exactly the same rate
    /// (`fieldform-gpu/examples/body_order_probe.rs`, the third scene). One node per write is the
    /// common case and was never affected; several at once was silently ordered by nothing.
    #[test]
    fn things_made_in_one_write_are_stamped_distinctly_and_in_order() {
        let now = 1_700_000_000_000_000u64;
        let stamps = crate::stamps_from(now, 40);
        assert_eq!(stamps.len(), 40);
        let unique: std::collections::HashSet<u64> = stamps.iter().copied().collect();
        assert_eq!(unique.len(), 40, "a stamp two things share orders them by nothing");
        let mut sorted = stamps.clone();
        sorted.sort_unstable();
        assert_eq!(stamps, sorted, "and the order given has to be the order kept");
        assert_eq!(stamps[0], now, "the first is the moment of the write itself");
    }

    /// **An import's solids come back in the file's order, not their ids'.**
    ///
    /// The case with the most explicit intended order in the application: the importer numbers
    /// them `file (1)`, `file (2)` and a re-import pairs them up *by that position*
    /// (`cad-import-block-spec.md` §4). They were named in order, matched in order, and listed in
    /// random order — because every solid of one import arrives in a single write, and a single
    /// write used to give them all one stamp.
    ///
    /// Asserted through `Document::listed_bodies` on what `stamp_in_order` writes, which is the whole
    /// path bar the mesh reading. Twenty rounds, for the usual reason: the ids are random, so one
    /// round agrees by luck often enough to prove nothing.
    #[test]
    fn an_imports_solids_are_listed_in_the_order_the_file_had_them() {
        use fieldform_scene::{Document, Graph, Node, NodeId, ParamRef, Place};
        let lit = ParamRef::Literal;
        for round in 0..20 {
            let mut doc = Document::new();
            let mut graph = Graph::empty();
            // Six solids, added in the file's order the way `add_imported_bodies` adds them.
            let solids: Vec<NodeId> = (0..6)
                .map(|i| {
                    graph.add(Node::Sphere {
                        centre: Place::At([lit(i as f64 * 20.0), lit(0.0), lit(0.0)]),
                        radius: lit(8.0),
                    })
                })
                .collect();
            // What `stamp_in_order` writes.
            let now = 1_700_000_000_000_000u64;
            for (id, at) in solids.iter().zip(crate::stamps_from(now, solids.len())) {
                doc.created.insert(*id, at);
            }
            assert_eq!(
                doc.listed_bodies(&graph),
                solids,
                "round {round}: file (1) through file (6), in that order",
            );
        }
    }

    /// A block that takes two operands finds them itself, without either slot being set.
    ///
    /// **The complaint was that a morph would not recognise the bodies in the part.** It read
    /// `boolean_a` and `boolean_b` directly and refused unless both held something, and those
    /// two are only ever written by the boolean tool's own pickers — so adding bodies, however
    /// many, never filled them and the morph stayed unavailable while saying only that it
    /// needed two bodies. The boolean beside it read the body list instead, which is why the
    /// same gesture worked for one block and not the other.
    ///
    /// So the property is about the *picker*, not about either block: given bodies and nothing
    /// chosen, it yields two of them, and they are distinct. Both blocks now go through this
    /// one function, which is what stops them drifting again (D9).
    #[test]
    fn a_pair_is_found_from_the_bodies_when_nothing_has_been_chosen() {
        use fieldform_scene::{BooleanOp, Graph, Node, ParamRef, Place};

        let lit = ParamRef::Literal;
        let ball = |x: f64| Node::Sphere {
            centre: Place::At([lit(x), lit(0.0), lit(0.0)]),
            radius: lit(10.0),
        };
        let mut graph = Graph::empty();
        let a = graph.add(ball(0.0));
        let b = graph.add(ball(12.0));
        let c = graph.add(ball(24.0));
        // When each was made, as `Document::created` would hold it. Written out rather than
        // stubbed flat, because *which* two the picker reaches for is the property now: it used
        // to be whichever two happened to hold the lowest random ids.
        let made = move |id| [a, b, c].iter().position(|x| *x == id).unwrap_or(9) as u64;

        // Nothing chosen, three bodies: the first two *made*, and not the same one twice.
        let (first, second) = crate::starting_pair_in(&graph, made, None, None)
            .expect("three bodies is enough to start from");
        assert_ne!(first, second, "a block cannot take one body as both its operands");
        assert_eq!((first, second), (a, b), "the two oldest blocks, not the two lowest ids");

        // One chosen: it is honoured, and the other is filled in around it.
        let (first, second) = crate::starting_pair_in(&graph, made, Some(c), None)
            .expect("a choice plus the list is enough");
        assert_eq!(first, c);
        assert_eq!(second, a, "and the oldest of the rest fills the other slot");

        // **A consumed block is reachable whether or not it has a name.** Two spheres and the
        // union of them leave one body, and the two operands are still right there -- which is
        // the reported case: a part with one root left and the morph tool refusing. It used to
        // need a name, and `Pattern` and `Mirror` wrote one on the user's behalf to keep it
        // reachable; `pair_candidates_in` says why the gate went instead.
        let mut consumed = Graph::empty();
        let a = consumed.add(ball(0.0));
        let b = consumed.add(ball(12.0));
        let union = consumed.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![a.into(), b.into()] });
        let made = move |id| [a, b].iter().position(|x| *x == id).unwrap_or(9) as u64;
        assert_eq!(
            crate::pair_candidates_in(&consumed, &made).len(),
            3,
            "the boolean and the two blocks inside it"
        );
        let (first, second) = crate::starting_pair_in(&consumed, made, None, None)
            .expect("two blocks inside a boolean are still a pair");
        assert_ne!(first, second);
        assert_eq!(
            first, union,
            "and a body still sorts first, so a tool's starting pair opens on one"
        );
        assert_eq!(
            crate::starting_pair_in(&consumed, made, Some(a), Some(b)),
            Some((a, b)),
            "asking for those two by name gives exactly them"
        );
    }

    /// An idle ramp is offered back before another is built.
    ///
    /// **The reported pile-up, as a property.** Turning "varies through space" off leaves the
    /// ramp standing, because §7.2 makes it a block and deleting a visible block as a side
    /// effect of a toggle is what the delete cascade was already changed to stop. Turning it
    /// back on then built a *second* ramp and a second point, and the two rules together made
    /// the panel accumulate blocks nothing read. Reaching for the idle one closes the loop.
    ///
    /// A ramp that is driving something is not idle: taking it would silently rewire whatever
    /// was using it, and wanting a second, differently-shaped ramp is the ordinary case.
    #[test]
    fn an_idle_ramp_is_the_one_reached_for_and_a_busy_one_is_not() {
        use fieldform_scene::{Graph, Node, NodeId, ParamRef, Place, RampShape};

        let lit = ParamRef::Literal;
        let ramp_on = |graph: &mut Graph, point: NodeId| {
            graph.add(Node::Ramp {
                input: point.into(),
                in_min: lit(0.0),
                in_max: lit(10.0),
                out_min: lit(0.0),
                out_max: lit(1.0),
                shape: RampShape::Linear,
            })
        };
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: Place::At([lit(0.0); 3]),
            radius: lit(30.0),
        });
        let point = graph.add(Node::Point { at: [lit(0.0); 3] });
        let busy = ramp_on(&mut graph, point);
        graph.add(Node::Offset { input: ball.into(), distance: ParamRef::Field(busy) });

        // Consumed, so not spare. This is the half that stops a second socket stealing a ramp.
        let consumed = graph.consumed();
        assert!(consumed.contains(&busy), "a ramp that drives something is not idle");

        let idle = ramp_on(&mut graph, point);
        let consumed = graph.consumed();
        assert!(!consumed.contains(&idle), "a ramp nothing reads is idle");
        assert!(consumed.contains(&busy), "and the busy one still is not");
    }

    /// Moving a block that carries its own XYZ writes that XYZ, keeping its shape.
    ///
    /// **The reported behaviour: drag a sphere and its centre should follow.** It used to wrap
    /// the block in a placement and write the motion there, so the coordinates in the panel
    /// stayed at whatever the shape was built with while the shape sat somewhere else — two
    /// places holding one fact, with only one of them moving.
    ///
    /// A two-ended shape moves both ends by the same vector, which is what keeps a cylinder the
    /// length it was: moving one end alone would reshape it, and a drag on the block is meant to
    /// move it. Asserted here because it is the half that is easy to get wrong.
    #[test]
    fn moving_a_block_with_its_own_xyz_writes_that_xyz_and_keeps_its_shape() {
        use fieldform_scene::{Ends, Graph, Node, ParamRef, Place};
        use nalgebra::{Point3, Vector3};

        let lit = ParamRef::Literal;
        let xyz = |place: &Place| {
            let typed = place.typed().expect("written as a typed place");
            let n = |r: &ParamRef| match r {
                ParamRef::Literal(v) => *v,
                _ => panic!("a dragged coordinate is written as a plain number"),
            };
            Point3::new(n(&typed[0]), n(&typed[1]), n(&typed[2]))
        };
        let read = |graph: &Graph, id| -> Vec<Point3<f64>> {
            graph.node(id).map(|n| n.places().iter().map(|p| xyz(p)).collect()).unwrap_or_default()
        };

        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: Place::At([lit(1.0), lit(2.0), lit(3.0)]),
            radius: lit(10.0),
        });
        let rod = graph.add(Node::Cylinder { edge: Default::default(),
            ends: Ends::At {
                from: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
                to: Place::At([lit(0.0), lit(0.0), lit(50.0)]),
            },
            radius: lit(4.0),
        });

        let shift = Vector3::new(5.0, -1.0, 2.0);
        for id in [ball, rod] {
            let before = read(&graph, id);
            let after: Vec<Point3<f64>> = before.iter().map(|p| p + shift).collect();
            crate::App::set_own_anchors(&mut graph, id, &after);
            for (w, a) in read(&graph, id).iter().zip(&after) {
                assert!((w - a).norm() < 1e-12, "{w:?} should be {a:?}");
            }
        }

        // No placement was added: the blocks moved themselves.
        assert!(
            !graph.nodes().any(|(_, n)| matches!(n, Node::Transform { .. })),
            "moving a block that carries its own XYZ must not wrap it"
        );
        // And the rod is the length it was.
        let ends = read(&graph, rod);
        assert!(
            ((ends[1] - ends[0]).norm() - 50.0).abs() < 1e-12,
            "a move must not reshape the block"
        );
    }

    /// A block sitting on a point is moved by moving that point.
    ///
    /// **The reported bug.** Make a sphere, turn its XYZ into a variable — which makes a point
    /// and puts the sphere on it — and dragging the sphere did nothing to any number on screen:
    /// its place was `From { node }`, so the block had no coordinates of its own to write and the
    /// drag fell back to wrapping it in a placement. The numbers in the panel are the *point's*,
    /// read through, so those are the ones a drag has to move.
    ///
    /// Writing literals into the place instead would have updated the panel by cutting the link
    /// the panel is showing: the point left in the list with nothing reading it, and the V that
    /// said "this is a variable" quietly false.
    ///
    /// One place with one source, because that is the only case with one answer — asserted here
    /// with a cylinder, whose two ends could name two different blocks.
    #[test]
    fn a_block_sitting_on_a_point_is_dragged_by_moving_that_point() {
        use fieldform_scene::{Ends, Graph, Node, NodeId, ParamRef, Place};

        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let point = graph.add(Node::Point { at: [lit(5.0), lit(0.0), lit(0.0)] });
        let seated = graph.add(Node::Sphere {
            centre: Place::From { node: point },
            radius: lit(10.0),
        });
        let standing = graph.add(Node::Sphere {
            centre: Place::At([lit(0.0); 3]),
            radius: lit(10.0),
        });
        let rod = graph.add(Node::Cylinder { edge: Default::default(),
            ends: Ends::At {
                from: Place::From { node: point },
                to: Place::At([lit(0.0), lit(0.0), lit(50.0)]),
            },
            radius: lit(4.0),
        });

        // The rule, as the resolver states it: exactly one place, and its source is a point.
        let follows = |graph: &Graph, id| -> Option<NodeId> {
            let node = graph.node(id)?;
            let places = node.places();
            let [place] = places.as_slice() else { return None };
            place
                .source()
                .filter(|s| matches!(graph.node(*s), Some(Node::Point { .. })))
        };

        assert_eq!(follows(&graph, seated), Some(point), "a seated block follows its point");
        assert_eq!(follows(&graph, standing), None, "a block with its own place moves itself");
        assert_eq!(
            follows(&graph, rod),
            None,
            "two ends could name two blocks, so there is no single one to move"
        );
    }

    /// A quoted word is one word, and is told apart from an argument.
    ///
    /// **The flag is the whole point.** `sphere 'left bearing'` is a name with a space in it and
    /// `sphere radius=20` is a key and a value; without knowing which arrived in quotes, a bare
    /// word would have to be guessed at, and the guess is wrong for the first command anybody
    /// mistypes.
    #[test]
    fn a_quoted_console_word_is_one_word_and_says_it_was_quoted() {
        let words = crate::console_words("sphere 'left bearing' radius=20");
        assert_eq!(
            words,
            vec![
                ("sphere".to_string(), false),
                ("left bearing".to_string(), true),
                ("radius=20".to_string(), false),
            ]
        );

        // Either quote character, since a name with an apostrophe is likelier than one with a
        // double quote and requiring the other is a rule nobody remembers.
        assert_eq!(
            crate::console_words("cube \"a name\""),
            vec![("cube".to_string(), false), ("a name".to_string(), true)]
        );

        // An empty name is still a word, so it reaches the code that turns a blank name back
        // into no name rather than being dropped on the floor here.
        assert_eq!(
            crate::console_words("sphere ''"),
            vec![("sphere".to_string(), false), (String::new(), true)]
        );

        // An unclosed quote runs to the end of the line: what was meant is obvious, and the
        // alternative is an error message about punctuation.
        assert_eq!(
            crate::console_words("sphere 'unclosed"),
            vec![("sphere".to_string(), false), ("unclosed".to_string(), true)]
        );

        // And nothing quoted behaves exactly as splitting on whitespace did.
        assert_eq!(
            crate::console_words("  mesh   cells=96 "),
            vec![("mesh".to_string(), false), ("cells=96".to_string(), false)]
        );
    }

    /// A placed ramp is a block in the list and never a body.
    ///
    /// A ramp draws nothing, so counting one as a body would put a meaningless surface on
    /// screen; hiding it from the list instead would make it unselectable, which is the state
    /// §7.2 exists to end. Both halves are asserted, because either alone is satisfiable by
    /// doing the wrong thing.
    #[test]
    fn a_placed_ramp_is_listed_but_is_not_a_body() {
        use fieldform_scene::{Graph, Node, ParamRef, Place, RampShape};

        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: Place::At([lit(0.0); 3]),
            radius: lit(30.0),
        });
        let point = graph.add(Node::Point { at: [lit(0.0); 3] });
        let ramp = graph.add(Node::Ramp {
            input: point.into(),
            in_min: lit(0.0),
            in_max: lit(10.0),
            out_min: lit(0.0),
            out_max: lit(1.0),
            shape: RampShape::Linear,
        });

        assert_eq!(graph.roots(), vec![ball], "only the sphere is a body");
        assert!(
            graph.nodes().any(|(id, _)| id == ramp),
            "the ramp is in the graph, so the block list walking every node shows it"
        );
        assert!(
            !graph.roots().is_empty(),
            "a part holding a ramp beside a body still has a body, so it must not read as empty"
        );
    }

    /// A locked box keeps its proportions whichever side is dragged.
    #[test]
    fn a_locked_size_moves_the_other_two_sides_by_the_same_factor() {
        // Y doubled, so X and Z double with it and the shape is the same shape.
        let (led, factor) = crate::leading_factor([10.0, 20.0, 40.0], [10.0, 40.0, 40.0])
            .expect("one side moved");
        assert_eq!(led, 1);
        assert!((factor - 2.0).abs() < 1e-12, "the factor is {factor}");
        assert!((10.0 * factor - 20.0).abs() < 1e-12);
        assert!((40.0 * factor - 80.0).abs() < 1e-12);

        // Nothing moved: a frame where the row was only drawn must not rewrite the other two,
        // or a locked box would drift every frame it is looked at.
        assert!(crate::leading_factor([10.0, 20.0, 40.0], [10.0, 20.0, 40.0]).is_none());

        // Two moved at once is not a drag of one, and picking a leader would be a guess.
        assert!(crate::leading_factor([10.0, 20.0, 40.0], [11.0, 21.0, 40.0]).is_none());

        // From zero every factor is zero, so there is no ratio to hold.
        assert!(crate::leading_factor([0.0, 20.0, 40.0], [5.0, 20.0, 40.0]).is_none());
    }

    /// Picking goes through the seam two touching solids leave, and lands on the real wall.
    ///
    /// **The bug this pins was reported as "a small gap between the two profiles".** Extrude a
    /// profile up, extrude it down, union the two and subtract the union: `min` reads exactly
    /// nought all along the face they share, because both surfaces genuinely are there and it
    /// cannot answer that neither bounds the union. Negated by the subtract, that sheet of zero
    /// sits in the middle of the cut, and a march that stops wherever the value is small stops
    /// on it -- drawn as a sliver of material left behind, picked as a pivot in thin air.
    ///
    /// The ray here starts inside the cut and points at its floor, 27 mm away. Stopping at the
    /// join instead is stopping at 12, so the two answers are nowhere near each other and this
    /// cannot pass by accident. `fieldform-field/examples/opposed_extrudes.rs` carries the
    /// numbers and the alternatives.
    #[test]
    fn picking_does_not_stop_on_the_seam_between_two_touching_solids() {
        use fieldform_field::ScalarField;
        use fieldform_scene::{BooleanOp, Curve, Document, Graph, Node, ParamRef, Place, Sweep};
        use nalgebra::{Point3, Vector3};

        let (half, depth) = (20.0f64, 15.0f64);
        let mut doc = Document::new();
        let id = fieldform_scene::id::CurveId::new();
        doc.curves.insert(
            id,
            Curve::polyline(
                id,
                "Square".to_string(),
                vec![
                    Point3::new(-half, -half, 0.0),
                    Point3::new(half, -half, 0.0),
                    Point3::new(half, half, 0.0),
                    Point3::new(-half, half, 0.0),
                ],
                true,
            )
            .expect("a closed square"),
        );
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let up = graph.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: Sweep::Normal,
            height: lit(depth),
            draft: lit(0.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            corners: Default::default(),
        });
        let down = graph.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: Sweep::Normal,
            height: lit(-depth),
            draft: lit(0.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            corners: Default::default(),
        });
        let union = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![up.into(), down.into()] });
        let block = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            size: [lit(80.0), lit(80.0), lit(60.0)],
            turn: fieldform_scene::no_turn(),
        });
        graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![block.into(), union.into()] });
        let field = fieldform_field::GraphField::compile_graph(&doc, &graph, 1e-4)
            .expect("the graph compiles");

        // The seam reads as a surface where the honest answer is fifteen millimetres of solid.
        assert!(
            field.distance(Point3::new(0.0, 0.0, 0.0)).abs() < 1e-9,
            "the premise: `min` reads nought at the join"
        );

        let hit = crate::trace_field(
            &field,
            Point3::new(0.0, 0.0, 12.0),
            Vector3::new(0.0, 0.0, -1.0),
            80.0,
            1e-3,
            512,
        )
        .expect("the ray leaves the cut through its floor");
        assert!(
            (hit.z + depth).abs() < 0.01,
            "stopped at z = {:.4}; the floor of the cut is at {}",
            hit.z,
            -depth
        );
    }

    /// A new curve is flat in the plane it is built in, at every degree it can be made at.
    ///
    /// **The bug this replaces was invisible until a drawing plane existed.** The starting shape
    /// is a half sine, and its rise was in Z -- fine on its own, because the curve simply arched
    /// upward in a world where Z is up. But a curve is built in the drawing plane's own
    /// coordinates and lifted into the world, so a local Z is the plane's *normal*: setting a
    /// plane and making a curve produced one arching out of the plane rather than lying on it,
    /// which is the one thing a drawing plane exists to prevent.
    ///
    /// Asserted at every offered degree, because the shape is generated per degree and a rise
    /// that crept back into Z for degree 9 alone would be no easier to spot than the first one.
    #[test]
    fn a_new_curve_is_flat_in_the_plane_it_is_built_in() {
        use crate::{single_span_bezier, CURVE_DEGREES};
        for degree in CURVE_DEGREES {
            let curve = single_span_bezier(fieldform_scene::id::CurveId::new(), degree, 60.0)
                .expect("every offered degree makes a curve");
            for (i, p) in curve.control_points().iter().enumerate() {
                assert!(
                    p.z.abs() < 1e-12,
                    "degree {degree} point {i} sat {} off the plane",
                    p.z
                );
            }
            // And it is not flat by being straight: something has to be off the chord, or the
            // test would pass just as well on a curve that had lost its shape entirely.
            assert!(
                curve.control_points().iter().any(|p| p.y.abs() > 1e-9) || degree == 1,
                "degree {degree} came out with no shape at all"
            );
        }
    }

    /// An arbitrary plane is stood up the way a cube face is.
    ///
    /// **Two rules for one question is how a plane picked from the menu ends up rolled
    /// differently from the same plane picked off the ViewCube** -- both looking down the same
    /// axis, one of them on its side, with nothing on screen explaining why. `up_for_normal`
    /// generalises `CubeRegion::up`, and this is what says it still agrees with it everywhere
    /// the two overlap.
    #[test]
    fn an_arbitrary_plane_is_stood_up_the_way_a_cube_face_is() {
        use crate::{up_for_normal, CubeRegion};
        for sign in [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] {
            let region = CubeRegion { sign };
            assert_eq!(
                up_for_normal(region.direction()),
                region.up(),
                "{sign:?} stood up differently"
            );
        }
    }

    /// A point built in a drawing plane ends up on that plane, for every plane you can name.
    ///
    /// **The whole promise of step 1**, and the reason it is worth asserting rather than reading:
    /// the lift is a rotation and a translation, and a rotation that takes the wrong axis to `+Z`
    /// puts the curve on a plane perpendicular to the one asked for — which looks plausible on
    /// screen from the wrong angle and is exactly the sort of thing a person discovers an hour
    /// later.
    ///
    /// Checked by the definition of the plane rather than by comparing coordinates: the vector
    /// from the plane's origin to the point has no component along its normal.
    #[test]
    fn a_point_built_in_a_drawing_plane_lies_on_it() {
        use crate::{drawing_frame_of, Axis, DrawingPlane};
        use nalgebra::Point3;

        let doc = fieldform_scene::Document::new();
        let graph = fieldform_scene::Graph::empty();
        let numbers = |r: &fieldform_scene::ParamRef| match r {
            fieldform_scene::ParamRef::Literal(v) => Some(*v),
            _ => None,
        };

        for normal in [Axis::X, Axis::Y, Axis::Z] {
            for offset_mm in [0.0, 25.0, -7.5] {
                let plane = DrawingPlane::World { normal, offset_mm };
                let frame = drawing_frame_of(plane).expect("a world plane names a frame");
                let (translation, rotation) =
                    fieldform_field::resolve_frame(&doc, &graph, &frame, &numbers)
                        .expect("a typed frame resolves");

                let up = match normal {
                    Axis::X => nalgebra::Vector3::x(),
                    Axis::Y => nalgebra::Vector3::y(),
                    Axis::Z => nalgebra::Vector3::z(),
                };
                // The origin is along the normal at the offset asked for.
                assert!(
                    (translation - up * offset_mm).norm() < 1e-9,
                    "{normal:?} at {offset_mm} started at {translation:?}"
                );

                // Anything with z = 0 locally is on the plane. Two points off the local origin,
                // because a rotation that is wrong about its own axis still fixes the origin.
                for local in [
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(40.0, 0.0, 0.0),
                    Point3::new(-13.0, 9.0, 0.0),
                ] {
                    let world = Point3::from(rotation * local.coords + translation);
                    let off = (world.coords - translation).dot(&up);
                    assert!(
                        off.abs() < 1e-9,
                        "{local:?} on the {normal:?} plane sat {off} mm off it"
                    );
                }
            }
        }
    }

    /// With no plane set, a point is where it was put — the identity, exactly.
    ///
    /// This is what lets every caller lift unconditionally instead of branching. If it were
    /// merely *close* to the identity, every curve in every existing document would move by a
    /// rounding error the first time it was touched.
    #[test]
    fn no_drawing_plane_leaves_a_point_exactly_where_it_was() {
        use crate::{drawing_frame_of, DrawingPlane};
        assert!(drawing_frame_of(DrawingPlane::None).is_none());
    }

    /// A drawing plane takes the click ahead of the model behind it.
    ///
    /// That is what setting one means: a profile drawn over a body stays flat on the plane
    /// rather than being pulled onto the body's surface, which would make it a curve the
    /// extrude could not use. The surface is not even asked.
    #[test]
    fn a_drawing_plane_takes_the_click_ahead_of_the_surface() {
        use crate::landing_point;
        use nalgebra::{Point3, Vector3};
        let origin = Point3::new(0.0, 0.0, 100.0);
        let dir = Vector3::new(0.3, 0.1, -1.0).normalize();
        let plane = Some((Point3::new(0.0, 0.0, 20.0), Vector3::z()));
        let asked = std::cell::Cell::new(false);
        let surface = || {
            asked.set(true);
            Some(Point3::new(1.0, 2.0, 3.0))
        };
        let at = landing_point(origin, dir, plane, surface, Point3::origin(), -Vector3::z())
            .expect("a plane seen squarely takes the click");
        assert!((at.z - 20.0).abs() < 1e-4, "landed {at:?}, off the plane");
        let t = (at - origin).dot(&dir);
        assert!((origin + dir * t - at).norm() < 1e-3, "landed {at:?}, off the ray");
        assert!(!asked.get(), "the surface was consulted with a plane set");
    }

    /// Within a few degrees of edge-on, the click is declined rather than placed far away.
    #[test]
    fn a_plane_seen_edge_on_declines_the_click() {
        use crate::landing_point;
        use nalgebra::{Point3, Vector3};
        // About three degrees off the XY plane, from a metre away: the intersection would be
        // twenty metres along the ray.
        let dir = Vector3::new(1.0, 0.0, 0.05).normalize();
        let plane = Some((Point3::origin(), Vector3::z()));
        let declined = landing_point(
            Point3::new(-1000.0, 0.0, 1.0),
            dir,
            plane,
            || Some(Point3::origin()),
            Point3::origin(),
            Vector3::x(),
        );
        assert!(declined.is_none());
    }

    /// With no plane, the point lands on whatever is under the cursor.
    #[test]
    fn with_no_plane_the_point_lands_on_the_surface_under_the_cursor() {
        use crate::landing_point;
        use nalgebra::{Point3, Vector3};
        let hit = Point3::new(12.0, -3.0, 7.5);
        let at = landing_point(
            Point3::new(0.0, 0.0, 100.0),
            Vector3::new(0.1, 0.0, -1.0).normalize(),
            None,
            || Some(hit),
            Point3::origin(),
            -Vector3::z(),
        )
        .expect("a surface hit is a landing");
        assert_eq!(at, hit);
    }

    /// With nothing under the cursor, the point lands level with the pivot, facing the camera.
    ///
    /// Predictable, which is the most that can be asked of a click into empty space: in a view
    /// square to the model it is the view plane, and the point is dragged from there.
    #[test]
    fn with_nothing_under_the_cursor_the_point_lands_level_with_the_pivot() {
        use crate::landing_point;
        use nalgebra::{Point3, Vector3};
        let forward = Vector3::new(0.0, 1.0, -0.5).normalize();
        let pivot = Point3::new(5.0, 5.0, 5.0);
        let origin = pivot - forward * 300.0;
        let dir = (forward + Vector3::new(0.1, 0.0, 0.05)).normalize();
        let at = landing_point(origin, dir, None, || None, pivot, forward)
            .expect("empty space still has a landing");
        assert!(
            (at - pivot).dot(&forward).abs() < 1e-3,
            "landed {at:?}, not level with the pivot"
        );
        let t = (at - origin).dot(&dir);
        assert!((origin + dir * t - at).norm() < 1e-3, "landed {at:?}, off the ray");
    }

    /// A placed curve runs from the first click to the second, straight, at every degree.
    ///
    /// Interior points evenly along the chord is the spec's "two clicks plus dragging": the
    /// curve starts as a straight run of the right degree and is pulled into shape afterwards.
    #[test]
    fn a_placed_curve_runs_from_the_first_click_to_the_second() {
        use crate::bezier_between;
        use fieldform_scene::CurveId;
        use nalgebra::Point3;
        let a = Point3::new(-10.0, 4.0, 2.5);
        let b = Point3::new(30.0, -6.0, 12.5);
        for degree in [1usize, 2, 3, 5, 9] {
            let curve =
                bezier_between(CurveId::new(), degree, a, b).expect("a chord is a valid curve");
            assert_eq!(curve.degree(), degree);
            let points = curve.control_points();
            assert_eq!(points.len(), degree + 1);
            for (i, p) in points.iter().enumerate() {
                let expected = a + (b - a) * (i as f64 / degree as f64);
                assert!(
                    (p - expected).norm() < 1e-12,
                    "degree {degree}, point {i} at {p:?}, expected {expected:?}"
                );
            }
            assert_eq!(
                curve.name,
                if degree == 1 { "Line".to_string() } else { format!("Curve {degree}") }
            );
        }
    }

    /// A short flag is the same as saying the key is true, and a number is not a flag.
    ///
    /// **One mechanism, in the console rather than in the command.** Every command that takes a
    /// switch gains `-x` at once this way, instead of each one growing its own spelling -- which
    /// is how two ways to say one thing start.
    #[test]
    fn a_dash_flag_is_the_key_said_yes_to() {
        use crate::{console_arguments, console_words, Value};

        // What `run_command` does with the words after the verb, by the function it does it
        // with: this test used to carry its own copy of the loop.
        let read = |line: &str| console_arguments(console_words(line)).map(|(args, _)| args);

        let switched = read("-u").expect("a flag");
        assert!(matches!(switched.get("u"), Some(Value::Bool(true))), "a short flag sets its own key");
        assert!(read("--subtract").expect("a flag").get("subtract").is_some(), "and a long one sets its name");
        // A number is not a flag, however it is signed: it still has to be said as key=value,
        // and the message that says so is worth more than a key called "5".
        assert!(read("-5").is_err());
        assert!(read("radius=3").expect("an argument").get("radius").is_some());
    }

    /// `sweep --taper-scale 0.5 --law g1 --twist 90 --starts 10 --ends 90` reaches the law as
    /// the numbers it means, and the two tapers together are refused.
    ///
    /// The grammar's own test is above; this is what the sweep's own words parse to, so that a
    /// renamed flag is caught here rather than in a session where nothing happens.
    #[test]
    fn a_sweeps_words_are_its_laws_numbers() {
        use crate::{console_arguments, console_words, Value};
        let read = |line: &str| {
            console_arguments(console_words(line)).map(|(args, _)| args).expect("parses")
        };

        let scaled = read("--taper_scale 0.5 --law g1 --twist 90 --extent 10,90");
        assert!(scaled.get("taper_scale").and_then(Value::as_f64) == Some(0.5));
        assert!(scaled.get("law").and_then(Value::as_str) == Some("g1"));
        assert!(scaled.get("twist").and_then(Value::as_f64) == Some(90.0));
        // The stretch of path is the law's own `extent`, one key of two percentages: `ends` is a
        // wall's word for a square or a round end, and `wall_wanted` reads any `ends` as asking
        // for a wall, so the law could not have been reached through it.
        assert_eq!(
            crate::console_keys::percent_pair("sweep", "extent", scaled.get("extent").expect("given")),
            Ok([10.0, 90.0])
        );
        assert!(crate::console_keys::percent_pair("sweep", "extent", &Value::Str("90,10".into())).is_err());
        assert!(crate::console_keys::percent_pair("sweep", "extent", &Value::Str("0,120".into())).is_err());

        // `corners` takes the word the panel shows; `-corners` alone still reads as sharp, which
        // is what `--sharp` meant before the key had the field's name.
        use fieldform_scene::Corners;
        assert_eq!(crate::console_keys::corners_named(&Value::Str("sharp".into())), Ok(Corners::Sharp));
        assert_eq!(crate::console_keys::corners_named(&Value::Str("round".into())), Ok(Corners::Round));
        assert_eq!(crate::console_keys::corners_named(&Value::Bool(true)), Ok(Corners::Sharp));
        assert_eq!(crate::console_keys::corners_named(&Value::Bool(false)), Ok(Corners::Round));
        assert!(crate::console_keys::corners_named(&Value::Str("mitred".into())).is_err());

        // A draft is the other taper, and a negative one is a number rather than a flag.
        let drafted = read("--taper_draft 5");
        assert!(drafted.get("taper_draft").and_then(Value::as_f64) == Some(5.0));
        assert!(read("taper_draft=-5").get("taper_draft").and_then(Value::as_f64) == Some(-5.0));

        // Both tapers at once parse; `apply_sweep_options` is what refuses them, with a
        // message, rather than taking whichever came last.
        let both = read("--taper_scale 0.5 --taper_draft 5");
        assert!(both.get("taper_scale").is_some() && both.get("taper_draft").is_some());
    }

    /// `loft sections=A,C spine=P --g1` names its sections in order, its centerline and its
    /// smoothness; `loft` alone takes every closed curve and loop in list order; a name that
    /// matches nothing is refused by name rather than skipped.
    #[test]
    fn a_lofts_words_are_its_sections_its_centerline_and_its_smoothness() {
        use crate::{console_arguments, console_words, loft_asked};
        use fieldform_scene::{Curve, CurveId, Document, LoftSmooth};
        use nalgebra::Point3;
        let mut doc = Document::new();
        let mut add = |name: &str, closed: bool, z: f64| {
            let points = vec![
                Point3::new(-5.0, -5.0, z),
                Point3::new(5.0, -5.0, z),
                Point3::new(5.0, 5.0, z),
                Point3::new(-5.0, 5.0, z),
            ];
            let curve = Curve::polyline(CurveId::new(), name, points, closed).expect("a curve");
            let id = curve.id;
            doc.curves.insert(id, curve);
            id
        };
        let (a, b, c) = (add("A", true, 0.0), add("B", true, 20.0), add("C", true, 40.0));
        let p = add("P", false, 0.0);
        let read = |line: &str| {
            console_arguments(console_words(line)).map(|(args, _)| args).expect("parses")
        };

        let (sections, spine, smooth) =
            loft_asked(&doc, &[], &read("")).expect("every closed curve");
        let mut all = vec![a, b, c];
        all.sort_by_key(|id| id.raw());
        assert_eq!((sections, spine, smooth), (all, None, LoftSmooth::G0), "the open P left out");

        let (sections, spine, smooth) =
            loft_asked(&doc, &[], &read("sections=C,a spine=P --g1")).expect("named");
        assert_eq!((sections, spine, smooth), (vec![c, a], Some(p), LoftSmooth::G1));

        assert!(loft_asked(&doc, &[], &read("sections=A,Q")).is_err(), "Q is nobody");
        assert!(loft_asked(&doc, &[], &read("spine=Q")).is_err());
    }

    /// `extrude --wall 2 --side left` is `extrude wall=2 side=left`: a long flag takes the word
    /// after it as its value, which is how `outline-wall-spec.md` §6 writes its commands.
    ///
    /// **Only a long one, and only a bare word.** A short flag stays a switch whatever follows,
    /// so a stray word after `boolean -u` is still reported rather than swallowed; and a flag,
    /// a key=value or a quoted name after a long flag is its own thing, which leaves the flag a
    /// switch.
    #[test]
    fn a_long_flag_takes_the_word_after_it_as_its_value() {
        use crate::{console_arguments, console_words, Value};
        let read = |line: &str| console_arguments(console_words(line));

        let (args, naming) = read("--wall 2 --side left").expect("two flags and their values");
        assert_eq!(args.get("wall").and_then(Value::as_f64), Some(2.0));
        assert_eq!(args.get("side").and_then(Value::as_str), Some("left"));
        assert_eq!(naming, None);
        // The same arguments, in the console's own spelling.
        let (plain, _) = read("wall=2 side=left").expect("two arguments");
        assert_eq!(plain.get("wall").and_then(Value::as_f64), Some(2.0));
        assert_eq!(plain.get("side").and_then(Value::as_str), Some("left"));

        // A negative number is a value, not a flag, so the thickness that is refused later is
        // the one that was typed.
        let (args, _) = read("--wall -2").expect("a flag and its value");
        assert_eq!(args.get("wall").and_then(Value::as_f64), Some(-2.0));

        // With nothing after it, or with something that is its own argument, it is a switch.
        for line in ["--wall", "--wall --side left", "--wall side=left", "--wall 'a rib'"] {
            let (args, naming) = read(line).expect("a switch");
            assert!(matches!(args.get("wall"), Some(Value::Bool(true))), "{line}");
            assert_eq!(naming.is_some(), line.ends_with('\''), "{line}");
        }

        // And a short flag does not take one: the stray word is still said to be stray.
        assert!(read("-u stray").is_err());
        let (args, _) = read("-u radius=3").expect("a switch and an argument");
        assert!(matches!(args.get("u"), Some(Value::Bool(true))));
        assert_eq!(args.get("radius").and_then(Value::as_f64), Some(3.0));
    }

    /// A wall's three sides are shown in the words the curve under it makes true: left and
    /// right along an open curve, inside and outside round a loop — and round a loop the
    /// button marked Inside stores whichever of left and right the field says is the inside,
    /// so the word is true of the wall whichever way the loop happens to run.
    #[test]
    fn a_walls_sides_are_offered_in_the_words_its_curve_makes_true() {
        use crate::{across, side_choices};
        use fieldform_scene::WallSide::{Centre, Left, Right};
        let offered = |inside| side_choices(inside).map(|(side, word, _)| (side, word));

        assert_eq!(offered(None), [(Centre, "Centre"), (Left, "Left"), (Right, "Right")]);
        assert_eq!(offered(Some(Left)), [(Centre, "Centre"), (Left, "Inside"), (Right, "Outside")]);
        assert_eq!(offered(Some(Right)), [(Centre, "Centre"), (Right, "Inside"), (Left, "Outside")]);
        // Every row offers each stored side once, so no wall is shown with nothing lit.
        for inside in [None, Some(Left), Some(Right)] {
            let mut sides = side_choices(inside).map(|(side, _, _)| side as u8);
            sides.sort_unstable();
            assert_eq!(sides, [Centre as u8, Left as u8, Right as u8]);
        }
        assert_eq!((across(Left), across(Right), across(Centre)), (Right, Left, Centre));
    }

    /// The console's words for a wall: `--side inside` means something only round a loop and
    /// is refused along an open curve rather than guessed at, left and right are taken as
    /// written wherever they are said, and a thickness is the number given or, given none, the
    /// last one.
    #[test]
    fn a_walls_console_words_mean_what_they_say_or_are_refused() {
        use crate::{ends_from_word, side_from_word, wall_thickness_asked, wall_wanted, Value};
        use fieldform_ops::Args;
        use fieldform_scene::{WallEnds, WallSide};

        assert_eq!(side_from_word("left", None), Ok(WallSide::Left));
        assert_eq!(side_from_word("Centre", None), Ok(WallSide::Centre));
        assert_eq!(side_from_word("center", Some(WallSide::Right)), Ok(WallSide::Centre));
        assert_eq!(side_from_word("inside", Some(WallSide::Right)), Ok(WallSide::Right));
        assert_eq!(side_from_word("outside", Some(WallSide::Right)), Ok(WallSide::Left));
        // Written as left on a loop, it is left: what a script says is what the file holds.
        assert_eq!(side_from_word("left", Some(WallSide::Right)), Ok(WallSide::Left));
        assert!(side_from_word("inside", None).is_err(), "an open curve has no inside");
        assert!(side_from_word("up", None).is_err());

        assert_eq!(ends_from_word("round"), Ok(WallEnds::Round));
        assert_eq!(ends_from_word("Square"), Ok(WallEnds::Square));
        assert!(ends_from_word("pointed").is_err());

        let with = |key: &str, value: Value| Args::new().with(key, value);
        // Asked for by `wall`, or by a side or an end, which mean nothing without one.
        assert!(wall_wanted(&with("wall", Value::Float(2.0))));
        assert!(wall_wanted(&with("wall", Value::Bool(true))));
        assert!(wall_wanted(&with("side", Value::Str("left".into()))));
        assert!(wall_wanted(&with("ends", Value::Str("round".into()))));
        assert!(!wall_wanted(&Args::new()), "a plain extrude is still a profile's");
        assert!(!wall_wanted(&with("wall", Value::Bool(false))));

        assert_eq!(wall_thickness_asked(&with("wall", Value::Int(3)), 2.0), Ok(3.0));
        assert_eq!(wall_thickness_asked(&with("wall", Value::Float(1.5)), 2.0), Ok(1.5));
        assert_eq!(wall_thickness_asked(&with("wall", Value::Bool(true)), 2.0), Ok(2.0));
        assert_eq!(wall_thickness_asked(&with("side", Value::Str("left".into())), 2.5), Ok(2.5));
        assert!(wall_thickness_asked(&with("wall", Value::Float(0.0)), 2.0).is_err());
        assert!(wall_thickness_asked(&with("wall", Value::Float(-2.0)), 2.0).is_err());
        assert!(wall_thickness_asked(&with("wall", Value::Str("thick".into())), 2.0).is_err());
    }

    /// A grid number sits inside the viewport, on the part of its line that is visible.
    ///
    /// The anchor is what makes the numbers readable at all: the obvious alternative -- writing
    /// each one where its line crosses the plane's own axis -- puts every number off screen the
    /// moment somebody zooms in on a detail, which is when they most want to know the size.
    #[test]
    fn a_grid_number_lands_inside_the_viewport_or_not_at_all() {
        use crate::{egui, grid_label_at, GRID_LABEL_INSET};
        let rect = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(400.0, 300.0));

        // A line crossing the viewport top to bottom, its ends well outside it.
        let at = grid_label_at(rect, egui::pos2(120.0, -500.0), egui::pos2(120.0, 900.0))
            .expect("a line through the viewport has somewhere to write");
        assert!(rect.contains(at), "{at:?} is outside the viewport");
        assert!(
            at.y >= rect.bottom() - GRID_LABEL_INSET - 1e-3,
            "a vertical line labels at the bottom, where a drawing's margin is",
        );

        // One that misses entirely writes nothing rather than being clamped into view, which
        // would put a number on a line that is not there.
        assert!(grid_label_at(rect, egui::pos2(-50.0, -50.0), egui::pos2(-50.0, 400.0)).is_none());

        // And one running left to right labels at the left, by the same rule.
        let at = grid_label_at(rect, egui::pos2(-900.0, 80.0), egui::pos2(900.0, 80.0))
            .expect("a horizontal line through the viewport has somewhere to write");
        assert!(rect.contains(at));
        assert!(at.x <= rect.left() + GRID_LABEL_INSET + 1e-3, "{at:?} is not at the left");
    }

    /// The grid pitch is the finest decade of millimetres whose cell is wide enough to aim at.
    ///
    /// Both halves matter: a cell narrower than the minimum is a target nobody can hit, and a
    /// pitch coarser than it needs to be throws away precision the zoom had earned.
    #[test]
    fn the_grid_pitch_is_the_finest_readable_decade() {
        use crate::{grid_pitch, GRID_FINEST_MM, GRID_MIN_CELL_PX};
        let mut px_per_mm = 0.005;
        while px_per_mm < 2000.0 {
            let pitch = grid_pitch(px_per_mm);
            let cell = pitch * px_per_mm;
            assert!(
                cell >= GRID_MIN_CELL_PX * (1.0 - 1e-9),
                "at {px_per_mm} px/mm a {pitch} mm pitch is a {cell} px cell"
            );
            // A power of ten, and never finer than the finest there is.
            assert!(pitch >= GRID_FINEST_MM * (1.0 - 1e-9), "{pitch} mm is finer than the finest");
            let exponent = pitch.log10();
            assert!(
                (exponent - exponent.round()).abs() < 1e-9,
                "{pitch} mm is not a decade of millimetres"
            );
            // And the decade below it would have been too narrow to aim at, unless this is
            // already the finest one.
            let finer = pitch / 10.0;
            assert!(
                finer < GRID_FINEST_MM
                    || finer * px_per_mm < GRID_MIN_CELL_PX * (1.0 + 1e-9),
                "at {px_per_mm} px/mm, {finer} mm would still have been readable"
            );
            px_per_mm *= 1.37;
        }
    }

    /// A snapped point sits on an intersection of the plane's grid and stays where it was
    /// across the plane — the grid rounds in the plane, never out of it.
    #[test]
    fn a_grid_snap_lands_on_an_intersection_and_stays_in_the_plane() {
        use crate::{drawing_frame_of, snap_to_grid, Axis, DrawingPlane};
        use nalgebra::{Point3, Vector3};
        let doc = fieldform_scene::Document::new();
        let graph = fieldform_scene::Graph::empty();
        let numbers = |r: &fieldform_scene::ParamRef| match r {
            fieldform_scene::ParamRef::Literal(v) => Some(*v),
            _ => None,
        };
        let plane = DrawingPlane::World { normal: Axis::X, offset_mm: 25.0 };
        let frame = drawing_frame_of(plane).expect("a world plane names a frame");
        let (translation, rotation) =
            fieldform_field::resolve_frame(&doc, &graph, &frame, &numbers)
                .expect("a typed frame resolves");
        let pitch = 5.0;
        for point in [
            Point3::new(25.0, 3.2, -7.9),
            Point3::new(25.0, -12.6, 41.1),
            // Off the plane by 4 mm along its normal, as a Ctrl snap to another curve leaves it.
            Point3::new(29.0, 0.4, 0.4),
        ] {
            let snapped = snap_to_grid(point, translation, rotation, pitch);
            let local = rotation.inverse() * (snapped.coords - translation);
            for c in [local.x, local.y] {
                let steps = c / pitch;
                assert!((steps - steps.round()).abs() < 1e-9, "{snapped:?} is between lines");
            }
            let moved = snapped - point;
            assert!(
                moved.dot(&Vector3::x()).abs() < 1e-9,
                "{point:?} moved along the plane's normal to {snapped:?}"
            );
            let before = rotation.inverse() * (point.coords - translation);
            assert!((local.x - before.x).abs() <= pitch * 0.5 + 1e-9);
            assert!((local.y - before.y).abs() <= pitch * 0.5 + 1e-9);
        }
    }

    /// The cube's buttons do not overlap one another or the cube, and stay inside the viewport
    /// at the smallest size the cube is shown at.
    ///
    /// The layout is three rectangles positioned by hand, and a fourth was just added beside the
    /// third; this is what says the row still fits — and that the viewport's own "was that press
    /// on the cube" test, which takes their union, covers each of them.
    #[test]
    fn the_view_cube_controls_do_not_overlap_or_leave_the_viewport() {
        use crate::{
            view_cube_controls_rect, view_cube_grid_rect, view_cube_home_rect,
            view_cube_projection_rect, view_cube_rect, VIEWCUBE_MIN_VIEWPORT,
        };
        use eframe::egui;
        let viewport = egui::Rect::from_min_size(
            egui::pos2(100.0, 40.0),
            egui::vec2(VIEWCUBE_MIN_VIEWPORT, VIEWCUBE_MIN_VIEWPORT),
        );
        let cube = view_cube_rect(viewport);
        let controls = [
            ("cube", cube),
            ("home", view_cube_home_rect(cube)),
            ("projection", view_cube_projection_rect(cube)),
            ("grid", view_cube_grid_rect(cube)),
        ];
        let all = view_cube_controls_rect(viewport);
        for (i, (name, rect)) in controls.iter().enumerate() {
            assert!(viewport.contains_rect(*rect), "{name} leaves the viewport: {rect:?}");
            assert!(all.contains_rect(*rect), "{name} is outside the controls rect");
            for (other, other_rect) in &controls[i + 1..] {
                assert!(
                    !rect.intersects(*other_rect),
                    "{name} overlaps {other}: {rect:?} vs {other_rect:?}"
                );
            }
        }
    }

    /// The two halves of naming a plane agree with each other.
    #[test]
    fn a_planes_name_and_its_normal_are_inverses() {
        use crate::{plane_normal, plane_word, Axis};
        for normal in [Axis::X, Axis::Y, Axis::Z] {
            let word = plane_word(normal);
            assert_eq!(
                plane_normal(word),
                Some(normal),
                "{word} did not read back as {normal:?}"
            );
            // Typed the other way round, since nobody agrees which order to say them in.
            let flipped: String = word.chars().rev().collect();
            assert_eq!(plane_normal(&flipped), Some(normal));
        }
        assert_eq!(plane_normal("xq"), None);
    }

    /// **A drawn variable is counted where it is drawn**, which random ids make the whole
    /// question.
    ///
    /// A named block that something consumes can be drawn beside the bodies, and it is drawn as
    /// a *copy* — a body is a root and that block is not one. The copy gets a fresh `NodeId`,
    /// and ids here are **random rather than sequential** (`id.rs` says why), while `roots()`
    /// sorts by that id. So a copy lands anywhere in the drawn list, and the obvious-looking
    /// assumption — bodies first, then the variables after them — is false at random.
    ///
    /// That matters because the index of a body in the drawn list is the number the shader is
    /// given to highlight it. Counted against a different order, the wrong body lights up, and
    /// only sometimes. This asserts the two readings are one: whatever `drawn_bodies` says is
    /// what `display_graph` contains, in that order.
    #[test]
    fn the_drawn_order_is_read_once_rather_than_recounted() {
        use crate::{BooleanOp, Graph, Node, NodeId, ParamRef};
        let lit = ParamRef::Literal;

        // Fifty attempts, because the failure this guards is a random id landing in front of a
        // body rather than behind it. One run proves nothing.
        for _ in 0..50 {
            let mut graph = Graph::empty();
            let hub = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                radius: lit(20.0),
            });
            let block = graph.add(Node::Cuboid { edge: Default::default(),
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                size: [lit(30.0); 3],
                turn: fieldform_scene::no_turn(),
            });
            let body =
                graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![block.into(), hub.into()] });

            // The drawn list, built the way the app builds it: the bodies in the document's
            // order, plus a *stand-in* for the consumed block, then read back in the caller's
            // own terms. Through `folded_bodies`, because that is what `drawn_bodies` asks and what
            // the shader's own fold asks -- a hand-rolled `roots()` here would be testing an
            // arrangement the app no longer has.
            let mut display = graph.clone();
            let copy = display.add_stand_in(hub).expect("the sphere is in the graph");
            let mut stand_ins = std::collections::HashMap::new();
            stand_ins.insert(copy, hub);
            let mut doc = fieldform_scene::Document::new();
            for (at, id) in [(1u64, hub), (2, block), (3, body)] {
                doc.created.insert(id, 1_700_000_000_000_000 + at);
            }
            let drawn: Vec<NodeId> = doc
                .folded_bodies(&display)
                .into_iter()
                .map(|id| stand_ins.get(&id).copied().unwrap_or(id))
                .collect();

            // Both are drawn, and each is at the index its own copy occupies -- not at an index
            // guessed from the order they were added in.
            assert_eq!(drawn.len(), 2, "the body and the shown variable");
            assert!(drawn.contains(&body) && drawn.contains(&hub));
            // And the stand-in sorts where the block it stands for does, so the shown ingredient
            // comes *before* the body built out of it rather than at one end of the list.
            assert_eq!(drawn, vec![hub, body], "the block it stands for is older than the body");
            let roots = doc.folded_bodies(&display);
            for (index, drawn_id) in drawn.iter().enumerate() {
                let on_screen = roots[index];
                let stands_for = stand_ins.get(&on_screen).copied().unwrap_or(on_screen);
                assert_eq!(
                    *drawn_id, stands_for,
                    "the drawn list and what is actually drawn must be the same order"
                );
            }
        }
    }

    /// **Nothing that already consumes a block is offered to it**, at any depth.
    ///
    /// Choosing one would make a loop — this block feeding the thing that feeds it — and a
    /// cyclic graph does not compile, so the result is a part that has stopped drawing and an
    /// error naming a node rather than the click that caused it. Worse, it saves, so the file
    /// opens to the same error.
    ///
    /// It mattered little while a boolean was the only block offering a choice. Every block that
    /// takes a body offers one now, which is the owner's rule and makes this reachable in one
    /// click from six different panels.
    #[test]
    fn a_block_is_never_offered_something_that_consumes_it() {
        use crate::{operand_choices_in, BooleanOp, Graph, Node, ParamRef};
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let box_ = graph.add(Node::Cuboid { edge: Default::default(), centre: fieldform_scene::Place::At([lit(0.0); 3]), size: [lit(20.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        // A chain above the sphere: shell consumes it, the boolean consumes the shell.
        let shell = graph.add(Node::Shell { input: ball.into(), thickness: lit(1.0) });
        let joined =
        graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![shell.into(), box_.into()] });

        // The only thing that can keep a block out of this list is that taking it would close
        // a loop. Every stamp the same: these assert the *filter*, and a tie leaves the id order
        // the list had before stamps existed.
        let offered = operand_choices_in(&graph, ball, |_| 0);
        assert!(
            !offered.contains(&shell),
            "the block directly consuming it would be an immediate loop"
        );
        assert!(
            !offered.contains(&joined),
            "and one further up is the same loop with a step in it"
        );
        assert!(offered.contains(&box_), "something off to the side is still fine");

        // And the guard is about direction, not about being related: from the top of the chain,
        // everything below is still a legitimate choice.
        let from_above = operand_choices_in(&graph, joined, |_| 0);
        assert!(from_above.contains(&ball) && from_above.contains(&shell));
    }

    /// Every block is offered to every other, named or not. Only a loop is refused.
    ///
    /// **This reverses what the list used to do, on the owner's instruction**: "both un-named and
    /// named objects should be available for boolean". It used to withhold a block that something
    /// already consumed until it had a name, on the reading that a name is what declares a block
    /// reusable -- and the test that stood here said the negative half was the load-bearing one,
    /// because dropping the filter "would fill every dropdown with the insides of blocks nobody
    /// meant to share".
    ///
    /// That cost is real and is now accepted: a dropdown lists the interior of every block, and
    /// what makes it usable is the filter box the dropdown already has for the same reason a part
    /// with a hundred bodies needed one. What the old rule got wrong is that it answered "can
    /// this be taken" with a fact about *naming*, which is how a person tells two rows apart --
    /// two questions, one flag.
    #[test]
    fn every_block_is_offered_to_every_other_and_only_a_loop_is_refused() {
        use crate::{operand_choices_in, BooleanOp, Graph, Node, ParamRef};
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let hub = graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(20.0) });
        let plain = graph.add(Node::Sphere { centre: fieldform_scene::Place::At([lit(0.0); 3]), radius: lit(5.0) });
        let block = graph.add(Node::Cuboid { edge: Default::default(), centre: fieldform_scene::Place::At([lit(0.0); 3]), size: [lit(30.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        // One boolean consumes both spheres, so neither is a body any more.
        let first = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![hub.into(), plain.into()] });
        let other = graph.add(Node::Boolean { op: BooleanOp::Difference, inputs: vec![block.into(), first.into()] });

        // Nothing is named, and everything is offered anyway -- including the two spheres
        // buried inside the first boolean, which is the case the old rule withheld.
        let offered = operand_choices_in(&graph, other, |_| 0);
        assert!(offered.contains(&hub), "an unnamed inner node is offered");
        assert!(offered.contains(&plain), "and so is its neighbour");
        assert!(offered.contains(&block) && offered.contains(&first), "as are its own operands");

        // And never itself: a node is not its own operand.
        assert!(!offered.contains(&other));

        // A ramp is the one block still withheld, and not for want of a name: it is a number
        // rather than a field, which `Node::is_body` is what says.
        let ramp = graph.add(Node::Ramp {
            input: fieldform_scene::Operand::empty(),
            shape: fieldform_scene::RampShape::Linear,
            in_min: lit(0.0),
            in_max: lit(1.0),
            out_min: lit(0.0),
            out_max: lit(1.0),
        });
        assert!(!operand_choices_in(&graph, other, |_| 0).contains(&ramp));
    }

    /// The cage's own geometry hides what is behind it.
    ///
    /// **Facing was not enough**, which is what backlog item 22 kept open. A vertex on a face
    /// pointing at the camera is on the near side of *that face* and can still sit behind a
    /// different part of a cage that folds over itself. This builds the simplest thing that
    /// folds — a cube, whose far face points away but whose *near* face is squarely in the way
    /// of it — and checks the far side is unreachable through the near one.
    #[test]
    fn a_cage_hides_what_is_behind_its_own_faces() {
        use crate::occluded;
        let cage = fieldform_cage::Cage::cube_of(100.0);
        let mesh = crate::cage_mesh::CageMesh::from_cage(&cage);
        let eye = nalgebra::Point3::new(0.0f32, 0.0, 400.0);

        // Straight through the middle of the near face at the middle of the far one. Nothing on
        // the far side should be reachable along it.
        let tolerance = 0.1f32;
        let far = nalgebra::Point3::new(0.0f32, 0.0, -50.0);
        assert!(
            occluded(&mesh, eye, far, tolerance),
            "the near face should hide the far one"
        );

        // And the near face itself is not hidden by anything, including by itself — the ray
        // reaches it at exactly its own distance, which the tolerance is there to allow.
        let near = nalgebra::Point3::new(0.0f32, 0.0, 50.0);
        assert!(
            !occluded(&mesh, eye, near, tolerance),
            "a face in plain view should not be hidden"
        );

        // A corner of the near face, where several faces meet: still in view.
        let corner = nalgebra::Point3::new(50.0f32, 50.0, 50.0);
        assert!(
            !occluded(&mesh, eye, corner, tolerance),
            "a corner of the near face should not hide itself"
        );
    }
    /// **Four faces picked together move as one piece.** Shift adds and takes away, and what
    /// drags is the union of the parts: two faces sharing an edge move their shared corners once,
    /// not twice as far, which is what averaging the faces' own centres would have done.
    #[test]
    fn shift_adds_to_the_selection_and_the_parts_move_as_one() {
        use crate::{Picked, Selection};
        let cage = fieldform_cage::Cage::cube_of(10.0);
        let mesh: &fieldform_cage::QuadMesh = &cage;
        let placement = nalgebra::Similarity3::identity();

        let faces: Vec<_> = mesh.face_ids().collect();
        let mut picked = Picked::one(Selection::Face(faces[0]));
        assert!(picked.contains(Selection::Face(faces[0])));
        assert_eq!(picked.vertices(mesh).len(), 4);

        // A second face beside the first: eight corners between them, and on a cube every pair
        // of faces that meets shares two, so the union is smaller than the sum.
        picked.toggle(Selection::Face(faces[1]));
        let union = picked.vertices(mesh);
        let mut once = union.clone();
        once.sort_by_key(|v| v.index());
        once.dedup();
        assert_eq!(once.len(), union.len(), "a shared corner appears once");
        assert_eq!(picked.parts().len(), 2);

        // Shift again on the same face takes it away, which is what makes a mis-click cheap.
        picked.toggle(Selection::Face(faces[1]));
        assert_eq!(picked.parts().len(), 1);
        assert!(!picked.contains(Selection::Face(faces[1])));

        // The pivot is the middle of the points that will move.
        picked.toggle(Selection::Face(faces[1]));
        let centre = picked.centroid(mesh, &placement).expect("something picked");
        let verts = picked.vertices(mesh);
        let mean = verts.iter().fold(nalgebra::Vector3::zeros(), |acc, v| {
            acc + mesh.vertex_position(*v).cast::<f32>().coords
        }) / verts.len() as f32;
        assert!((centre.coords - mean).norm() < 1e-5);

        // Two faces have two normals, so the handles stay world-aligned; one face turns them.
        assert!(picked.normal(mesh, &placement).is_none());
        assert!(Picked::one(Selection::Face(faces[0])).normal(mesh, &placement).is_some());

        // And a lone point is the one case with nothing to turn or scale about.
        assert!(Picked::one(Selection::Vertex(mesh.vertex_ids().next().unwrap())).is_single_vertex());
        assert!(!picked.is_single_vertex());

        // A part naming geometry that has gone is dropped; the rest of the selection stays.
        let mut mixed = Picked::one(Selection::Vertex(fieldform_cage::VertexId::from_index(0)));
        mixed.toggle(Selection::Vertex(fieldform_cage::VertexId::from_index(99)));
        mixed.keep_valid_for(mesh.vertex_count(), mesh.face_count());
        assert_eq!(mixed.parts().len(), 1);
    }

    /// **A surface's points can be picked from underneath.** The cage's picking offers only the
    /// points on faces turned toward the camera, which is right for a closed solid and wrong for
    /// the open sheet a conformal lattice follows: seen from below, every face of a panel turns
    /// away, and with the cage's rule no point on the underside could be clicked. Checked for a
    /// point, and for a face, which the raycast has to find from its back.
    #[test]
    fn a_surface_is_picked_from_either_side() {
        use crate::{pick, OrbitCamera, Selection};
        use eframe::egui;
        let n = 3usize;
        let mut positions = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                positions.push(nalgebra::Point3::new(i as f64 * 10.0, j as f64 * 10.0, 0.0));
            }
        }
        let at = |i: usize, j: usize| (j * (n + 1) + i) as u32;
        let mut faces = Vec::new();
        for j in 0..n {
            for i in 0..n {
                faces.push(vec![at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        let panel = fieldform_cage::QuadMesh::from_faces(positions, &faces).expect("a panel");
        let mesh = crate::cage_mesh::CageMesh::from_cage(&panel);
        assert!(mesh.two_sided, "an open surface is drawn from both sides");
        let placement = nalgebra::Similarity3::identity();
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        let aspect = rect.width() / rect.height();

        for below in [false, true] {
            let mut camera = OrbitCamera::default();
            camera.target = nalgebra::Point3::new(15.0, 15.0, 0.0);
            camera.eye = nalgebra::Point3::new(12.0, 9.0, if below { -80.0 } else { 80.0 });
            camera.up = nalgebra::Vector3::y();
            let side = if below { "below" } else { "above" };

            let point = nalgebra::Point3::new(10.0f32, 20.0, 0.0);
            let screen = camera.project_to_screen(aspect, rect, point).expect("in view");
            let picked = pick(&panel, &placement, &mesh, &camera, aspect, rect, screen);
            assert!(
                matches!(picked, Selection::Vertex(v) if v.index() == at(1, 2) as usize),
                "the point at (10, 20) from {side}"
            );

            let middle = nalgebra::Point3::new(15.0f32, 15.0, 0.0);
            let screen = camera.project_to_screen(aspect, rect, middle).expect("in view");
            let picked = pick(&panel, &placement, &mesh, &camera, aspect, rect, screen);
            assert!(
                matches!(picked, Selection::Face(f) if f.index() == 4),
                "the middle face from {side}"
            );
        }
    }

    /// The splash is embedded, so a missing or corrupt file is a build that starts and then
    /// shows nothing rather than one that fails to build. This is the guard for that.
    #[test]
    fn the_splash_screen_decodes() {
        const BYTES: &[u8] = include_bytes!("../assets/splash-screen.png");
        let image = image::load_from_memory(BYTES).expect("the embedded splash is a valid PNG");
        assert!(image.width() > 0 && image.height() > 0);
    }

    /// The window icon is embedded, so a corrupt one would put the system's icon back without
    /// failing the build. It is also the one place the emblem's colours are written down in code:
    /// white on a pure black tile, and big enough to be sharp on a high-DPI taskbar.
    ///
    /// The white fraction is measured, not asserted: the mark is 12.3% of the tile in
    /// `window-icon.svg`, so a band of 8 to 18% catches an empty tile and a flooded one without
    /// pinning the exact figure.
    #[test]
    fn the_window_icon_is_a_white_mark_on_a_black_tile() {
        let icon = super::window_icon();
        assert!(icon.width >= 256 && icon.width == icon.height, "{}x{}", icon.width, icon.height);
        assert_eq!(icon.rgba.len(), (icon.width * icon.height * 4) as usize);
        let px = |x: u32, y: u32| {
            let i = ((y * icon.width + x) * 4) as usize;
            [icon.rgba[i], icon.rgba[i + 1], icon.rgba[i + 2], icon.rgba[i + 3]]
        };
        for (x, y) in [(0, 0), (icon.width - 1, 0), (0, icon.height - 1), (icon.width - 1, icon.height - 1)] {
            assert_eq!(px(x, y), [0, 0, 0, 255], "the corner at {x},{y} is opaque black");
        }
        let white = icon.rgba.chunks(4).filter(|p| p[0] >= 128).count() as f64;
        let share = white / (icon.width * icon.height) as f64;
        assert!((0.08..0.18).contains(&share), "white share {share:.3}");
        assert!(icon.rgba.chunks(4).all(|p| p[3] == 255), "every pixel is opaque");
    }

    use super::{
        console_commands, single_span_bezier, view_cube_cells, view_cube_faces, CubeRegion,
        OrbitCamera, PrimitiveKind, Projection, CURVE_DEGREES,
    };
    use std::collections::HashSet;

    /// An RMB dolly has to zoom under *both* projections, and the two measure it differently.
    ///
    /// The dolly slides the eye toward the point of interest and re-plants the target a fixed
    /// distance ahead of it, which is right for perspective — the image comes from where the
    /// eye is. Under orthographic it did nothing at all: the rays are parallel, the near plane
    /// sits behind the eye, and what actually sets the zoom is `ortho_half_height`, which reads
    /// the eye-to-target distance — precisely the quantity the dolly was holding constant.
    ///
    /// So this asserts the visible quantity rather than the camera's coordinates, because the
    /// eye did move and the picture still did not change.
    #[test]
    fn an_rmb_dolly_zooms_under_both_projections() {
        for projection in [Projection::Perspective, Projection::Orthographic] {
            let mut camera = OrbitCamera::default();
            camera.projection = projection;
            let at = camera.target;
            let before = camera.view_half_height(at);

            camera.begin_dolly();
            // Past the axis-lock threshold, downward, which the lock reads as zoom.
            camera.dolly_drag(super::egui::Vec2::new(0.0, 40.0));
            let after = camera.view_half_height(at);

            assert!(
                after > before * 1.05,
                "{projection:?}: dragging down should widen the view, \
                 but half-height went {before} -> {after}"
            );

            // And back the other way, from a fresh drag.
            camera.end_dolly();
            let before = camera.view_half_height(at);
            camera.begin_dolly();
            camera.dolly_drag(super::egui::Vec2::new(0.0, -40.0));
            let after = camera.view_half_height(at);
            assert!(
                after < before * 0.95,
                "{projection:?}: dragging up should narrow the view, \
                 but half-height went {before} -> {after}"
            );
        }
    }

    /// Every tool the Create menu offers has to be typeable too.
    ///
    /// The console's list used to be written out by hand beside the menu's, and the plane and
    /// the point landed in the menu alone — reachable by mouse and not by keyboard, which is
    /// half an app. The list is now derived from the same place the menu is; this is what says
    /// so, and what would fail if anyone wrote a second one.
    #[test]
    fn every_primitive_is_reachable_from_the_console() {
        let commands = console_commands();
        for kind in PrimitiveKind::ALL {
            let word = kind.word();
            let found = commands.iter().find(|(w, _)| *w == word);
            let (_, op) = found
                .unwrap_or_else(|| panic!("{:?} has no console word {word:?}", kind.label()));
            assert_eq!(*op, kind.command(), "{word} dispatches to the wrong operation");
        }
        // And the menu offers them all, in this order and under their own labels: the registry
        // lists them by hand now, so this is what keeps its list and this one the same.
        let primitives: Vec<&str> = crate::commands::COMMANDS
            .iter()
            .filter(|c| c.category == ["Create", "Primitives"])
            .filter(|c| PrimitiveKind::ALL.iter().any(|k| k.command() == c.op))
            .map(|c| c.display_name)
            .collect();
        let labels: Vec<&str> = PrimitiveKind::ALL.iter().map(|k| k.label()).collect();
        assert_eq!(primitives, labels, "Create › Primitives offers every kind, in order");
    }

    /// **Which way each arrow moves the depth.** Side one's face moves the depth as far as it
    /// moves; a symmetric extrude's face stands at half the depth, so it moves it twice as far; and
    /// side two's face moves *against* the sweep as its depth grows. Ctrl rounds to whole
    /// millimetres, after the arithmetic and not before it.
    #[test]
    fn a_depth_arrow_changes_the_depth_by_the_amount_its_face_moves() {
        use super::dragged_depth;
        assert_eq!(dragged_depth(0, false, 20.0, 5.0, false), 25.0);
        assert_eq!(dragged_depth(0, false, 20.0, -5.0, false), 15.0, "pulled in, it shortens");
        assert_eq!(dragged_depth(0, true, 20.0, 5.0, false), 30.0, "half each way: twice as far");
        assert_eq!(dragged_depth(1, false, 8.0, -3.0, false), 11.0, "side two grows against the sweep");
        assert_eq!(dragged_depth(1, false, 8.0, 3.0, false), 5.0);
        assert_eq!(dragged_depth(0, false, 20.0, 5.4, true), 25.0);
        assert_eq!(dragged_depth(0, false, 20.0, 5.6, true), 26.0);
        assert_eq!(dragged_depth(0, true, 20.0, 0.3, true), 21.0, "rounded after doubling");
    }

    /// An arrow stands over the extrude only where nothing above it moves the solid: a boolean or a
    /// blend leaves its face where it is, and a placement, a pattern or a mirror does not.
    #[test]
    fn a_depth_arrow_is_only_offered_where_the_extrude_is_drawn_where_it_is() {
        use super::drawn_where_it_is;
        use fieldform_scene::{BooleanOp, Fillet, Frame, Graph, Node, NodeId, ParamRef, PatternKind};
        let lit = ParamRef::Literal;
        let build = |above: &dyn Fn(&mut Graph, NodeId) -> NodeId| {
            let mut graph = Graph::empty();
            let body = graph.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(0.0); 3]),
                radius: lit(5.0),
            });
            let other = above(&mut graph, body);
            (graph, body, other)
        };
        let (graph, body, _) = build(&|g, b| {
            let c = g.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(9.0); 3]),
                radius: lit(5.0),
            });
            g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![b.into(), c.into()] })
        });
        assert!(drawn_where_it_is(&graph, body), "under a union it is where it was");
        let (graph, body, _) = build(&|g, b| {
            let c = g.add(Node::Sphere {
                centre: fieldform_scene::Place::At([lit(9.0); 3]),
                radius: lit(5.0),
            });
            g.add(Node::Blend { op: BooleanOp::Union, a: b.into(), b: c.into(), fillet: Fillet::of(lit(1.0)) })
        });
        assert!(drawn_where_it_is(&graph, body), "and under a blend");
        let (graph, body, _) = build(&|g, b| {
            g.add(Node::Pattern {
                input: b.into(),
                count: lit(3.0),
                kind: PatternKind::Radial { frame: Frame::default() },
                join: true,
                fillet: Fillet::hard(),
                options: Default::default(),
            })
        });
        assert!(!drawn_where_it_is(&graph, body), "but not under a pattern");
        let (graph, body, _) = build(&|g, b| {
            let inner = g.add(Node::Offset { input: b.into(), distance: lit(1.0) });
            g.add(Node::Mirror { input: inner.into(), frame: Frame::default(), join: true })
        });
        assert!(!drawn_where_it_is(&graph, body), "nor under a mirror, however far up");
    }

    /// What a console `pattern` can ask beyond the count: centring, a whole run, an arc and copies
    /// to leave out — and a typo in the skip list is refused by name, because dropping the word
    /// would leave out the wrong copies.
    #[test]
    fn a_console_pattern_reads_its_options_and_refuses_a_skip_list_with_a_word_in_it() {
        use super::pattern_options_from;
        use fieldform_ops::{Args, Value};
        use fieldform_scene::{ParamRef, Spacing};

        let plain = pattern_options_from(&Args::new()).expect("no options is fine");
        assert!(plain.is_plain(), "nothing asked is nothing changed: {plain:?}");

        let asked = Args::new()
            .with("symmetric", Value::Bool(true))
            .with("extent", Value::Bool(true))
            .with("angle", Value::Int(180))
            .with("skip", Value::Str("5, 2,5".into()));
        let options = pattern_options_from(&asked).expect("these are all fine");
        assert!(options.symmetric);
        assert_eq!(options.spacing, Spacing::Extent);
        assert_eq!(options.angle, ParamRef::Literal(180.0));
        assert_eq!(options.skip, vec![2, 5], "sorted, and a repeated number once");

        let one = pattern_options_from(&Args::new().with("skip", Value::Int(3))).unwrap();
        assert_eq!(one.skip, vec![3], "`skip=3` parses as a number, not a list");

        // `rows=` makes a grid along Y, by `step2=` millimetres or fifty.
        let grid = pattern_options_from(
            &Args::new().with("rows", Value::Int(4)).with("step2", Value::Float(60.0)),
        )
        .unwrap();
        let second = grid.second.expect("rows= asks for a second direction");
        assert_eq!(second.count, ParamRef::Literal(4.0));
        assert_eq!(second.step[1], ParamRef::Literal(60.0));
        let fifty = pattern_options_from(&Args::new().with("rows", Value::Int(2))).unwrap();
        assert_eq!(fifty.second.expect("a second direction").step[1], ParamRef::Literal(50.0));

        for bad in ["2,x", "-1", "2.5"] {
            let args = Args::new().with("skip", Value::Str(bad.into()));
            assert!(pattern_options_from(&args).is_err(), "{bad:?} is not a list of copies");
        }
        assert!(pattern_options_from(&Args::new().with("skip", Value::Int(-4))).is_err());
    }

    /// **What Create makes of every solid has to be a solid.** A kind whose default numbers
    /// compile to nothing — a skewed box whose three edges happen to lie flat, say — would be an
    /// entry that adds a block and draws an empty viewport, found by clicking it. Each kind's
    /// default is compiled and asked whether its own centre is inside; the point and the plane
    /// are not solids and are left out by name.
    #[test]
    fn every_solid_create_offers_arrives_with_volume() {
        use super::DEFAULT_BODY_MM;
        use fieldform_field::{GraphField, ScalarField};
        for kind in &PrimitiveKind::ALL[PrimitiveKind::SOLIDS_START..] {
            let mut graph = fieldform_scene::Graph::empty();
            graph.add(kind.node([fieldform_scene::ParamRef::Literal(0.0); 3], DEFAULT_BODY_MM));
            let field = GraphField::compile_graph(&fieldform_scene::Document::new(), &graph, 1e-4)
                .unwrap_or_else(|e| panic!("{} does not compile: {e}", kind.label()));
            // A torus is hollow at its centre, so ask each kind for a point it surely holds:
            // somewhere in the middle of the ring's tube for that one, the centre for the rest.
            let probe = match kind {
                PrimitiveKind::Torus => {
                    nalgebra::Point3::new(DEFAULT_BODY_MM * 0.35, 0.0, 0.0)
                }
                _ => nalgebra::Point3::origin(),
            };
            assert!(
                field.distance(probe) < 0.0,
                "{} arrives with nothing inside it at {probe:?}",
                kind.label()
            );
        }
    }

    /// A skewed box is offered under one console word with no space in it, since a word with one
    /// would be read as a word and an argument.
    #[test]
    fn the_skewed_box_is_typed_as_one_word() {
        assert_eq!(PrimitiveKind::SkewedBox.word(), "skewedbox");
        assert!(console_commands().iter().any(|(w, op)| w == "skewedbox" && *op == "create.skewbox"));
    }

    /// The name a field is promoted under has to be one an expression can say, and has to be
    /// new.
    #[test]
    fn a_promoted_fields_name_is_something_an_expression_can_say() {
        use super::{unique_parameter_name, Document};
        let mut doc = Document::new();
        assert_eq!(unique_parameter_name(&doc, "Radius:"), "radius");
        assert_eq!(unique_parameter_name(&doc, "Start radius:"), "start_radius");
        assert_eq!(unique_parameter_name(&doc, "Degrees / mm:"), "degrees_mm");
        assert_eq!(unique_parameter_name(&doc, "2nd"), "_2nd");
        assert_eq!(unique_parameter_name(&doc, ":"), "value");
        doc.add_parameter("radius", 1.0);
        assert_eq!(unique_parameter_name(&doc, "Radius:"), "radius_2");
        for label in ["Radius:", "Start radius:", "Degrees / mm:", "2nd", ":"] {
            assert!(Document::is_parameter_name(&unique_parameter_name(&doc, label)), "{label}");
        }
    }

    #[test]
    fn a_readout_drops_the_zeros_nobody_typed() {
        use super::readout;
        assert_eq!(readout(23.0), "23");
        assert_eq!(readout(2.5), "2.5");
        assert_eq!(readout(-0.00001), "0");
        assert_eq!(readout(1.23456), "1.2346");
    }

    /// The three qualities are the table in `DisplayQuality`'s doc, and `Full` is what the
    /// viewport did before the setting existed: every cap at the constant, eight frames.
    #[test]
    fn display_quality_is_one_ladder_and_each_rung_asks_at_least_what_the_last_did() {
        use super::DisplayQuality;
        assert_eq!(DisplayQuality::Fast.still_frames(), 1);
        assert!(!DisplayQuality::Fast.occlusion());
        assert_eq!(DisplayQuality::Fast.drag_pixels(), 480);
        assert_eq!(DisplayQuality::Fast.still_pixels(), 720);
        assert_eq!(DisplayQuality::Fast.cells(), 64);
        assert_eq!(DisplayQuality::Balanced.still_frames(), 4);
        assert_eq!(DisplayQuality::Balanced.still_pixels(), 1080);
        assert_eq!(DisplayQuality::Balanced.cells(), 128);
        assert_eq!(DisplayQuality::Full.still_frames(), 8);
        assert_eq!(DisplayQuality::Full.drag_pixels(), 720);
        assert_eq!(DisplayQuality::Full.still_pixels(), 1440);
        assert!(DisplayQuality::Full.occlusion());
        // The top rung asks for the viewport's own size, and every dial at its finest.
        assert_eq!(DisplayQuality::Enchilada.still_pixels(), 0);
        assert_eq!(DisplayQuality::Enchilada.cells(), 192);
        assert_eq!(DisplayQuality::Enchilada.spine_degrees(), 0.25);
        assert_eq!(DisplayQuality::Enchilada.profile_degrees(), 0.5);
        // Each rung asks for at least what the one below it does.
        for pair in DisplayQuality::ALL.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            assert!(a.cells() <= b.cells() && a.still_frames() <= b.still_frames());
            assert!(a.spine_degrees() >= b.spine_degrees());
            assert!(a.profile_degrees() >= b.profile_degrees());
        }
        for quality in DisplayQuality::ALL {
            assert_eq!(DisplayQuality::from_name(quality.label()), Some(quality));
            assert_eq!(DisplayQuality::from_name(&quality.label().to_uppercase()), Some(quality));
        }
        assert_eq!(DisplayQuality::from_name("fastest"), None);
    }

    /// A rung somebody chose in an earlier session is where the display starts, whatever the
    /// adapter, and it starts as a choice — which is what keeps the first draft's correction off
    /// it, so a slow first frame cannot quietly undo what was picked. With nothing saved the start
    /// is the adapter's guess, `Full` on a discrete card and `Balanced` on anything else, and a
    /// guess stays open to that correction.
    #[test]
    fn a_saved_display_quality_is_where_the_display_starts_and_counts_as_chosen() {
        use super::display_quality::starting_quality;
        use super::DisplayQuality;
        assert_eq!(starting_quality(None, true), (DisplayQuality::Full, false));
        assert_eq!(starting_quality(None, false), (DisplayQuality::Balanced, false));
        for saved in DisplayQuality::ALL {
            for discrete in [true, false] {
                assert_eq!(
                    starting_quality(Some(saved), discrete),
                    (saved, true),
                    "{saved:?} saved, discrete {discrete}"
                );
            }
        }
    }

    /// A settings file reads back as the rung it saved; writing one keeps every other entry, so a
    /// setting kept beside it later is not erased each time this one is chosen; and a file that
    /// says nothing usable — empty, unparseable, a rung by a name no longer offered — reads as
    /// nothing saved, so the display starts on the adapter's guess rather than not at all.
    #[test]
    fn a_saved_display_quality_reads_back_as_the_rung_saved_and_keeps_the_other_settings() {
        use super::display_quality::{saved_quality, with_saved_quality};
        use super::DisplayQuality;
        for quality in DisplayQuality::ALL {
            assert_eq!(saved_quality(&with_saved_quality(None, quality)), Some(quality));
        }
        let before = r#"{ "a_later_setting": [1, 2], "display_quality": "Fast" }"#;
        let after = with_saved_quality(Some(before), DisplayQuality::Full);
        assert_eq!(saved_quality(&after), Some(DisplayQuality::Full));
        let kept: serde_json::Value = serde_json::from_str(&after).unwrap();
        assert_eq!(kept["a_later_setting"], serde_json::json!([1, 2]));
        // A file this build cannot read is replaced rather than standing in the way of a choice.
        let over = with_saved_quality(Some("not json"), DisplayQuality::Fast);
        assert_eq!(saved_quality(&over), Some(DisplayQuality::Fast));
        for unusable in [
            "",
            "not json",
            "[]",
            "{}",
            r#"{ "display_quality": "Fastest" }"#,
            r#"{ "display_quality": 2 }"#,
        ] {
            assert_eq!(saved_quality(unusable), None, "{unusable}");
        }
    }

    /// The row that holds a fillet's number says what the number is, and the bend it implies is
    /// quoted from the continuity rather than typed into the panel.
    #[test]
    fn a_fillets_row_says_reach_where_the_number_is_not_the_radius_and_quotes_the_bend() {
        use super::{fillet_bend_note, fillet_label};
        use fieldform_scene::Continuity;
        assert_eq!(fillet_label(Continuity::G1), "Fillet:");
        assert_eq!(fillet_label(Continuity::G2), "Reach:");
        // At G1 the tightest bend is the number itself between square faces: nothing to add.
        assert_eq!(fillet_bend_note(Continuity::G1, 10.0), None);
        let note = fillet_bend_note(Continuity::G2, 10.0).expect("G2 explains its bend");
        assert!(note.contains("10.00 mm from the other"), "{note}");
        assert!(note.contains("5.61 mm"), "0.5612 of ten: {note}");
        assert!(note.contains("0.56"), "{note}");
        assert!(note.contains("right angle"), "the figure is a corner's, and says so: {note}");
        // A chamfer's number is its leg, said as one, and its note is the bevel's, not a bend.
        assert_eq!(fillet_label(Continuity::Chamfer), "Chamfer:");
        let bevel = fillet_bend_note(Continuity::Chamfer, 2.0).expect("a chamfer says its leg");
        assert!(bevel.contains("2.00 mm along each face"), "{bevel}");
        assert!(!bevel.contains("bend"), "a bevel has no bend to quote: {bevel}");
    }

    /// Everything the panel offers is something the field can evaluate: a shape in the list that
    /// the kernel had no exponent for would be a button that makes a document that draws nothing.
    #[test]
    fn every_blend_shape_the_panel_offers_is_one_the_kernel_evaluates() {
        use super::BLEND_MODES;
        assert_eq!(BLEND_MODES[0], (None, "G0"), "a plain boolean comes first");
        for (mode, label) in &BLEND_MODES[1..] {
            let continuity = mode.expect("every entry after G0 is a fillet");
            assert_eq!(continuity.label(), *label);
            // One is the chamfer -- the family's flat member, `a + b − k` across the join --
            // and below one the form bulges outward past the faces, which is no join at all.
            assert!(continuity.exponent() >= 1.0, "{label}: below 1 is not a join");
        }
        let shapes: Vec<&str> = BLEND_MODES.iter().map(|(_, label)| *label).collect();
        assert_eq!(shapes, ["G0", "G1", "G2", "Chamfer"]);
    }

    /// What a reference field offers while it has the focus: not what it holds, narrowed by what
    /// is typed, capped. The optional current is the new part — a field with nothing yet offers
    /// every choice, which is what the ramp-driver field does before a ramp is picked.
    #[test]
    fn a_reference_fields_suggestions_offer_everything_and_narrow_by_what_is_typed() {
        use super::{suggested, RefChoice};
        let choices = vec![
            RefChoice::named(1u32, "Ramp"),
            RefChoice::named(2, "Ramp 2"),
            RefChoice::named(3, "Slope"),
        ];
        let keys = |found: Vec<(u32, String, Option<&'static str>)>| {
            found.into_iter().map(|(k, _, _)| k).collect::<Vec<_>>()
        };

        assert_eq!(
            keys(suggested(&choices, "", false)),
            [1, 2, 3],
            "the held one is offered too -- a list missing what you are looking for reads as a \
             list that will not let you pick it"
        );
        assert_eq!(keys(suggested(&choices, "", false)), [1, 2, 3], "nothing held, all offered");
        // Typing narrows — but only once the text has left the settled name, or a field showing
        // "Ramp 2" would offer nothing but things containing "ramp 2".
        assert_eq!(keys(suggested(&choices, "slo", true)), [3]);
        assert_eq!(keys(suggested(&choices, "ramp", true)), [1, 2]);
        assert_eq!(
            keys(suggested(&choices, "ramp", false)),
            [1, 2, 3],
            "showing: no narrowing"
        );
        assert!(suggested(&choices, "zzz", true).is_empty());

        // **A refused choice is offered and marked, not left out** -- the owner's ruling: a
        // missing row answers nothing, and why a body is not an option is the question somebody
        // has while looking at the list rather than after submitting a name.
        let mut with_refusal = vec![RefChoice::named(1u32, "Plane"), RefChoice::named(2, "Sphere")];
        with_refusal[0].refusal = Some("a point has no interior");
        let rows = suggested(&with_refusal, "", false);
        assert_eq!(keys(rows.clone()), [1, 2], "both rows are offered");
        assert_eq!(
            rows.iter().map(|(_, _, r)| r.is_some()).collect::<Vec<_>>(),
            [true, false],
            "and the refused one carries its reason, which is what greys it"
        );

        let many: Vec<RefChoice<u32>> =
            (0..25).map(|i| RefChoice::named(i, &format!("Block {i}"))).collect();
        assert_eq!(suggested(&many, "", false).len(), 10, "a long list is cut at ten");
    }

    /// A ramp is addressed in a ramp's words, not a block's — a mistyped name is told "no ramp is
    /// called", not "no block".
    #[test]
    fn a_ramp_field_speaks_of_ramps() {
        use super::RefKind;
        for text in [
            RefKind::Ramp.hint(),
            RefKind::Ramp.why_not(),
            RefKind::Ramp.nothing(),
            RefKind::Ramp.how(),
        ] {
            assert!(text.to_lowercase().contains("ramp"), "{text}");
        }
        assert_ne!(RefKind::Ramp.hint(), RefKind::Block.hint());
    }

    /// **A quick edit after an undo never replaces the entry below the one undone.**
    ///
    /// A drag goes on the history once because each new instant of it replaces the entry on top,
    /// which is right only while the entry on top is the drag's own. An undo moves the top down one,
    /// and the gesture used to outlive it: a same-structure edit inside `GESTURE_WINDOW` took itself
    /// for the drag's next instant and replaced the entry *below* -- an edit nobody undid, gone from
    /// the history, so the undo after it went straight past (refactor plan §3.3, root cause 4).
    ///
    /// Asserted through the two functions `set_part_graph` and Ctrl+Z run, since an `App` needs a
    /// window and a device. A script cannot reach this one, because the runner ends the graph's
    /// gesture between lines; `benchmarks/smoke/gesture-undo.ffc` holds the material's.
    #[test]
    fn a_quick_edit_after_an_undo_never_replaces_the_entry_below() {
        use super::{record_graph_edit, step_history};
        use fieldform_ops::{CommandStack, Document};
        use fieldform_scene::{Graph, Node, ParamRef, Place};

        let lit = ParamRef::Literal;
        let ball = |x: f64| Node::Sphere {
            centre: Place::At([lit(x), lit(0.0), lit(0.0)]),
            radius: lit(10.0),
        };
        let resized = |graph: &Graph, id, r: f64| {
            let mut graph = graph.clone();
            if let Some(Node::Sphere { radius, .. }) = graph.node_mut(id) {
                *radius = lit(r);
            }
            graph
        };
        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        doc.parts.get_mut(&part).expect("one part").graph = Graph::empty();
        let radius_of = |doc: &Document, id| match doc.parts[&part].graph.node(id) {
            Some(Node::Sphere { radius: ParamRef::Literal(r), .. }) => Some(*r),
            _ => None,
        };
        let mut stack = CommandStack::default();
        let mut gesture = None;

        // A ball, which is an entry of its own.
        let mut one = Graph::empty();
        let first = one.add(ball(0.0));
        record_graph_edit(&mut stack, &mut doc, &mut gesture, part, one.clone()).expect("a ball");
        // A second ball and a drag on its radius, which continues the same act: one entry.
        let mut two = one.clone();
        let second = two.add(ball(40.0));
        record_graph_edit(&mut stack, &mut doc, &mut gesture, part, two.clone()).expect("another");
        record_graph_edit(&mut stack, &mut doc, &mut gesture, part, resized(&two, second, 15.0))
            .expect("the drag");

        // Ctrl+Z takes the second ball back, drag and all, and the first ball's radius is changed
        // straight after -- the same structure as what is on screen, and well inside the window.
        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        assert_eq!(radius_of(&doc, second), None, "the second ball is gone");
        record_graph_edit(&mut stack, &mut doc, &mut gesture, part, resized(&one, first, 20.0))
            .expect("the first ball's radius");

        // That edit is its own entry, and the first ball's is still under it to take back.
        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        assert_eq!(radius_of(&doc, first), Some(10.0), "the radius came off on its own");
        assert!(
            step_history(&mut stack, &mut doc, &mut gesture, true).is_some(),
            "the first ball's entry is still on the history",
        );
        assert_eq!(radius_of(&doc, first), None, "and undoing it takes the ball away");
    }

    /// **Undoing a drag that followed a new block leaves no block nested with no reader**
    /// (`block-ownership-spec.md` §4.3, the boolean-then-fillet case).
    ///
    /// A blend adopts its two operands, and dragging its radius straight after continues the
    /// same gesture: each instant replaces the blend's entry with one reaching back to the graph
    /// before the blend, since that is where the gesture started. What an undo puts back of who
    /// owns what has to be from there too. Taken at each instant instead, it held both operands
    /// nested, and the undo left them nested with the only block that read them gone. No parent
    /// is stored since 112, so the invariant `Inside(p)`'s "the readers are exactly `{p}`" became
    /// is what that breaks: a `Nested` block has at least one reader (`notebook-spec.md` §12).
    ///
    /// Through the function `set_part_graph` runs, for the reason the test above gives; the
    /// command's own half is `fieldform-ops`' test of the same case.
    #[test]
    fn undoing_a_drag_that_followed_a_new_block_leaves_no_block_nested_with_no_reader() {
        use super::{record_graph_edit, step_history};
        use fieldform_ops::{CommandStack, Document};
        use fieldform_scene::{BlockKey, BooleanOp, Fillet, Graph, Node, Ownership, ParamRef, Place};

        let lit = ParamRef::Literal;
        let ball = |x: f64| Node::Sphere {
            centre: Place::At([lit(x), lit(0.0), lit(0.0)]),
            radius: lit(10.0),
        };
        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        doc.parts.get_mut(&part).expect("one part").graph = Graph::empty();
        let mut stack = CommandStack::default();
        let mut gesture = None;

        let mut balls = Graph::empty();
        let (a, b) = (balls.add(ball(0.0)), balls.add(ball(15.0)));
        record_graph_edit(&mut stack, &mut doc, &mut gesture, part, balls).expect("two balls");
        let before_blend = (doc.ownership_nodes.clone(), doc.ownership_curves.clone());

        // The blend, an entry of its own, and its radius dragged at once: the same structure, well
        // inside the window, so each instant replaces the blend's entry.
        let mut blended = doc.parts[&part].graph.clone();
        let fillet = Fillet { radius: lit(0.5), ..Fillet::hard() };
        let blend = blended.add(Node::Blend { op: BooleanOp::Union, a: a.into(), b: b.into(), fillet });
        record_graph_edit(&mut stack, &mut doc, &mut gesture, part, blended.clone())
            .expect("the blend");
        assert_eq!(doc.ownership_nodes.get(&a), Some(&Ownership::Nested), "adopted");
        for radius in [1.0, 2.0, 3.0] {
            if let Some(Node::Blend { fillet, .. }) = blended.node_mut(blend) {
                fillet.radius = lit(radius);
            }
            record_graph_edit(&mut stack, &mut doc, &mut gesture, part, blended.clone())
                .expect("the drag");
        }

        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        let graph = &doc.parts[&part].graph;
        assert!(graph.node(blend).is_none(), "one undo takes the blend and its drag");
        for id in [a, b] {
            let nested = doc.ownership_nodes.get(&id) == Some(&Ownership::Nested);
            assert!(
                !nested || !doc.block_readers(BlockKey::Node(id)).is_empty(),
                "{id} is nested with nothing reading it",
            );
        }
        assert_eq!(
            (doc.ownership_nodes.clone(), doc.ownership_curves.clone()),
            before_blend,
            "who owned what where the gesture started",
        );
    }

    /// **A gesture's next instant settles from where the gesture began, so the entry it leaves owns
    /// what one edit from there would** (`block-ownership-spec.md` §4.3, refactor plan 2.22).
    ///
    /// The next instant replaces the entry on top without undoing it, so it runs on a document
    /// that already holds the first instant's settle, and a settle never changes a variable. A
    /// curve is not an input, so pointing a sweep at another spine continues the act that added
    /// it, which is how the console's `sweep profile= spine=` ran: built on a spine of its own
    /// choosing, then pointed at the one it was told, in one undo step. When its choice was the
    /// curve a pipe held, the sweep was that curve's second reader for one instant -- a shared
    /// instance for that instant and nothing more, since 112 (`notebook-spec.md` §7.3) -- and once
    /// the line is done nothing but the pipe reads it, so the pipe is the one place it is drawn.
    ///
    /// Each arm is held to the same edit made as one: both ownership maps entry for entry, the pipe
    /// still holding its curve, what is said, and the one undo that takes it all back. The second
    /// arm's first choice is a curve no block reads, and it is the control for the maps: a release
    /// puts an adoption back where it was, so they came out as one edit's even unfixed. What was
    /// said gave that arm away instead -- that the curve was on its own again, which it had never
    /// stopped being.
    #[test]
    fn a_gesture_s_next_instant_settles_from_where_the_gesture_began() {
        use super::{ownership_news, record_graph_edit, step_history};
        use fieldform_ops::{CommandStack, Document, OwnershipSnapshot};
        use fieldform_scene::{BlockKey, Curve, CurveId, Graph, Node, Outline, ParamRef};
        use nalgebra::Point3;

        let line = |doc: &mut Document, name: &str, a: [f64; 3], b: [f64; 3]| {
            let p = |v: [f64; 3]| Point3::new(v[0], v[1], v[2]);
            let curve = Curve::polyline(CurveId::new(), name.to_string(), vec![p(a), p(b)], false)
                .expect("a line");
            let id = curve.id;
            doc.curves.insert(id, curve);
            id
        };
        let mut found = Vec::new();
        let mut wanted = Vec::new();
        for (first_choice, spare_first) in
            [("the curve the pipe holds", false), ("a curve no block reads", true)]
        {
            let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
            let part = *doc.parts.keys().next().expect("with_cage makes one part");
            doc.parts.get_mut(&part).expect("one part").graph = Graph::empty();
            // The pipe's path, the square the sweep carries, the spine it is told, and a spare.
            let path = line(&mut doc, "path", [0.0, -100.0, 0.0], [0.0, -100.0, 50.0]);
            let corners = [[298.0, -2.0], [302.0, -2.0], [302.0, 2.0], [298.0, 2.0]];
            let square: Vec<CurveId> = (0..4)
                .map(|i| {
                    let ([x0, y0], [x1, y1]) = (corners[i], corners[(i + 1) % 4]);
                    line(&mut doc, &format!("side {i}"), [x0, y0, 0.0], [x1, y1, 0.0])
                })
                .collect();
            let up = line(&mut doc, "up", [300.0, 0.0, 0.0], [300.0, 0.0, 30.0]);
            let spare = line(&mut doc, "spare", [500.0, 0.0, 0.0], [500.0, 0.0, 30.0]);
            let mut stack = CommandStack::default();
            let mut g = Graph::empty();
            let tube = g.add(Node::Pipe { curve: path, radius: ParamRef::Literal(3.0) });
            record_graph_edit(&mut stack, &mut doc, &mut None, part, g.clone()).expect("the pipe");
            assert_eq!(
                doc.nesting_in(&doc.parts[&part].graph).curve_owners_of(path).to_vec(),
                vec![tube],
                "the pipe holds its curve",
            );
            doc.pending_ownership_changes.clear();
            let started = doc.clone();

            // The sweep on a spine of its own choosing, then pointed at `up`: one gesture.
            let mut swept = g.clone();
            let bar = swept.add(Node::Sweep { rims: Default::default(),
                outline: Outline::closed(square[0]),
                spine: if spare_first { spare } else { path },
                section: Default::default(),
                law: Default::default(),
            });
            let mut told = swept.clone();
            if let Some(Node::Sweep { spine, .. }) = told.node_mut(bar) {
                *spine = up;
            }
            let mut gesture = None;
            record_graph_edit(&mut stack, &mut doc, &mut gesture, part, swept).expect("the sweep");
            record_graph_edit(&mut stack, &mut doc, &mut gesture, part, told.clone())
                .expect("its spine");

            // The same edit made as one.
            let mut one = started.clone();
            record_graph_edit(&mut CommandStack::default(), &mut one, &mut None, part, told)
                .expect("the sweep on its spine");

            let label = |doc: &Document, block: BlockKey| match block {
                BlockKey::Curve(id) => doc.curves[&id].name.clone(),
                BlockKey::Node(id) if id == tube => "tube".to_string(),
                BlockKey::Node(_) => "bar".to_string(),
            };
            let said = |doc: &mut Document| {
                let changes = std::mem::take(&mut doc.pending_ownership_changes);
                ownership_news(doc, &changes, |block| label(doc, block))
            };
            let heard = said(&mut doc);
            let as_one = OwnershipSnapshot::of(&doc) == OwnershipSnapshot::of(&one);
            let held =
                doc.nesting_in(&doc.parts[&part].graph).curve_owners_of(path).to_vec() == [tube];
            let undid = step_history(&mut stack, &mut doc, &mut gesture, true).is_some();
            let graph = &doc.parts[&part].graph;
            let taken_back = undid
                && graph.node(bar).is_none()
                && graph.node(tube).is_some()
                && OwnershipSnapshot::of(&doc) == OwnershipSnapshot::of(&started);
            found.push((first_choice, as_one, held, heard, taken_back));
            wanted.push((first_choice, true, true, said(&mut one), true));
        }
        assert_eq!(
            found,
            wanted,
            "first choice, maps as one edit's, the pipe holds its curve, what is said, one undo",
        );
    }

    /// **A move a gesture made is said once, however many frames the gesture spans**
    /// (`block-ownership-spec.md` §4.3, §6.3).
    ///
    /// The application hears what settling moved at the top of every frame, and a drag is an
    /// instant a frame, each settled from where the gesture began. Settled from there, every
    /// instant after the first finds again what the first one moved, and would say it again:
    /// delete a shell and drag its sphere inside the gesture window, and the release was said on
    /// that frame and then on every frame of the drag, sixty lines a second.
    ///
    /// **The placement arm says nothing at all, since 112**, and that is the other half of the
    /// rule rather than a case that lapsed: a body dragged with the gizmo for the first time is
    /// wrapped in a placement that takes every reader the body had, and two readers are a shared
    /// instance now, not a variable (`notebook-spec.md` §12, R5) — so the frame that wraps it has
    /// nothing to announce, and neither has any frame after it.
    ///
    /// Each frame's lines, from what that frame's drain takes, as `ownership_news` makes them.
    /// The console drains once a line, so `a_gesture_s_next_instant_settles_from_where_the_gesture_began`
    /// could not see this.
    #[test]
    fn a_move_a_gesture_made_is_said_once_however_many_frames_it_spans() {
        use super::{ownership_news, record_graph_edit};
        use fieldform_ops::{CommandStack, Document};
        use fieldform_scene::{BlockKey, Graph, Node, NodeId, ParamRef, Place};

        let lit = ParamRef::Literal;
        let ball = || Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) };
        let shell = |input| Node::Shell { input, thickness: lit(1.0) };
        let mut found = Vec::new();
        let mut wanted = Vec::new();
        for released in [false, true] {
            let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
            let part = *doc.parts.keys().next().expect("with_cage makes one part");
            doc.parts.get_mut(&part).expect("one part").graph = Graph::empty();
            let mut stack = CommandStack::default();
            let mut labels: std::collections::HashMap<NodeId, &str> = Default::default();
            let mut stamp = |doc: &mut Document, id: NodeId, label: &'static str| {
                doc.created.insert(id, 100 + doc.created.len() as u64);
                labels.insert(id, label);
            };

            let mut g = Graph::empty();
            let sphere = g.add(ball());
            stamp(&mut doc, sphere, "Sphere");
            let hollow = g.add(shell(sphere.into()));
            stamp(&mut doc, hollow, "Shell");
            if !released {
                let rim = g.add(shell(sphere.into()));
                stamp(&mut doc, rim, "Rim");
            }
            record_graph_edit(&mut stack, &mut doc, &mut None, part, g.clone()).expect("the start");
            doc.pending_ownership_changes.clear();

            // The first frame changes what reads what; every frame after it moves a number.
            let mut frame = g.clone();
            let dragged = if released {
                frame.remove(hollow);
                sphere
            } else {
                let placement = frame.add(Node::Transform {
                    input: sphere.into(),
                    translation: [lit(0.0); 3],
                    rotation: [lit(0.0); 3],
                    scale: lit(1.0),
                    about: Place::At([lit(0.0); 3]),
                });
                frame.rewire_consumers(sphere, placement);
                stamp(&mut doc, placement, "Placement");
                placement
            };
            let mut gesture = None;
            let mut heard = Vec::new();
            for step in 0..4 {
                match frame.node_mut(dragged) {
                    Some(Node::Sphere { radius, .. }) => *radius = lit(10.0 + step as f64),
                    Some(Node::Transform { translation, .. }) => translation[0] = lit(step as f64),
                    _ => unreachable!("the sphere or its placement"),
                }
                record_graph_edit(&mut stack, &mut doc, &mut gesture, part, frame.clone())
                    .expect("one frame of the gesture");
                let changes = std::mem::take(&mut doc.pending_ownership_changes);
                heard.push(ownership_news(&doc, &changes, |block| match block {
                    BlockKey::Node(id) => labels.get(&id).copied().unwrap_or("?").to_string(),
                    BlockKey::Curve(_) => "a curve".to_string(),
                }));
            }
            found.push(heard);
            // The placement's two readers declare it (§17), which is said on the frame the wrap
            // happens and no other; the release is said on its own frame and no other. Each line
            // is one frame's, which is what this test is about.
            let first: Vec<String> = match released {
                false => vec!["Placement is now a variable \u{2014} Shell and Rim both use it".to_string()],
                true => vec!["Sphere is on its own again".to_string()],
            };
            wanted.push(vec![first, vec![], vec![], vec![]]);
        }
        assert_eq!(found, wanted, "what each frame of the drag says, a placement then a release");
    }

    /// **A block let go of is said once, while it is still true, and nothing else is said at
    /// all** (`block-ownership-spec.md` §4.2, §6.3).
    ///
    /// Adoption is what every consume does, and saying so would be a line for every boolean.
    /// **A second reader is said again, 2026-10-24** (R1 inverted, `notebook-spec.md` §17): it
    /// declares the block, which takes it out of the slot it was drawn in, and that is a move
    /// nobody asked for in so many words. So two lines are left — the promotion and the release —
    /// and both name who was involved. And news an edit took back before the frame heard it is not
    /// news: an undo puts the ownership back without settling, so the move it took back is still
    /// in the list, and so is a release that a later edit in the same frame adopted straight back.
    ///
    /// The lines as `ownership_news` makes them from the changes real graph edits left, with the
    /// blocks given plain labels; what the application calls a block is `body_label`'s business.
    #[test]
    fn a_block_let_go_of_is_said_once_while_it_is_still_true_and_nothing_else_is() {
        use super::{ownership_news, record_graph_edit, step_history};
        use fieldform_ops::{CommandStack, Document};
        use fieldform_scene::{BlockKey, BooleanOp, Graph, Node, NodeId, ParamRef, Place};

        let lit = ParamRef::Literal;
        let ball = || Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) };
        let shell = |input| Node::Shell { input, thickness: lit(1.0) };
        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        doc.parts.get_mut(&part).expect("one part").graph = Graph::empty();
        let mut stack = CommandStack::default();
        let mut gesture = None;
        let mut labels: std::collections::HashMap<NodeId, &str> = Default::default();
        // Each step is its own act, as two clicks are, and each block is stamped as the
        // application stamps what it makes, in the order made.
        let mut edit = |doc: &mut Document,
                        stack: &mut CommandStack,
                        graph: Graph,
                        made: &[(NodeId, &'static str)]| {
            for (id, label) in made {
                doc.created.insert(*id, 100 + doc.created.len() as u64);
                labels.insert(*id, *label);
            }
            record_graph_edit(stack, doc, &mut None, part, graph).expect("the edit");
            let changes = std::mem::take(&mut doc.pending_ownership_changes);
            ownership_news(doc, &changes, |block| match block {
                BlockKey::Node(id) => labels.get(&id).copied().unwrap_or("?").to_string(),
                BlockKey::Curve(_) => "a curve".to_string(),
            })
        };

        let mut g = Graph::empty();
        let (sphere, cube) = (g.add(ball()), g.add(ball()));
        let heard = edit(&mut doc, &mut stack, g.clone(), &[(sphere, "Sphere"), (cube, "Cube")]);
        assert_eq!(heard, Vec::<String>::new(), "a new block is free, which is not a move");

        let joined = g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![sphere.into(), cube.into()] });
        let heard = edit(&mut doc, &mut stack, g.clone(), &[(joined, "Union")]);
        assert_eq!(heard, Vec::<String>::new(), "adopting the two is not said");

        let hollow = g.add(shell(sphere.into()));
        let heard = edit(&mut doc, &mut stack, g.clone(), &[(hollow, "Shell")]);
        assert_eq!(
            heard,
            ["Sphere is now a variable \u{2014} Union and Shell both use it"],
            "a second reader declares it, and the line names both",
        );

        // Three readers at once: declared in one move, with all three named.
        let hub = g.add(ball());
        let spokes: Vec<NodeId> = (0..3).map(|_| g.add(shell(hub.into()))).collect();
        let mut made = vec![(hub, "Hub")];
        made.extend(spokes.iter().zip(["First", "Second", "Third"]).map(|(id, l)| (*id, l)));
        let heard = edit(&mut doc, &mut stack, g.clone(), &made);
        assert_eq!(
            heard,
            ["Hub is now a variable \u{2014} First, Second and Third all use it"],
            "three readers at once are one promotion, with all three named in the order made",
        );

        // The union deleted, which lets the cube go, and the delete taken back before anything
        // heard of it.
        let mut taken_back = g.clone();
        taken_back.remove(joined);
        record_graph_edit(&mut stack, &mut doc, &mut gesture, part, taken_back)
            .expect("the union deleted");
        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        let changes = std::mem::take(&mut doc.pending_ownership_changes);
        assert!(!changes.is_empty(), "the edit that was taken back did say something");
        let heard = ownership_news(&doc, &changes, |_| "?".to_string());
        assert_eq!(heard, Vec::<String>::new(), "and none of it is true any more");

        // The union and its shell deleted: the cube is let go of, which is the release line. The
        // sphere is **not** — its second reader declared it earlier in this test, and a settle
        // leaves a variable alone however few read it, so losing every reader is not a move for a
        // declared block (R1 inverted, §17; the way back is Put back inside, §6.6).
        g.remove(hollow);
        g.remove(joined);
        let mut heard = edit(&mut doc, &mut stack, g.clone(), &[]);
        heard.sort();
        assert_eq!(heard, ["Cube is on its own again"]);

        // Let go of and taken straight back by the next edit, before the frame heard either.
        let mut freed = g.clone();
        let wrap = freed.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![cube.into()] });
        record_graph_edit(&mut stack, &mut doc, &mut None, part, freed.clone()).expect("taken");
        freed.remove(wrap);
        record_graph_edit(&mut stack, &mut doc, &mut None, part, freed.clone()).expect("let go");
        let again = freed.add(shell(cube.into()));
        record_graph_edit(&mut stack, &mut doc, &mut None, part, freed).expect("taken again");
        let changes = std::mem::take(&mut doc.pending_ownership_changes);
        let heard = ownership_news(&doc, &changes, |_| "?".to_string());
        assert_eq!(heard, Vec::<String>::new(), "inside {again} now, so its release is not news");
    }

    /// **However the readers changed before the frame heard of it, only losing every one of
    /// them is said** (`block-ownership-spec.md` §6.3, `notebook-spec.md` §12).
    ///
    /// What a settle reports is who read the block when it ran, and more than one edit can come
    /// before anything hears of it: every line of a script runs in one frame, and a reader can be
    /// added and deleted again inside one. **This is where the "is now a variable" line gets its
    /// readers right**, which is the reason the line is built from the document rather than from
    /// the change: a reader deleted since does not use the block, and one added since would
    /// otherwise go unnamed. The line came back with R1's inversion (`notebook-spec.md` §17), and
    /// so did this hazard.
    ///
    /// **And once declared, it stays declared**: `settle_one` leaves a `Variable` alone however
    /// many read it, so taking every reader away does *not* say "is on its own again" — that line
    /// is a nested block's, and the way back for a declared one is *Put back inside* (§6.6), which
    /// with no readers makes it a plain top-level block. Asserted here because the difference is
    /// invisible on screen: both end up as a row of their own.
    #[test]
    fn the_readers_a_promotion_names_are_the_ones_it_has_when_the_frame_hears() {
        use super::{ownership_news, record_graph_edit};
        use fieldform_ops::{CommandStack, Document};
        use fieldform_scene::{BlockKey, BooleanOp, Graph, Node, ParamRef, Place};

        let lit = ParamRef::Literal;
        let ball = || Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) };
        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        doc.parts.get_mut(&part).expect("one part").graph = Graph::empty();
        let mut stack = CommandStack::default();

        let mut g = Graph::empty();
        let sphere = g.add(ball());
        let joined = g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![sphere.into()] });
        record_graph_edit(&mut stack, &mut doc, &mut None, part, g.clone()).expect("adopted");
        let hollow = g.add(Node::Shell { input: sphere.into(), thickness: lit(1.0) });
        record_graph_edit(&mut stack, &mut doc, &mut None, part, g.clone())
            .expect("a second reader");
        // Before anything hears of it: the shell deleted, and another union pointed at the sphere.
        g.remove(hollow);
        let pair = g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![sphere.into()] });
        record_graph_edit(&mut stack, &mut doc, &mut None, part, g.clone()).expect("the readers");
        for (at, id) in [joined, hollow, pair].into_iter().enumerate() {
            doc.created.insert(id, 100 + at as u64);
        }

        let labels = [(sphere, "Sphere"), (joined, "Union"), (hollow, "Shell"), (pair, "Pair")];
        let said = |doc: &Document, changes: &[fieldform_scene::OwnershipChange]| {
            ownership_news(doc, changes, |block| match block {
                BlockKey::Node(id) => labels
                    .iter()
                    .find(|(of, _)| *of == id)
                    .map_or("?", |(_, label)| *label)
                    .to_string(),
                BlockKey::Curve(_) => "a curve".to_string(),
            })
        };
        let changes = std::mem::take(&mut doc.pending_ownership_changes);
        assert_eq!(
            said(&doc, &changes),
            ["Sphere is now a variable \u{2014} Union and Pair both use it"],
            "one reader, then two: the promotion is the line, naming who reads it now",
        );

        // And both readers deleted: the declaration stands, so there is nothing to say. A
        // *nested* block losing its one reader is what says "is on its own again", which
        // `a_block_let_go_of_is_said_once_while_it_is_still_true_and_nothing_else_is` holds.
        g.remove(joined);
        g.remove(pair);
        record_graph_edit(&mut stack, &mut doc, &mut None, part, g).expect("both readers gone");
        let changes = std::mem::take(&mut doc.pending_ownership_changes);
        assert_eq!(
            said(&doc, &changes),
            Vec::<String>::new(),
            "a declared block stays declared with nothing reading it, and says nothing",
        );
        assert_eq!(
            doc.ownership(BlockKey::Node(sphere)),
            fieldform_scene::Ownership::Variable,
            "which is the state Put back inside is the way out of",
        );
    }

    /// **Opening a file names the unnamed variables two blocks read, and no other**
    /// (`block-ownership-spec.md` §9.2, §4.2).
    ///
    /// A file from before 3.15 places a block two others read as a variable, by the rule its list
    /// was laid out by, and has no name for it, so the drain after a load names it, quietly. That
    /// rule is the one thing 112 deliberately leaves alone (`notebook-spec.md` §15): the old list
    /// drew such a block as a row of its own, and an old file has to open drawing the same things,
    /// so it opens a variable rather than the shared instance a new document would make of it.
    ///
    /// An unnamed variable with fewer readers is not one of those. It is one whose name was
    /// cleared while it was a variable — which the panel refuses and a file can still hold — and
    /// naming it on opening lifted it straight back out while the list was laid out by names, so
    /// its row moved between a save and the next open.
    ///
    /// Through a save and an open, since the question is what an opened document holds. The two
    /// variables are written as the old rule places them, since the save itself is at the current
    /// format. A frame names nothing at all now, which is the first assertion: settling no longer
    /// declares (§12, R5), so the load's own rule is the only thing left that can name anything.
    #[test]
    fn opening_a_file_names_the_unnamed_variables_two_blocks_read_and_no_other() {
        use super::{record_graph_edit, variables_to_name};
        use fieldform_ops::{CommandStack, Document};
        use fieldform_scene::{BooleanOp, Graph, Node, Ownership, ParamRef, Place};

        let lit = ParamRef::Literal;
        let ball = || Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) };
        let shell = |input| Node::Shell { input, thickness: lit(1.0) };
        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        doc.parts.get_mut(&part).expect("one part").graph = Graph::empty();
        let mut stack = CommandStack::default();

        // A sphere a union and a shell both read, and a hub two shells read, both stored as
        // variables: where `legacy_ownership` puts a named block and an unnamed block two blocks
        // read. The sphere is named below, the hub never is.
        let mut g = Graph::empty();
        let sphere = g.add(ball());
        g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![sphere.into()] });
        let hollow = g.add(shell(sphere.into()));
        let hub = g.add(ball());
        for _ in 0..2 {
            g.add(shell(hub.into()));
        }
        record_graph_edit(&mut stack, &mut doc, &mut None, part, g.clone()).expect("the blocks");
        doc.pending_ownership_changes.clear();
        for id in [sphere, hub] {
            doc.ownership_nodes.insert(id, Ownership::Variable);
        }
        doc.node_meta.entry(sphere).or_default().name = Some("Sphere".into());

        // The shell deleted, leaving the sphere a variable the union alone reads, and its name
        // cleared, which is the state the control is in.
        g.remove(hollow);
        record_graph_edit(&mut stack, &mut doc, &mut None, part, g).expect("the shell deleted");
        doc.node_meta.entry(sphere).or_default().name = None;
        assert_eq!(
            variables_to_name(&doc, false),
            [hub],
            "a frame names what the settle declared \u{2014} the hub, which two blocks read \u{2014} and \
             not the sphere, whose name was cleared while one block reads it (R1 inverted, §17)",
        );

        let saved = fieldform_io::to_archive(&doc).expect("saved");
        let opened = fieldform_io::from_archive(&saved).expect("opened");
        assert_eq!(opened.ownership_nodes.get(&sphere), Some(&Ownership::Variable), "as saved");
        assert_eq!(opened.ownership_nodes.get(&hub), Some(&Ownership::Variable), "as saved");
        assert_eq!(
            variables_to_name(&opened, true),
            [hub],
            "the hub, which two blocks read, and not the sphere, whose name was cleared",
        );
    }

    /// Balls in `part`, one for each radius given, put on the history as one edit -- and the gesture
    /// that put them there long over, so whatever a test does next is an entry of its own.
    fn balls_on_the_history(
        doc: &mut fieldform_ops::Document,
        part: fieldform_scene::PartId,
        radii: &[fieldform_scene::ParamRef],
    ) -> (Vec<fieldform_scene::NodeId>, fieldform_ops::CommandStack) {
        use fieldform_scene::{Graph, Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let balls = radii
            .iter()
            .enumerate()
            .map(|(i, &radius)| {
                let x = 40.0 * i as f64;
                graph.add(Node::Sphere { centre: Place::At([lit(x), lit(0.0), lit(0.0)]), radius })
            })
            .collect();
        let mut stack = fieldform_ops::CommandStack::default();
        super::record_graph_edit(&mut stack, doc, &mut None, part, graph).expect("the balls");
        (balls, stack)
    }

    /// What ball `id` reads its radius from, while it is there.
    fn radius_of(
        doc: &fieldform_ops::Document,
        part: fieldform_scene::PartId,
        id: fieldform_scene::NodeId,
    ) -> Option<fieldform_scene::ParamRef> {
        match doc.parts[&part].graph.node(id) {
            Some(fieldform_scene::Node::Sphere { radius, .. }) => Some(*radius),
            _ => None,
        }
    }

    /// `graph` with ball `id`'s radius read from `radius`.
    fn with_radius(
        graph: &fieldform_scene::Graph,
        id: fieldform_scene::NodeId,
        radius: fieldform_scene::ParamRef,
    ) -> fieldform_scene::Graph {
        let mut graph = graph.clone();
        if let Some(fieldform_scene::Node::Sphere { radius: r, .. }) = graph.node_mut(id) {
            *r = radius;
        }
        graph
    }

    /// Every parameter a block in `doc` names that the document does not hold.
    fn parameters_read_and_gone(doc: &fieldform_ops::Document) -> Vec<fieldform_scene::ParamId> {
        let mut gone = Vec::new();
        for part in doc.parts.values() {
            for (_, node) in part.graph.nodes() {
                for r in node.param_refs() {
                    if let fieldform_scene::ParamRef::Named(id) = *r
                        && !doc.parameters.contains_key(&id)
                    {
                        gone.push(id);
                    }
                }
            }
        }
        gone
    }

    /// **Undoing the × beside a named value brings back the parameter it removed.**
    ///
    /// The × puts the field back to a number and removes the parameter if that was its last
    /// reader. The removal was written straight into the document and only the graph edit went on
    /// the history, so Ctrl+Z restored a `ParamRef::Named` to a parameter that was gone, and the
    /// block read a value nobody could see or set (refactor plan §3.3, root cause 7). The control
    /// is a parameter a second ball still reads, which the × leaves where it is.
    ///
    /// Asserted through the functions the field, `set_part_graph` and the end of the block list
    /// run, since an `App` needs a window and a device.
    #[test]
    fn undoing_the_x_beside_a_named_value_brings_back_the_parameter_it_removed() {
        use super::{record_graph_edit_with, record_parameter_edits, step_history, ParameterEdits};
        use fieldform_ops::Document;
        use fieldform_scene::ParamRef;

        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        let own = doc.add_parameter("radius", 10.0);
        let shared = doc.add_parameter("shared", 12.0);
        let radii = [ParamRef::Named(own), ParamRef::Named(shared), ParamRef::Named(shared)];
        let (balls, mut stack) = balls_on_the_history(&mut doc, part, &radii);
        let mut gesture = None;
        let mut edits = ParameterEdits::default();

        // The × on the first ball, the one reader of its parameter: the field goes back to the
        // number, and the parameter goes once the graph no longer reads it.
        edits.released.push(own);
        let unnamed = with_radius(&doc.parts[&part].graph, balls[0], ParamRef::Literal(10.0));
        record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, part, unnamed)
            .expect("the ×");
        record_parameter_edits(&mut stack, &mut doc, &mut gesture, &mut edits).expect("the frame");
        assert!(!doc.parameters.contains_key(&own), "its last reader let go of it");

        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        assert_eq!(parameters_read_and_gone(&doc), [], "nothing reads a parameter that is gone");
        assert_eq!(radius_of(&doc, part, balls[0]), Some(ParamRef::Named(own)));
        assert_eq!(doc.parameter_value(own), Some(10.0));

        // The control: the second ball's × leaves the parameter the third still reads.
        edits.released.push(shared);
        let unnamed = with_radius(&doc.parts[&part].graph, balls[1], ParamRef::Literal(12.0));
        record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, part, unnamed)
            .expect("the second ×");
        record_parameter_edits(&mut stack, &mut doc, &mut gesture, &mut edits).expect("the frame");
        assert!(doc.parameters.contains_key(&shared), "the third ball still reads it");
        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        assert_eq!(radius_of(&doc, part, balls[1]), Some(ParamRef::Named(shared)));
        assert_eq!(parameters_read_and_gone(&doc), []);
    }

    /// **Undoing a delete never leaves a block reading a parameter the row's × removed.**
    ///
    /// Deleting the one block that read a parameter leaves it unread, and its row then offers the
    /// ×. The × removed it straight from the document, so undoing the delete brought the block
    /// back reading nothing: the history had the delete on it and not the removal after it
    /// (refactor plan §3.3, root cause 7). On the history, the first undo puts the parameter back
    /// and the second the block that reads it.
    #[test]
    fn undoing_a_delete_never_leaves_a_block_reading_a_parameter_the_row_s_x_removed() {
        use super::{record_graph_edit_with, remove_unread_parameter, step_history, ParameterEdits};
        use fieldform_ops::Document;
        use fieldform_scene::{Graph, ParamRef};

        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        let radius = doc.add_parameter("radius", 10.0);
        let (balls, mut stack) = balls_on_the_history(&mut doc, part, &[ParamRef::Named(radius)]);
        let mut gesture = None;
        let mut edits = ParameterEdits::default();

        // The ball deleted, then the parameter nothing reads now removed from its row.
        record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, part, Graph::empty())
            .expect("the delete");
        let taken = remove_unread_parameter(&mut stack, &mut doc, &mut gesture, radius);
        assert!(taken.expect("the row's ×"), "nothing reads it, so the row's × takes it");
        assert!(!doc.parameters.contains_key(&radius));

        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        assert_eq!(parameters_read_and_gone(&doc), []);
        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        assert_eq!(radius_of(&doc, part, balls[0]), Some(ParamRef::Named(radius)), "the ball");
        assert_eq!(parameters_read_and_gone(&doc), [], "and what it reads is there");
        assert_eq!(doc.parameter_value(radius), Some(10.0));

        // A parameter something reads is never the row's to take.
        let taken = remove_unread_parameter(&mut stack, &mut doc, &mut gesture, radius);
        assert!(!taken.expect("asked"));
        assert!(doc.parameters.contains_key(&radius));
    }

    /// **What the overlay keeps from frame to frame is what reading the part afresh gives, after
    /// every kind of edit** (refactor plan step 3.3).
    ///
    /// The readings are kept against the count of the document's moves and the application's own
    /// view of the part (`OverlayKey`). A change the count misses is a point left where it was, or
    /// a pick box sized to where a body used to be, and nothing would say so. Each edit here runs
    /// through the function the application runs it through and moves the count where the
    /// application moves it, since an `App` needs a window and a device; so this holds the key to
    /// the count, not the application to moving it. Every one of them changes what the overlay
    /// reads, so a reading kept past it would be caught. Switching the part, drawing a body as its
    /// mesh, hiding a joint with its row and meshing finer move nothing in the document, only the
    /// key's own view.
    #[test]
    fn the_kept_overlay_readings_are_a_fresh_reading_after_every_kind_of_edit() {
        use super::{eye_command, record_graph_edit_with, step_history, OverlayKey, ParameterEdits};
        use fieldform_ops::Document;
        use fieldform_scene::{Node, NodeId, ParamRef, Part, PartId, Place, Turn};

        let lit = ParamRef::Literal;
        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let first = *doc.parts.keys().next().expect("with_cage makes one part");
        let cage = *doc.cages.keys().next().expect("and one cage");
        let other_cage = doc.add_cage(fieldform_cage::Cage::cube_of(300.0));
        let second = doc.add_part(Part::from_cage("Second", other_cage));
        let r = doc.add_parameter("r", 12.0);
        let opened = fieldform_io::to_archive(&doc).expect("saved as it was opened");

        let mut stack = fieldform_ops::CommandStack::default();
        let (mut gesture, mut edits) = (None, ParameterEdits::default());
        let mut kept = fieldform_field::KeptReadings::default();
        let mut moves = 0;
        let key = |moves: u64, part: PartId, meshed: Vec<NodeId>| OverlayKey {
            moves,
            part,
            resolution: 64,
            hidden_with_rows: Vec::new(),
            meshed,
        };

        let mut last = kept_overlay_frame("opened", &mut kept, &doc, &key(moves, first, vec![]), None);

        // A graph edit, as `App::set_part_graph_with` records it and moves the count.
        let mut graph = doc.parts[&first].graph.clone();
        let ball = graph.add(Node::Sphere {
            centre: Place::at(80.0, 0.0, 0.0),
            radius: ParamRef::Named(r),
        });
        let joint = graph.add(Node::Point { at: [lit(0.0), lit(0.0), lit(60.0)] });
        graph.add(Node::Plane {
            origin: Place::at(0.0, 0.0, -20.0),
            flipped: false,
            turn: Turn::default(),
        });
        record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, first, graph)
            .expect("the graph edit");
        moves += 1;
        let now = key(moves, first, vec![]);
        last = kept_overlay_frame("a graph edit", &mut kept, &doc, &now, Some(&last));

        // Undo and redo, as `edit.undo` and `edit.redo` step and `after_document_change` marks.
        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        moves += 1;
        let now = key(moves, first, vec![]);
        last = kept_overlay_frame("an undo", &mut kept, &doc, &now, Some(&last));
        assert!(step_history(&mut stack, &mut doc, &mut gesture, false).is_some());
        moves += 1;
        let now = key(moves, first, vec![]);
        last = kept_overlay_frame("a redo", &mut kept, &doc, &now, Some(&last));

        // A parameter's number, as its row types it and `after_parameter_change` marks it.
        assert!(doc.set_parameter_literal(r, 30.0));
        moves += 1;
        let now = key(moves, first, vec![]);
        last = kept_overlay_frame("a parameter change", &mut kept, &doc, &now, Some(&last));

        // An eye, as `App::set_visible_all` issues it and marks it.
        let graph = doc.parts[&first].graph.clone();
        stack.execute(&mut doc, Box::new(eye_command(&graph, ball, false))).expect("the eye");
        moves += 1;
        let now = key(moves, first, vec![]);
        last = kept_overlay_frame("a body hidden", &mut kept, &doc, &now, Some(&last));

        // A drag of the cage, written live through `App::mesh_of_mut`, which moves the count on
        // every frame of it and records nothing until the hand comes off.
        let corner = doc.cage(cage).expect("the cage").vertex_ids().next().expect("a corner");
        doc.cage_mesh_mut(cage)
            .expect("the cage")
            .set_vertex_position(corner, nalgebra::Point3::new(0.0, -150.0, 0.0));
        moves += 1;
        let now = key(moves, first, vec![]);
        last = kept_overlay_frame("a cage dragged", &mut kept, &doc, &now, Some(&last));

        // The cage's body drawn as its mesh, which moves nothing in the document.
        let caged = doc.parts[&first]
            .graph
            .roots()
            .into_iter()
            .find(|id| matches!(doc.parts[&first].graph.node(*id), Some(Node::Cage(_))))
            .expect("the cage's body");
        let now = key(moves, first, vec![caged]);
        last = kept_overlay_frame("a body drawn as its mesh", &mut kept, &doc, &now, Some(&last));

        // Another part made active, and back, which move nothing in the document either.
        let now = key(moves, second, vec![]);
        last = kept_overlay_frame("a part switch", &mut kept, &doc, &now, Some(&last));
        let now = key(moves, first, vec![]);
        last = kept_overlay_frame("and back", &mut kept, &doc, &now, Some(&last));

        // The point taken for a joint whose profile row is hidden, which is what
        // `points_hidden_with_rows` hands the key, and then a finer meshing resolution, which sets
        // the pick box's margin. Neither moves the document, and each step changes one field of
        // the key, so a key that ignored that field would hand back the step before's readings.
        let now = OverlayKey { hidden_with_rows: vec![joint], ..now };
        last = kept_overlay_frame("a profile row hidden", &mut kept, &doc, &now, Some(&last));
        let now = OverlayKey { resolution: 128, ..now };
        last = kept_overlay_frame("a finer meshing resolution", &mut kept, &doc, &now, Some(&last));

        // The file opened again over the edits, which `open_document` ends with a rebuild of what
        // is drawn: the same parts under the same ids, and none of the edits.
        doc = fieldform_io::from_archive(&opened).expect("opened again");
        stack.clear();
        moves += 1;
        let now = OverlayKey { moves, ..now };
        kept_overlay_frame("an open", &mut kept, &doc, &now, Some(&last));
    }

    /// The readings `kept` holds for `key`, held to a fresh reading of `doc`, and to `last`, which
    /// the edit since must have changed for the check to test anything.
    fn kept_overlay_frame(
        what: &str,
        kept: &mut fieldform_field::KeptReadings<super::OverlayKey>,
        doc: &fieldform_ops::Document,
        key: &super::OverlayKey,
        last: Option<&std::rc::Rc<fieldform_field::OverlayReadings>>,
    ) -> std::rc::Rc<fieldform_field::OverlayReadings> {
        // `App::resolve`: a number that cannot be resolved reads as zero, for drawing and sizing.
        let numbers = |r: &fieldform_scene::ParamRef| Some(doc.resolve(r).unwrap_or(0.0));
        let got = kept.get(key.clone(), |key| key.read(doc, &numbers));
        let fresh = key.read(doc, &numbers);
        assert_eq!(got.points, fresh.points, "{what}: the points");
        assert_eq!(got.planes, fresh.planes, "{what}: the planes");
        assert_eq!(got.pick_bounds, fresh.pick_bounds, "{what}: the pick box");
        assert_eq!(got.values, fresh.values, "{what}: the parameter table");
        if let Some(last) = last {
            assert!(
                last.points != fresh.points
                    || last.planes != fresh.planes
                    || last.pick_bounds != fresh.pick_bounds
                    || last.values != fresh.values,
                "{what} changed nothing the overlay reads, so it tests nothing",
            );
        }
        got
    }

    /// **What the block list keeps from frame to frame is what reading the part afresh gives,
    /// after every kind of edit** (refactor plan step 3.4).
    ///
    /// The list's readings are kept against the count of the document's moves and the part on
    /// screen (`ListKey`), as the overlay's are, and held here the way those are: each edit runs
    /// through the function the application runs it through, and moves the count where the
    /// application moves it. Afresh is what the list asked every frame before: `nesting_in`,
    /// `listed_blocks`, and `Graph::roots` for every row's "is this a body". Every step changes
    /// what the list reads, so a reading kept past one would be caught: the rows' order, where a
    /// block is drawn, a curve's readers in another part, or the bodies.
    #[test]
    fn the_kept_list_readings_are_a_fresh_reading_after_every_kind_of_edit() {
        use super::{make_variable, record_graph_edit_with, step_history, ListKey, ParameterEdits};
        use fieldform_ops::Document;
        use fieldform_scene::{BlockKey, Curve, CurveId, Node, NodeId, ParamRef, Part, Place};
        use nalgebra::Point3;
        use std::rc::Rc;

        let lit = ParamRef::Literal;
        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let first = *doc.parts.keys().next().expect("with_cage makes one part");
        let other_cage = doc.add_cage(fieldform_cage::Cage::cube_of(300.0));
        let second = doc.add_part(Part::from_cage("Second", other_cage));
        let spine = Curve::polyline(
            CurveId::new(),
            "Spine".to_string(),
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 50.0)],
            false,
        )
        .expect("a line");
        let spine = {
            let id = spine.id;
            doc.curves.insert(id, spine);
            id
        };
        doc.settle_ownership();
        let opened = fieldform_io::to_archive(&doc).expect("saved as it was opened");

        let mut stack = fieldform_ops::CommandStack::default();
        let (mut gesture, mut edits) = (None, ParameterEdits::default());
        let mut kept = fieldform_field::KeptReadings::default();
        let reads = std::cell::Cell::new(0);
        let mut moves = 0;
        let key = |moves: u64, part| ListKey { moves, part };
        let mut frame = |what: &str, doc: &Document, key: ListKey, last: Option<&Rc<_>>| {
            kept_list_frame(what, &mut kept, doc, &key, last, &reads)
        };

        // Frames with nothing moved between them read the part once, where the list used to read
        // it every frame.
        let mut last = None;
        for _ in 0..4 {
            last = Some(frame("opened", &doc, key(moves, first), None));
        }
        assert_eq!(reads.get(), 1, "four frames at one revision");
        let mut last = last.expect("four frames");

        // A graph edit, as `App::set_part_graph_with` records it and moves the count: a body made
        // from a ball, which it adopts, and a point, which nothing reads and is never a body.
        let mut graph = doc.parts[&first].graph.clone();
        let ball = graph.add(Node::Sphere { centre: Place::at(80.0, 0.0, 0.0), radius: lit(12.0) });
        graph.add(Node::Offset { input: ball.into(), distance: lit(2.0) });
        graph.add(Node::Point { at: [lit(0.0), lit(0.0), lit(60.0)] });
        record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, first, graph)
            .expect("the graph edit");
        moves += 1;
        last = frame("a graph edit", &doc, key(moves, first), Some(&last));

        // Undo and redo, as `edit.undo` and `edit.redo` step and `after_document_change` marks.
        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        moves += 1;
        last = frame("an undo", &doc, key(moves, first), Some(&last));
        assert!(step_history(&mut stack, &mut doc, &mut gesture, false).is_some());
        moves += 1;
        last = frame("a redo", &doc, key(moves, first), Some(&last));

        // The ball made a variable, as its row's V does through `App::set_ownership`.
        let variable = make_variable(&doc, BlockKey::Node(ball), || "Ball".to_string())
            .expect("the ball is inside the body made from it");
        gesture = None;
        stack.execute(&mut doc, Box::new(variable)).expect("made a variable");
        moves += 1;
        last = frame("a variable", &doc, key(moves, first), Some(&last));

        // The bodies dragged into the other order, as `App::arrange_block` writes the ranks, and
        // the order reset, as `App::clear_arrangement` clears them; each marks the change.
        let mut rows = doc.listed_bodies(&doc.parts[&first].graph);
        rows.reverse();
        for (i, id) in rows.iter().enumerate() {
            doc.arrangement.insert(*id, i as u32);
        }
        moves += 1;
        last = frame("a reorder", &doc, key(moves, first), Some(&last));
        let mine: Vec<NodeId> = doc.parts[&first].graph.nodes().map(|(id, _)| id).collect();
        for id in mine {
            doc.arrangement.remove(&id);
        }
        moves += 1;
        last = frame("a reset order", &doc, key(moves, first), Some(&last));

        // The other part shown, a pipe made there along the spine, and back. A curve's readers are
        // counted in every part (spec I2), so the pipe changes this part's list as well.
        let here = Rc::clone(&last);
        last = frame("a part switch", &doc, key(moves, second), Some(&last));
        let mut graph = doc.parts[&second].graph.clone();
        graph.add(Node::Pipe { curve: spine, radius: lit(3.0) });
        record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, second, graph)
            .expect("the pipe");
        moves += 1;
        last = frame("a pipe there", &doc, key(moves, second), Some(&last));
        last = frame("and back", &doc, key(moves, first), Some(&last));
        assert_eq!(here.nesting.readers_of_curve(spine), 0, "nothing here read the spine");
        assert_eq!(last.nesting.readers_of_curve(spine), 1, "and the pipe there is counted here");

        // The file opened again over the edits, which `open_document` ends with a rebuild of what
        // is drawn: the same parts under the same ids, and none of the edits.
        doc = fieldform_io::from_archive(&opened).expect("opened again");
        stack.clear();
        moves += 1;
        frame("an open", &doc, key(moves, first), Some(&last));
    }

    /// The list's readings `kept` holds for `key`, counting each read in `reads`, held to what the
    /// list asked afresh every frame before step 3.4, and to `last`, which the edit since must have
    /// changed for the check to test anything.
    fn kept_list_frame(
        what: &str,
        kept: &mut fieldform_field::KeptReadings<super::ListKey, fieldform_scene::ListReadings>,
        doc: &fieldform_ops::Document,
        key: &super::ListKey,
        last: Option<&std::rc::Rc<fieldform_scene::ListReadings>>,
        reads: &std::cell::Cell<usize>,
    ) -> std::rc::Rc<fieldform_scene::ListReadings> {
        let got = kept.get(*key, |key| {
            reads.set(reads.get() + 1);
            key.read(doc)
        });
        let graph = &doc.parts[&key.part].graph;
        let fresh = fieldform_scene::ListReadings {
            nesting: doc.nesting_in(graph),
            listed: doc.listed_blocks(graph),
        };
        assert_eq!(got.listed, fresh.listed, "{what}: the order of the rows");
        assert_eq!(got.nesting, fresh.nesting, "{what}: where each block is drawn");
        let roots = graph.roots();
        for (id, _) in graph.nodes() {
            assert_eq!(
                got.nesting.is_root(id),
                roots.contains(&id),
                "{what}: whether {id:?} is a body",
            );
        }
        if let Some(last) = last {
            assert!(**last != fresh, "{what} changed nothing the list reads, so it tests nothing");
        }
        got
    }

    /// **Naming a value is one act, and one undo takes all of it back.**
    ///
    /// "Name this value" makes a parameter and points the field at it. The parameter was written
    /// straight into the document and the graph edit alone went on the history, so Ctrl+Z put the
    /// number back and left the parameter behind with nothing reading it -- and, coming inside the
    /// window of the edit before it, the naming folded into that edit's entry, so the same Ctrl+Z
    /// took back the edit before as well (refactor plan §3.3, root cause 7). An edit straight after
    /// the naming is an entry of its own, so it cannot fold the naming into itself either.
    #[test]
    fn naming_a_value_is_one_undo_that_takes_the_parameter_back_with_it() {
        use super::{name_value, record_graph_edit_with, step_history, ParameterEdits};
        use fieldform_ops::{CommandStack, Document};
        use fieldform_scene::{Graph, Node, ParamRef, Place};

        let lit = ParamRef::Literal;
        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        let mut stack = CommandStack::default();
        let mut gesture = None;
        let mut edits = ParameterEdits::default();

        // A ball, and at once -- inside the window, as a click on its V can be -- its radius named.
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: Place::At([lit(0.0), lit(0.0), lit(0.0)]),
            radius: lit(10.0),
        });
        record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, part, graph.clone())
            .expect("a ball");
        let radius = name_value(&mut doc, &mut edits, "radius".into(), 10.0);
        let named = with_radius(&graph, ball, ParamRef::Named(radius));
        record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, part, named.clone())
            .expect("the naming");
        // And the ball moved straight after: values only, the same structure.
        let mut moved = named.clone();
        if let Some(Node::Sphere { centre, .. }) = moved.node_mut(ball) {
            *centre = Place::At([lit(5.0), lit(0.0), lit(0.0)]);
        }
        record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, part, moved)
            .expect("the move");

        let held = |doc: &Document| radius_of(doc, part, ball);
        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        assert_eq!(held(&doc), Some(ParamRef::Named(radius)), "the move came off alone");
        assert!(doc.parameters.contains_key(&radius));

        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        assert_eq!(held(&doc), Some(lit(10.0)), "one undo puts the number back");
        assert!(!doc.parameters.contains_key(&radius), "and takes the parameter with it");

        assert!(step_history(&mut stack, &mut doc, &mut gesture, false).is_some());
        assert_eq!(held(&doc), Some(ParamRef::Named(radius)), "redo is the same act again");
        assert_eq!(doc.parameters.get(&radius).map(|p| p.name.as_str()), Some("radius"));
    }

    /// `graph` with ball `id`'s radius named, as the field's "Name this value" does it: the
    /// parameter made and the edit pointing the ball at it put on the history as one entry.
    fn name_radius(
        doc: &mut fieldform_ops::Document,
        stack: &mut fieldform_ops::CommandStack,
        part: fieldform_scene::PartId,
        id: fieldform_scene::NodeId,
        value: f64,
    ) -> fieldform_scene::ParamId {
        use super::{name_value, record_graph_edit_with, unique_parameter_name, ParameterEdits};
        let mut edits = ParameterEdits::default();
        let name = unique_parameter_name(doc, "Radius:");
        let radius = name_value(doc, &mut edits, name, value);
        let named =
            with_radius(&doc.parts[&part].graph, id, fieldform_scene::ParamRef::Named(radius));
        record_graph_edit_with(stack, doc, &mut None, &mut edits, part, named).expect("the naming");
        radius
    }

    /// **Undoing a naming leaves the parameter while an expression has come to read it.**
    ///
    /// Expressions are typed off the history, so `boss = radius * 5 - 2` can come to read a value
    /// after the naming that made `radius`, and so can a Number saying `radius + 10`. Undoing the
    /// naming took `radius` out regardless: `boss`, which a second ball reads, computed nothing,
    /// and a save wrote a file refused on load. The graph half still comes off -- the first ball is
    /// back on its number -- and the parameter stays for what reads it.
    #[test]
    fn undoing_a_naming_leaves_the_parameter_while_an_expression_has_come_to_read_it() {
        use super::step_history;
        use fieldform_ops::Document;
        use fieldform_scene::ParamRef;

        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        let boss = doc.add_parameter("boss", 48.0);
        let radii = [ParamRef::Literal(10.0), ParamRef::Named(boss)];
        let (balls, mut stack) = balls_on_the_history(&mut doc, part, &radii);
        let radius = name_radius(&mut doc, &mut stack, part, balls[0], 10.0);
        doc.set_parameter_expression(boss, "radius * 5 - 2").expect("boss computed from it");
        let number = doc.add_parameter("number", 0.0);
        doc.set_parameter_expression(number, "radius + 10").expect("a Number reading it");

        let undone = step_history(&mut stack, &mut doc, &mut None, true);
        assert_eq!(undone.as_deref(), Some("parameter.name_value"));
        assert_eq!(radius_of(&doc, part, balls[0]), Some(ParamRef::Literal(10.0)));
        assert!(doc.parameters.contains_key(&radius), "expressions read it");
        assert_eq!(doc.parameter_value(boss), Some(48.0), "the second ball's radius is unmoved");
        assert_eq!(doc.parameter_value(number), Some(20.0));
    }

    /// **Undo and redo of a naming keep the name and number given it since.**
    ///
    /// The field focuses the new parameter's name so the placeholder is typed over, and that
    /// rename, like a number typed or dragged, is made off the history. Redo put the parameter
    /// back as it was made, so naming, renaming to `wall` and setting 20, then Ctrl+Z and Ctrl+Y,
    /// gave back `radius` at 10 -- and moved the ball reading it.
    #[test]
    fn undo_and_redo_of_a_naming_keep_the_name_and_number_given_it_since() {
        use super::step_history;
        use fieldform_ops::Document;
        use fieldform_scene::ParamRef;

        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        let (balls, mut stack) = balls_on_the_history(&mut doc, part, &[ParamRef::Literal(10.0)]);
        let radius = name_radius(&mut doc, &mut stack, part, balls[0], 10.0);
        doc.rename_parameter(radius, "wall").expect("the placeholder typed over");
        doc.set_parameter_literal(radius, 20.0);

        assert!(step_history(&mut stack, &mut doc, &mut None, true).is_some());
        assert!(step_history(&mut stack, &mut doc, &mut None, false).is_some());
        let back = &doc.parameters[&radius];
        assert_eq!((back.name.as_str(), back.value.as_literal()), ("wall", Some(20.0)));
        assert_eq!(radius_of(&doc, part, balls[0]), Some(ParamRef::Named(radius)));
        assert_eq!(doc.resolve(&ParamRef::Named(radius)), Some(20.0));
    }

    /// **Undoing and redoing two namings never leaves two parameters with one name.**
    ///
    /// Name the first ball's radius and rename it `wall`, and the second ball's radius is free to
    /// be `radius` too. Redo put each back as it was made, so after two undos and two redos both
    /// were `radius`: a document `rename_parameter` exists to refuse, and a file refused on load
    /// as soon as any expression is in it.
    #[test]
    fn undoing_and_redoing_two_namings_never_leaves_two_parameters_with_one_name() {
        use super::step_history;
        use fieldform_ops::Document;
        use fieldform_scene::ParamRef;

        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        let radii = [ParamRef::Literal(10.0), ParamRef::Literal(12.0)];
        let (balls, mut stack) = balls_on_the_history(&mut doc, part, &radii);
        let first = name_radius(&mut doc, &mut stack, part, balls[0], 10.0);
        doc.rename_parameter(first, "wall").expect("the placeholder typed over");
        let second = name_radius(&mut doc, &mut stack, part, balls[1], 12.0);
        assert_eq!(doc.parameters[&second].name, "radius", "free again once renamed");

        for back in [true, true, false, false] {
            assert!(step_history(&mut stack, &mut doc, &mut None, back).is_some());
        }
        let mut names: Vec<&str> = doc.parameters.values().map(|p| p.name.as_str()).collect();
        names.sort();
        assert_eq!(names, ["radius", "wall"]);
    }

    /// **A parameter put back by undo takes a free name if another has taken its own meanwhile.**
    ///
    /// Both ×s remove a parameter on the history, but a parameter can be made and renamed off it
    /// -- a Number, renamed. Undoing the × put the first back beside the second under one name.
    /// It comes back as `radius_2`, which is what naming a second value `radius` gives, and the
    /// ball the field's × let go of reads it again.
    #[test]
    fn a_parameter_put_back_by_undo_takes_a_free_name_if_another_has_taken_its_own() {
        use super::{record_graph_edit_with, remove_unread_parameter, step_history, ParameterEdits};
        use fieldform_ops::Document;
        use fieldform_scene::ParamRef;

        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        let radius = doc.add_parameter("radius", 10.0);
        let (balls, mut stack) = balls_on_the_history(&mut doc, part, &[ParamRef::Named(radius)]);
        let mut gesture = None;
        let mut edits = ParameterEdits::default();

        // The field's ×, then a Number made and renamed `radius`.
        edits.released.push(radius);
        let unnamed = with_radius(&doc.parts[&part].graph, balls[0], ParamRef::Literal(10.0));
        record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, part, unnamed)
            .expect("the ×");
        assert!(!doc.parameters.contains_key(&radius));
        let number = doc.add_parameter("number", 3.0);
        doc.rename_parameter(number, "radius").expect("free now");
        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        assert_eq!(radius_of(&doc, part, balls[0]), Some(ParamRef::Named(radius)));
        assert_eq!(doc.parameters[&radius].name, "radius_2");
        assert_eq!(doc.parameters[&number].name, "radius");
        assert_eq!(doc.parameter_value(radius), Some(10.0));

        // The row's ×, the same.
        let wall = doc.add_parameter("wall", 2.0);
        let taken = remove_unread_parameter(&mut stack, &mut doc, &mut gesture, wall);
        assert!(taken.expect("the row's ×"));
        let other = doc.add_parameter("wall", 4.0);
        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        assert_eq!(doc.parameters[&wall].name, "wall_2");
        assert_eq!(doc.parameters[&other].name, "wall");
    }

    /// **A value named where no graph edit takes it stays off the history.**
    ///
    /// The draft analysis's pull offers "Name this value" on X, Y and Z. The pull is the panel's
    /// own, not the document's, so no graph edit follows the naming, and the parameter went on the
    /// history alone: the next Ctrl+Z took it out while the pull still read it, so the field said
    /// `missing` and the pull resolved to nought. A parameter goes on the history with the
    /// reference to it or not at all, so this one stays in the document, and Ctrl+Z reaches the
    /// edit before.
    #[test]
    fn a_value_named_where_no_graph_edit_takes_it_stays_off_the_history() {
        use super::{name_value, record_parameter_edits, step_history, ParameterEdits};
        use fieldform_ops::Document;
        use fieldform_scene::ParamRef;

        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        let (balls, mut stack) = balls_on_the_history(&mut doc, part, &[ParamRef::Literal(10.0)]);
        let mut gesture = None;
        let mut edits = ParameterEdits::default();

        let pulled = name_value(&mut doc, &mut edits, "x".into(), 0.0);
        let recorded = record_parameter_edits(&mut stack, &mut doc, &mut gesture, &mut edits);
        assert!(!recorded.expect("the frame"), "nothing went on the history");
        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        assert_eq!(radius_of(&doc, part, balls[0]), None, "the undo took the ball");
        assert!(doc.parameters.contains_key(&pulled), "and left what the pull reads");
    }

    /// **Undoing a naming leaves the draft pull on the number it showed.**
    ///
    /// The pull can be pointed at a parameter that exists ("or use"), and being the panel's own
    /// state it is nothing the history knows reads one. Undoing the naming that made the parameter
    /// takes it out, and the pull read `missing` and resolved to nought. It goes back to the number
    /// it showed, as the × beside a named value does; a parameter still there is still read.
    #[test]
    fn undoing_a_naming_leaves_the_draft_pull_on_the_number_it_showed() {
        use super::{keep_typed_direction, step_history, typed_direction};
        use fieldform_ops::Document;
        use fieldform_scene::{ParamRef, Sweep};

        let lit = ParamRef::Literal;
        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        let kept = doc.add_parameter("kept", 1.0);
        let (balls, mut stack) = balls_on_the_history(&mut doc, part, &[lit(10.0)]);
        let radius = name_radius(&mut doc, &mut stack, part, balls[0], 10.0);
        let mut pull =
            Sweep::Toward { xyz: [ParamRef::Named(kept), lit(0.0), ParamRef::Named(radius)] };

        let was = typed_direction(&doc, &pull);
        assert!(step_history(&mut stack, &mut doc, &mut None, true).is_some());
        assert!(!doc.parameters.contains_key(&radius), "nothing in the document reads it");
        keep_typed_direction(&doc, &mut pull, was);
        assert_eq!(pull, Sweep::Toward { xyz: [ParamRef::Named(kept), lit(0.0), lit(10.0)] });
    }

    /// **Putting a number back turns its socket into the value** (`block-ownership-spec.md` §11,
    /// test 19), **and is refused wherever there is not one socket for it to go back into.**
    ///
    /// A parameter is a variable already: *Name this value* takes a number out of its field into a
    /// row of its own, and the bright V on that row is the way back. It goes back where there is
    /// exactly one place for it — a number, not an expression, that one socket in the document
    /// holds and no other parameter's expression reads — and the socket gets the number while the
    /// parameter goes, in one entry, so one undo brings back the parameter and the reference to it
    /// together. A socket in another part counts as one on screen does, since a parameter is the
    /// document's. Anything else is refused saying who uses it, or why an expression cannot go
    /// back: a field holds a number or a name, never a formula.
    #[test]
    fn putting_a_number_back_turns_its_socket_into_the_value() {
        use super::{put_number_back, record_graph_edit, step_history};
        use fieldform_ops::Document;
        use fieldform_scene::{Graph, Node, NodeId, ParamRef, Part, PartId, Place};

        let (lit, named) = (ParamRef::Literal, ParamRef::Named);
        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        let radius = doc.add_parameter("radius", 10.0);
        let shared = doc.add_parameter("shared", 12.0);
        let wall = doc.add_parameter("wall", 2.0);
        let boss = doc.add_parameter("boss", 0.0);
        let side = doc.add_parameter("side", 20.0);
        let across = doc.add_parameter("across", 4.0);
        let elsewhere = doc.add_parameter("elsewhere", 6.0);
        let spare = doc.add_parameter("spare", 1.0);
        let radii = [named(radius), named(shared), named(wall), named(boss), named(across)];
        let (balls, mut stack) = balls_on_the_history(&mut doc, part, &radii);
        doc.set_parameter_expression(boss, "wall * 5 - 2").expect("boss reads the wall");
        let mut graph = doc.parts[&part].graph.clone();
        let shell = graph.add(Node::Shell { input: balls[1].into(), thickness: named(shared) });
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [named(side), named(side), lit(5.0)],
            turn: fieldform_scene::no_turn(),
        });
        record_graph_edit(&mut stack, &mut doc, &mut None, part, graph).expect("a shell and a box");
        let mut there = Graph::empty();
        let ball =
            |x: f64, radius| Node::Sphere { centre: Place::At([lit(x), lit(0.0), lit(0.0)]), radius };
        let far = there.add(ball(0.0, named(across)));
        let away = there.add(ball(40.0, named(elsewhere)));
        let other = doc.add_part(Part {
            id: PartId::new(),
            name: "Other".into(),
            material: None,
            graph: there,
            inputs: Vec::new(),
        });
        // Stamped in the order they were made, as the application stamps every block, since a
        // refusal names the blocks in that order.
        for (at, id) in balls.iter().chain([&shell, &cube, &far, &away]).enumerate() {
            doc.created.insert(*id, at as u64 + 1);
        }
        let kind = |doc: &Document, part: PartId, node: NodeId| {
            match doc.parts[&part].graph.node(node) {
                Some(Node::Sphere { .. }) => "Sphere",
                Some(Node::Shell { .. }) => "Shell",
                Some(Node::Cuboid { .. }) => "Box",
                _ => "something else",
            }
            .to_string()
        };
        let refusal = |doc: &Document, id| match put_number_back(doc, id, |p, n| kind(doc, p, n)) {
            Ok(_) => panic!("{} went back, and should have been refused", doc.parameters[&id].name),
            Err(why) => why,
        };
        let held = |doc: &Document, part: PartId, id: NodeId| match doc.parts[&part].graph.node(id) {
            Some(Node::Sphere { radius, .. }) => *radius,
            _ => panic!("a ball"),
        };

        for (id, said) in [
            (shared, "used by Sphere and Shell"),
            (boss, "an expression has to stay a variable"),
            (wall, "used by Sphere and boss"),
            (side, "used by Box twice"),
            (across, "used by Sphere and Sphere"),
            (spare, "nothing uses it"),
        ] {
            let why = refusal(&doc, id);
            assert!(why.contains(said), "{}: {why}", doc.parameters[&id].name);
        }

        // One socket holds `radius`: it gets the number, and the parameter goes.
        let back = put_number_back(&doc, radius, |p, n| kind(&doc, p, n))
            .unwrap_or_else(|why| panic!("one socket holds it, so it can go back: {why}"));
        stack.execute(&mut doc, Box::new(back)).expect("the put back");
        assert_eq!(held(&doc, part, balls[0]), lit(10.0), "the socket holds the value");
        assert!(!doc.parameters.contains_key(&radius), "and the parameter is gone");
        assert_eq!(held(&doc, part, balls[1]), named(shared), "and nothing else changed");

        let undone = step_history(&mut stack, &mut doc, &mut None, true);
        assert_eq!(undone.as_deref(), Some("parameter.put_back_inside"), "one entry");
        assert_eq!(held(&doc, part, balls[0]), named(radius), "one undo names it again");
        assert_eq!(doc.parameters.get(&radius).map(|p| p.name.as_str()), Some("radius"));
        assert_eq!(doc.parameter_value(radius), Some(10.0), "with the value it had");
        assert!(step_history(&mut stack, &mut doc, &mut None, false).is_some());
        assert_eq!(held(&doc, part, balls[0]), lit(10.0), "and a redo is the same act again");
        assert!(!doc.parameters.contains_key(&radius));

        // In another part as in this one.
        let back = put_number_back(&doc, elsewhere, |p, n| kind(&doc, p, n))
            .unwrap_or_else(|why| panic!("one socket, in the other part: {why}"));
        stack.execute(&mut doc, Box::new(back)).expect("the put back");
        assert_eq!(held(&doc, other, away), lit(6.0));
        assert!(!doc.parameters.contains_key(&elsewhere));
        assert_eq!(parameters_read_and_gone(&doc), [], "and nothing reads a parameter that is gone");
    }

    /// **Deleting either point of a two-point frame leaves its block aimed where it was.**
    ///
    /// A frame between two points reads both, and the delete let go only of what
    /// `Frame::source` names -- the one block a `From` frame is taken from, and nothing for
    /// `Between`. So deleting one of the two points left the frame holding it, and the cascade
    /// moved that end onto the block's surviving input: a mirror between two points became a
    /// mirror aimed from the very body it reflects (refactor plan §3.3, root cause 2). It is let
    /// go of now the way a frame taken from one block is, typed at the place and the direction it
    /// resolved to.
    ///
    /// Asserted through `after_deleting`, the graph half of `delete_body`, since an `App` needs a
    /// window and a device.
    #[test]
    fn deleting_either_point_of_a_two_point_frame_leaves_its_block_aimed_where_it_was() {
        use super::App;
        use fieldform_ops::Document;
        use fieldform_scene::{Frame, Graph, Node, ParamRef, Place};

        let lit = ParamRef::Literal;
        let doc = Document::new();
        let numbers = |r: &ParamRef| doc.resolve(r);
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: Place::At([lit(20.0), lit(0.0), lit(0.0)]),
            radius: lit(10.0),
        });
        let from = graph.add(Node::Point { at: [lit(5.0), lit(0.0), lit(0.0)] });
        let to = graph.add(Node::Point { at: [lit(5.0), lit(10.0), lit(10.0)] });
        let mirror =
            graph.add(Node::Mirror { input: ball.into(), frame: Frame::Between { from, to }, join: true });
        // Where the mirror plane is and which way it faces, as the kernel reads it.
        let aim = |graph: &Graph| {
            let frame = *graph.node(mirror).expect("the mirror is still there").frames()[0];
            let (origin, rotation) = fieldform_field::resolve_frame(&doc, graph, &frame, &numbers)
                .expect("the frame resolves");
            (origin, rotation * nalgebra::Vector3::z())
        };
        let (origin, axis) = aim(&graph);

        for (gone, kept) in [(from, to), (to, from)] {
            let after = App::after_deleting(&doc, &graph, gone, &|e| e.to_string())
                .unwrap_or_else(|why| panic!("deleting a point the mirror is aimed by was refused: {why}"));
            assert!(after.node(gone).is_none(), "the point is deleted");
            assert!(after.node(kept).is_some(), "and the other one stays");
            assert_eq!(
                after.node(mirror).map(|n| n.inputs()),
                Some(vec![ball]),
                "the mirror reads its body and nothing else",
            );
            let (now_origin, now_axis) = aim(&after);
            assert!(
                (now_origin - origin).norm() < 1e-9,
                "the mirror plane stays where it was: {now_origin:?} against {origin:?}",
            );
            assert!(
                (now_axis - axis).norm() < 1e-9,
                "and faces the way it faced: {now_axis:?} against {axis:?}",
            );
        }
    }

    /// **Deleting either point of a two-point turn leaves its block facing where it was.**
    ///
    /// The same fault as a two-point frame's, through `Turn::source`: a plane turned to face from
    /// one point to another lost its facing when either point was deleted, the cascade leaving it
    /// turned from the surviving point to itself -- no direction, so no turn at all.
    ///
    /// The facing is asserted to be a real turn first, so that a plane left unturned cannot pass
    /// for one that kept its turn.
    #[test]
    fn deleting_either_point_of_a_two_point_turn_leaves_its_block_facing_where_it_was() {
        use super::App;
        use fieldform_ops::Document;
        use fieldform_scene::{Graph, Node, ParamRef, Place, Turn};

        let lit = ParamRef::Literal;
        let doc = Document::new();
        let numbers = |r: &ParamRef| doc.resolve(r);
        let mut graph = Graph::empty();
        let from = graph.add(Node::Point { at: [lit(0.0); 3] });
        let to = graph.add(Node::Point { at: [lit(10.0), lit(0.0), lit(10.0)] });
        let plane = graph.add(Node::Plane {
            origin: Place::At([lit(0.0); 3]),
            flipped: false,
            turn: Turn::Between { from, to },
        });
        let facing = |graph: &Graph| {
            let turn = *graph.node(plane).and_then(|n| n.turn()).expect("the plane is still there");
            fieldform_field::turn_rotation(&doc, graph, &turn, &numbers)
        };
        let before = facing(&graph);
        assert!(before.angle() > 0.5, "the two points turn the plane, by {} rad", before.angle());

        for (gone, kept) in [(from, to), (to, from)] {
            let after = App::after_deleting(&doc, &graph, gone, &|e| e.to_string())
                .unwrap_or_else(|why| panic!("deleting a point the plane faces by was refused: {why}"));
            assert!(after.node(gone).is_none(), "the point is deleted");
            assert!(after.node(kept).is_some(), "and the other one stays");
            let turn = after.node(plane).and_then(|n| n.turn()).copied();
            assert!(
                matches!(turn, Some(Turn::By(_))),
                "the plane's facing is typed now, reading neither point: {turn:?}",
            );
            let drift = (before.inverse() * facing(&after)).angle();
            assert!(drift < 1e-9, "the plane faces the way it faced, but is {drift} rad off");
        }
    }

    /// **A delete that would break a part that builds is refused, in the words of the check every
    /// other graph edit asks.**
    ///
    /// `delete_body` committed without asking `refuse_reason` (refactor plan §3.2). Deleting the
    /// point an attract pulls toward hands the attract its surviving operand, the very body it
    /// warps, as the point to pull toward -- and an attractor that is not a point cannot be
    /// built, so the part went dark on a click that then said "Deleted". Refused now, before
    /// anything is written, with the reason the check gives for the graph the delete would have
    /// left. That reason is a sentence written to stand alone, so where the refusal quotes it
    /// after a colon it loses its capital; and the refusal says the part builds now, since a part
    /// that did not is never refused a delete.
    ///
    /// Two controls: the part builds before the delete, so it is the delete being refused; and
    /// deleting the body instead takes the attract with it and is not refused. So is deleting the
    /// body of a twist aimed at a plane, the case this was first written on: the twist was handed
    /// the plane, which only aimed it, and a stand-in comes from the operands now (§3.3, root
    /// cause 1).
    #[test]
    fn a_delete_that_would_break_a_part_that_builds_is_refused_with_the_check_s_reason() {
        use super::App;
        use fieldform_ops::Document;
        use fieldform_scene::{Attractor, Frame, Graph, Node, ParamRef, Place};

        let lit = ParamRef::Literal;
        let doc = Document::new();
        let explain = |e: &fieldform_field::CompileError| e.to_string();
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: Place::At([lit(0.0), lit(0.0), lit(30.0)]),
            radius: lit(10.0),
        });
        let target = graph.add(Node::Point { at: [lit(20.0), lit(0.0), lit(30.0)] });
        graph.add(Node::Attract {
            input: ball.into(),
            toward: Attractor::Point(target),
            strength: lit(0.5),
            falloff: lit(10.0),
        });

        assert_eq!(App::refusal_of(&doc, &graph, &explain), None, "the part builds before the delete");
        let why = App::after_deleting(&doc, &graph, target, &explain)
            .expect_err("an attract pulled toward its own body cannot be built");
        let mut left = graph.clone();
        left.remove_cascading(target);
        assert_eq!(
            Some(why.clone()),
            App::refusal_of(&doc, &left, &explain),
            "refused in the check's own words, for the graph the delete would have left",
        );
        // The check's own sentences open with a capital, being written to stand alone as any other
        // edit's warning, and a cycle's is one of them. Taken from a graph that loops, so it is the
        // check's sentence and not one written here.
        let mut looped = Graph::empty();
        let inner =
            looped.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let outer = looped.add(Node::Shell { input: inner.into(), thickness: lit(1.0) });
        let shell = looped.add(Node::Shell { input: outer.into(), thickness: lit(1.0) });
        if let Some(Node::Shell { input, .. }) = looped.node_mut(outer) {
            *input = shell.into();
        }
        let circular = App::refusal_of(&doc, &looped, &explain).expect("a loop cannot be built");
        for why in [circular, why] {
            let said = App::delete_refused("Point", &why);
            let (head, reason) = said.split_once(": ").expect("the check's reason follows a colon");
            assert!(!reason.starts_with(char::is_uppercase), "no capital after the colon: {said}");
            assert!(head.starts_with("Deleting Point "), "the refusal names the block: {said}");
            assert!(head.contains("a part that builds now"), "and says the part built: {said}");
        }

        let after = App::after_deleting(&doc, &graph, ball, &explain)
            .unwrap_or_else(|why| panic!("the attract goes with its body, unrefused: {why}"));
        assert_eq!(
            after.nodes().map(|(id, _)| id).collect::<Vec<_>>(),
            vec![target],
            "the point is all that is left",
        );

        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere {
            centre: Place::At([lit(0.0), lit(0.0), lit(30.0)]),
            radius: lit(10.0),
        });
        let plane = graph.add(Node::Plane {
            origin: Place::At([lit(0.0); 3]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let frame = Frame::From { node: plane };
        graph.add(Node::Twist { input: ball.into(), degrees_per_mm: lit(2.0), frame });
        let after = App::after_deleting(&doc, &graph, ball, &explain).unwrap_or_else(|why| {
            panic!("a twist aimed at a plane goes with its body, unrefused: {why}")
        });
        assert_eq!(
            after.nodes().map(|(id, _)| id).collect::<Vec<_>>(),
            vec![plane],
            "the plane is all that is left",
        );
    }

    /// **A part with two broken blocks lets either be deleted.**
    ///
    /// The check a delete asks was asked of the graph the delete leaves and of nothing else, so in
    /// a part that already could not be built every delete was refused unless that one delete
    /// mended it -- and a part with two faults has no such delete. Two twists each left reading
    /// the plane it was aimed at, as the stand-in rule did before it took operands only (refactor
    /// plan §3.3, root cause 1), are two faults a document saved then can hold, and neither could
    /// be deleted: each delete left the other. A delete makes nothing worse in a part already
    /// broken, and is the way out of one, so it is refused only where the part built before it.
    ///
    /// Each first delete is asserted to leave the part still unable to be built, so that it is the
    /// narrowing letting it through rather than a delete that happened to mend the part; and the
    /// second delete then leaves a part that builds.
    #[test]
    fn a_part_with_two_broken_blocks_lets_either_be_deleted() {
        use super::App;
        use fieldform_ops::Document;
        use fieldform_scene::{Frame, Graph, Node, NodeId, ParamRef, Place};

        let lit = ParamRef::Literal;
        let doc = Document::new();
        let explain = |e: &fieldform_field::CompileError| e.to_string();
        let mut graph = Graph::empty();
        let twists: Vec<NodeId> = [0.0, 50.0]
            .into_iter()
            .map(|z| {
                let plane = graph.add(Node::Plane {
                    origin: Place::At([lit(0.0), lit(0.0), lit(z)]),
                    flipped: false,
                    turn: fieldform_scene::no_turn(),
                });
                let frame = Frame::From { node: plane };
                graph.add(Node::Twist { input: plane.into(), degrees_per_mm: lit(2.0), frame })
            })
            .collect();
        assert!(
            App::refusal_of(&doc, &graph, &explain).is_some(),
            "a twist reading the plane it is aimed at cannot be built",
        );

        for (gone, kept) in [(twists[0], twists[1]), (twists[1], twists[0])] {
            let after = App::after_deleting(&doc, &graph, gone, &explain)
                .unwrap_or_else(|why| panic!("deleting one broken twist was refused: {why}"));
            assert!(after.node(gone).is_none(), "the twist is deleted");
            assert!(after.node(kept).is_some(), "and the other one stays");
            assert!(
                App::refusal_of(&doc, &after, &explain).is_some(),
                "and the part still cannot be built, for the other twist",
            );
            let mended = App::after_deleting(&doc, &after, kept, &explain)
                .unwrap_or_else(|why| panic!("deleting the other broken twist was refused: {why}"));
            assert_eq!(App::refusal_of(&doc, &mended, &explain), None, "two deletes mend the part");
        }
    }

    /// **Deleting the last body leaves an empty part rather than a refusal.**
    ///
    /// The check that guards a delete asks the compilers, and both refuse a part with no bodies,
    /// rightly for a viewport: there is nothing to draw. But nothing to draw is not something that
    /// cannot be built, and a delete is the edit that arrives there on purpose -- refusing it would
    /// leave the last body of every part undeletable. A part left holding only a point counts,
    /// since a point is never a body.
    #[test]
    fn deleting_the_last_body_empties_the_part_rather_than_being_refused() {
        use super::App;
        use fieldform_ops::Document;
        use fieldform_scene::{Graph, Node, ParamRef, Place};

        let lit = ParamRef::Literal;
        let doc = Document::new();
        let mut graph = Graph::empty();
        let at = graph.add(Node::Point { at: [lit(0.0); 3] });
        let ball = graph.add(Node::Sphere { centre: Place::From { node: at }, radius: lit(10.0) });

        let after = App::after_deleting(&doc, &graph, ball, &|e| e.to_string())
            .unwrap_or_else(|why| panic!("deleting the last body was refused: {why}"));
        assert!(after.node(ball).is_none(), "the ball is deleted");
        assert!(after.node(at).is_some(), "the point it sat on stays");
        assert!(after.roots().is_empty(), "and no body is left");
    }

    /// **A block let go of is not drawn beside whatever consumes it next.**
    ///
    /// Pattern and Mirror leave the body they consume drawn as well, since it was on screen a
    /// moment before, and nothing cleared that when the body was let go of. So: pattern a sphere,
    /// delete the pattern, union the sphere with another -- and the sphere was drawn beside the
    /// union, asked for by a pattern that was no longer there (refactor plan §3.3, root cause 9).
    /// The flag goes with the delete, in the same undo entry: undoing the delete puts the sphere
    /// back beside its pattern, and redoing it lets go of it again.
    ///
    /// The control is a sphere a mirror still reads once its pattern is deleted. It is not let go
    /// of, so it goes on being drawn beside the mirror.
    ///
    /// Asserted through the rule the viewport draws by (`shown_ingredients_in` into `drawn_graph`,
    /// read through `folded_bodies` as `drawn_bodies` reads it) and the functions `delete_body`,
    /// `set_part_graph` and Ctrl+Z run, since an `App` needs a window and a device.
    #[test]
    fn a_block_let_go_of_is_not_drawn_beside_what_consumes_it_next() {
        use super::{
            drawn_graph, record_graph_edit_with, shown_ingredients_in, step_history, App,
            ParameterEdits,
        };
        use fieldform_ops::{CommandStack, Document};
        use fieldform_scene::{
            Axis, BooleanOp, Fillet, Frame, Graph, Node, NodeId, ParamRef, PatternKind, Place,
        };

        let lit = ParamRef::Literal;
        let ball = |x: f64| Node::Sphere {
            centre: Place::At([lit(x), lit(0.0), lit(0.0)]),
            radius: lit(10.0),
        };
        let pattern = |input| Node::Pattern {
            options: Default::default(),
            input,
            count: lit(4.0),
            kind: PatternKind::Radial { frame: Frame::at_along(Axis::Z, [lit(0.0); 3]) },
            join: false,
            fillet: Fillet::hard(),
        };
        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        let mut stack = CommandStack::default();
        let mut gesture = None;
        let mut edits = ParameterEdits::default();
        // What the viewport draws, by the block each is, in no particular order.
        let drawn = |doc: &Document| -> Vec<NodeId> {
            let graph = &doc.parts[&part].graph;
            let hidden = |id: NodeId| doc.node_meta.get(&id).is_some_and(|m| m.hidden);
            let (display, stand_ins) = drawn_graph(graph, &shown_ingredients_in(doc, graph), &hidden);
            let mut ids: Vec<NodeId> = doc
                .folded_bodies(&display)
                .into_iter()
                .map(|id| stand_ins.get(&id).copied().unwrap_or(id))
                .collect();
            ids.sort();
            ids
        };
        let sorted = |mut ids: Vec<NodeId>| {
            ids.sort();
            ids
        };
        let shown = |doc: &Document, id| doc.node_meta.get(&id).is_some_and(|m| m.shown);

        // A sphere patterned, with a second beside it to union it with.
        let mut graph = Graph::empty();
        let sphere = graph.add(ball(0.0));
        let other = graph.add(ball(30.0));
        let ring = graph.add(pattern(sphere.into()));
        record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, part, graph)
            .expect("the pattern");
        // As `apply_pattern` leaves it: drawn beside the pattern. Written straight in, since what
        // the pattern's own entry does with the flag is held by
        // `a_pattern_source_still_draws_after_undo_and_redo`.
        doc.node_meta.entry(sphere).or_default().shown = true;
        assert_eq!(drawn(&doc), sorted(vec![sphere, other, ring]), "the sphere beside its pattern");

        // The pattern deleted, which lets go of the sphere: a body again, drawn as one.
        let after = App::after_deleting(&doc, &doc.parts[&part].graph, ring, &|e| e.to_string())
            .unwrap_or_else(|why| panic!("deleting the pattern was refused: {why}"));
        record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, part, after)
            .expect("the delete");
        assert_eq!(drawn(&doc), sorted(vec![sphere, other]));

        // And unioned: the union is drawn, and the sphere is inside it rather than beside it.
        let mut joined = doc.parts[&part].graph.clone();
        let union = joined.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![sphere.into(), other.into()] });
        record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, part, joined)
            .expect("the union");
        assert_eq!(drawn(&doc), vec![union], "the union alone, with no sphere beside it");

        // Undoing the union and then the delete draws the sphere beside its pattern again.
        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
        assert_eq!(drawn(&doc), sorted(vec![sphere, other, ring]), "undo puts back what drew");
        assert!(shown(&doc, sphere));
        // And redoing the delete lets go of it again.
        assert!(step_history(&mut stack, &mut doc, &mut gesture, false).is_some());
        assert!(!shown(&doc, sphere), "redo is the same delete");
        assert_eq!(drawn(&doc), sorted(vec![sphere, other]));

        // The control: a sphere a mirror still reads is not let go of by its pattern's delete.
        let mut graph = Graph::empty();
        let read = graph.add(ball(0.0));
        let ring = graph.add(pattern(read.into()));
        let mirror = graph.add(Node::Mirror {
            input: read.into(),
            frame: Frame::at_along(Axis::X, [lit(0.0); 3]),
            join: false,
        });
        record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, part, graph)
            .expect("the pattern and the mirror");
        doc.node_meta.entry(read).or_default().shown = true;
        let after = App::after_deleting(&doc, &doc.parts[&part].graph, ring, &|e| e.to_string())
            .unwrap_or_else(|why| panic!("deleting the pattern was refused: {why}"));
        record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, part, after)
            .expect("the delete");
        assert!(shown(&doc, read), "the mirror still reads it, so nothing let go of it");
        assert_eq!(drawn(&doc), sorted(vec![read, mirror]), "drawn beside the mirror");
    }

    /// **One eye: it means `hidden` while a block is a body and `shown` once something consumes
    /// it, and a plane or a point means `hidden` whatever consumes it** (`block-ownership-spec.md`
    /// §7.2). Whichever flag that is, the eye draws or stops drawing the block, as one undo step.
    ///
    /// The eye used to speak to the two flags from places that chose for themselves, so the H key
    /// on a consumed block wrote `hidden`, which nothing draws by until the block is let go of, and
    /// the row beside it wrote `shown`. The rule is one function now and every eye asks it.
    /// Asserted through the rule the viewport draws by, read through the same visibility the app
    /// reads, since an `App` needs a window and a device.
    #[test]
    fn the_eye_draws_a_block_or_stops_whichever_flag_its_place_in_the_graph_gives_it() {
        use super::{
            drawn_graph, eye_command, eye_flag_in, is_visible_among, is_visible_in,
            shown_ingredients_in,
        };
        use fieldform_ops::{CommandStack, Document};
        use fieldform_scene::{
            no_turn, EyeFlag, Frame, Graph, Node, NodeId, NodeMeta, ParamRef, Place,
        };

        let lit = ParamRef::Literal;
        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let part = *doc.parts.keys().next().expect("with_cage makes one part");
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere { centre: Place::origin(), radius: lit(10.0) });
        let grown = graph.add(Node::Offset { input: ball.into(), distance: lit(2.0) });
        let pin = graph.add(Node::Point { at: [lit(30.0), lit(0.0), lit(0.0)] });
        let plane =
            graph.add(Node::Plane { origin: Place::origin(), flipped: false, turn: no_turn() });
        let other = graph.add(Node::Sphere { centre: Place::origin(), radius: lit(5.0) });
        let mirror = graph.add(Node::Mirror {
            input: other.into(),
            frame: Frame::From { node: plane },
            join: false,
        });
        doc.parts.get_mut(&part).expect("the part").graph = graph;

        let graph = doc.parts[&part].graph.clone();
        assert_eq!(eye_flag_in(&graph, grown), EyeFlag::Hidden, "a body");
        assert_eq!(eye_flag_in(&graph, ball), EyeFlag::Shown, "consumed by the offset");
        assert_eq!(eye_flag_in(&graph, pin), EyeFlag::Hidden, "a point, which is never a body");
        assert_eq!(eye_flag_in(&graph, plane), EyeFlag::Hidden, "a plane a frame reads");
        assert_eq!(eye_flag_in(&graph, mirror), EyeFlag::Hidden);

        let drawn = |doc: &Document| -> Vec<NodeId> {
            let graph = &doc.parts[&part].graph;
            // As `display_graph_with_stand_ins` asks it: the consumed set built once.
            let consumed = graph.consumed();
            let hidden = |id: NodeId| !is_visible_among(doc, graph, &consumed, id);
            let (display, stand_ins) = drawn_graph(graph, &shown_ingredients_in(doc, graph), &hidden);
            let mut ids: Vec<NodeId> = doc
                .folded_bodies(&display)
                .into_iter()
                .map(|id| stand_ins.get(&id).copied().unwrap_or(id))
                .collect();
            ids.sort();
            ids
        };
        let sorted = |mut ids: Vec<NodeId>| {
            ids.sort();
            ids
        };
        let bodies = sorted(vec![grown, mirror]);
        assert_eq!(drawn(&doc), bodies, "the bodies, and nothing they consume");
        assert!(is_visible_in(&doc, &graph, pin) && is_visible_in(&doc, &graph, plane));
        assert!(!is_visible_in(&doc, &graph, ball));

        let mut stack = CommandStack::default();
        // The consumed ball's eye on: drawn beside what consumes it.
        stack.execute(&mut doc, Box::new(eye_command(&graph, ball, true))).expect("the eye");
        assert_eq!(drawn(&doc), sorted(vec![grown, mirror, ball]), "the ball beside its offset");
        assert_eq!(doc.node_meta.get(&ball), Some(&NodeMeta { shown: true, ..NodeMeta::default() }));
        // The body's eye off: it goes, and the ball asked for stays.
        stack.execute(&mut doc, Box::new(eye_command(&graph, grown, false))).expect("the eye");
        assert_eq!(drawn(&doc), sorted(vec![mirror, ball]));
        // The plane's eye off, which is the flag its overlay reads, though a frame consumes it.
        stack.execute(&mut doc, Box::new(eye_command(&graph, plane, false))).expect("the eye");
        assert!(!is_visible_in(&doc, &graph, plane));
        assert!(doc.node_meta.get(&plane).is_some_and(|m| m.hidden && !m.shown));

        for _ in 0..3 {
            stack.undo(&mut doc).expect("one step each");
        }
        assert_eq!(drawn(&doc), bodies, "undone, the picture it started with");
        assert!(doc.node_meta.is_empty(), "and nothing left behind: {:?}", doc.node_meta);
    }

    /// **A walk over many blocks reads each eye exactly as asking that block alone does.** The
    /// viewport reads the eye of every body each frame, and the eye's rule asks whether anything
    /// reads the block. Asked of the graph a block at a time, that walks the whole graph per block,
    /// so the frame's drawn list grew as bodies times blocks where it had been one flag read a
    /// body. A walk builds the graph's consumed set once and hands it to `is_visible_among`, and a
    /// second reading of one rule is a second chance to disagree with it: so every kind of block
    /// the rule tells apart -- a body, one something consumes, a plane a frame reads and one on its
    /// own, a point a ramp reads and one on its own, and the ramp -- with no entry and under every
    /// setting of the two flags.
    #[test]
    fn a_walk_over_every_block_reads_each_eye_as_asking_that_block_alone_does() {
        use super::{is_visible_among, is_visible_in};
        use fieldform_ops::Document;
        use fieldform_scene::{
            no_turn, Frame, Graph, Node, NodeId, NodeMeta, ParamRef, Place, RampShape,
        };

        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let ball = graph.add(Node::Sphere { centre: Place::origin(), radius: lit(10.0) });
        graph.add(Node::Offset { input: ball.into(), distance: lit(2.0) });
        let plane =
            graph.add(Node::Plane { origin: Place::origin(), flipped: false, turn: no_turn() });
        graph.add(Node::Plane { origin: Place::origin(), flipped: false, turn: no_turn() });
        let other = graph.add(Node::Sphere { centre: Place::origin(), radius: lit(5.0) });
        graph.add(Node::Mirror { input: other.into(), frame: Frame::From { node: plane }, join: false });
        let pin = graph.add(Node::Point { at: [lit(30.0), lit(0.0), lit(0.0)] });
        graph.add(Node::Point { at: [lit(0.0); 3] });
        graph.add(Node::Ramp {
            input: pin.into(),
            in_min: lit(0.0),
            in_max: lit(10.0),
            out_min: lit(0.0),
            out_max: lit(1.0),
            shape: RampShape::Linear,
        });

        let consumed = graph.consumed();
        let ids: Vec<NodeId> = graph.nodes().map(|(id, _)| id).collect();
        let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
        let flags = [
            None,
            Some((false, false)),
            Some((true, false)),
            Some((false, true)),
            Some((true, true)),
        ];
        for set in flags {
            doc.node_meta.clear();
            if let Some((hidden, shown)) = set {
                for id in &ids {
                    doc.node_meta.insert(*id, NodeMeta { hidden, shown, ..NodeMeta::default() });
                }
            }
            for id in &ids {
                assert_eq!(
                    is_visible_among(&doc, &graph, &consumed, *id),
                    is_visible_in(&doc, &graph, *id),
                    "{:?} with (hidden, shown) {set:?}",
                    graph.node(*id),
                );
            }
        }
    }

    /// **A pattern's source still draws after the pattern is undone and redone**
    /// (`block-ownership-spec.md` §7.3, test 13), and a mirror's the same.
    ///
    /// Each consumes the body it is made from and turns that body's eye on, since it was on screen
    /// a moment before. The graph edit and the eye were two entries, so Ctrl+Z took two presses to
    /// take the pattern back, and the one Ctrl+Y after them brought the pattern back with its
    /// source no longer drawn: the eye was the other entry, and nothing had redone it. As one
    /// entry, the eye after the edit, a redo replays both in that order.
    ///
    /// And a drag straight after is an entry of its own. The pattern opens no gesture, so changing
    /// a number inside the window does not fold the pattern's entry into the drag's, which would
    /// take the eye off the history while the document kept it drawn. Held before any undo, since
    /// an undo ends every gesture and would hide one the pattern had opened.
    ///
    /// Through the functions `set_part_graph_with` and Ctrl+Z run, and the rule the viewport draws
    /// by, since an `App` needs a window and a device.
    #[test]
    fn a_pattern_source_still_draws_after_undo_and_redo() {
        use super::{
            drawn_graph, eye_command, is_visible_among, record_graph_edit_then,
            record_graph_edit_with, shown_ingredients_in, step_history, ParameterEdits,
        };
        use fieldform_ops::{Command, CommandStack, Document};
        use fieldform_scene::{
            Axis, Fillet, Frame, Graph, Node, NodeId, ParamRef, PatternKind, Place,
        };

        let lit = ParamRef::Literal;
        let pattern = |input: NodeId| Node::Pattern {
            options: Default::default(),
            input: input.into(),
            count: lit(4.0),
            kind: PatternKind::Radial { frame: Frame::at_along(Axis::Z, [lit(0.0); 3]) },
            join: false,
            fillet: Fillet::hard(),
        };
        let mirror = |input: NodeId| Node::Mirror {
            input: input.into(),
            frame: Frame::at_along(Axis::X, [lit(0.0); 3]),
            join: false,
        };
        let sorted = |mut ids: Vec<NodeId>| {
            ids.sort();
            ids
        };
        let consumers: [(&str, &dyn Fn(NodeId) -> Node); 2] =
            [("pattern", &pattern), ("mirror", &mirror)];
        for (what, consume) in consumers {
            let mut doc = Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
            let part = *doc.parts.keys().next().expect("with_cage makes one part");
            let mut graph = Graph::empty();
            let ball = graph.add(Node::Sphere { centre: Place::origin(), radius: lit(10.0) });
            doc.parts.get_mut(&part).expect("the part").graph = graph.clone();
            // What the viewport draws, by the block each is, as `display_graph_with_stand_ins`
            // asks it.
            let drawn = |doc: &Document| -> Vec<NodeId> {
                let graph = &doc.parts[&part].graph;
                let consumed = graph.consumed();
                let hidden = |id: NodeId| !is_visible_among(doc, graph, &consumed, id);
                let (display, stand_ins) =
                    drawn_graph(graph, &shown_ingredients_in(doc, graph), &hidden);
                let ids: Vec<NodeId> = doc
                    .folded_bodies(&display)
                    .into_iter()
                    .map(|id| stand_ins.get(&id).copied().unwrap_or(id))
                    .collect();
                sorted(ids)
            };
            let mut stack = CommandStack::default();
            let mut gesture = None;
            let mut edits = ParameterEdits::default();

            // As `apply_pattern` and `apply_mirror` issue it: the eye asked of the graph the edit
            // writes, where the ball is consumed.
            let made = graph.add(consume(ball));
            let keep: Vec<Box<dyn Command>> = vec![Box::new(eye_command(&graph, ball, true))];
            record_graph_edit_then(
                &mut stack,
                &mut doc,
                &mut gesture,
                &mut edits,
                part,
                graph,
                keep,
            )
            .unwrap_or_else(|e| panic!("the {what}: {e}"));
            assert_eq!(drawn(&doc), sorted(vec![ball, made]), "{what}: the ball beside it");

            // The ball's radius dragged straight after, inside the window: its own entry, so one
            // Ctrl+Z takes back the drag and leaves the ball drawn beside what consumes it.
            let mut dragged = doc.parts[&part].graph.clone();
            if let Some(Node::Sphere { radius, .. }) = dragged.node_mut(ball) {
                *radius = lit(12.0);
            }
            record_graph_edit_with(&mut stack, &mut doc, &mut gesture, &mut edits, part, dragged)
                .expect("the drag");
            assert!(step_history(&mut stack, &mut doc, &mut gesture, true).is_some());
            assert!(
                doc.parts[&part].graph.node(made).is_some(),
                "{what}: undoing the drag took the {what} with it",
            );
            assert_eq!(drawn(&doc), sorted(vec![ball, made]), "{what}: and the ball still drawn");

            // Ctrl+Z until it is gone, then Ctrl+Y once.
            let mut presses = 0;
            while doc.parts[&part].graph.node(made).is_some() {
                assert!(
                    step_history(&mut stack, &mut doc, &mut gesture, true).is_some(),
                    "{what}: the history ran out with it still there",
                );
                presses += 1;
            }
            assert_eq!(drawn(&doc), vec![ball], "{what}: undone, the ball is a body again");
            assert!(step_history(&mut stack, &mut doc, &mut gesture, false).is_some());
            assert_eq!(
                drawn(&doc),
                sorted(vec![ball, made]),
                "{what}: redone, the ball is still drawn beside it",
            );
            assert_eq!(presses, 1, "{what}: one act, one undo");
        }
    }

    /// **Realign mends a relation found broken on open, and one undo breaks it again.** The part of
    /// the relations work that had no test of its own: the check that marks a relation broken and
    /// the re-derivation that runs after are tested elsewhere; this is the clearing between them.
    #[test]
    fn realigning_a_broken_relation_mends_it_and_one_undo_breaks_it_again() {
        use super::realign_commands;
        use fieldform_ops::{Batch, CommandStack, Document};
        use nalgebra::Point3;
        use fieldform_scene::relation::{check, rederive};
        use fieldform_scene::{
            Curve, CurveEnd, CurveId, End, JointContinuity, Relation, RelationRecord,
        };

        let master = Curve::polyline(
            CurveId::new(),
            "Line",
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 0.0, 0.0)],
            false,
        )
        .expect("a line");
        let degree = 3;
        let knots: Vec<f64> = std::iter::repeat_n(0.0, degree + 1)
            .chain(std::iter::repeat_n(1.0, degree + 1))
            .collect();
        let slave = Curve::new(
            CurveId::new(),
            "Curve",
            degree,
            vec![
                Point3::new(10.0, 0.0, 0.0),
                Point3::new(14.0, 3.0, 0.0),
                Point3::new(20.0, 6.0, 0.0),
                Point3::new(25.0, 0.0, 0.0),
            ],
            vec![1.0; 4],
            knots,
            false,
        )
        .expect("a Bézier");
        let (m, s) = (master.id, slave.id);
        let mut doc = Document::new();
        doc.curves.insert(m, master);
        doc.curves.insert(s, slave);
        let id = fieldform_scene::RelationId::new();
        doc.relations.insert(
            id,
            RelationRecord {
                relation: Relation::Joint {
                    master: CurveEnd { curve: m, end: End::End },
                    slave: CurveEnd { curve: s, end: End::Start },
                    continuity: JointContinuity::G1,
                },
                broken: None,
            },
        );
        rederive(&mut doc);
        assert!(check(&doc).is_empty(), "it holds as derived");
        let held = doc.curve(s).unwrap().control_points()[1];

        // An older build moved the derived point off the tangent and saved; open found it broken.
        let off = Point3::new(14.0, 3.0, 0.0);
        doc.curve_mut(s).unwrap().set_control_point(1, off);
        let found = check(&doc);
        assert_eq!(found.len(), 1);
        doc.relations.get_mut(&id).unwrap().broken = Some(found[0].1.clone());
        assert!(rederive(&mut doc).is_empty(), "a broken relation is not applied");

        let (commands, moved, said) = realign_commands(&doc, &[id]).expect("something to mend");
        assert_eq!(moved.len(), 1, "one point goes back on the tangent");
        assert_eq!(said.len(), 1);
        let mut stack = CommandStack::default();
        stack
            .execute(&mut doc, Box::new(Batch::new("relation.realign", commands)))
            .unwrap();
        assert!(doc.relations[&id].broken.is_none(), "the mark is cleared");
        assert!(
            (doc.curve(s).unwrap().control_points()[1] - held).norm() < 1e-9,
            "re-derived from the master as it is now"
        );
        assert!(check(&doc).is_empty(), "and it holds again");

        // One undo, because it was one batch: the mark and the point both come back.
        assert_eq!(stack.undo(&mut doc).as_deref(), Some("relation.realign"));
        assert!(doc.relations[&id].broken.is_some(), "broken again");
        assert_eq!(doc.curve(s).unwrap().control_points()[1], off, "and the point is where it was");

        // A relation that is not broken has nothing to mend.
        doc.relations.get_mut(&id).unwrap().broken = None;
        let nothing: Vec<fieldform_scene::RelationId> = Vec::new();
        assert!(realign_commands(&doc, &nothing).is_none());
    }

    /// No word may mean two things, whichever list it came from.
    #[test]
    fn no_console_word_is_claimed_twice() {
        let commands = console_commands();
        let mut seen = std::collections::HashSet::new();
        for (word, _) in &commands {
            assert!(seen.insert(word.clone()), "{word:?} is listed more than once");
        }
    }

    /// A degree-9 curve saved earlier still opens and edits.
    ///
    /// Degree 9 left `CURVE_DEGREES` on 2026-09-24 (`profile-group-spec.md` §7): 7 already
    /// carries G3 at both ends. Only the picker changed. A curve of degree 9 from a file
    /// written before that day is still built, still moves a control point, and can still be
    /// taken to an offered degree — refusing it would destroy saved work. The reading half of
    /// this is `a_degree_nine_curve_saved_earlier_still_opens` in `fieldform-io`.
    #[test]
    fn a_degree_nine_curve_saved_earlier_still_opens_and_edits() {
        assert!(!CURVE_DEGREES.contains(&9), "9 is no longer offered");
        let id = fieldform_scene::CurveId::new();
        let mut nine = single_span_bezier(id, 9, 50.0).expect("a saved degree-9 curve still builds");
        assert_eq!(nine.degree(), 9);
        assert_eq!(nine.control_points().len(), 10);

        // Editing: a point moves, and the curve follows it.
        let moved = nalgebra::Point3::new(20.0, 30.0, 5.0);
        nine.set_control_point(4, moved);
        assert_eq!(nine.control_points()[4], moved);
        let (lo, hi) = nine.domain();
        let mid = nine.evaluate((lo + hi) * 0.5);
        assert!(mid.z > 0.0, "the curve did not follow the moved point: {mid:?}");

        // And it can be brought to an offered degree, exactly upward and approximately down.
        let seven = nine.with_degree(7).expect("lowering to 7 approximates and says so");
        assert_eq!(seven.degree(), 7);
        let (a, b) = nine.ends();
        let (c, d) = seven.ends();
        assert!((a - c).norm() < 1e-9 && (b - d).norm() < 1e-9, "the ends are pinned");
    }

    /// Phase 2 of `profile-group-spec.md`: joints, at the level of the document and its
    /// commands, which is where the rules live so that they can be run without a window.
    mod joints {
        use super::super::{
            drag_targets, ends_on_point, group_by_curve, join_loop_commands, joint_choices,
            move_point_node, moved_to_start_at, newly_closed_rows, point_literals,
            points_hidden_with_rows, points_in_box, set_targets, settle_curves_on_points,
            single_span_bezier, split_selection, toggle_in, ungroup_commands,
        };
        use fieldform_cage::Cage;
        use fieldform_ops::{Batch, CommandStack, Document, SetCurve};
        use fieldform_scene::{Curve, CurveId, Graph, Node, NodeId, PartId};
        use nalgebra::Point3;

        /// Four lines making a square, in a document with one part and nothing in its graph.
        fn square_document() -> (Document, PartId, Vec<CurveId>) {
            let mut doc = Document::with_cage(Cage::cube_of(10.0));
            let part = *doc.parts.keys().next().expect("with_cage makes one part");
            doc.parts.get_mut(&part).expect("one part").graph = Graph::empty();
            let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
            let sides = [
                (p(0.0, 0.0), p(10.0, 0.0)),
                (p(10.0, 0.0), p(10.0, 10.0)),
                (p(10.0, 10.0), p(0.0, 10.0)),
                (p(0.0, 10.0), p(0.0, 0.0)),
            ];
            let mut square = Vec::new();
            for (n, (a, b)) in sides.into_iter().enumerate() {
                let id = CurveId::new();
                doc.curves.insert(
                    id,
                    Curve::polyline(id, format!("Line {}", n + 1), vec![a, b], false)
                        .expect("a line"),
                );
                doc.curves_created.insert(id, n as u64 + 1);
                square.push(id);
            }
            (doc, part, square)
        }

        fn points_in(doc: &Document, part: PartId) -> Vec<NodeId> {
            doc.parts
                .get(&part)
                .expect("the part")
                .graph
                .nodes()
                .filter(|(_, n)| matches!(n, Node::Point { .. }))
                .map(|(id, _)| id)
                .collect()
        }

        /// How many ends of `curves` sit on a point.
        fn ends_held(doc: &Document, curves: &[CurveId]) -> usize {
            curves
                .iter()
                .filter_map(|id| doc.curve(*id))
                .map(|c| {
                    let n = c.control_points().len();
                    [0, n - 1].iter().filter(|i| c.placed_on(**i).is_some()).count()
                })
                .sum()
        }

        fn every_control_point_bits(doc: &Document, curves: &[CurveId]) -> Vec<[u64; 3]> {
            curves
                .iter()
                .filter_map(|id| doc.curve(*id))
                .flat_map(|c| c.control_points().iter().map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]))
                .collect()
        }

        /// **Every curve is drawn in exactly one place — a row of its own, or inside the row that
        /// heads its loop.**
        ///
        /// The invariant a curve vanished for. `profile_block` decided which members to skip by reading
        /// `declared_curves` itself, which disagreed with `folded_curve_row_in` about the loop's *key*:
        /// the key's row **is** the profile row, so declaring it gives it no row of its own, but the
        /// member list skipped it anyway and it was drawn nowhere. Reported as "I nested a profile in
        /// an extrude then went to the V on the first curve and now the curve is gone completely."
        /// Held over every ownership a member can have, since that is what the V writes now, and
        /// with and without a block reading the variables, since a member a block reads stays in.
        #[test]
        fn every_curve_is_drawn_in_exactly_one_place() {
            use super::super::{folded_curve_row_in, Block, ProfileRows};
            use fieldform_scene::Ownership;
            let square: Vec<CurveId> = (0..4).map(|_| CurveId::new()).collect();
            let loose = CurveId::new();
            let profiles = ProfileRows { at: None, loops: vec![square.clone()] };

            // Each case: the variables, how every other curve is held, and whether the variables
            // are read — a block holding a curve reads it, and a free one is read by nothing.
            for (variables, other, variables_read) in [
                (vec![], Ownership::Free, false),
                (vec![square[0]], Ownership::Free, false),
                (vec![square[2]], Ownership::Nested, false),
                (vec![square[2]], Ownership::Nested, true),
                (square.clone(), Ownership::Free, false),
                (square.clone(), Ownership::Free, true),
                (vec![square[0], square[2]], Ownership::Nested, false),
                (vec![square[0], square[2]], Ownership::Free, true),
            ] {
                let mut doc = Document::new();
                for &id in square.iter().chain(std::iter::once(&loose)) {
                    let ownership =
                        if variables.contains(&id) { Ownership::Variable } else { other };
                    doc.ownership_curves.insert(id, ownership);
                }
                let read = |id: CurveId| {
                    let held = !variables.contains(&id) && other == Ownership::Nested;
                    usize::from(held || (variables.contains(&id) && variables_read))
                };
                let declared = &variables;
                for &id in square.iter().chain(std::iter::once(&loose)) {
                    let row = folded_curve_row_in(&profiles, &doc, &read, id);
                    let own_row = matches!(row, Some(Block::Curve(_)));
                    let heads_row = matches!(row, Some(Block::Profile(_)));
                    // `None` is what `profile_block` draws as a member, so it is a place, not nowhere.
                    let inside = row.is_none();
                    assert_eq!(
                        [own_row, heads_row, inside].iter().filter(|b| **b).count(),
                        1,
                        "curve is in no place or two with {declared:?}",
                    );
                    // The key heads its row whatever is declared, because it has nowhere else to be.
                    if id == square[0] {
                        assert!(heads_row, "the key stopped heading its row with {declared:?}");
                    } else if id != loose {
                        // And a member's place is its ownership's and its readers': out if it is a
                        // variable nothing reads, which is what its own V makes it, and inside the
                        // row however else it is held — a block reading it reads the loop.
                        assert_eq!(
                            own_row,
                            declared.contains(&id) && read(id) == 0,
                            "{id:?} with {declared:?}, read: {variables_read}",
                        );
                    }
                }
            }
        }

        /// **A join leaves a corner loose rather than binding an end a relation derives.**
        ///
        /// The join was the one path not asking `owners_of` — the `set` word and the curve panel's
        /// bind both consult it before putting a point on a block. So an ordinary loop close bound an
        /// end a mirror or a concentric claims, and `Curve::on_point` became a lie: `placed_on` said
        /// the end was the block's while `control_points` held the relation's answer. The other end on
        /// that same joint did follow the block, so the corner tore open.
        ///
        /// The whole meeting is skipped, not the claimed end: binding only the free end would make a
        /// half-joint, which looks joined and behaves as though it is not.
        #[test]
        fn a_join_leaves_an_end_a_relation_derives_loose() {
            use fieldform_scene::{
                relation::{Relation, RelationRecord},
                RelationId,
            };
            let (mut doc, part, square) = square_document();

            // Unclaimed, so all four corners are joined: the state before any relation exists.
            let (_, joints, loose) = join_loop_commands(&doc, part, &square);
            assert_eq!(joints, 4, "four corners, four joints");
            assert!(loose.is_empty(), "and nothing left loose");

            // A concentric claims every index of its slave, so both of that curve's ends are owned
            // and the two corners it sits in must be left alone.
            doc.relations.insert(
                RelationId::new(),
                RelationRecord {
                    relation: Relation::Concentric { master: square[0], slave: square[2] },
                    broken: None,
                },
            );
            let (commands, joints, loose) = join_loop_commands(&doc, part, &square);
            assert_eq!(joints, 2, "the two corners the relation claims are left loose");
            assert_eq!(loose.len(), 1, "and the relation is named once: {loose:?}");

            // **Nothing that was bound got bound to a claimed end.** The point of skipping the
            // meeting rather than the end is that no half-joint is made, so every end this did bind
            // is one the relation does not own.
            let mut doc_after = doc.clone();
            let mut stack = CommandStack::default();
            stack
                .execute(&mut doc_after, Box::new(Batch::new("profile.join", commands)))
                .expect("the joins apply");
            for id in &square {
                let Some(c) = doc_after.curve(*id) else { continue };
                let n = c.control_points().len();
                for i in [0, n - 1] {
                    if c.placed_on(i).is_some() {
                        assert!(
                            fieldform_scene::relation::owners_of(&doc_after, *id, i, None)
                                .is_empty(),
                            "an end a relation derives was bound to a block anyway",
                        );
                    }
                }
            }
        }

        /// **Moving a master retraces the view, because its slave shapes a body.**
        ///
        /// The gate asked only about the curve in the hand — does a block read it, does it sit on a
        /// point a block reads — so a loose master whose *slave* is an extrude's profile answered no.
        /// The solid then stayed frozen while both overlay lines moved, and stayed frozen: orbiting,
        /// zooming and selecting all redraw from the same stale shader and the same stale
        /// `pick_field`, so clicking the solid picked the old shape too. Written down as known at the
        /// profile-groups merge, never run, and reproduced on 2026-10-01 as worse than described.
        ///
        /// Chained deliberately — master drives middle drives profile — because relations chain and a
        /// walk that stopped at one step would pass a one-link test and still lie.
        #[test]
        fn moving_a_master_counts_as_shaping_what_its_slave_shapes() {
            use super::super::curve_shapes_a_body_in;
            use fieldform_scene::{
                relation::{Relation, RelationRecord},
                ParamRef, RelationId,
            };
            let (mut doc, part, square) = square_document();
            let (master, middle, profile) = (square[0], square[1], square[2]);

            // Only the third curve is read by anything: an extrude made from it.
            let extrude = doc.parts.get_mut(&part).expect("one part").graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(profile),
                sweep: fieldform_scene::Sweep::Normal,
                height: ParamRef::Literal(10.0),
                start: ParamRef::Literal(0.0),
                sides: fieldform_scene::Sides::One,
                draft: ParamRef::Literal(0.0),
                corners: Default::default(),
            });
            let _ = extrude;
            let graph = || doc.parts.get(&part).expect("one part").graph.clone();

            assert!(
                curve_shapes_a_body_in(&doc, &graph(), profile),
                "the extrude reads it directly, which always worked",
            );
            assert!(
                !curve_shapes_a_body_in(&doc, &graph(), master),
                "and nothing reads the master yet",
            );

            // master -> middle -> profile, so the master's move reaches the extrude two steps out.
            for (a, b) in [(master, middle), (middle, profile)] {
                doc.relations.insert(
                    RelationId::new(),
                    RelationRecord {
                        relation: Relation::EqualLength { master: a, slave: b },
                        broken: None,
                    },
                );
            }
            assert!(
                curve_shapes_a_body_in(&doc, &graph(), master),
                "moving the master re-derives the middle, which re-derives the profile",
            );

            // **A broken relation is skipped, because `rederive` skips it.** The gate has to agree
            // with what will actually move rather than with what was declared.
            for record in doc.relations.values_mut() {
                record.broken = Some("by hand, for the test".to_string());
            }
            assert!(
                !curve_shapes_a_body_in(&doc, &graph(), master),
                "a broken relation moves nothing, so there is nothing to retrace for",
            );
        }

        /// **Deleting a whole row takes its joints with it.**
        ///
        /// It did not, from the profile-groups merge until 2026-10-01: `delete_profile` went straight
        /// to `delete_curves_commands`, which knows about curves and relations and nothing about
        /// points, so a deleted square left four rows reading "Point" and four dots at corners of a
        /// square that was gone — saved and reopened with them, and offered in every blend's operand
        /// list. Written down as known and never run, which is why it survived four months.
        ///
        /// Asserted here rather than in the panel because an `App` needs a window and a device, and
        /// this is a rule about commands.
        #[test]
        fn deleting_a_row_takes_its_joints_with_it() {
            use super::super::delete_row_commands;
            let (mut doc, part, square, mut stack) = joined_square();
            assert_eq!(points_in(&doc, part).len(), 4, "four corners, four joints");

            let commands = delete_row_commands(&doc, part, &square);
            stack
                .execute(&mut doc, Box::new(Batch::new("profile.delete", commands)))
                .expect("the delete applies");
            assert!(
                points_in(&doc, part).is_empty(),
                "the joints went with the row: {:?}",
                points_in(&doc, part),
            );
            assert!(
                square.iter().all(|id| doc.curve(*id).is_none()),
                "and so did the curves",
            );

            // **One entry, undone in one go, and the points come back before the curves that sit on
            // them.** The graph edit is last in the batch so it is undone first; the other order
            // restores a curve whose end names a point that does not exist yet.
            stack.undo(&mut doc).expect("one undo");
            assert_eq!(points_in(&doc, part).len(), 4, "the joints came back");
            assert!(
                square.iter().all(|id| doc.curve(*id).is_some()),
                "and the curves with them",
            );
            assert_eq!(
                ends_held(&doc, &square),
                8,
                "every end back on the point it was on",
            );
        }

        /// **A curve can be taken out of a loop a block is built on, and the block follows.**
        /// Asked for the first day a block could be built on an open chain: deleting a side of an
        /// extruded square was refused, since the extrude read the loop. The side goes, the two
        /// joints at its ends go (the chain's other two stay, each still joining two curves), and
        /// the extrude is a wall of the last thickness along the three sides left — pointed past
        /// the deleted curve when that was the one it named. One entry, undone in one go.
        #[test]
        fn deleting_a_side_of_an_extruded_square_leaves_a_wall_along_the_other_three() {
            use super::super::{delete_loop_member_commands, SetPartGraph};
            use fieldform_scene::{Outline, ParamRef, WallSide};
            let (mut doc, part, square, mut stack) = joined_square();
            let gone = square[0];
            // An extrude naming the very curve that goes, so it has to be pointed elsewhere.
            let before = doc.parts[&part].graph.clone();
            let mut graph = before.clone();
            let extrude = graph.add(Node::Extrude { rims: Default::default(),
                outline: Outline::closed(gone),
                sweep: fieldform_scene::Sweep::Normal,
                height: ParamRef::Literal(10.0),
                draft: ParamRef::Literal(0.0),
                start: ParamRef::Literal(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
            });
            stack
                .execute(&mut doc, Box::new(SetPartGraph::new(part, before, graph)))
                .expect("the extrude is added");
            let mates = fieldform_field::loop_from(&doc, gone, None).expect("a loop");
            assert_eq!(mates.len(), 4);

            let (commands, _, done) =
                delete_loop_member_commands(&doc, part, gone, &mates, 2.5).expect("a loop member");
            assert_eq!(done.walled, vec![extrude]);
            assert_eq!(done.joints, 2, "the two joints at the deleted side's ends");
            stack
                .execute(&mut doc, Box::new(Batch::new("curve.delete", commands)))
                .expect("the delete applies");
            assert!(doc.curve(gone).is_none());
            assert_eq!(points_in(&doc, part).len(), 2, "the other two joints stay");
            assert_eq!(ends_held(&doc, &square[1..]), 4, "two curves still held at each of them");
            match doc.parts[&part].graph.node(extrude).and_then(Node::outline) {
                Some(Outline::Wall { curve, thickness, side, .. }) => {
                    assert!(square[1..].contains(curve), "pointed at a curve that is still there");
                    assert_eq!(*thickness, ParamRef::Literal(2.5));
                    assert_eq!(*side, WallSide::Centre);
                }
                other => panic!("the extrude should be a wall now, not {other:?}"),
            }
            // And the chain the wall runs along is the three sides, open.
            let chain = fieldform_field::outline_chain(&doc, square[1], None).expect("a chain");
            assert!(!chain.closed);
            assert_eq!(chain.curves.len(), 3);

            stack.undo(&mut doc).expect("one undo");
            assert!(doc.curve(gone).is_some(), "the side is back");
            assert_eq!(points_in(&doc, part).len(), 4, "and its joints");
            assert_eq!(ends_held(&doc, &square), 8, "every end back on its point");
            assert_eq!(
                doc.parts[&part].graph.node(extrude).and_then(Node::outline),
                Some(&Outline::closed(gone)),
                "and the extrude reads the loop again"
            );
        }

        /// **The curve an extrude names comes out by its V when its loop opens, and goes back** —
        /// backlog item 73's reported case, as the list draws it (refactor plan 2.19,
        /// `block-ownership-spec.md` §10).
        ///
        /// "All of the curves went outside of the extrude body, except for the first one. That one
        /// is now trapped inside extrude. And I cannot release it." An extrude names one curve of
        /// its loop and walks the rest, so that curve alone is the extrude's own. While the loop
        /// closes, its profile row is drawn inside the extrude; break one joint and the loop opens,
        /// the row is gone, the three curves the extrude does not name stand on their own, and the
        /// one it names is inside it. Its row's V — `make_variable` and `put_back_inside`, which
        /// are what it calls — takes it out and puts it back, each one undo.
        #[test]
        fn the_curve_an_extrude_names_comes_out_by_its_v_when_its_loop_opens_and_goes_back() {
            use super::super::{make_variable, put_back_inside, Block, SetPartGraph};
            use fieldform_scene::BlockKey;
            let (mut doc, part, square, mut stack) = joined_square();
            let extrude = read_through(&mut stack, &mut doc, part, extrusion(square[0]));
            let rows = |doc: &Document| square_rows(doc, part, &square, &[extrude]);
            let key = fieldform_field::profile_partition(&doc)[0][0];
            assert_eq!(
                rows(&doc),
                vec![(Block::Node(extrude), vec![Block::Profile(key)])],
                "closed, the profile row is inside the extrude",
            );

            // One joint broken, as pulling a corner apart breaks it: the two ends on the corner the
            // second and third sides share taken off its point, the point gone, and the second
            // side's end drawn 4 mm clear.
            let corner = doc.curve(square[1]).and_then(|c| c.placed_on(1)).expect("held");
            let second = doc.curve(square[1]).cloned().expect("a side");
            let third = doc.curve(square[2]).cloned().expect("a side");
            let (mut pulled, mut let_go) = (second.clone(), third.clone());
            pulled.place_on(1, None);
            pulled.set_control_point(1, Point3::new(10.0, 14.0, 0.0));
            let_go.place_on(0, None);
            let before = doc.parts[&part].graph.clone();
            let mut graph = before.clone();
            assert!(graph.remove(corner), "the joint is a point of the part");
            let broken: Vec<Box<dyn fieldform_ops::Command>> = vec![
                Box::new(SetCurve::new(square[1], Some(second), Some(pulled))),
                Box::new(SetCurve::new(square[2], Some(third), Some(let_go))),
                Box::new(SetPartGraph::new(part, before, graph)),
            ];
            stack
                .execute(&mut doc, Box::new(Batch::new("joint.break", broken)))
                .expect("the joint is broken");
            assert!(fieldform_field::profile_partition(&doc).is_empty(), "the loop is open");
            let trapped = vec![
                (Block::Curve(square[1]), vec![]),
                (Block::Curve(square[2]), vec![]),
                (Block::Curve(square[3]), vec![]),
                (Block::Node(extrude), vec![Block::Curve(square[0])]),
            ];
            assert_eq!(rows(&doc), trapped, "the one the extrude names is inside it");

            let named = BlockKey::Curve(square[0]);
            let name = doc.curve(square[0]).map(|c| c.name.clone());
            let out = make_variable(&doc, named, || panic!("a curve is born named"))
                .expect("inside the extrude, so its V can make it a variable");
            stack.execute(&mut doc, Box::new(out)).expect("its V");
            let free: Vec<(Block, Vec<Block>)> = square
                .iter()
                .map(|id| (Block::Curve(*id), vec![]))
                .chain([(Block::Node(extrude), vec![])])
                .collect();
            assert_eq!(rows(&doc), free, "out of the extrude, which still reads it");
            stack.undo(&mut doc).expect("one undo");
            assert_eq!(rows(&doc), trapped, "undone, inside it again");
            stack.redo(&mut doc).expect("one redo");
            assert_eq!(rows(&doc), free);

            // The extrude reads it, and nothing else, so it goes back inside the extrude.
            let back = put_back_inside(&doc, named).expect("one reader");
            stack.execute(&mut doc, Box::new(back)).expect("its V again");
            assert_eq!(rows(&doc), trapped, "back inside the extrude");
            assert_eq!(doc.curve(square[0]).map(|c| c.name.clone()), name, "its name kept");
            stack.undo(&mut doc).expect("one undo");
            assert_eq!(rows(&doc), free, "undone, a variable again");
        }

        /// **A console line calls only the blocks the list gives a row, so never a held loop's
        /// joint.**
        ///
        /// `variable` and `inline` found their row among every block of the part, joints included
        /// -- unnamed points the list draws inside their profile row (§4.3) and gives no V. So
        /// beside a held square, the one point made from the menu, which the list shows as one
        /// "Point" row, was refused as what five blocks show; and with no such point,
        /// `variable 'Point'` made a joint a variable, which no V can do.
        #[test]
        fn a_console_line_calls_only_the_blocks_the_list_gives_a_row_so_never_a_joint() {
            use super::super::ProfileRows;
            use crate::blocks::list::blocks_a_line_can_call;
            use fieldform_scene::ParamRef;
            let (mut doc, part, _, mut stack) = joined_square();
            let joints = points_in(&doc, part);
            assert_eq!(joints.len(), 4, "four corners, four joints");
            let alone = read_through(&mut stack, &mut doc, part, Node::Point { at: [ParamRef::Literal(0.0); 3] });
            let profiles = ProfileRows { at: None, loops: fieldform_field::profile_partition(&doc) };
            assert_eq!(profiles.loops.len(), 1, "the square is one held loop");
            assert_eq!(
                blocks_a_line_can_call(&doc, part, &profiles),
                vec![alone],
                "the point made on its own, and none of the joints {joints:?}",
            );
        }

        /// **A profile row's V takes its loop out of the extrude and puts it back, whichever of
        /// its curves the extrude names.**
        ///
        /// The V wrote the loop's first curve in walk order, the key the row is drawn under. An
        /// extrude names whichever curve was selected, and the row goes inside whatever holds any
        /// curve of its loop, so with the second side selected the V wrote a curve nothing read:
        /// it turned bright, said the profile had a row of its own, and the row stayed inside the
        /// extrude. It writes the curve that puts the row where it is now
        /// (`profile_v_curve_in`), asked of the nesting the row is drawn from. Asserted for every
        /// side, because asserting it for one would pass on the walk order rather than on the fix.
        #[test]
        fn a_profile_row_s_v_takes_its_loop_out_of_the_extrude_whichever_curve_it_names() {
            use super::super::{make_variable, profile_v_curve_in, put_back_inside, Block};
            use fieldform_scene::{BlockKey, Ownership};
            for named in 0..4 {
                let (mut doc, part, square, mut stack) = joined_square();
                let extrude = read_through(&mut stack, &mut doc, part, extrusion(square[named]));
                let rows = |doc: &Document| square_rows(doc, part, &square, &[extrude]);
                let members = fieldform_field::profile_partition(&doc)[0].clone();
                let key = members[0];
                let inside = vec![(Block::Node(extrude), vec![Block::Profile(key)])];
                // No curve row of its own: all four are drawn in the profile row.
                let out = vec![(Block::Profile(key), vec![]), (Block::Node(extrude), vec![])];
                assert_eq!(rows(&doc), inside, "side {named} named: the row is inside the extrude");

                let v = |doc: &Document| {
                    let nesting = doc.nesting_in(&doc.parts[&part].graph);
                    BlockKey::Curve(profile_v_curve_in(doc, &nesting, &members).expect("curves"))
                };
                assert_ne!(doc.ownership(v(&doc)), Ownership::Variable, "side {named}: a dim V");
                let make = make_variable(&doc, v(&doc), || panic!("a curve is born named"))
                    .expect("not a variable, so the V makes it one");
                stack.execute(&mut doc, Box::new(make)).expect("the dim V");
                assert_eq!(rows(&doc), out, "side {named}: out of the extrude, and whole");
                assert_eq!(doc.ownership(v(&doc)), Ownership::Variable, "side {named}: bright");
                stack.undo(&mut doc).expect("one undo");
                assert_eq!(rows(&doc), inside, "side {named}: undone, inside again");
                stack.redo(&mut doc).expect("one redo");
                assert_eq!(rows(&doc), out, "side {named}: redone");

                let back = put_back_inside(&doc, v(&doc)).expect("one reader");
                stack.execute(&mut doc, Box::new(back)).expect("the bright V");
                assert_eq!(rows(&doc), inside, "side {named}: back inside the extrude");
                stack.undo(&mut doc).expect("one undo");
                assert_eq!(rows(&doc), out, "side {named}: undone, out again");
            }
        }

        /// **A loop two blocks read is one row, drawn inside each of them, whichever of its
        /// curves they name.**
        ///
        /// Two blocks reading one curve made it a variable until 112, and a variable has a row of
        /// its own; but a curve a block reads is the loop to that block. Lifted out for being a
        /// variable, it split the loop exactly when the two were made with a side other than the
        /// first selected, since the first heads the row and cannot leave it — whether a loop
        /// split was a coin flip on walk order. And a file from before ownership was stored, whose
        /// list folded such a curve in, opened split.
        ///
        /// **A second reader declares the curve** (R1 inverted 2026-10-24, `notebook-spec.md`
        /// §17), so the loop is one profile row of its **own**, above the two blocks that read it,
        /// and neither of them holds it. Between 112 and the inversion it was nested and drawn in
        /// full inside both readers. The loop still may not split, which is what the one row
        /// holding every curve of it says. Asserted for every side, so a rule that holds for the
        /// key alone fails.
        #[test]
        fn a_loop_two_blocks_read_is_one_row_of_its_own_whichever_curve_they_name() {
            use super::super::Block;
            use fieldform_scene::{Axis, BlockKey, Frame, Outline, Ownership, ParamRef};
            for named in 0..4 {
                let (mut doc, part, square, mut stack) = joined_square();
                let extrude = read_through(&mut stack, &mut doc, part, extrusion(square[named]));
                let revolve = Node::Revolve { rims: Default::default(),
                    outline: Outline::closed(square[named]),
                    frame: Frame::at_along(Axis::Z, [ParamRef::Literal(0.0); 3]),
                    angle: ParamRef::Literal(360.0),
                    sides: fieldform_scene::AngleSides::One,
                };
                let revolve = read_through(&mut stack, &mut doc, part, revolve);
                assert_eq!(
                    doc.ownership(BlockKey::Curve(square[named])),
                    Ownership::Variable,
                    "side {named}: two blocks read it, so it is declared (R1 inverted)",
                );
                let key = fieldform_field::profile_partition(&doc)[0][0];
                assert_eq!(
                    square_rows(&doc, part, &square, &[extrude, revolve]),
                    vec![
                        (Block::Profile(key), vec![]),
                        (Block::Node(extrude), vec![]),
                        (Block::Node(revolve), vec![]),
                    ],
                    "side {named}: the one profile row is its own, and holds every curve of the \
                     loop whichever of them the two blocks name",
                );
            }
        }

        /// An extrude of the loop `named` closes, as Extrude makes one from the selected curve.
        fn extrusion(named: CurveId) -> Node {
            use fieldform_scene::{Outline, ParamRef};
            Node::Extrude { rims: Default::default(),
                outline: Outline::closed(named),
                sweep: fieldform_scene::Sweep::Normal,
                height: ParamRef::Literal(10.0),
                draft: ParamRef::Literal(0.0),
                start: ParamRef::Literal(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
            }
        }

        /// `node` added to `part` as an edit adds it: through the history, so settled, one undo.
        fn read_through(
            stack: &mut CommandStack,
            doc: &mut Document,
            part: PartId,
            node: Node,
        ) -> NodeId {
            use super::super::SetPartGraph;
            let before = doc.parts[&part].graph.clone();
            let mut graph = before.clone();
            let id = graph.add(node);
            stack
                .execute(doc, Box::new(SetPartGraph::new(part, before, graph)))
                .expect("the block is added");
            id
        }

        /// The rows as the list lays them out over `square` and `blocks`: the curves folded into
        /// the profile rows their loops make, by the counts of the nesting the list is drawn from,
        /// then the blocks in the order given, each row with the rows drawn inside it. The joints
        /// are drawn inside their loop's row, so not here.
        fn square_rows(
            doc: &Document,
            part: PartId,
            square: &[CurveId],
            blocks: &[NodeId],
        ) -> Vec<(super::super::Block, Vec<super::super::Block>)> {
            use super::super::{folded_curve_row_in, nest_rows, Block, ProfileRows};
            let profiles = ProfileRows { at: None, loops: fieldform_field::profile_partition(doc) };
            let nesting = doc.nesting_in(&doc.parts[&part].graph);
            let readers = |id| nesting.readers_of_curve(id);
            let mut flat: Vec<Block> = square
                .iter()
                .filter_map(|id| folded_curve_row_in(&profiles, doc, &readers, *id))
                .collect();
            flat.extend(blocks.iter().map(|id| Block::Node(*id)));
            let loop_of = |seed| fieldform_field::loop_from(doc, seed, None).unwrap_or_default();
            nest_rows(flat, &nesting, &loop_of)
                .iter()
                .map(|r| (r.block, r.inside.iter().map(|i| i.block).collect()))
                .collect()
        }

        /// A second part called `name`, with nothing in its graph.
        fn second_part(doc: &mut Document, name: &str) -> PartId {
            doc.add_part(fieldform_scene::Part {
                id: PartId::new(),
                name: name.to_string(),
                material: None,
                graph: Graph::empty(),
                inputs: Vec::new(),
            })
        }

        /// An extrude of the loop `profile` closes, added to `part`.
        fn extrude_in(doc: &mut Document, part: PartId, profile: CurveId) -> NodeId {
            extrude_of(doc, part, fieldform_scene::Outline::closed(profile))
        }

        /// An extrude of `outline`, added to `part`.
        fn extrude_of(
            doc: &mut Document,
            part: PartId,
            outline: fieldform_scene::Outline,
        ) -> NodeId {
            use fieldform_scene::ParamRef;
            doc.parts.get_mut(&part).expect("the part").graph.add(Node::Extrude { rims: Default::default(),
                outline,
                sweep: fieldform_scene::Sweep::Normal,
                height: ParamRef::Literal(10.0),
                draft: ParamRef::Literal(0.0),
                start: ParamRef::Literal(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
            })
        }

        /// A line of its own at height `y`, clear of the square, so no walk round the square
        /// reaches it.
        fn lone_line(doc: &mut Document, name: &str, y: f64) -> CurveId {
            let id = CurveId::new();
            let ends = vec![Point3::new(50.0, y, 0.0), Point3::new(60.0, y, 0.0)];
            let line = Curve::polyline(id, name.to_string(), ends, false).expect("a line");
            doc.curves.insert(id, line);
            id
        }

        /// **A block in another part that reads a curve is named, with the part it is in.**
        ///
        /// A curve belongs to the document, not to a part, so one curve can be what blocks in two
        /// parts are built on. The deletes asked the part on screen alone (refactor plan §3.3, root
        /// cause 10), so a curve another part was built on could go, and that part stopped
        /// evaluating out of sight. Readers are counted as `block-ownership-spec.md` I2 counts them —
        /// nodes in any part whose `Node::curves` names the curve — with the blocks whose section is
        /// walked through it besides, and asked from both sides, since which part is the other one
        /// depends on which is on screen.
        ///
        /// The label is asked of the reader's own graph: looked up in the part on screen, a block
        /// in another part is not there, and the app's list would have called it "gone". The pipe
        /// and the spare line are clear of the square, since the lid's extrude reads every side of
        /// it (`a_block_in_another_part_reads_every_curve_its_section_is_walked_through`).
        #[test]
        fn a_block_in_another_part_that_reads_a_curve_is_named_with_that_part() {
            use super::super::readers_in_other_parts;
            let (mut doc, base, square) = square_document();
            let lid = second_part(&mut doc, "Lid");
            extrude_in(&mut doc, lid, square[0]);
            let rail = lone_line(&mut doc, "Rail", 30.0);
            let spare = lone_line(&mut doc, "Spare", 60.0);
            doc.parts.get_mut(&base).expect("the base").graph.add(Node::Pipe {
                curve: rail,
                radius: fieldform_scene::ParamRef::Literal(1.0),
            });
            let label = |graph: &Graph, node: NodeId| {
                match graph.node(node) {
                    Some(Node::Extrude { .. }) => "Extrude",
                    Some(Node::Pipe { .. }) => "Pipe",
                    _ => "not in the graph it was asked of",
                }
                .to_string()
            };

            assert_eq!(
                readers_in_other_parts(&doc, base, &[square[0]], label).readers,
                vec!["Extrude in Lid".to_string()],
                "the lid is built on the first side, so the base may not delete it",
            );
            assert_eq!(
                readers_in_other_parts(&doc, lid, &[rail], label).readers,
                vec!["Pipe in Part".to_string()],
                "and the other way round, the base's pipe is another part's from the lid",
            );
            assert_eq!(
                readers_in_other_parts(&doc, base, &[square[0], rail], label).readers,
                vec!["Extrude in Lid".to_string()],
                "asked over both, only the block outside the part on screen is named",
            );
            assert!(
                readers_in_other_parts(&doc, base, &[rail], label).is_empty(),
                "a reader in the part on screen is the delete's own business, as it always was",
            );
            assert!(
                readers_in_other_parts(&doc, base, &[spare], label).is_empty(),
                "nothing reads the spare line",
            );

            // **And the line counts the parts**, so readers in two are not "another part": a pipe
            // along the same side in a third part makes it two.
            let one = readers_in_other_parts(&doc, base, &[square[0]], label);
            assert_eq!((one.parts, one.built_on()), (1, "another part is built on it, so it stays"));
            let cap = second_part(&mut doc, "Cap");
            doc.parts.get_mut(&cap).expect("the cap").graph.add(Node::Pipe {
                curve: square[0],
                radius: fieldform_scene::ParamRef::Literal(1.0),
            });
            let two = readers_in_other_parts(&doc, base, &[square[0]], label);
            assert_eq!(two.readers, vec!["Extrude in Lid".to_string(), "Pipe in Cap".to_string()]);
            assert_eq!((two.parts, two.built_on()), (2, "other parts are built on it, so it stays"));
        }

        /// **A block in another part reads every curve its section is walked through, not only
        /// the one it names.**
        ///
        /// An extrude names one curve and is built on what is walked from it: the loop it closes,
        /// or, as a wall, the chain it is part of, open or closed. Deleting any other curve of that
        /// walk changes the block as surely as deleting the one it names. The open chain is the
        /// case asking over the deleted curve's loop could not see: `loop_from` finds no loop
        /// there, so the curve was asked about alone, and the other part's wall lost a curve out of
        /// sight.
        #[test]
        fn a_block_in_another_part_reads_every_curve_its_section_is_walked_through() {
            use super::super::readers_in_other_parts;
            use fieldform_scene::{Outline, ParamRef, WallEnds, WallSide};
            let label = |_: &Graph, _: NodeId| "Extrude".to_string();
            let lid_extrude = vec!["Extrude in Lid".to_string()];

            // Round a loop, named by its second side.
            let (mut doc, base, square) = square_document();
            let lid = second_part(&mut doc, "Lid");
            extrude_in(&mut doc, lid, square[1]);
            for (n, side) in square.iter().enumerate() {
                let read = readers_in_other_parts(&doc, base, &[*side], label);
                assert_eq!(
                    read.readers,
                    lid_extrude,
                    "side {} is part of the loop the lid's extrude is made from",
                    n + 1,
                );
                // Said as one of the curves it is made from, except the side it names, which it
                // uses by name.
                assert_eq!(read.through_sections, *side != square[1], "side {}", n + 1);
            }

            // Along an open chain of three sides, as a wall named by the middle one.
            let (mut doc, base, square) = square_document();
            doc.curves.remove(&square[3]);
            let lid = second_part(&mut doc, "Lid");
            extrude_of(
                &mut doc,
                lid,
                Outline::Wall {
                    curve: square[1],
                    thickness: ParamRef::Literal(2.0),
                    side: WallSide::Centre,
                    ends: WallEnds::Square,
                },
            );
            for (n, side) in square[..3].iter().enumerate() {
                assert_eq!(
                    readers_in_other_parts(&doc, base, &[*side], label).readers,
                    lid_extrude,
                    "side {} is part of the chain the lid's wall runs along",
                    n + 1,
                );
            }
        }

        /// **A pipe in another part along a different side does not stop this side going.**
        ///
        /// The control for the two tests either side of it. Asked over the whole loop, a refusal
        /// would count every block in another part naming any side of it, whatever it named the
        /// side for — so a pipe along the second side would refuse a delete of the first, named as
        /// what the loop is made from. A pipe is built on its own curve alone, and taking the
        /// first side out only sets the second's end loose where it already is. Only a block
        /// whose section is walked round the loop reads the sides it does not name.
        #[test]
        fn a_pipe_in_another_part_along_a_different_side_does_not_stop_this_side_going() {
            use super::super::{delete_loop_member_commands, readers_in_other_parts};
            let (mut doc, base, square, mut stack) = joined_square();
            let walled = extrude_in(&mut doc, base, square[0]);
            let lid = second_part(&mut doc, "Lid");
            let pipe = doc.parts.get_mut(&lid).expect("the lid").graph.add(Node::Pipe {
                curve: square[1],
                radius: fieldform_scene::ParamRef::Literal(1.0),
            });
            let mates = fieldform_field::loop_from(&doc, square[0], None).expect("a loop");

            assert!(
                readers_in_other_parts(&doc, base, &[square[0]], |_, _| "Pipe".to_string())
                    .is_empty(),
                "the lid's pipe reads the second side, not the first",
            );
            let (commands, _, done) = delete_loop_member_commands(&doc, base, square[0], &mates, 2.5)
                .expect("a pipe along another side is no reason to keep this one");
            assert_eq!(done.walled, vec![walled], "the base's own extrude follows, as it did");
            stack
                .execute(&mut doc, Box::new(Batch::new("curve.delete", commands)))
                .expect("the delete applies");
            assert!(doc.curve(square[0]).is_none());
            assert!(doc.curve(square[1]).is_some(), "the pipe's side stays");
            assert!(
                matches!(
                    doc.parts[&lid].graph.node(pipe),
                    Some(Node::Pipe { curve, .. }) if *curve == square[1]
                ),
                "and the lid's pipe still runs along it",
            );
        }

        /// **A side of a loop another part is built on is not taken out of it.**
        ///
        /// Taking a side out of a loop walls every block that read the loop — in the part it is
        /// handed, and only there. A block in another part built on the same loop was left reading
        /// an open chain as a closed profile, and that part stopped evaluating with nothing said
        /// where it broke. Refused where the commands are built, so no caller can get round it; the
        /// app asks first and names the block (refactor plan §7, 2.6). The control is
        /// `deleting_a_side_of_an_extruded_square_leaves_a_wall_along_the_other_three`: the same
        /// extrude in the part being edited is followed.
        #[test]
        fn a_side_of_a_loop_another_part_is_built_on_is_not_taken_out_of_it() {
            use super::super::delete_loop_member_commands;
            let (mut doc, base, square, _) = joined_square();
            let lid = second_part(&mut doc, "Lid");
            extrude_in(&mut doc, lid, square[1]);
            let mates = fieldform_field::loop_from(&doc, square[0], None).expect("a loop");
            assert_eq!(mates.len(), 4);

            assert!(
                delete_loop_member_commands(&doc, base, square[0], &mates, 2.5).is_none(),
                "the lid's extrude reads the loop, so the base may not open it",
            );
        }

        /// **A part already broken elsewhere lets a curve be taken out of a loop.**
        ///
        /// Taking a side out of a held loop asked the check of the graph it left and of nothing
        /// else, as a block's delete did until it was refused only for breaking a part that built
        /// (refactor plan 2.10b). So in a part that already could not be built, every side of every
        /// loop was refused, for a fault the delete had not made. Here the fault is a twist reading
        /// the plane it is aimed at, clear of the square, and the side goes with the extrude walled
        /// as it would be in a part that builds.
        ///
        /// The graph left is asserted to be still unable to be built, for the twist, so it is the
        /// narrowing letting the delete through rather than a delete that mended the part.
        #[test]
        fn a_part_already_broken_elsewhere_lets_a_curve_be_taken_out_of_a_loop() {
            use super::super::{delete_loop_member_commands, App};
            use fieldform_scene::{Frame, ParamRef, Place};
            let lit = ParamRef::Literal;
            let explain = |e: &fieldform_field::CompileError| e.to_string();
            let (mut doc, part, square, _) = joined_square();
            let walled = extrude_in(&mut doc, part, square[0]);
            let graph = &mut doc.parts.get_mut(&part).expect("the part").graph;
            let plane = graph.add(Node::Plane {
                origin: Place::At([lit(0.0), lit(0.0), lit(50.0)]),
                flipped: false,
                turn: fieldform_scene::no_turn(),
            });
            let frame = Frame::From { node: plane };
            graph.add(Node::Twist { input: plane.into(), degrees_per_mm: lit(2.0), frame });
            assert!(
                App::refusal_of(&doc, &doc.parts[&part].graph, &explain).is_some(),
                "a twist reading the plane it is aimed at cannot be built",
            );
            let mates = fieldform_field::loop_from(&doc, square[0], None).expect("a loop");

            let (_, done) = App::loop_member_deletion(&doc, part, square[0], &mates, 2.5, &explain)
                .unwrap_or_else(|why| panic!("taking a side out of the loop was refused: {why}"));
            assert_eq!(done.walled, vec![walled], "the extrude follows, as in a part that builds");
            let (_, left, _) = delete_loop_member_commands(&doc, part, square[0], &mates, 2.5)
                .expect("a loop member");
            assert!(
                App::refusal_of(&doc, &left, &explain).is_some(),
                "and the part still cannot be built, for the twist",
            );
        }

        /// **Taking a curve out of a loop that would break a part that builds is refused, as a
        /// block's delete is.**
        ///
        /// A drafted extrude with sharp corners builds on a closed profile and not on a wall, which
        /// is what taking a side out of its loop makes it. So the delete is refused before anything
        /// is written, and in the words `delete_refused` gives a block's delete, since it is the
        /// same refusal for the same reason. It had a sentence of its own, which said the delete
        /// "would leave" a block unable to be built and opened the check's reason with a capital
        /// after a colon.
        ///
        /// Two controls: the part builds before the delete, so it is the delete being refused; and
        /// the same extrude with round corners is walled and not refused, so it is breaking the
        /// part that is refused rather than taking a curve out of a loop.
        #[test]
        fn taking_a_curve_out_of_a_loop_that_would_break_a_part_that_builds_is_refused_as_a_delete_is(
        ) {
            use super::super::{delete_loop_member_commands, App};
            use fieldform_scene::{Corners, Outline, ParamRef};
            let explain = |e: &fieldform_field::CompileError| e.to_string();
            for corners in [Corners::Sharp, Corners::Round] {
                let (mut doc, part, square, _) = joined_square();
                let extrude = doc.parts.get_mut(&part).expect("the part").graph.add(Node::Extrude { rims: Default::default(),
                    outline: Outline::closed(square[0]),
                    sweep: fieldform_scene::Sweep::Normal,
                    height: ParamRef::Literal(10.0),
                    draft: ParamRef::Literal(5.0),
                    start: ParamRef::Literal(0.0),
                    sides: fieldform_scene::Sides::One,
                    corners,
                });
                assert_eq!(
                    App::refusal_of(&doc, &doc.parts[&part].graph, &explain),
                    None,
                    "a drafted extrude of the square builds, with {corners:?} corners",
                );
                let mates = fieldform_field::loop_from(&doc, square[0], None).expect("a loop");

                let deleting =
                    App::loop_member_deletion(&doc, part, square[0], &mates, 2.5, &explain);
                match corners {
                    Corners::Sharp => {
                        let (_, left, _) =
                            delete_loop_member_commands(&doc, part, square[0], &mates, 2.5)
                                .expect("a loop member");
                        let why = App::refusal_of(&doc, &left, &explain)
                            .expect("a wall's corners cannot be mitred");
                        assert_eq!(
                            deleting.err(),
                            Some(App::delete_refused("Line 1", &why)),
                            "refused in a block delete's words, with the check's reason",
                        );
                    }
                    Corners::Round => {
                        let (_, done) = deleting.unwrap_or_else(|why| {
                            panic!("a wall can round its corners, so this was not refused: {why}")
                        });
                        assert_eq!(done.walled, vec![extrude], "the extrude is walled");
                    }
                }
            }
        }

        /// The square, held: one entry on a fresh history.
        fn joined_square() -> (Document, PartId, Vec<CurveId>, CommandStack) {
            let (mut doc, part, square) = square_document();
            let mut stack = CommandStack::default();
            let (commands, joints, _) = join_loop_commands(&doc, part, &square);
            assert_eq!(joints, 4, "four corners, four joints");
            stack
                .execute(&mut doc, Box::new(Batch::new("profile.join", commands)))
                .expect("the joins apply");
            (doc, part, square, stack)
        }

        /// A box picks the handles inside it on every curve, and the selection it makes keeps
        /// the selected curve as the one holding the handles when a point of it is inside.
        #[test]
        fn a_box_picks_handles_across_curves_and_the_selected_curve_keeps_the_handles() {
            use eframe::egui::{pos2, Rect};
            let (a, b) = (CurveId::new(), CurveId::new());
            let handles = vec![
                (a, 0, pos2(10.0, 10.0)),
                (a, 1, pos2(50.0, 50.0)),
                (b, 0, pos2(20.0, 20.0)),
                (b, 3, pos2(200.0, 200.0)),
            ];
            let inside = points_in_box(&handles, Rect::from_two_pos(pos2(0.0, 0.0), pos2(60.0, 60.0)));
            assert_eq!(inside, vec![(a, 0), (a, 1), (b, 0)]);
            assert_eq!(split_selection(&inside, Some(b)), Some((b, vec![0], vec![(a, 0), (a, 1)])));
            assert_eq!(split_selection(&inside, None), Some((a, vec![0, 1], vec![(b, 0)])));
            let stranger = CurveId::new();
            assert_eq!(split_selection(&inside, Some(stranger)), Some((a, vec![0, 1], vec![(b, 0)])));
            assert_eq!(split_selection(&[], None), None);
            assert_eq!(group_by_curve(&inside), vec![(a, vec![0, 1]), (b, vec![0])]);
            // Shift toggles: in, then out again.
            let mut picked = vec![(a, 0)];
            toggle_in(&mut picked, (b, 2));
            assert_eq!(picked, vec![(a, 0), (b, 2)]);
            toggle_in(&mut picked, (a, 0));
            assert_eq!(picked, vec![(b, 2)]);
        }

        /// `set` names a point by its curve and index or end, and refuses what is not there.
        #[test]
        fn set_names_a_point_by_curve_and_index_and_refuses_what_is_not_there() {
            let (doc, _part, square) = square_document();
            let named = |n: &str| doc.curves.values().find(|c| c.name == n).map(|c| c.id);
            let ask = |pairs: &[(&str, &str)]| {
                let pairs: Vec<(String, String)> =
                    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
                set_targets(&doc, &pairs, named)
            };
            assert_eq!(
                ask(&[("Line 1.p1", "10,20,0")]).unwrap(),
                vec![(square[0], 1, Point3::new(10.0, 20.0, 0.0))]
            );
            assert_eq!(
                ask(&[("Line 2.end", "1, 2, 3"), ("Line 2.start", "0,0,0")]).unwrap(),
                vec![(square[1], 1, Point3::new(1.0, 2.0, 3.0)), (square[1], 0, Point3::origin())]
            );
            assert!(ask(&[("Line 1.p2", "0,0,0")]).unwrap_err().contains("p0 to p1"));
            assert!(ask(&[("Nope.p0", "0,0,0")]).unwrap_err().contains("no curve is called"));
            assert!(ask(&[("Line 1.p0", "1,2")]).unwrap_err().contains("three numbers"));
            assert!(ask(&[("radius", "5")]).unwrap().is_empty(), "a key that names no point");
        }

        /// A joint's two ends are found from its point, and a corner offers none and four.
        #[test]
        fn a_joints_two_ends_are_found_from_its_point_and_its_corner_offers_four_ways() {
            let (doc, part, square, _stack) = joined_square();
            for point in points_in(&doc, part) {
                let ends = ends_on_point(&doc, point);
                assert_eq!(ends.len(), 2, "a corner is where two ends meet");
                assert_ne!(ends[0].curve, ends[1].curve, "of two different curves");
                assert!(ends.iter().all(|e| square.contains(&e.curve)));
                let choices = joint_choices(&doc, ends[0], ends[1]);
                assert_eq!(choices.len(), 5);
                assert!(choices[0].1.is_none(), "none comes first");
                let kinds: Vec<&str> =
                    choices[1..].iter().map(|(_, r)| r.as_ref().expect("a joint").kind()).collect();
                assert_eq!(kinds, ["G1", "G1", "G2", "G2"]);
            }
            let stray = Graph::empty().add(Node::Point { at: [fieldform_scene::ParamRef::Literal(0.0); 3] });
            assert!(ends_on_point(&doc, stray).is_empty(), "a point no end sits on has none");
        }

        /// Hiding a row hides the points holding it, and nothing is written onto a point.
        #[test]
        fn a_hidden_rows_joints_go_unseen_with_it_and_one_curve_shown_brings_them_back() {
            let (doc, part, square, _stack) = joined_square();
            let points = points_in(&doc, part);
            assert_eq!(points.len(), 4);
            let rows = vec![square.clone()];
            let mut hidden: std::collections::HashSet<CurveId> = square.iter().copied().collect();
            let unseen = points_hidden_with_rows(&doc, part, &hidden, &rows);
            assert_eq!(unseen.len(), 4, "every joint of a hidden row is unseen");
            assert!(points.iter().all(|p| unseen.contains(p)));
            assert!(
                doc.node_meta.values().all(|m| !m.hidden),
                "the row's eye writes nothing onto a point, so a reopen shows them with the curves"
            );
            // One curve shown is a row no longer hidden, and its corners come back with it.
            hidden.remove(&square[2]);
            assert!(points_hidden_with_rows(&doc, part, &hidden, &rows).is_empty());
        }

        /// Closing a loop makes one joint per meeting, and one undo entry.
        #[test]
        fn closing_a_loop_makes_one_joint_per_meeting_and_one_undo_entry() {
            let (mut doc, part, square, mut stack) = joined_square();
            assert_eq!(points_in(&doc, part).len(), 4);
            assert_eq!(ends_held(&doc, &square), 8, "both ends of every line sit on a point");
            // Held already, so nothing more is asked for.
            let (again, joints, _) = join_loop_commands(&doc, part, &square);
            assert!(again.is_empty() && joints == 0, "a held loop asks for nothing");
            // One Ctrl+Z takes the lot.
            assert!(stack.undo(&mut doc).is_some(), "there is one entry to undo");
            assert!(points_in(&doc, part).is_empty(), "the points went with it");
            assert_eq!(ends_held(&doc, &square), 0, "and so did every placing");
            assert!(stack.undo(&mut doc).is_none(), "and it was the only entry");
        }

        /// No joint is made where three ends meet.
        #[test]
        fn no_joint_is_made_where_three_ends_meet() {
            let (mut doc, part, square) = square_document();
            let stray = CurveId::new();
            doc.curves.insert(
                stray,
                Curve::polyline(
                    stray,
                    "Stray",
                    vec![Point3::new(0.0, 0.0, 0.0), Point3::new(-30.0, -30.0, 0.0)],
                    false,
                )
                .expect("a line"),
            );
            let (commands, joints, _) = join_loop_commands(&doc, part, &square);
            assert_eq!(joints, 3, "the branched corner is skipped");
            let mut stack = CommandStack::default();
            stack.execute(&mut doc, Box::new(Batch::new("profile.join", commands))).expect("applies");
            assert_eq!(points_in(&doc, part).len(), 3);
            let first = doc.curve(square[0]).expect("the first line");
            let last = doc.curve(square[3]).expect("the last line");
            assert!(first.placed_on(0).is_none() && last.placed_on(1).is_none(), "the corner with three ends sits on nothing");
            assert!(doc.curve(stray).expect("the stray").sits_on().next().is_none());
        }

        /// Dragging a joint moves both curve ends, and the loop stays closed.
        #[test]
        fn dragging_a_joint_moves_both_curve_ends_and_the_loop_stays_closed() {
            let (mut doc, part, square, _stack) = joined_square();
            let joint = doc.curve(square[0]).expect("line 1").placed_on(1).expect("its end is held");
            let to = Point3::new(13.0, -2.0, 1.0);
            let graph = &mut doc.parts.get_mut(&part).expect("the part").graph;
            assert!(move_point_node(graph, joint, to), "a typed point moves");
            assert!(settle_curves_on_points(&mut doc, part), "the ends followed");
            assert_eq!(doc.curve(square[0]).expect("line 1").control_points()[1], to);
            assert_eq!(doc.curve(square[1]).expect("line 2").control_points()[0], to);
            let read = fieldform_field::profile_loop(&doc, square[0], None).expect("still closed");
            assert_eq!(read.curves.len(), 4);
        }

        /// Dragging a curve end on a point moves the point, and never unbinds it.
        #[test]
        fn dragging_a_curve_end_on_a_point_moves_the_point_and_never_unbinds_it() {
            let (mut doc, part, square, _stack) = joined_square();
            let line = doc.curve(square[0]).expect("line 1").clone();
            let (bound, free) = drag_targets(&line, &[1]);
            assert!(free.is_empty(), "a held end is not free to move on its own");
            let [(index, node)] = bound[..] else { panic!("one bound end, got {bound:?}") };
            assert_eq!(index, 1);
            let to = Point3::new(12.0, 0.5, 0.0);
            let graph = &mut doc.parts.get_mut(&part).expect("the part").graph;
            assert!(move_point_node(graph, node, to));
            settle_curves_on_points(&mut doc, part);
            let line = doc.curve(square[0]).expect("line 1");
            assert_eq!(line.placed_on(1), Some(node), "still on its point");
            assert_eq!(line.control_points()[1], to, "and where the point went");
            let graph = &doc.parts.get(&part).expect("the part").graph;
            assert_eq!(point_literals(graph, node), Some(to));
        }

        /// Ungroup moves nothing: every control point is bit-identical before and after.
        #[test]
        fn ungroup_moves_nothing() {
            let (mut doc, part, square, mut stack) = joined_square();
            let before = every_control_point_bits(&doc, &square);
            let (commands, removed) = ungroup_commands(&doc, part, &square);
            assert_eq!(removed, 4, "nothing else read the four points");
            stack.execute(&mut doc, Box::new(Batch::new("profile.ungroup", commands))).expect("applies");
            assert_eq!(ends_held(&doc, &square), 0);
            assert!(points_in(&doc, part).is_empty());
            assert_eq!(every_control_point_bits(&doc, &square), before, "nothing moved");
            // Still a loop, still a row.
            assert_eq!(fieldform_field::profile_partition(&doc).len(), 1);
            // And one undo holds it again.
            assert!(stack.undo(&mut doc).is_some());
            assert_eq!(ends_held(&doc, &square), 8);
            assert_eq!(points_in(&doc, part).len(), 4);
        }

        /// Dragging a member curve never takes it out of a joined loop.
        #[test]
        fn dragging_a_member_curve_never_takes_it_out_of_a_joined_loop() {
            let (mut doc, part, square, _stack) = joined_square();
            // The whole line is picked, both of its points. Both sit on points, so the drag
            // moves the two blocks and the neighbours' ends come along.
            let line = doc.curve(square[0]).expect("line 1").clone();
            let (bound, free) = drag_targets(&line, &[0, 1]);
            assert!(free.is_empty() && bound.len() == 2);
            let delta = nalgebra::Vector3::new(0.0, -5.0, 0.0);
            for (i, node) in bound {
                let to = line.control_points()[i] + delta;
                let graph = &mut doc.parts.get_mut(&part).expect("the part").graph;
                assert!(move_point_node(graph, node, to));
            }
            settle_curves_on_points(&mut doc, part);
            let rows = fieldform_field::profile_partition(&doc);
            assert_eq!(rows.len(), 1, "{rows:?}");
            assert_eq!(rows[0].len(), 4, "the loop kept every curve: {rows:?}");
            assert_eq!(doc.curve(square[1]).expect("line 2").control_points()[0], Point3::new(10.0, -5.0, 0.0));
        }

        /// Cut takes a curve out, and undo puts it back joined.
        #[test]
        fn cut_takes_a_curve_out_and_undo_puts_it_back_joined() {
            let (mut doc, part, square, mut stack) = joined_square();
            let taken = doc.curve(square[1]).expect("line 2").clone();
            stack.execute(&mut doc, Box::new(SetCurve::remove(taken))).expect("cut");
            assert!(fieldform_field::profile_partition(&doc).is_empty(), "the loop opened");
            assert_eq!(points_in(&doc, part).len(), 4, "the joint points stay, holding the neighbours");
            assert!(stack.undo(&mut doc).is_some());
            assert_eq!(fieldform_field::profile_partition(&doc).len(), 1, "the row is back");
            let back = doc.curve(square[1]).expect("line 2 is back");
            assert!(back.placed_on(0).is_some() && back.placed_on(1).is_some(), "and still held");
        }

        /// A curve pasted back in place rejoins the row, but is not held.
        #[test]
        fn a_curve_pasted_back_in_place_rejoins_the_row_but_is_not_held() {
            let (mut doc, part, square, mut stack) = joined_square();
            let taken = doc.curve(square[1]).expect("line 2").clone();
            stack.execute(&mut doc, Box::new(SetCurve::remove(taken.clone()))).expect("cut");
            let pasted = CurveId::new();
            let copy = taken.copy_as(pasted, "Line 2 copy").expect("a copy");
            stack.execute(&mut doc, Box::new(SetCurve::insert(copy))).expect("paste");
            let rows = fieldform_field::profile_partition(&doc);
            assert_eq!(rows.len(), 1, "{rows:?}");
            assert!(rows[0].contains(&pasted), "the copy touches the loop and the row re-forms");
            assert!(doc.curve(pasted).expect("the copy").sits_on().next().is_none(), "but sits on nothing");
            let rest: Vec<CurveId> = square.iter().copied().filter(|id| *id != square[1]).collect();
            assert_eq!(ends_held(&doc, &rest), 6, "its neighbours are still held");
            assert_eq!(points_in(&doc, part).len(), 4);
        }

        /// Paste here moves the curve rigidly, so its start lands on the click.
        #[test]
        fn paste_here_moves_the_curve_rigidly_so_its_start_lands_on_the_click() {
            let curve = single_span_bezier(CurveId::new(), 3, 40.0).expect("a cubic");
            let at = Point3::new(5.0, -6.0, 7.5);
            let moved = moved_to_start_at(&curve, CurveId::new(), "copy", at).expect("moves");
            assert_eq!(moved.control_points()[0], at);
            for (a, b) in curve.control_points().iter().zip(moved.control_points()) {
                let was = a - curve.control_points()[0];
                let is = b - moved.control_points()[0];
                assert!((was - is).norm() < 1e-12, "the shape changed: {was:?} vs {is:?}");
            }
            assert!(moved.sits_on().next().is_none(), "and it lands loose");
        }

        /// A square extruded from its first side and a pipe along a line of its own, unioned —
        /// both ways a block reads a curve, by name and round a loop from the one it names. The
        /// document, its part, the square, the extrude, the line, the pipe and the union.
        fn extrude_and_pipe_unioned(
        ) -> (Document, PartId, Vec<CurveId>, NodeId, CurveId, NodeId, NodeId) {
            use fieldform_scene::{BooleanOp, ParamRef};
            let (mut doc, part, square) = square_document();
            let extrude = extrude_in(&mut doc, part, square[0]);
            let rail = lone_line(&mut doc, "Rail", 30.0);
            let graph = &mut doc.parts.get_mut(&part).expect("the part").graph;
            let pipe = graph.add(Node::Pipe { curve: rail, radius: ParamRef::Literal(1.0) });
            let inputs = vec![extrude.into(), pipe.into()];
            let union = graph.add(Node::Boolean { op: BooleanOp::Union, inputs });
            (doc, part, square, extrude, rail, pipe, union)
        }

        /// Copies `root` out of `part` and pastes it back in on `stack`, as Copy then Paste does,
        /// and answers the pasted root.
        fn copy_and_paste(
            stack: &mut CommandStack,
            doc: &mut Document,
            part: PartId,
            root: NodeId,
        ) -> NodeId {
            use super::super::{copied_body, paste_body, Clipboard};
            let Clipboard::Body { nodes, root, cages, curves } =
                copied_body(&*doc, &doc.parts[&part].graph, root)
            else {
                panic!("a block is copied as a body");
            };
            paste_body(stack, doc, part, &nodes, root, &cages, &curves).expect("the paste applies")
        }

        /// **One undo takes back everything a paste made.**
        ///
        /// A paste was three entries on the history, and the curves it copied were written into
        /// the document beside them rather than through any of them: undoing it took the pasted
        /// blocks out and left their curves behind, read by nothing, and the two undos after it
        /// changed nothing anybody could see (refactor plan §3.3, root cause 5). It is one entry
        /// now, the copied curves and the graph edit in one batch, so one undo takes back every
        /// block and every curve it made, and one redo puts back exactly those.
        #[test]
        fn one_undo_takes_back_everything_a_paste_made() {
            use std::collections::BTreeSet;
            let (mut doc, part, _, _, _, _, union) = extrude_and_pipe_unioned();
            let curves = |doc: &Document| doc.curves.keys().copied().collect::<BTreeSet<CurveId>>();
            let nodes = |doc: &Document| {
                doc.parts[&part].graph.nodes().map(|(id, _)| id).collect::<BTreeSet<NodeId>>()
            };
            let (curves_before, nodes_before) = (curves(&doc), nodes(&doc));
            let mut stack = CommandStack::default();
            let pasted = copy_and_paste(&mut stack, &mut doc, part, union);
            let (curves_pasted, nodes_pasted) = (curves(&doc), nodes(&doc));
            assert_eq!(nodes_pasted.len(), 2 * nodes_before.len(), "the union and its two again");
            assert!(nodes_pasted.contains(&pasted));

            stack.undo(&mut doc).expect("the paste is on the history");
            assert_eq!(curves(&doc), curves_before, "every curve the paste copied goes with it");
            assert_eq!(nodes(&doc), nodes_before, "and every block it made");
            assert!(stack.undo(&mut doc).is_none(), "and the paste was the one entry");

            stack.redo(&mut doc).expect("and one redo");
            assert_eq!(curves(&doc), curves_pasted, "puts back the same curves");
            assert_eq!(nodes(&doc), nodes_pasted, "and the same blocks");
        }

        /// **A paste leaves each of the original's curves with the one reader it had.**
        ///
        /// Copy took the curve under a pipe and no other, so a pasted extrude named the original's
        /// profile. That profile then had two readers, which is one curve drawn inside both the
        /// original block and the pasted one — a shared instance of a curve nobody shared
        /// (refactor plan §3.3, root cause 5; before 112 it lost its owner and stood on its own).
        /// Counted as the block list counts it, distinct readers by `Node::curves`
        /// (`block-ownership-spec.md` I2), for the curve each block names. The pipe's line is the
        /// control, since a pipe's curve was always copied.
        #[test]
        fn a_paste_leaves_each_of_the_original_s_curves_with_the_one_reader_it_had() {
            let (mut doc, part, square, extrude, rail, pipe, union) = extrude_and_pipe_unioned();
            let mut stack = CommandStack::default();
            copy_and_paste(&mut stack, &mut doc, part, union);
            let nesting = doc.nesting_in(&doc.parts[&part].graph);
            let named_by = [(square[0], extrude, "the profile"), (rail, pipe, "the line")];
            for (curve, reader, what) in named_by {
                assert_eq!(nesting.readers_of_curve(curve), 1, "{what} is read by its block alone");
                assert_eq!(
                    nesting.curve_owners_of(curve).to_vec(),
                    vec![reader],
                    "{what} is drawn in its own block's row and nowhere else",
                );
            }
        }

        /// The square extruded from its second side, so the loop has curves either side of the one
        /// named, then copied and pasted: the document, its part, the square, the extrude and the
        /// paste of it.
        fn a_square_extruded_and_pasted() -> (Document, PartId, Vec<CurveId>, NodeId, NodeId) {
            let (mut doc, part, square) = square_document();
            let extrude = extrude_in(&mut doc, part, square[1]);
            let mut stack = CommandStack::default();
            let pasted = copy_and_paste(&mut stack, &mut doc, part, extrude);
            (doc, part, square, extrude, pasted)
        }

        /// Stamps every curve that has no stamp yet as younger than every curve that has one, which
        /// is what `stamp_new_nodes` does to a paste's copies once it has landed.
        fn stamped_as_the_paste_lands(doc: &mut Document) {
            let mut next = doc.curves_created.values().max().copied().unwrap_or(0);
            let stamped = |c: &CurveId| doc.curves_created.contains_key(c);
            let mut fresh: Vec<CurveId> = doc.curves.keys().copied().filter(|c| !stamped(c)).collect();
            fresh.sort_by_key(|c| c.raw());
            for curve in fresh {
                next += 1;
                doc.curves_created.insert(curve, next);
            }
        }

        /// The curves `extrude`'s profile is walked through and the area they enclose, found as
        /// the compile finds them.
        fn walked_profile(
            doc: &Document,
            part: PartId,
            extrude: NodeId,
        ) -> (std::collections::BTreeSet<CurveId>, f64) {
            let named = match doc.parts[&part].graph.node(extrude).and_then(Node::outline) {
                Some(fieldform_scene::Outline::Closed { profile }) => *profile,
                other => panic!("an extrude of a closed profile, not {other:?}"),
            };
            let found = fieldform_field::profile_loop(doc, named, None).expect("a closed profile");
            let points = &found.points;
            let twice: f64 = (0..points.len())
                .map(|i| {
                    let (a, b) = (points[i], points[(i + 1) % points.len()]);
                    a.x * b.y - b.x * a.y
                })
                .sum();
            (found.curves.into_iter().collect(), (twice * 0.5).abs())
        }

        /// **A pasted extrude is built on its own copy of every curve its profile is walked
        /// through.**
        ///
        /// An extrude names one curve of its loop and is built on the loop walked from it, so a
        /// paste that copied the named curve alone would still walk round the original's other
        /// sides. The pasted extrude names a copy, and the copies are the square again, side for
        /// side and in place. **Walked in the document as the paste leaves it**, with the
        /// original's sides still under the copies: every side of the walk is a copy, and it
        /// measures the square. Before the walk chose between curves lying on each other it went
        /// round the copies alone in about one paste in twenty. Both before and after the copies
        /// are stamped, since the walk's choice reads the stamps; over many sets of random ids,
        /// since those decided it.
        #[test]
        fn a_pasted_extrude_is_built_on_its_own_copy_of_every_curve_of_its_profile() {
            use fieldform_scene::Outline;
            use std::collections::BTreeSet;
            for _ in 0..32 {
                let (mut doc, part, square, _, pasted) = a_square_extruded_and_pasted();
                let originals: BTreeSet<CurveId> = square.iter().copied().collect();
                let named = match doc.parts[&part].graph.node(pasted).and_then(Node::outline) {
                    Some(Outline::Closed { profile }) => *profile,
                    other => panic!("the paste should extrude a closed profile, not {other:?}"),
                };
                assert!(!originals.contains(&named), "the pasted extrude names a curve of its own");
                let copies: BTreeSet<CurveId> =
                    doc.curves.keys().copied().filter(|c| !originals.contains(c)).collect();
                assert_eq!(copies.len(), square.len(), "one copy for each side of the loop");
                assert!(copies.contains(&named));
                let shapes = |ids: &mut dyn Iterator<Item = &CurveId>| {
                    let mut out: Vec<Vec<[u64; 3]>> =
                        ids.map(|id| every_control_point_bits(&doc, &[*id])).collect();
                    out.sort();
                    out
                };
                assert_eq!(
                    shapes(&mut copies.iter()),
                    shapes(&mut square.iter()),
                    "the square again, in place"
                );

                for stamped in [false, true] {
                    if stamped {
                        stamped_as_the_paste_lands(&mut doc);
                    }
                    let (walked, area) = walked_profile(&doc, part, pasted);
                    assert_eq!(walked, copies, "round its own four sides, stamped {stamped}");
                    assert!((area - 100.0).abs() < 1e-9, "the square's 100 mm², not {area}");
                }
            }
        }

        /// **A paste leaves the original built on its own curves, in the shape it had.**
        ///
        /// A paste lands in place, so every curve it copies lies on the one it was copied from, and
        /// the walk that finds an extrude's loop searches every open curve in the document. Before
        /// it chose between curves lying on each other, pasting an extrude left the original on
        /// its own four sides about one paste in nineteen. Otherwise it went round a mix of its
        /// own sides and the copies or, in more than half, closed with no area, out along a side
        /// and back along its copy, and the original stopped building. The original's walk is its
        /// own four sides and measures the square, and the paste is built on none of them. Both
        /// before and after the copies are stamped, over many sets of random ids.
        #[test]
        fn a_paste_leaves_the_original_built_on_its_own_curves_in_the_shape_it_had() {
            use std::collections::BTreeSet;
            for _ in 0..32 {
                let (mut doc, part, square, extrude, pasted) = a_square_extruded_and_pasted();
                let originals: BTreeSet<CurveId> = square.iter().copied().collect();
                for stamped in [false, true] {
                    if stamped {
                        stamped_as_the_paste_lands(&mut doc);
                    }
                    let (walked, area) = walked_profile(&doc, part, extrude);
                    assert_eq!(walked, originals, "round its own four sides, stamped {stamped}");
                    assert!((area - 100.0).abs() < 1e-9, "the square's 100 mm², not {area}");
                    let (theirs, _) = walked_profile(&doc, part, pasted);
                    assert!(theirs.is_disjoint(&originals), "and the paste is on none of them");
                }
            }
        }

        /// Deleting a curve deletes its relations in the same undo.
        #[test]
        fn deleting_a_curve_deletes_its_relations_in_the_same_undo() {
            use super::super::delete_curves_commands;
            use fieldform_scene::{Relation, RelationId, RelationRecord};
            let (mut doc, _part, square) = square_document();
            let id = RelationId::new();
            doc.relations.insert(
                id,
                RelationRecord {
                    relation: Relation::Parallel { master: square[0], slave: square[2] },
                    broken: None,
                },
            );
            let mut stack = CommandStack::default();
            let commands = delete_curves_commands(&doc, &[square[2]]);
            assert_eq!(commands.len(), 2, "the relation and the curve");
            stack.execute(&mut doc, Box::new(Batch::new("curve.delete", commands))).expect("deletes");
            assert!(doc.curve(square[2]).is_none());
            assert!(doc.relations.is_empty(), "the relation went with the curve");
            assert!(stack.undo(&mut doc).is_some(), "one entry");
            assert!(doc.curve(square[2]).is_some());
            assert_eq!(doc.relations.len(), 1, "and came back with it");
            assert!(stack.undo(&mut doc).is_none(), "and it was the only entry");
        }

        /// Opening an old document adds no joints.
        ///
        /// A loop found on the first partition of a document closed before the document was
        /// opened, and mutating a document because it was opened is a change nobody asked
        /// for. The rule is in `newly_closed_rows`: with no previous partition there is nothing
        /// newly closed, so nothing to hold.
        #[test]
        fn opening_an_old_document_adds_no_joints() {
            let (doc, part, square) = square_document();
            let rows = fieldform_field::profile_partition(&doc);
            assert_eq!(rows.len(), 1, "the file holds a loop");
            assert!(newly_closed_rows(None, &rows).is_empty(), "found, not newly closed");
            assert!(points_in(&doc, part).is_empty(), "and no joints were made for it");
            // Whereas the same loop appearing against an empty previous partition is an edit.
            assert_eq!(newly_closed_rows(Some(&[]), &rows), vec![0]);
            assert_eq!(rows[0].len(), square.len());
        }
    }

    /// Snapping to Top from a tipped side view keeps the roll, and repeating it moves nothing.
    ///
    /// Reported from a Left view tipped a little toward Top: clicking Top turned the model the
    /// other way round, because the top view's up was a fixed Y. The orbit that would reach
    /// Top smoothly ends with the up pointing where the camera was looking, so that is what
    /// the snap chooses, snapped to an axis so a second click asks for what the first gave.
    #[test]
    fn snapping_to_top_from_a_tipped_side_view_keeps_the_roll_and_repeating_it_moves_nothing() {
        use crate::{CubeRegion, OrbitCamera};
        use nalgebra::{Point3, Vector3};
        let top = CubeRegion { sign: [0, 0, 1] };
        let bottom = CubeRegion { sign: [0, 0, -1] };
        let up_after = |camera: &OrbitCamera| {
            camera.snap.as_ref().expect("a snap began").to * Vector3::y()
        };
        let mut camera = OrbitCamera::default();
        camera.target = Point3::origin();

        // A Left view -- eye toward +Y, looking along -Y -- tipped up a little.
        let tip = 0.2_f32;
        camera.eye = Point3::new(0.0, 300.0 * tip.cos(), 300.0 * tip.sin());
        camera.up = Vector3::new(0.0, -tip.sin(), tip.cos());
        camera.begin_snap(top);
        let up = up_after(&camera);
        assert!((up - Vector3::new(0.0, -1.0, 0.0)).norm() < 1e-4, "tipped: up became {up:?}");

        // Exactly square, the forward decides, and it agrees with the tipped answer.
        camera.eye = Point3::new(0.0, 300.0, 0.0);
        camera.up = Vector3::z();
        camera.begin_snap(top);
        let up = up_after(&camera);
        assert!((up - Vector3::new(0.0, -1.0, 0.0)).norm() < 1e-4, "square: up became {up:?}");

        // Under the bottom the up ends the other way.
        camera.begin_snap(bottom);
        let up = up_after(&camera);
        assert!((up - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-4, "bottom: up became {up:?}");

        // Already in that top view, clicking Top again asks for exactly what it has.
        camera.eye = Point3::new(0.0, 0.0, 300.0);
        camera.up = Vector3::new(0.0, -1.0, 0.0);
        camera.begin_snap(top);
        let up = up_after(&camera);
        assert!((up - Vector3::new(0.0, -1.0, 0.0)).norm() < 1e-4, "repeat: up became {up:?}");

        // A side view keeps world up whatever the camera was doing.
        camera.begin_snap(CubeRegion { sign: [-1, 0, 0] });
        let up = up_after(&camera);
        assert!((up - Vector3::z()).norm() < 1e-4, "front: up became {up:?}");
    }

    /// **A block's readers are counted once per block, however many times it names it.**
    ///
    /// The count behind "in how many places is this drawn". It was the V's guard — "if it's used
    /// in more than one place, you can't undeclare it. Just like in programming." — and 112
    /// reversed that: a block goes back inside every reader and is one block in each
    /// (`notebook-spec.md` §6.6, §7.3), so no count refuses the move and the number says how many
    /// places the one block appears in.
    ///
    /// The third case is what makes deduplication load-bearing rather than tidy, and it is the
    /// same case either way: a boolean given the same body twice holds two edges to it and is one
    /// block reading it, so the body is drawn in one place and not two.
    #[test]
    fn a_block_s_readers_are_counted_once_per_block_however_many_times_it_names_it() {
        // **Asserted against the document's own counter, not the app's.** This used to ask
        // `blocks_reading`, a copy in the app kept "free of the app so the property can be
        // asserted" — of a rule that no longer exists, since no count refuses the move. The
        // deduplication it was guarding is still load-bearing, but it is load-bearing *in
        // `Nesting`*, whose owner lists must not name one reader twice, so that is what is asked.
        use fieldform_scene::{BlockKey, BooleanOp, Document, Graph, Node, ParamRef, Part, PartId,
            Place};
        let readers = |graph: &Graph, id| {
            let mut doc = Document::new();
            doc.add_part(Part {
                id: PartId::new(),
                name: "Part".into(),
                material: None,
                graph: graph.clone(),
                inputs: Vec::new(),
            });
            doc.block_readers(BlockKey::Node(id)).len()
        };
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let sphere =
            graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [lit(20.0); 3],
            turn: fieldform_scene::no_turn(),
        });

        let cut = graph.add(Node::Boolean {
            op: BooleanOp::Difference,
            inputs: vec![cube.into(), sphere.into()],
        });
        assert_eq!(readers(&graph, sphere), 1, "one block reads it, so it is drawn once");

        let second = graph.add(Node::Boolean {
            op: BooleanOp::Difference,
            inputs: vec![cut.into(), sphere.into()],
        });
        assert_eq!(
            readers(&graph, sphere),
            2,
            "two blocks read it, so it is one block drawn in each of them",
        );

        // The same body for both operands: two edges, one reader.
        let twice = graph.add(Node::Boolean {
            op: BooleanOp::Union,
            inputs: vec![second.into(), second.into()],
        });
        assert_eq!(
            readers(&graph, second),
            1,
            "one block naming it twice is one block, not two",
        );
        assert_eq!(readers(&graph, twice), 0, "a body nothing reads");
    }

    /// **The blocks a variable goes back inside are named as a line reads them**
    /// (`notebook-spec.md` §6.6).
    ///
    /// This was the refusal the V and the row menu's Variable untick both asked for: two or more
    /// readers were no single block to go back inside, so the answer was a reason naming them.
    /// There is no refusal left — the block goes inside every reader and is one block in each
    /// (§7.3) — and the names are what the line that *reports* the move is built from, which is
    /// the same question asked for the opposite answer: "“Sphere” is part of Union and Shell
    /// again".
    ///
    /// So it is the joining that is held here, as a line reads it rather than as a list prints
    /// it: no comma before the last name, and none at all for two.
    #[test]
    fn the_blocks_a_variable_goes_back_inside_are_named_as_a_line_reads_them() {
        use crate::names_in_a_row;
        let names = |list: &[&str]| list.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        for (readers, said) in [
            (names(&[]), "nothing"),
            (names(&["Union"]), "Union"),
            (names(&["Union", "Shell"]), "Union and Shell"),
            (names(&["Union", "Shell", "Pattern"]), "Union, Shell and Pattern"),
            (names(&["Union", "Shell", "Pattern", "Loft"]), "Union, Shell, Pattern and Loft"),
        ] {
            assert_eq!(names_in_a_row(&readers), said, "{readers:?}");
        }
    }

    /// The sphere and cube a union reads, in a one-part document every edit has settled, and the
    /// stack the commands go through. The list is asked through `nesting_in`, as the panel asks.
    struct UnionOfTwo {
        doc: fieldform_ops::Document,
        stack: fieldform_ops::CommandStack,
        part: fieldform_scene::PartId,
        sphere: fieldform_scene::NodeId,
        cube: fieldform_scene::NodeId,
        joined: fieldform_scene::NodeId,
    }

    impl UnionOfTwo {
        fn new() -> Self {
            use super::record_graph_edit;
            use fieldform_scene::{BooleanOp, Graph, Node, ParamRef, Place};
            let lit = ParamRef::Literal;
            let ball = || Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) };
            let mut doc = fieldform_ops::Document::with_cage(fieldform_cage::Cage::cube_of(10.0));
            let part = *doc.parts.keys().next().expect("with_cage makes one part");
            doc.parts.get_mut(&part).expect("one part").graph = Graph::empty();
            let mut stack = fieldform_ops::CommandStack::default();
            let mut g = Graph::empty();
            let (sphere, cube) = (g.add(ball()), g.add(ball()));
            let joined = g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![sphere.into(), cube.into()] });
            record_graph_edit(&mut stack, &mut doc, &mut None, part, g).expect("the union");
            Self { doc, stack, part, sphere, cube, joined }
        }

        /// Every block `id` is drawn inside, ascending — empty for a row of its own, and two
        /// entries for a block two blocks read, which is one block in both (`notebook-spec.md`
        /// §7.3).
        fn owners(&self, id: fieldform_scene::NodeId) -> Vec<fieldform_scene::NodeId> {
            self.doc.nesting_in(&self.doc.parts[&self.part].graph).owners_of(id).to_vec()
        }

        fn name(&self, id: fieldform_scene::NodeId) -> Option<String> {
            self.doc.node_meta.get(&id).and_then(|m| m.name.clone())
        }

        fn run(&mut self, cmd: fieldform_ops::SetOwnership) {
            self.stack.execute(&mut self.doc, Box::new(cmd)).expect("the command runs");
        }

        fn step(&mut self, undo: bool) {
            let stepped = super::step_history(&mut self.stack, &mut self.doc, &mut None, undo);
            assert!(stepped.is_some(), "there was an entry to step over");
        }
    }

    /// **Making a variable and putting it back are each one undo, and the name stays**
    /// (`block-ownership-spec.md` §5.1, §5.2, §8).
    ///
    /// The V, the row menu and a drop on the list all reach these two, so what holds here holds
    /// for each. Making one takes the row out of the block that reads it and gives the block a
    /// name if it has none, since a variable is typed by its name (I4); one undo takes back the
    /// row and the invented name together. Putting one back goes inside whatever reads it and
    /// keeps the name, which is the difference from the gesture it replaced: clearing the name was
    /// the move back inside, so the name could not survive it.
    ///
    /// **With two readers it goes inside both, since 112** (`notebook-spec.md` §6.6): there is no
    /// one block to be inside, and with no parent stored that is not a contradiction to refuse but
    /// one block drawn in each of the two — a shared instance (§7.3). It used to be the one
    /// refusal, answered with who the readers were. With no readers at all it stands on its own.
    #[test]
    fn making_a_variable_and_putting_it_back_are_each_one_undo_and_the_name_stays() {
        use super::{make_variable, put_back_inside, record_graph_edit};
        use fieldform_scene::{BlockKey, Node, Ownership, ParamRef};
        let mut at = UnionOfTwo::new();
        let (sphere, cube, joined) = (at.sphere, at.cube, at.joined);
        assert_eq!(at.owners(sphere), vec![joined], "adopted by the union");

        let make = make_variable(&at.doc, BlockKey::Node(sphere), || "Sphere".to_string())
            .expect("inside a block, so it can become a variable");
        at.run(make);
        assert_eq!(at.doc.ownership_nodes.get(&sphere), Some(&Ownership::Variable));
        assert!(at.owners(sphere).is_empty(), "a row of its own");
        assert_eq!(at.name(sphere).as_deref(), Some("Sphere"), "and a name to be typed by");
        assert!(
            make_variable(&at.doc, BlockKey::Node(sphere), || "Again".to_string()).is_none(),
            "a variable already is one, so there is nothing to make",
        );
        at.step(true);
        assert_eq!(at.owners(sphere), vec![joined], "one undo puts the row back");
        assert_eq!(at.name(sphere), None, "and takes the name it was given with it");
        at.step(false);
        assert!(at.owners(sphere).is_empty(), "and a redo makes it again");

        // A block with a name keeps it; the invented one is only for a block that has none.
        at.doc.node_meta.entry(cube).or_default().name = Some("box".to_string());
        let make =
            make_variable(&at.doc, BlockKey::Node(cube), || "Cube".to_string()).expect("inside");
        at.run(make);
        assert_eq!(at.name(cube).as_deref(), Some("box"));

        let back = put_back_inside(&at.doc, BlockKey::Node(sphere)).expect("one reader");
        at.run(back);
        assert_eq!(at.doc.ownership_nodes.get(&sphere), Some(&Ownership::Nested));
        assert_eq!(at.owners(sphere), vec![joined], "back inside the one block reading it");
        assert_eq!(at.name(sphere).as_deref(), Some("Sphere"), "with its name");
        at.step(true);
        assert!(at.owners(sphere).is_empty(), "one undo makes it a variable again");

        // **A second reader refuses it, and names them** (`notebook-spec.md` §6.6 as amended
        // 2026-10-24, §17). 112 had it go back inside both and be one block in each; a second
        // reader declares the block now, so there is nowhere for it to go and a settle would put
        // it straight back. The wiring itself has already made it a variable here, which is the
        // other half of the same rule.
        let mut g = at.doc.parts[&at.part].graph.clone();
        let hollow = g.add(Node::Shell { input: sphere.into(), thickness: ParamRef::Literal(1.0) });
        record_graph_edit(&mut at.stack, &mut at.doc, &mut None, at.part, g.clone())
            .expect("a shell of it");
        // Ascending, since ids are random and that is the order the rows come in.
        let mut readers = vec![joined, hollow];
        readers.sort();
        assert_eq!(
            at.doc.ownership_nodes.get(&sphere),
            Some(&Ownership::Variable),
            "the second reader declared it",
        );
        assert_eq!(
            put_back_inside(&at.doc, BlockKey::Node(sphere)).err(),
            Some(readers.clone()),
            "so putting it back is refused, naming who reads it",
        );
        assert!(at.owners(sphere).is_empty(), "and its row stays its own");

        // None: it stands on its own.
        g.remove(hollow);
        g.remove(joined);
        record_graph_edit(&mut at.stack, &mut at.doc, &mut None, at.part, g).expect("both gone");
        let back = put_back_inside(&at.doc, BlockKey::Node(sphere)).expect("nothing reads it");
        at.run(back);
        assert_eq!(at.doc.ownership_nodes.get(&sphere), Some(&Ownership::Free));
    }

    /// **Clearing a variable's name is refused, and a rename moves no block**
    /// (`block-ownership-spec.md` §5.3, I4).
    ///
    /// A name is a label now, so writing one changes nothing about where a block is drawn,
    /// whatever it was: a block inside a union stays inside it named or not, a body stays a body,
    /// and a variable stays a variable under its new name. The one name that may not go is a
    /// variable's, because a variable is what other blocks refer to by name and an unnamed one
    /// could be reused and never typed; the refusal says what to do instead.
    #[test]
    fn clearing_a_variable_s_name_is_refused_and_a_rename_moves_no_block() {
        use super::{record_graph_edit, rename_block};
        use fieldform_scene::{BlockKey, Node, Ownership, ParamRef, Place};
        let mut at = UnionOfTwo::new();
        let (sphere, cube, joined) = (at.sphere, at.cube, at.joined);
        let mut g = at.doc.parts[&at.part].graph.clone();
        let lit = ParamRef::Literal;
        let loose = g.add(Node::Sphere {
            centre: Place::At([lit(50.0), lit(0.0), lit(0.0)]),
            radius: lit(5.0),
        });
        record_graph_edit(&mut at.stack, &mut at.doc, &mut None, at.part, g).expect("a body");
        let block = BlockKey::Node(cube);
        at.run(fieldform_ops::SetOwnership::new(block, Ownership::Variable, Some("box")));

        let renamed = rename_block(&at.doc, cube, "crate").expect("a variable may be renamed");
        at.run(renamed);
        assert_eq!(at.name(cube).as_deref(), Some("crate"));
        assert_eq!(at.doc.ownership_nodes.get(&cube), Some(&Ownership::Variable), "still one");
        for blank in ["", "   "] {
            let why = rename_block(&at.doc, cube, blank).err().expect("a variable needs a name");
            assert!(why.contains("put it back inside first"), "{why}");
        }

        for name in ["hub", ""] {
            let renamed = rename_block(&at.doc, sphere, name).expect("not a variable");
            at.run(renamed);
            assert_eq!(at.owners(sphere), vec![joined], "named {name:?}, and still the union's");
        }
        assert_eq!(at.name(sphere), None, "and the empty name cleared it");

        let renamed = rename_block(&at.doc, loose, "spare").expect("a body");
        at.run(renamed);
        assert_eq!(at.doc.ownership_nodes.get(&loose), Some(&Ownership::Free), "still a body");
        at.step(true);
        assert_eq!(at.name(loose), None, "one undo per rename");
    }

    /// What a click on the row menu's Variable item asks for, drawn alone in a headless egui and
    /// clicked the way a window delivers it: two quiet frames, since egui hit-tests against the
    /// previous frame's rectangles, then a move, a press and a release over the item's centre.
    fn variable_item_clicked(named: bool) -> Option<bool> {
        variable_item_clicked_where(named, None)
    }

    /// [`variable_item_clicked`] on a row whose V is somewhere else, for the reason `elsewhere`
    /// gives, as the menu draws a curve's row ([`crate::curve_v_elsewhere`]).
    fn variable_item_clicked_where(named: bool, elsewhere: Option<&str>) -> Option<bool> {
        use crate::{egui, variable_item};
        let ctx = egui::Context::default();
        let mut centre = egui::Pos2::ZERO;
        let mut asked = None;
        for step in 0..7u8 {
            let button = |pressed| egui::Event::PointerButton {
                pos: centre,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            };
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(400.0, 300.0),
                )),
                time: Some(0.05 * f64::from(step)),
                events: match step {
                    2 => vec![egui::Event::PointerMoved(centre)],
                    3 => vec![button(true)],
                    4 => vec![button(false)],
                    _ => vec![],
                },
                ..Default::default()
            };
            let _ = ctx.run_ui(raw, |ui| {
                let item = ui.scope(|ui| variable_item(ui, named, elsewhere));
                centre = item.response.rect.center();
                asked = asked.or(item.inner);
            });
        }
        asked
    }

    /// **The row menu's Variable item is live both ways, and dead only where the V is another
    /// row's.**
    ///
    /// The untick puts the block back inside exactly as the bright V does, and since 112 that
    /// move always has an answer: the block goes inside every reader and is one block in each, so
    /// no count can leave the item with nothing to do (`notebook-spec.md` §6.6, §12). The item went
    /// dead above one reader until then, and the reader names it was given to say why are gone
    /// with the guard -- the count does not reach it at all now. Where it goes back *to* is
    /// `making_a_variable_and_putting_it_back_are_each_one_undo_and_the_name_stays`.
    ///
    /// Clicked rather than inspected, in both directions, because what has to hold is that the
    /// click is delivered -- and, for a row whose V is its loop's, that it is not.
    #[test]
    fn the_row_menu_s_variable_item_is_live_both_ways_unless_the_v_is_another_row_s() {
        assert_eq!(
            variable_item_clicked(true),
            Some(false),
            "unticking puts the block back inside whatever reads it",
        );
        assert_eq!(variable_item_clicked(false), Some(true), "and ticking makes it a variable");
        for named in [false, true] {
            assert_eq!(
                variable_item_clicked_where(named, Some("its loop's row carries the V")),
                None,
                "a variable: {named} -- greyed where the V is another row's, so no click lands",
            );
        }
    }

    /// **A curve's Variable item is offered where its own V acts, and greyed where that V is its
    /// loop's.**
    ///
    /// The V on the curve heading a loop, and on one a block reads, shows the state and does
    /// nothing, because the profile row's V carries the gesture for the loop
    /// (`curve_v_elsewhere`). The row menu offered an enabled Variable item there all the same,
    /// and it made the curve a variable — which moved the whole profile row when that curve was
    /// the one placing it, from a menu on a row whose V says it cannot. A member nothing reads and
    /// a curve in no loop keep both. Clicked, in both states, since what has to hold is that the
    /// click does nothing.
    #[test]
    fn a_curve_s_variable_item_is_offered_only_where_its_own_v_acts() {
        use crate::{curve_v_elsewhere, ProfileRows};
        use fieldform_scene::CurveId;
        let square: Vec<CurveId> = (0..4).map(|_| CurveId::new()).collect();
        let loose = CurveId::new();
        let profiles = ProfileRows { at: None, loops: vec![square.clone()] };
        for (id, readers, its_own, what) in [
            (square[0], 0, false, "the curve heading the loop, nothing reading it"),
            (square[0], 1, false, "the curve heading the loop, an extrude reading it"),
            (square[1], 1, false, "a member an extrude reads"),
            (square[1], 2, false, "a member two blocks read"),
            (square[2], 0, true, "a member nothing reads"),
            (loose, 0, true, "a curve in no loop"),
            (loose, 1, true, "a curve in no loop that a pipe reads"),
        ] {
            let elsewhere = curve_v_elsewhere(&profiles, id, readers);
            assert_eq!(elsewhere.is_none(), its_own, "{what}: whose V is it");
            for is_variable in [false, true] {
                let clicked = variable_item_clicked_where(is_variable, elsewhere);
                let expected = match (its_own, is_variable) {
                    (false, _) => None,
                    (true, false) => Some(true),
                    (true, true) => Some(false),
                };
                assert_eq!(clicked, expected, "{what}, a variable: {is_variable}");
            }
        }
    }

    /// `graph` as a part of `doc`, settled as every edit leaves a document, so that
    /// `Document::nesting_in` reads each block where its readers put it.
    fn settled_in(doc: &mut fieldform_scene::Document, graph: &fieldform_scene::Graph) {
        doc.add_part(fieldform_scene::Part {
            id: fieldform_scene::PartId::new(),
            name: "Part".into(),
            material: None,
            graph: graph.clone(),
            inputs: Vec::new(),
        });
        doc.settle_ownership();
        doc.pending_ownership_changes.clear();
    }

    /// **A block whose input is made a variable still accounts for it** (backlog item 108).
    ///
    /// The owner's report: a sweep with one of its profiles made a variable stopped showing it.
    /// Before, a union reading a sphere and a cube held both rows; make the sphere a variable and
    /// the union's row held only the cube — one input fewer than it reads, and nothing saying
    /// which or where. Every input is now one of the three things the ownership spec reads it as:
    /// a row inside (owned), or a reference on the row (a variable). The count adds up again.
    #[test]
    fn a_block_whose_input_is_made_a_variable_still_accounts_for_it() {
        use crate::{nest_rows, Block};
        use fieldform_scene::{
            BlockKey, BooleanOp, Document, Graph, Node, Ownership, ParamRef, Place,
        };
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let sphere =
            graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [lit(20.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let union = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![sphere.into(), cube.into()] });
        let flat = vec![Block::Node(sphere), Block::Node(cube), Block::Node(union)];
        settled_in(&mut doc, &graph);

        let row_of = |rows: &[crate::BlockRow], id| {
            rows.iter().find(|r| r.block == Block::Node(id)).map(|r| {
                (r.inside.iter().map(|i| i.block).collect::<Vec<_>>(), r.references.clone())
            })
        };

        // Both owned: two rows inside, nothing referred to.
        let rows = nest_rows(flat.clone(), &doc.nesting_in(&graph), &|_| Vec::new());
        let (inside, refs) = row_of(&rows, union).expect("the union's row");
        assert_eq!(inside.len() + refs.len(), 2, "two inputs, two accounted for");
        assert!(refs.is_empty());

        // The sphere made a variable: its own row at the top, and the union still says it reads it.
        doc.ownership_nodes.insert(sphere, Ownership::Variable);
        let rows = nest_rows(flat, &doc.nesting_in(&graph), &|_| Vec::new());
        let (inside, refs) = row_of(&rows, union).expect("the union's row");
        assert_eq!(inside, vec![Block::Node(cube)], "the cube is still the union's own");
        assert_eq!(refs, vec![BlockKey::Node(sphere)], "and the sphere is referred to, not lost");
        assert_eq!(inside.len() + refs.len(), 2, "the count adds up again — the reported bug");
        assert!(
            rows.iter().any(|r| r.block == Block::Node(sphere) && !r.nested),
            "the variable keeps its row of its own: referred to here, edited there",
        );
    }

    /// **A union of a sphere and a cube nobody named is one row holding two.**
    ///
    /// The layout the owner asked for, and the shape of the tree the panel draws:
    /// `Document::nesting_in` reads who owns whom and `nest_rows` turns the flat list into rows
    /// within rows. Asserted here rather than in the panel because an `App` needs a window and a
    /// device, which is the same reason the order it reads lives on `Document`.
    #[test]
    fn a_union_of_two_unnamed_blocks_is_one_row_holding_two() {
        use crate::{nest_rows, Block};
        use fieldform_scene::{BooleanOp, Document, Graph, Node, Ownership, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let sphere =
            graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let cube = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [lit(20.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let union = graph.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![sphere.into(), cube.into()] });
        // The flat list, in the order `Document::listed_blocks` gives it: ingredients first.
        let flat = vec![Block::Node(sphere), Block::Node(cube), Block::Node(union)];
        settled_in(&mut doc, &graph);

        let rows = nest_rows(flat.clone(), &doc.nesting_in(&graph), &|_| Vec::new());
        assert_eq!(rows.len(), 1, "one top-level row, which is the body");
        assert_eq!(rows[0].block, Block::Node(union));
        assert!(!rows[0].nested, "a body is nobody's content");
        assert_eq!(
            rows[0].inside.iter().map(|r| r.block).collect::<Vec<_>>(),
            vec![Block::Node(sphere), Block::Node(cube)],
            "both ingredients inside it, in the order the list had them",
        );
        assert!(rows[0].inside.iter().all(|r| r.nested), "and each knows it is inside one");

        // Making one a variable lifts it back out, beside the union rather than inside it — and
        // above, which is where the flat list already had it.
        doc.ownership_nodes.insert(sphere, Ownership::Variable);
        let rows = nest_rows(flat, &doc.nesting_in(&graph), &|_| Vec::new());
        assert_eq!(
            rows.iter().map(|r| r.block).collect::<Vec<_>>(),
            vec![Block::Node(sphere), Block::Node(union)],
            "the variable takes a row of its own, above what consumes it",
        );
        assert_eq!(
            rows[1].inside.iter().map(|r| r.block).collect::<Vec<_>>(),
            vec![Block::Node(cube)],
            "and the cube is still the union's",
        );
    }

    /// **The insertion line is drawn where the block will land, which is the top of a branch.**
    ///
    /// A drag moves a body together with what only it reads, so a variable pinned above the body
    /// that consumes it travels with it. A line drawn on the body's own row therefore promised one
    /// place and delivered the one above it — seen by running the list, which is what
    /// `block-order-spec.md` §8.4 asked for before anything else was built on this drag.
    #[test]
    fn the_insertion_line_sits_at_the_top_of_a_branch_rather_than_on_its_body() {
        use crate::{branch_starts, nest_rows, Block, BlockRow};
        use fieldform_scene::{BooleanOp, Document, Graph, Node, Ownership, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let hub = graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let inner = graph.add(Node::Cuboid { edge: Default::default(),
            centre: Place::At([lit(0.0); 3]),
            size: [lit(20.0); 3],
            turn: fieldform_scene::no_turn(),
        });
        let cut = graph.add(Node::Boolean {
            op: BooleanOp::Difference,
            inputs: vec![hub.into(), inner.into()],
        });
        let loose = graph.add(Node::Sphere {
            centre: Place::At([lit(60.0), lit(0.0), lit(0.0)]),
            radius: lit(9.0),
        });
        // The hub is a variable, so it has a row of its own — directly above the block that
        // consumes it, which is where dependency pins it and why it moves when that block moves.
        settled_in(&mut doc, &graph);
        doc.ownership_nodes.insert(hub, Ownership::Variable);

        let rows = nest_rows(
            vec![Block::Node(loose), Block::Node(hub), Block::Node(cut)],
            &doc.nesting_in(&graph),
            // No profile in this case, so nothing asks.
            &|_| Vec::new(),
        );
        assert_eq!(
            rows.iter().map(|r| r.block).collect::<Vec<_>>(),
            vec![Block::Node(loose), Block::Node(hub), Block::Node(cut)],
            "the cube is inside the boolean and the other three are rows",
        );
        assert_eq!(
            branch_starts(&rows, &graph),
            vec![Some(loose), Some(cut), Some(cut)],
            "the boolean's branch starts at the named block above it, not at the boolean",
        );

        // A row the body cannot reach is not part of its branch and does not move with it, so the
        // line does not go above that either.
        let orphan = BlockRow {
            block: Block::Parameter(fieldform_scene::ParamId::new()),
            inside: Vec::new(),
            nested: false,
            readers: 0,
            references: Vec::new(),
        };
        let rows = vec![orphan, rows.into_iter().next_back().expect("the boolean's row")];
        assert_eq!(
            branch_starts(&rows, &graph),
            vec![None, Some(cut)],
            "a variable's row is a section of its own, and nothing lands in front of it",
        );
    }

    /// **A row never disappears, whatever the nesting says.**
    ///
    /// Two ways it could: an owner that is not a row in the list at all — a block folded into an
    /// import's row is the real case — and an owner that comes *above* its own content, which
    /// dependency forbids for a graph edge but not for a curve, since curves are merged into the
    /// list by when they were made and a file decides that. Either one silently swallows a block,
    /// and a block you cannot reach is worse than a row in an odd place.
    #[test]
    fn a_row_whose_owner_is_not_in_the_list_stays_in_the_list() {
        use crate::{nest_rows, Block};
        use fieldform_scene::{Document, Graph, Node, ParamRef, Place};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();
        let sphere =
            graph.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
        let shell = graph.add(Node::Shell { input: sphere.into(), thickness: lit(2.0) });
        settled_in(&mut doc, &graph);
        let nesting = doc.nesting_in(&graph);
        assert_eq!(
            nesting.owners_of(sphere).to_vec(),
            vec![shell],
            "the sphere is drawn inside the shell",
        );

        // The owner is missing from the list.
        let rows = nest_rows(vec![Block::Node(sphere)], &nesting, &|_| Vec::new());
        assert_eq!(rows.iter().map(|r| r.block).collect::<Vec<_>>(), vec![Block::Node(sphere)]);
        assert!(!rows[0].nested, "nothing to be nested in");

        // The owner is above its content, which would mean gathering a row that had already gone
        // past.
        let rows = nest_rows(vec![Block::Node(shell), Block::Node(sphere)], &nesting, &|_| Vec::new());
        assert_eq!(
            rows.iter().map(|r| r.block).collect::<Vec<_>>(),
            vec![Block::Node(shell), Block::Node(sphere)],
            "both rows are still there",
        );
        assert!(rows.iter().all(|r| r.inside.is_empty()));
    }

    /// The twist tool starts from a body it could actually wind, not from the first one listed.
    ///
    /// **Reported**: "I make a point, a plane, and then I twist it, the twist tool automatically
    /// selects I think the plane. I don't think that really makes sense." It does not: a plane
    /// and a point are both bodies by `Node::is_body`, so a part holding them before it holds a
    /// solid handed the twist one of those — and then its own panel greyed it, because
    /// `operand_refusal` refuses an unbounded body to a twist. The tool was disagreeing with its
    /// own panel.
    ///
    /// Asserted through `twist_would_take`, which is the question `apply_twist` asks, so the two
    /// cannot drift from the picker's rule.
    #[test]
    fn the_twist_tool_starts_from_a_body_it_could_wind() {
        use crate::twist_would_take;
        use fieldform_scene::{Document, Graph, Node, ParamRef};
        let lit = ParamRef::Literal;
        let doc = Document::new();
        let mut g = Graph::empty();

        // The order the report built them in: a point, then a plane, then a solid.
        let dot = g.add(Node::Point { at: [lit(0.0); 3] });
        let plane = g.add(Node::Plane {
            origin: fieldform_scene::Place::At([lit(0.0); 3]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });
        let cube = g.add(Node::Cuboid { edge: Default::default(),
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            size: [lit(20.0); 3],
            turn: fieldform_scene::no_turn(),
        });

        assert!(!twist_would_take(&doc, &g, dot), "a point has nothing to wind");
        assert!(!twist_would_take(&doc, &g, plane), "a plane has no extent to wind over");
        assert!(twist_would_take(&doc, &g, cube), "a solid is what a twist is for");

        // **The rule the tool applies is the rule the panel applies.** Were these two to drift,
        // the tool would hand itself an operand its own panel greys -- which is the bug.
        let twisted = g.add(Node::Twist {
            input: cube.into(),
            degrees_per_mm: lit(5.0),
            frame: fieldform_scene::Frame::default(),
        });
        for candidate in [dot, plane, cube] {
            assert_eq!(
                crate::operand_refusal(&doc, &g, twisted, candidate).is_none(),
                twist_would_take(&doc, &g, candidate),
                "the tool and the panel disagree about a candidate"
            );
        }
    }

    /// Every operand slot refuses what its block could not do anything with, and nothing else.
    ///
    /// **The whole table, because a table of rules is the thing that drifts.** Asked for after
    /// finding that every block offered the same list: "we used to be able to boolean two points
    /// together, I don't think that's valid ... similarly, you shouldn't be able to twist a
    /// plane." Both hold. What the audit added is that it is not "points and planes are never
    /// valid" — the rule is what the block does to the field, so an offset of a point is a sphere
    /// and a mirror carries anything.
    #[test]
    fn an_operand_slot_refuses_what_its_block_could_not_use() {
        use crate::operand_refusal;
        use fieldform_scene::{Continuity, Document, Fillet, Frame, Graph, Node, ParamRef};
        let lit = ParamRef::Literal;
        let doc = Document::new();
        let mut g = Graph::empty();

        let ball = g.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(10.0),
        });
        let dot = g.add(Node::Point { at: [lit(20.0), lit(0.0), lit(0.0)] });
        let plane = g.add(Node::Plane {
            origin: fieldform_scene::Place::At([lit(0.0); 3]),
            flipped: false,
            turn: fieldform_scene::no_turn(),
        });

        let round = Fillet { radius: lit(2.0), continuity: Continuity::G1, second: None };
        let takers = [
            ("boolean", g.add(Node::Boolean {
                op: fieldform_scene::BooleanOp::Union,
                inputs: vec![ball.into()],
            })),
            ("blend", g.add(Node::Blend {
                op: fieldform_scene::BooleanOp::Union,
                a: ball.into(),
                b: ball.into(),
                fillet: round.clone(),
            })),
            ("shell", g.add(Node::Shell { input: ball.into(), thickness: lit(2.0) })),
            ("morph", g.add(Node::Morph { a: ball.into(), b: ball.into(), t: lit(0.5) })),
            ("twist", g.add(Node::Twist {
                input: ball.into(),
                degrees_per_mm: lit(5.0),
                frame: Frame::default(),
            })),
            ("offset", g.add(Node::Offset { input: ball.into(), distance: lit(3.0) })),
            ("mirror", g.add(Node::Mirror { input: ball.into(), frame: Frame::default(), join: true })),
            ("pattern", g.add(Node::Pattern {
                options: Default::default(),
                input: ball.into(),
                count: lit(6.0),
                kind: fieldform_scene::PatternKind::Radial { frame: Frame::default() },
                join: false,
                fillet: round,
            })),
        ];
        let taker = |name: &str| takers.iter().find(|(n, _)| *n == name).expect("a taker").1;
        let takes = |block: &str, given| operand_refusal(&doc, &g, taker(block), given).is_none();

        // A point: refused wherever the block needs an interior, taken where the block is a
        // motion, adds to the field, or reaches to one on purpose.
        for block in ["boolean", "shell", "morph", "twist", "pattern"] {
            assert!(!takes(block, dot), "{block} should refuse a point");
        }
        for block in ["blend", "offset", "mirror"] {
            assert!(takes(block, dot), "{block} should take a point");
        }

        // Unbounded: refused wherever the block needs a bound. A boolean takes one, because
        // trimming a part with a half-space is what `Node::Plane` exists for.
        for block in ["twist", "pattern"] {
            assert!(!takes(block, plane), "{block} should refuse an unbounded body");
        }
        for block in ["boolean", "blend", "offset", "mirror", "shell", "morph"] {
            assert!(takes(block, plane), "{block} should take an unbounded body");
        }

        // **The control**: an ordinary solid goes anywhere. A gate that refused everything would
        // pass every assertion above.
        for (name, _) in &takers {
            assert!(takes(name, ball), "{name} should take a plain solid");
        }

        // And the refusals say why, since the reason is what the row shows on hover.
        let why = operand_refusal(&doc, &g, taker("twist"), plane).expect("refused");
        assert!(why.contains("extent"), "the reason should name the cause, not just refuse: {why}");
    }

    /// A profile of several curves is drawn inside the block made from it, whichever of its
    /// curves that block happens to name.
    ///
    /// **Reported**: "I made two curves into a profile, I then went to extrude the profile, the
    /// profile is no longer inside the extrude ... for some reason it's now outside." It was a
    /// coin flip, which is why it looked arbitrary. A profile row is keyed by its loop's first
    /// curve in walk order; an extrude names *one* curve of the loop, whichever was picked. When
    /// those were the same curve the row nested, and when they were not the key had no reader at
    /// all, so the row had no owner and sat outside the extrude while one of its own curves sat
    /// inside it.
    ///
    /// Asserted for **both** curves, because asserting it for one would pass on the walk order
    /// rather than on the fix.
    #[test]
    fn a_profile_of_two_curves_nests_in_the_extrude_whichever_curve_it_names() {
        use crate::{nest_rows, Block};
        use fieldform_scene::{Curve, CurveId, Document, Graph, Node, ParamRef};
        let lit = ParamRef::Literal;

        // Two arcs-worth of polyline meeting end to end, so they close as one loop.
        let mut doc = Document::new();
        let a = CurveId::new();
        let b = CurveId::new();
        let p = |x: f64, y: f64| nalgebra::Point3::new(x, y, 0.0);
        doc.curves.insert(
            a,
            Curve::polyline(a, "Line", vec![p(0.0, 0.0), p(40.0, 0.0), p(40.0, 30.0)], false)
                .expect("open"),
        );
        doc.curves.insert(
            b,
            Curve::polyline(b, "Line 2", vec![p(40.0, 30.0), p(0.0, 30.0), p(0.0, 0.0)], false)
                .expect("open"),
        );
        let walk = |seed| fieldform_field::loop_from(&doc, seed, None).unwrap_or_default();
        let members = walk(a);
        assert_eq!(members.len(), 2, "the two curves close as one loop");
        // The key the row is drawn under: the loop's first curve, in walk order.
        let key = members[0];

        for named in [a, b] {
            let mut graph = Graph::empty();
            let extrude = graph.add(Node::Extrude { rims: Default::default(),
                outline: fieldform_scene::Outline::closed(named),
                sweep: fieldform_scene::Sweep::Normal,
                height: lit(10.0),
                start: lit(0.0),
                sides: fieldform_scene::Sides::One,
                draft: lit(0.0),
                corners: Default::default(),
            });
            // A document of its own for each, so the two extrudes are not two readers of one curve.
            let mut settled = doc.clone();
            settled_in(&mut settled, &graph);
            let rows = nest_rows(
                vec![Block::Profile(key), Block::Node(extrude)],
                &settled.nesting_in(&graph),
                &walk,
            );
            assert_eq!(
                rows.iter().map(|r| r.block).collect::<Vec<_>>(),
                vec![Block::Node(extrude)],
                "with the extrude naming {}, the profile row is not inside it",
                if named == key { "the loop's key" } else { "the loop's other curve" }
            );
            assert_eq!(
                rows[0].inside.iter().map(|r| r.block).collect::<Vec<_>>(),
                vec![Block::Profile(key)],
                "and the profile is the row inside it"
            );
        }
    }

    /// Two patterns are unioned with each other, not with what one of them is made of.
    ///
    /// **Reported**: a pattern of an extrude and a pattern of a pipe, unioned, and "it kicks the
    /// pipe out of the pattern and it tries to union the pipe with the pattern instead of the
    /// pattern with the pattern". The eviction is the nesting rule working correctly on a wrong
    /// operand — a block nests when exactly one thing consumes it, so the moment a boolean takes
    /// the pipe as well as its pattern, the pipe has two readers and gets a row of its own.
    ///
    /// The wrong operand is the thing to fix. A nested row is the block a pattern is *made of*,
    /// and reaching past a pattern to it is not what "union these two" means.
    #[test]
    fn a_boolean_starts_from_the_block_a_row_belongs_to_and_not_from_what_is_inside_it() {
        use crate::starting_pair_among;
        use fieldform_scene::{Document, Graph, Node, ParamRef};
        let lit = ParamRef::Literal;
        let mut doc = Document::new();
        let mut graph = Graph::empty();

        // A pattern of an extrude, six copies.
        let id = fieldform_scene::CurveId::new();
        let corners = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)];
        doc.curves.insert(
            id,
            fieldform_scene::Curve::polyline(
                id,
                "square",
                corners.iter().map(|&(x, y)| nalgebra::Point3::new(x, y, 0.0)).collect(),
                true,
            )
            .expect("closed"),
        );
        let extrude = graph.add(Node::Extrude { rims: Default::default(),
            outline: fieldform_scene::Outline::closed(id),
            sweep: fieldform_scene::Sweep::Normal,
            height: lit(10.0),
            start: lit(0.0),
            sides: fieldform_scene::Sides::One,
            draft: lit(0.0),
            corners: Default::default(),
        });
        let blades = graph.add(Node::Pattern {
            options: Default::default(),
            input: extrude.into(),
            count: lit(6.0),
            kind: fieldform_scene::PatternKind::Radial {
                frame: fieldform_scene::Frame::default(),
            },
            join: false,
            fillet: fieldform_scene::Fillet {
                radius: lit(0.0),
                continuity: fieldform_scene::Continuity::G1,
                second: None,
            },
        });

        // Then a pattern of a pipe.
        let spine = fieldform_scene::CurveId::new();
        doc.curves.insert(
            spine,
            fieldform_scene::Curve::polyline(
                spine,
                "spine",
                vec![nalgebra::Point3::new(0.0, 0.0, 0.0), nalgebra::Point3::new(40.0, 0.0, 0.0)],
                false,
            )
            .expect("a line"),
        );
        let pipe = graph.add(Node::Pipe { curve: spine, radius: lit(3.0) });
        let pipes = graph.add(Node::Pattern {
            options: Default::default(),
            input: pipe.into(),
            count: lit(6.0),
            kind: fieldform_scene::PatternKind::Radial {
                frame: fieldform_scene::Frame::default(),
            },
            join: false,
            fillet: fieldform_scene::Fillet {
                radius: lit(0.0),
                continuity: fieldform_scene::Continuity::G1,
                second: None,
            },
        });

        let order = [extrude, blades, pipe, pipes];
        let made = move |n| order.iter().position(|x| *x == n).unwrap_or(99) as u64;
        settled_in(&mut doc, &graph);
        let nesting = doc.nesting_in(&graph);

        // The arrangement the report starts from: each pattern draws its own body inside itself.
        assert_eq!(
            nesting.owners_of(pipe).to_vec(),
            vec![pipes],
            "the pipe is drawn inside its pattern",
        );
        assert_eq!(
            nesting.owners_of(extrude).to_vec(),
            vec![blades],
            "and the extrude inside its own",
        );

        // Nothing selected: the two patterns, which is what the roots give and always did.
        assert_eq!(
            starting_pair_among(&graph, made, None, None, |_| true),
            Some((blades, pipes)),
            "with nothing chosen the pair is the two bodies"
        );

        // **The report.** The nested pipe's row is selected -- which is what clicking the row
        // inside the pattern does -- and the pair must still be the two patterns. Composed the
        // way `App::starting_pair` composes it, since the lifting is what the fix is.
        let selected = |id| Some(crate::outermost_in(&nesting, id));
        assert_eq!(
            starting_pair_among(&graph, made, selected(pipe), None, |_| true),
            Some((pipes, blades)),
            "a boolean started from a nested row reaches past the pattern to the pipe"
        );

        // And the same from the other side, so this is not an accident of the ordering.
        assert_eq!(
            starting_pair_among(&graph, made, selected(extrude), None, |_| true),
            Some((blades, pipes)),
            "the same, from the extrude's row"
        );

        // **The reported path, with two screenshots behind it.** `add_pipe_body` points
        // `boolean_b` at the raw pipe it just made; a pattern is built over that pipe afterwards;
        // `boolean -u` then runs with a seed naming a node that is now inside something. Both
        // seeds are lifted, so the union is still of the two patterns -- unlifted, it took the
        // pipe, which then has the union for a second reader and is drawn inside it as well as
        // inside its pattern (`notebook-spec.md` §7.3), which is not the body anybody pointed at.
        let seed = |id| Some(crate::outermost_in(&nesting, id));
        assert_eq!(
            starting_pair_among(&graph, made, seed(extrude), seed(pipe), |_| true),
            Some((blades, pipes)),
            "a seed naming a block that has since been wrapped still reaches the inner block"
        );
        // And the shape of the failure, so this case cannot pass by the lift being a no-op:
        // unlifted, the same two seeds give the two *inner* blocks.
        assert_eq!(
            starting_pair_among(&graph, made, Some(extrude), Some(pipe), |_| true),
            Some((extrude, pipe)),
            "unlifted, the seeds name what the patterns are made of"
        );

        // And a row that is already outermost is its own answer, so nothing moved for the
        // ordinary case of two bodies neither of which is inside anything.
        assert_eq!(crate::outermost_in(&nesting, pipes), pipes);
        assert_eq!(crate::outermost_in(&nesting, blades), blades);
    }

    /// A boolean's starting pair never includes a point, and a blend's may.
    #[test]
    fn a_boolean_starts_from_no_point_and_a_blend_may() {
        use crate::{operand_choices_in, starting_pair_among, starting_pair_in};
        use fieldform_scene::{Graph, Node, ParamRef};
        let lit = ParamRef::Literal;
        let mut graph = Graph::empty();
        let point = graph.add(Node::Point { at: [lit(0.0); 3] });
        let sphere = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(0.0); 3]),
            radius: lit(10.0),
        });
        let ball = graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([lit(30.0), lit(0.0), lit(0.0)]),
            radius: lit(10.0),
        });
        let not_a_point = |id| !matches!(graph.node(id), Some(Node::Point { .. }));
        // When each was made, as `Document::created` holds it -- the picker orders its candidates
        // by that now, so it has to be told. Written out in the order they were added above.
        let made = move |id| [point, sphere, ball].iter().position(|x| *x == id).unwrap_or(9) as u64;
        // Asked to start from the point, a boolean starts from the solids instead.
        let (a, b) = starting_pair_among(&graph, made, Some(point), None, not_a_point)
            .expect("two solids");
        assert!(a != point && b != point);
        assert_eq!((a, b), (sphere, ball), "the two solids, oldest first");
        // A blend is allowed to reach to it.
        let (a, _) = starting_pair_in(&graph, made, Some(point), None).expect("a pair");
        assert_eq!(a, point);
        // And every block's operand list offers it, although a point is not a body.
        assert!(!graph.roots().contains(&point), "a point is never a body");
        assert!(operand_choices_in(&graph, sphere, made).contains(&point));
    }

    /// Every degree the picker offers has to actually construct. The knot vector is the part
    /// that can be silently wrong: `Curve::new` requires exactly `control_points + degree + 1`
    /// knots and a non-degenerate span between them, so an off-by-one in either half of it
    /// fails only at the degree where the arithmetic happens to disagree.
    #[test]
    fn every_offered_degree_makes_a_curve_of_that_degree() {
        for degree in CURVE_DEGREES {
            let id = fieldform_scene::CurveId::new();
            let curve = single_span_bezier(id, degree, 50.0)
                .unwrap_or_else(|e| panic!("degree {degree} did not construct: {e}"));
            assert_eq!(curve.degree(), degree);
            assert_eq!(
                curve.control_points().len(),
                degree + 1,
                "a single-span Bézier of degree {degree} has {} control points",
                degree + 1,
            );
            // One span, which is what "Bézier" means here: the whole domain is between the two
            // knot values, with nothing in between.
            let (lo, hi) = curve.domain();
            assert!(hi > lo, "degree {degree} has an empty domain");
        }
    }

    /// A degree-one curve and a line are the same object, so the picker's first entry must not
    /// quietly produce something else. The half-sine is zero at both ends, which is what makes
    /// this true rather than a coincidence.
    #[test]
    fn a_degree_one_curve_is_a_straight_line() {
        let id = fieldform_scene::CurveId::new();
        let curve = single_span_bezier(id, 1, 50.0).unwrap();
        let points = curve.control_points();
        assert_eq!(points.len(), 2);
        assert!(points[0].coords.norm() < 1e-12, "starts at the origin");
        assert!((points[1] - points[0]).normalize().z.abs() < 1e-12, "runs flat");
        // And the curve really is straight: its middle is the midpoint of its ends.
        let middle = curve.evaluate(0.5);
        let expected = nalgebra::center(&points[0], &points[1]);
        assert!((middle - expected).norm() < 1e-9);
    }

    /// Higher degrees must actually bend, or the picker is six ways of making the same line.
    #[test]
    fn a_higher_degree_curve_leaves_the_chord_it_spans() {
        for degree in [2usize, 3, 5, 9] {
            let id = fieldform_scene::CurveId::new();
            let curve = single_span_bezier(id, degree, 50.0).unwrap();
            let points = curve.control_points();
            let chord = nalgebra::center(&points[0], &points[points.len() - 1]);
            let middle = curve.evaluate(0.5);
            assert!(
                (middle - chord).norm() > 1.0,
                "degree {degree} drew a straight line",
            );
        }
    }

    /// The cube is drawn out of the same quads it is picked from, so the drawing is also the
    /// proof that the twenty-six regions are all reachable. Six faces of nine cells is
    /// fifty-four quads naming twenty-six regions: a face is named once, an edge by the two
    /// faces that meet along it, a corner by the three that meet at it.
    ///
    /// A dead region — one no cell names — would be a target the user can see and cannot hit,
    /// and it would be invisible in a screenshot.
    #[test]
    fn the_faces_cells_name_every_one_of_the_twenty_six_regions() {
        let mut seen: HashSet<[i8; 3]> = HashSet::new();
        let mut cells = 0;
        for face in view_cube_faces() {
            for (region, _) in view_cube_cells(face) {
                seen.insert(region.sign);
                cells += 1;
            }
        }
        assert_eq!(cells, 54, "six faces of nine cells");
        assert_eq!(seen.len(), 26, "six faces, twelve edges, eight corners");
        assert!(!seen.contains(&[0, 0, 0]), "the centre is not a region");

        let of_kind = |n: usize| {
            seen.iter().filter(|s| s.iter().filter(|c| **c != 0).count() == n).count()
        };
        assert_eq!(of_kind(1), 6, "faces");
        assert_eq!(of_kind(2), 12, "edges");
        assert_eq!(of_kind(3), 8, "corners");
    }

    /// Every region has to give a usable camera basis. Where the view direction is world up —
    /// straight down and straight up — the ordinary choice of up is degenerate, its cross
    /// product with the direction being zero, and a camera built from it has no orientation at
    /// all.
    #[test]
    fn no_region_leaves_the_camera_without_an_up_vector() {
        for face in view_cube_faces() {
            for (region, _) in view_cube_cells(face) {
                let d = region.direction();
                let u = region.up();
                assert!(d.norm().is_finite() && (d.norm() - 1.0).abs() < 1e-6, "{:?}", region.sign);
                assert!(
                    d.cross(&u).norm() > 1e-3,
                    "region {:?} has an up parallel to its view direction",
                    region.sign,
                );
            }
        }
    }

    /// Top and Bottom take a *fixed* up rather than one derived from where the camera happens
    /// to be. Deriving it is how repeated Top/Bottom clicks slowly rotate the view — the model
    /// creeps a little further round each time and nothing looks obviously wrong on any single
    /// click. This asserts the cure: the same click always gives the same answer.
    #[test]
    fn looking_straight_down_gives_the_same_view_every_time() {
        let top = CubeRegion { sign: [0, 0, 1] };
        let bottom = CubeRegion { sign: [0, 0, -1] };
        for _ in 0..5 {
            assert_eq!(top.up(), CubeRegion { sign: [0, 0, 1] }.up());
            assert_eq!(bottom.up(), CubeRegion { sign: [0, 0, -1] }.up());
        }
    }

    /// A corner looks equally at the three faces that meet there, which is what makes it read
    /// as an isometric view rather than as an oblique one.
    #[test]
    fn a_corner_looks_equally_at_its_three_faces() {
        for sign in [[1i8, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]] {
            let d = CubeRegion { sign }.direction();
            assert!((d.x.abs() - d.y.abs()).abs() < 1e-6, "{sign:?}");
            assert!((d.y.abs() - d.z.abs()).abs() < 1e-6, "{sign:?}");
        }
    }

/// **A part with an empty slot refuses to be measured, rather than measuring what is left.**
///
/// The owner's §7.2 ruling has two halves that pull opposite ways, and this is the one that is
/// easy to lose: the viewport *prunes* a holed body so the rest still draws, and the three
/// statements about the part read that same pruned graph — so without this they would answer
/// confidently about a part with a body silently missing. Hiding a body is something somebody
/// chose; a hole is not.
///
/// Asserted over the free rule rather than through `App`, which needs a window and a device
/// before it will exist — the arrangement `operand_choices_in` and `ownership_news` already have.
#[test]
fn a_part_with_an_empty_slot_refuses_to_be_measured_rather_than_measuring_what_is_left() {
    use crate::app::no_statement_while_a_slot_is_empty;
    use fieldform_scene::{BooleanOp, Graph, Node, Operand, ParamRef, Place};
    let lit = ParamRef::Literal;
    let name = |_: fieldform_scene::NodeId| "Shell".to_string();

    let mut whole = Graph::empty();
    let ball = whole.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
    let _ = whole.add(Node::Shell { input: ball.into(), thickness: lit(1.0) });
    assert!(
        no_statement_while_a_slot_is_empty(&whole, name, "measure").is_none(),
        "a part with nothing missing is measured as it always was",
    );

    let mut holed = Graph::empty();
    let fine = holed.add(Node::Sphere { centre: Place::At([lit(0.0); 3]), radius: lit(10.0) });
    let gap = holed.add(Node::Shell { input: Operand::empty(), thickness: lit(1.0) });
    let _ = holed.add(Node::Boolean {
        op: BooleanOp::Union,
        inputs: vec![fine.into(), gap.into()],
    });
    let why = no_statement_while_a_slot_is_empty(&holed, name, "measure")
        .expect("a hole stops a statement about the part");
    assert!(why.contains("Cannot measure"), "says what it will not do: {why}");
    assert!(why.contains("Shell"), "and names the block to fill in: {why}");
    assert!(
        why.contains("Put a block in it"),
        "and what to do about it, since a refusal somebody cannot act on is a dead end: {why}",
    );
}
