// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The Expression block through a real `App` without a window (backlog item 40,
//! `expression-field-spec.md` §12): the sphere case end to end, the one property most likely
//! to regress silently -- a parameter's drag writes four bytes and recompiles nothing, while an
//! edit to the text recompiles -- and what the verb refuses.

use super::*;
use crate::app_tests::{frame, headless_app};

/// Types `line` and runs a frame, answering what the console said after it.
fn run(app: &mut App, line: &str) -> Vec<(bool, String)> {
    let before = app.console_log.len();
    app.run_command(line);
    frame(app, Vec::new());
    app.console_log.iter().skip(before).map(|l| (l.ok, l.text.clone())).collect()
}

/// Frames until the traced view has nothing stale and nothing compiling, or a bounded wait
/// runs out, and answers whether a shader was seen compiling on the way. The compile is what
/// a change to the shader's structure costs, and it is observed rather than counted: the
/// application drains the rebuild costs itself each frame for its perf line.
fn settle(app: &mut App) -> bool {
    let started = std::time::Instant::now();
    let mut compiled = false;
    loop {
        frame(app, Vec::new());
        compiled |= app.traced.is_compiling().is_some();
        if !app.traced.is_stale() && app.traced.is_compiling().is_none() {
            frame(app, Vec::new());
            return compiled;
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(120), "the view never settled");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

/// Whether the last `expect` passed, and what it said.
fn last_expect(app: &App) -> (bool, String) {
    let record = app.expect_results.last().expect("an expect ran");
    (record.verdict == expect::Verdict::Pass, record.said.clone())
}

#[test]
fn a_written_sphere_measures_its_volume_and_is_remade_by_its_parameter_without_a_recompile() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    let said = run(&mut app, "param name=r value=5");
    assert!(said.iter().all(|(ok, _)| *ok), "{said:?}");
    let said = run(&mut app, "expression text=\"sqrt(x*x + y*y + z*z) - r\" size=30,30,30 claim=distance 'ball'");
    assert!(said.iter().all(|(ok, _)| *ok), "{said:?}");
    let ball = app.part_graph().nodes().find_map(|(id, n)| matches!(n, Node::Expression { .. }).then_some(id)).expect("the block is in the graph");
    assert!(app.part_graph().roots().contains(&ball), "a written field is a body");
    run(&mut app, "expect target=ball volume=523.60 tol=0.5%");
    let (pass, said) = last_expect(&app);
    assert!(pass, "{said}");

    // The §7 property: a parameter it reads is a slot. Settle, change the number, settle
    // again, and no shader was compiled -- while the volume follows.
    settle(&mut app);
    let said = run(&mut app, "param name=r value=6");
    assert!(said.iter().all(|(ok, _)| *ok), "{said:?}");
    assert!(!settle(&mut app), "a parameter's new value recompiled the shader");
    run(&mut app, "expect target=ball volume=904.78 tol=0.5%");
    let (pass, said) = last_expect(&app);
    assert!(pass, "{said}");

    // And the text is structure: editing it recompiles, and the new picture is the new formula.
    let said = run(&mut app, "set ball.text=\"max(abs(x) - 4, max(abs(y) - 4, abs(z) - 4))\"");
    assert!(said.iter().all(|(ok, _)| *ok), "{said:?}");
    assert!(settle(&mut app), "the text changed and nothing recompiled");
    run(&mut app, "expect target=ball volume=512.0 tol=0.5%");
    let (pass, said) = last_expect(&app);
    assert!(pass, "{said}");
}

#[test]
fn a_text_that_does_not_parse_is_refused_at_its_column_and_the_document_is_unchanged() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    let before = app.part_graph().nodes().count();
    let said = run(&mut app, "expression text=\"2 * pi * x\" 'bad'");
    assert!(said.iter().any(|(ok, t)| !ok && t.contains("degrees") && t.contains("column 5")), "{said:?}");
    assert_eq!(app.part_graph().nodes().count(), before, "nothing was made");
    let said = run(&mut app, "expression text=\"x - wall\" 'bad'");
    assert!(said.iter().any(|(ok, t)| !ok && t.contains("`wall` is not a parameter")), "{said:?}");
    let said = run(&mut app, "expression claim=maybe 'bad'");
    assert!(said.iter().any(|(ok, t)| !ok && t.contains("claim=")), "{said:?}");
    assert_eq!(app.part_graph().nodes().count(), before);
    // The bare verb makes a written sphere, selected, with the keys its own fields.
    let said = run(&mut app, "expression");
    assert!(said.iter().any(|(ok, t)| *ok && t.contains("Added a written field")), "{said:?}");
    assert_eq!(app.part_graph().nodes().count(), before + 1);
}

#[test]
fn verify_measures_a_written_sphere_as_exact_and_a_raw_gyroid_as_overstating() {
    let Some(mut app) = headless_app() else { return };
    frame(&mut app, Vec::new());
    run(&mut app, "expression text=\"sqrt(x*x + y*y + z*z) - 10\" size=30,30,30 claim=distance 'ball'");
    let ball = app.part_graph().nodes().find_map(|(id, n)| matches!(n, Node::Expression { .. }).then_some(id)).unwrap();
    app.verify_written(ball);
    let v = app.written_verdicts.get(&ball).cloned().expect("verified");
    assert!(v.overstatement < 1e-6, "a written sphere never overstates: {v:?}");
    assert!(v.deviation < 1e-3, "a written sphere is a distance: {v:?}");
    assert_eq!(v.missing, 0);
    assert!(v.suggested.is_none());

    run(&mut app, "expression text=\"sin(x * 36) * cos(y * 36) + sin(y * 36) * cos(z * 36) + sin(z * 36) * cos(x * 36) + 0.3\" size=30,30,30 'raw'");
    let raw = app.part_graph().nodes().filter_map(|(id, n)| matches!(n, Node::Expression { .. }).then_some(id)).find(|id| *id != ball).unwrap();
    app.verify_written(raw);
    let v = app.written_verdicts.get(&raw).cloned().expect("verified");
    assert!(v.overstatement > 1e-3, "a raw gyroid at k = 1 overstates: {v:?}");
    assert!(v.suggested.is_some_and(|k| k > 1.0), "and a divisor is suggested: {v:?}");
}
