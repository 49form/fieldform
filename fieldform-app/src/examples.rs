// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! One worked console line per common verb (`mcp-server-spec.md` §5.3).
//!
//! **Each is run, not trusted.** `app_tests::every_example_runs_on_a_new_document` plays the lines
//! in order on an empty document and fails on a refusal, so an example cannot outlive a key it
//! uses. A line may need blocks made by the ones before it, so an example is a short script and
//! the last line is the one shown. Verbs not listed have no example yet: reported as none,
//! never invented.

/// `(verb, lines)`: the lines are newline-separated; the last is the example proper.
pub const EXAMPLES: &[(&str, &str)] = &[
    ("sphere", "sphere radius=10 'ball'"),
    ("expression", "expression text=\"sqrt(x*x + y*y + z*z) - 10\" size=30,30,30 claim=distance 'ball'"),
    ("cube", "cube size=80,40,10 'plate'"),
    ("cylinder", "cylinder radius=4 'rod'"),
    ("cone", "cone base_radius=20 top_radius=5 'taper'"),
    ("torus", "torus major_radius=30 minor_radius=5 'ring'"),
    ("capsule", "capsule start_radius=5 end_radius=5 'pill'"),
    ("point", "point at=10,0,0"),
    ("plane", "plane origin=0,0,10"),
    ("param", "param name=wall value=3"),
    ("move", "sphere radius=10 'ball'\nmove target=ball by=40,0,0"),
    ("turn", "cube size=80,40,10 'plate'\nturn target=plate by=0,0,45"),
    ("mirror", "sphere radius=10 'ball'\nmirror target=ball"),
    ("shell", "sphere radius=20 'ball'\nshell target=ball thickness=2"),
    ("offset", "sphere radius=20 'ball'\noffset target=ball distance=2"),
    ("pattern", "sphere radius=5 'bead'\npattern target=bead kind=linear count=4 step=20,0,0"),
    ("boolean", "sphere radius=10 'a'\nsphere radius=10 'b'\nboolean in=a,b -union"),
    ("continuity", "line p0=0,0,0 p1=40,0,0 'a'\ncurve p=40,0,0;50,10,0;60,10,0;70,0,0 'b'\ncontinuity at=a.end g=1"),
];

pub fn example_of(verb: &str) -> Option<&'static str> {
    EXAMPLES.iter().find(|(v, _)| *v == verb).and_then(|(_, lines)| lines.lines().last())
}
