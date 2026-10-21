// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Does a swept solid follow its spine to the end? Asked of the owner's `swept-on-curve.fform`,
//! 2026-10-02: a 500 mm outline of nine curves, made a 6 mm wall and carried along a quadratic
//! that turns through sixty degrees, drew as a fan that stopped well short of the curve's end.
//!
//! The spine's own points are inside the wall's band at every station -- the chain passes
//! through the spine's start, and a centre wall covers its chain -- so the field read *on the
//! spine* should be about minus half the thickness from one end to the other. Where it turns
//! positive is where the solid stops. Printed for each way the section can be held, for the
//! wall and for the same outline as a profile, so the one that stops short names itself.
//!
//! **What it found, 2026-10-02.** Every one of the five stopped, at 693 to 769 mm of the
//! 1050 mm spine, and read +116 mm at the spine's second sample: the section was carried by
//! its plane's origin, the loop's point-centroid, 408 mm from the corner the path was drawn
//! from, and the whole solid ran along a copy of the spine moved by that much. Carried by the
//! point the path sets off from (`sweep_outline_section`), all five read inside from the first
//! interior sample to the last, 1034 mm, with nothing carried; the stretch rose from 2.1 to
//! 3.2, since the outline reaches further from its corner than from its middle.
//!
//! `cargo run --release -p fieldform-io --example sweep_follows_spine -- <file.fform>`

use fieldform_field::{GraphField, ScalarField};
use fieldform_scene::{Node, Outline, ParamRef, Section, WallEnds, WallSide};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: sweep_follows_spine <document.fform>");
    let (doc, _) = fieldform_io::open(std::path::Path::new(&path)).expect("opens");
    let part = doc.parts.values().next().expect("a part");
    let (node, profile, spine) = part
        .graph
        .nodes()
        .find_map(|(id, n)| match n {
            Node::Sweep { outline, spine, .. } => Some((id, outline.curve(), *spine)),
            _ => None,
        })
        .expect("a sweep");
    let path_curve = &doc.curves[&spine];
    let stations: Vec<_> = path_curve.tessellate(0.05);
    let length: f64 = stations.windows(2).map(|w| (w[1] - w[0]).norm()).sum();
    println!(
        "spine {} : {} samples, {:.1} mm, from {:?} to {:?}",
        path_curve.name,
        stations.len(),
        length,
        stations[0].map(|c| (c * 10.0).round() / 10.0),
        stations[stations.len() - 1].map(|c| (c * 10.0).round() / 10.0)
    );

    let wall = Outline::Wall {
        curve: profile,
        thickness: ParamRef::Literal(6.0),
        side: WallSide::Centre,
        ends: WallEnds::Square,
    };
    let cases = [
        ("wall, square", wall.clone(), Section::Square),
        ("wall, flush start", wall.clone(), Section::Skewed),
        ("wall, parallel", wall, Section::Parallel),
        ("profile, square", Outline::closed(profile), Section::Square),
        ("profile, flush start", Outline::closed(profile), Section::Skewed),
    ];
    for (label, outline, section) in cases {
        let mut graph = part.graph.clone();
        *graph.node_mut(node).expect("the sweep") = Node::Sweep { rims: Default::default(), outline, spine, section, law: Default::default() };
        let field = match GraphField::compile_graph(&doc, &graph, 0.5) {
            Ok(f) => f,
            Err(e) => {
                println!("\n{label}: REFUSED -- {e}");
                continue;
            }
        };
        let outline = match graph.node(node) {
            Some(Node::Sweep { outline, .. }) => outline.clone(),
            _ => unreachable!(),
        };
        let stretch = fieldform_field::sweep_stretch(&doc, &outline, spine, &Default::default());
        // Where the solid is carried to: the section's origin less the spine's start, which is
        // what the compile translates the swept leaf by.
        let offset = fieldform_field::sweep_offset(&doc, &outline, spine)
            .unwrap_or_else(nalgebra::Vector3::zeros);
        println!(
            "\n{label}: stretch {stretch:.2}; carried by ({:.1}, {:.1}, {:.1}) -- {:.1} mm",
            offset.x,
            offset.y,
            offset.z,
            offset.norm()
        );
        // Along the spine as drawn, and along the spine carried by the offset: how far along
        // each the field reads inside.
        for (what, shift) in [("the spine as drawn", nalgebra::Vector3::zeros()), ("the spine carried by the offset", offset)] {
            let mut arc = 0.0;
            let mut last_inside = None;
            let mut first_outside = None;
            let mut deepest: f64 = 0.0;
            for (i, p) in stations.iter().enumerate() {
                if i > 0 {
                    arc += (stations[i] - stations[i - 1]).norm();
                }
                let d = field.distance(*p + shift) * stretch;
                deepest = deepest.min(d);
                if d < 0.0 {
                    last_inside = Some((arc, d));
                } else if first_outside.is_none() {
                    first_outside = Some((arc, d));
                }
            }
            match (last_inside, first_outside) {
                (Some((arc, _)), None) => println!(
                    "  along {what}: inside all the way to {arc:.1} of {length:.1} mm; deepest {deepest:.2}"
                ),
                (Some((at, d)), Some((off, e))) => println!(
                    "  along {what}: last inside at {at:.1} mm ({d:.2}), first outside at {off:.1} mm \
                     ({e:.2}) of {length:.1} mm"
                ),
                (None, _) => println!("  along {what}: never inside"),
            }
        }
    }
}
