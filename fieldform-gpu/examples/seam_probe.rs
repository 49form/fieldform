// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Where the surface is on either side of a seam between bodies, read off the CPU field — whether
//! a step drawn at a seam is in the shape or only in the picture.
//!
//! For each body named, and for the union of all of them, finds the surface along lines through
//! the seam: down onto the top at x = 0, and in onto the side at z = `side_z`, at every y in a
//! range. A step in the shape is a jump in those positions across the seam; none means the field
//! is continuous there and a step on screen is the drawing's.
//!
//! `cargo run --release -p fieldform-gpu --example seam_probe -- <doc.fform> <y0> <y1> <dy> <side_z> <node>...`
//! prints the divisor each sweep's reading carries, and with `SEAM_Y` (and `SEAM_TOP`) what each body
//! reads a little way past the seam.
//!
//! On the owner's `swweeeeeping.fform` (backlog item 114): the top at z = 12.3214 and the side at
//! x = 14.3882 on both sides of the seam, so the shape is level across it; the forward sweep divided
//! by 82.9 and the backward by 1.18, reading 0.024 two millimetres past its start where the truth is 2.

use fieldform_field::ScalarField;
use fieldform_scene::NodeId;
use nalgebra::Point3;

/// The first crossing from outside to inside walking from `from` toward `to`, found by stepping
/// and then bisecting — the position of the surface along that line, to a micron.
fn crossing(field: &dyn ScalarField, from: Point3<f64>, to: Point3<f64>) -> Option<f64> {
    let steps = 4000;
    let mut prev = field.distance(from);
    for i in 1..=steps {
        let t = i as f64 / steps as f64;
        let here = field.distance(from + (to - from) * t);
        if prev > 0.0 && here <= 0.0 {
            let (mut lo, mut hi) = ((i - 1) as f64 / steps as f64, t);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if field.distance(from + (to - from) * mid) > 0.0 { lo = mid } else { hi = mid }
            }
            return Some((to - from).norm() * 0.5 * (lo + hi));
        }
        prev = here;
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let (y0, y1, dy, side_z): (f64, f64, f64, f64) =
        (args[2].parse().unwrap(), args[3].parse().unwrap(), args[4].parse().unwrap(), args[5].parse().unwrap());
    let wanted: Vec<NodeId> = args[6..].iter().map(|s| NodeId::from_raw(u64::from_str_radix(s, 16).unwrap())).collect();
    let (doc, _archive) = fieldform_io::open(std::path::Path::new(path)).expect("the document opens");
    let (_, part) = doc.parts.iter().next().expect("a part");
    let only = |keep: &[NodeId]| {
        let mut g = part.graph.clone();
        let bodies: Vec<NodeId> = g.nodes().filter(|(id, n)| n.is_body() && !keep.contains(id)).map(|(id, _)| id).collect();
        for id in bodies {
            if !g.nodes().any(|(_, n)| n.inputs().contains(&id)) {
                g.remove(id);
            }
        }
        // Whatever fed only the bodies removed is left; it is not a body, or it would have gone.
        let mut changed = true;
        while changed {
            changed = false;
            let roots: Vec<NodeId> = g
                .nodes()
                .filter(|(id, n)| n.is_body() && !keep.contains(id) && !g.nodes().any(|(_, m)| m.inputs().contains(id)))
                .map(|(id, _)| id)
                .collect();
            for id in roots {
                g.remove(id);
                changed = true;
            }
        }
        fieldform_field::GraphField::compile_graph(&doc, &g, 1e-4).expect("compiles")
    };
    for id in &wanted {
        if let Some(fieldform_scene::Node::Sweep { outline, spine, law, .. }) = part.graph.node(*id) {
            println!("{:x}: divided by {:.3}", id.raw(), fieldform_field::sweep_stretch(&doc, outline, *spine, law));
        }
    }
    println!("whole graph's value scale: {:.3}", fieldform_field::value_scale_with(&doc, &part.graph, &[]));
    let mut fields: Vec<(String, fieldform_field::GraphField)> =
        wanted.iter().map(|id| (format!("{:x}", id.raw()), only(&[*id]))).collect();
    fields.push(("union".to_string(), only(&wanted)));
    println!("{:>8} | {}", "y", fields.iter().map(|(n, _)| format!("{:>20}", format!("{n} top/side"))).collect::<Vec<_>>().join(" "));
    // What each body reads a little way past the seam, on the line where the other body's surface
    // carries on: the true distance there is the distance to this body's own start rim.
    if let Ok(seam) = std::env::var("SEAM_Y").map(|v| v.parse::<f64>().unwrap()) {
        let top = std::env::var("SEAM_TOP").map_or(12.3214, |v| v.parse::<f64>().unwrap());
        for (name, f) in &fields {
            println!("{name}: normalization {:?}, lipschitz {}", f.normalization(), f.lipschitz());
            for delta in [-2.0, -0.5, -0.1, -0.01, 0.01, 0.1, 0.5, 2.0] {
                let q = Point3::new(0.0, seam + delta, top);
                println!("    y = seam {delta:+.2}: reads {:+.5}", f.distance(q));
            }
        }
    }
    let mut y = y0;
    while y <= y1 + 1e-9 {
        let row: Vec<String> = fields
            .iter()
            .map(|(_, f)| {
                let top = crossing(f, Point3::new(0.0, y, 40.0), Point3::new(0.0, y, -40.0)).map(|d| 40.0 - d);
                let side = crossing(f, Point3::new(60.0, y, side_z), Point3::new(0.0, y, side_z)).map(|d| 60.0 - d);
                let show = |v: Option<f64>| v.map_or("    --  ".to_string(), |v| format!("{v:8.4}"));
                format!("{:>9} {:>9}  ", show(top), show(side))
            })
            .collect();
        println!("{y:8.3} | {}", row.join(" "));
        y += dy;
    }
}
