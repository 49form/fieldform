// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! How many bodies a part can hold before something breaks.
//!
//! An imported STEP assembly is one body per solid, so "many bodies" stopped being a
//! hypothetical the moment splitting landed. Both the CPU fold and the shader fold roots into
//! nested binary unions, and this measures what that costs.

use fieldform_field::GraphField;
use fieldform_gpu::codegen;
use fieldform_scene::{Document, Graph, Node, ParamRef, Part};
use nalgebra::Point3;

fn part_of(n: usize) -> (Document, Part) {
    let doc = Document::new();
    let mut graph = Graph::empty();
    for i in 0..n {
        let x = i as f64 * 3.0;
        graph.add(Node::Sphere {
            centre: fieldform_scene::Place::At([ParamRef::Literal(x), ParamRef::Literal(0.0), ParamRef::Literal(0.0)]),
            radius: ParamRef::Literal(1.0),
        });
    }
    let mut part = Part::from_cage("many", fieldform_scene::CageId::new());
    part.graph = graph;
    (doc, part)
}

fn main() {
    for n in [2, 10, 100, 500, 1000, 2000, 5000, 10000, 50000] {
        let (doc, part) = part_of(n);
        let started = std::time::Instant::now();
        let compiled = GraphField::compile(&doc, &part, 0.05);
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        // The shader half, which has its own limit: a nested expression per body is a parse
        // tree as deep as the part is wide, and the normal pass's was quadratic in length too.
        let started = std::time::Instant::now();
        let shader = codegen::generate(&doc, &part.graph);
        let gen_ms = started.elapsed().as_secs_f64() * 1000.0;
        match &shader {
            Ok(program) => println!(
                "{n:>6} bodies: shader {gen_ms:>7.1} ms, {:>9} chars of WGSL",
                program.source.len(),
            ),
            Err(e) => println!("{n:>6} bodies: shader refused — {e}"),
        }

        match compiled {
            Ok(field) => {
                use fieldform_field::ScalarField;
                let started = std::time::Instant::now();
                let mut acc = 0.0;
                for i in 0..1000 {
                    acc += field.distance(Point3::new(i as f64 * 0.01, 0.5, 0.5));
                }
                let eval_us = started.elapsed().as_secs_f64() * 1e6 / 1000.0;
                println!(
                    "{n:>6} bodies: compile {compile_ms:>8.1} ms, {eval_us:>9.2} us/sample  ({acc:.1})"
                );
            }
            Err(e) => println!("{n:>6} bodies: refused — {e}"),
        }
    }
}
