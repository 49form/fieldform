// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Whether rounding a join darkens the body, which it must not.
//!
//! Reported: "if you turn on fillet the shading goes dark after — so the whole object looks
//! darker". A pattern's fillet at a thousandth of a millimetre is a join nobody can see, so the
//! picture with it should be the picture without it. It was not: a rounded pattern divides its
//! value by W^(1/n) (blend-integrity-spec §3 A) and the shading march scales what it reads by
//! `step_multiplier`, which had not been told — so occlusion read a value 2.45 times too short
//! and called every surface crowded.
//!
//! This lights the same ring three ways and prints how bright the lit pixels are: hard; rounded
//! with the value scale the tracer used to be given; rounded with the one it is given now.
//!
//! `cargo run --release -p fieldform-gpu --example rounded_join_brightness`

use fieldform_gpu::{
    generate, BakeSettings, Camera, EnvironmentMaps, Look, Occlusion, Renderer, ShadingResources,
    Studio,
};
use fieldform_scene::{
    Axis, Continuity, Document, Fillet, Frame, Graph, Material, Node, ParamRef, PatternKind, Place,
};

const WIDTH: u32 = 720;
const HEIGHT: u32 = 480;

fn ring(radius: f64) -> Graph {
    let lit = ParamRef::Literal;
    let mut graph = Graph::empty();
    // Overlapping its neighbours, so the join is a real one and the window sums several copies.
    let body = graph.add(Node::Sphere {
        centre: Place::At([lit(35.0), lit(0.0), lit(0.0)]),
        radius: lit(24.0),
    });
    graph.add(Node::Pattern {
        options: Default::default(),
        input: body.into(),
        count: ParamRef::Literal(6.0),
        kind: PatternKind::Radial { frame: Frame::at_along(Axis::Z, [lit(0.0); 3]) },
        join: true,
        fillet: Fillet { radius: lit(radius), continuity: Continuity::G1 },
    });
    graph
}

/// Mean display value of the pixels that are not background, and how many there are.
fn brightness(doc: &Document, graph: &Graph, value_scale: f64) -> (f64, usize) {
    let program = generate(doc, graph).expect("the ring generates");
    let renderer = pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0))
        .expect("a GPU, which is where the shading runs");
    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
        .collect();
    renderer.set_params(&values);
    let maps = EnvironmentMaps::bake(&Studio::default(), BakeSettings::default());
    let mut shading = ShadingResources::new(renderer.device(), renderer.queue(), &maps)
        .expect("the lighting pipeline builds");
    let (lo, hi) = fieldform_field::part_extent(doc, graph).as_box().expect("finite bounds");
    let mut camera = Camera::framing((lo, hi), f64::from(WIDTH) / f64::from(HEIGHT));
    camera.value_scale = value_scale;
    let look = Look::studio(Material::default(), camera.forward, camera.right, camera.up);
    // Occlusion and the shadow are where the scale is read, so they are on, as in a still.
    camera.occlusion = Some(Occlusion::for_scene(look.lights[0].direction, (hi - lo).norm() * 0.5));
    let budget = (1024.0 * value_scale).ceil() as u32;
    renderer.present(&mut shading, &camera, WIDTH, HEIGHT, budget, &look, 0).expect("presents");
    let image = renderer.read_display(&shading).expect("reads back");
    // The background is one flat colour: whatever the corner is.
    let corner = [image.rgba[0], image.rgba[1], image.rgba[2]];
    let lit: Vec<f64> = image
        .rgba
        .chunks_exact(4)
        .filter(|p| [p[0], p[1], p[2]] != corner)
        .map(|p| (f64::from(p[0]) + f64::from(p[1]) + f64::from(p[2])) / 3.0)
        .collect();
    (lit.iter().sum::<f64>() / lit.len().max(1) as f64, lit.len())
}

fn main() {
    let doc = Document::new();
    let hard = ring(0.0);
    let rounded = ring(0.001);
    let scale = fieldform_field::step_multiplier(&doc, &rounded);
    println!("Six overlapping spheres about Z, 720 × 480, occlusion and shadow on. Mean of the lit pixels, 0–255.\n");
    println!("| join | value scale given to the tracer | lit pixels | mean brightness |");
    println!("|---|---|---|---|");
    let (base, n) = brightness(&doc, &hard, fieldform_field::step_multiplier(&doc, &hard));
    println!("| hard | 1.00 | {n} | **{base:.1}** |");
    let (was, n) = brightness(&doc, &rounded, 1.0);
    println!("| rounded at 0.001 mm | 1.00 — what it used to be given | {n} | **{was:.1}** |");
    let (now, n) = brightness(&doc, &rounded, scale);
    println!("| rounded at 0.001 mm | {scale:.2} — `step_multiplier` | {n} | **{now:.1}** |");
    println!(
        "\nUntold, the rounded ring is {:.0} % as bright as the hard one; told, {:.0} %.",
        100.0 * was / base,
        100.0 * now / base
    );
    assert!(
        (now - base).abs() <= 0.02 * base,
        "a join nobody can see changed the picture: {now:.1} against {base:.1}"
    );
}
