// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Renders the reported scene at settle quality and saves it, so the shading can be looked at.
//!
//! **Because "the shadows look wrong" is not a number.** The surface's position is measured by
//! `gpu_matches_cpu` and its normal by `rounded_fold_normal`, and both are exact — so whatever is
//! left is in the lighting, and the only way to see lighting is to look at it. This renders the
//! same ring the report was made against and writes it four ways: with and without ambient
//! occlusion, with and without the key light's shadow. Which tile the black appears in is what
//! names the term responsible, and it named ambient occlusion.
//!
//! Run with: `cargo run --release -p fieldform-gpu --example rounded_fold_look`

use fieldform_gpu::{
    generate, BakeSettings, Camera, EnvironmentMaps, Look, Occlusion, Renderer, ShadingResources,
    Studio,
};
use fieldform_scene::{
    Axis, Document, Fillet, Frame, Graph, Material, Node, ParamRef, PatternKind, Place,
};
use nalgebra::Point3;

const BALL_MM: f64 = 50.0;
const RING_MM: f64 = 95.0;
const COUNT: u32 = 10;
const TILE: u32 = 420;

fn main() {
    let lit = ParamRef::Literal;
    let doc = Document::new();

    let maps = EnvironmentMaps::bake(&Studio::default(), BakeSettings::default());
    let reach = RING_MM + BALL_MM;
    let bounds = (
        Point3::new(-reach, -reach, -BALL_MM * 1.2),
        Point3::new(reach * 2.0, reach, BALL_MM * 1.2),
    );

    let mut sheet = image::RgbaImage::from_pixel(
        TILE * 2,
        TILE * 2,
        image::Rgba([24, 26, 30, 255]),
    );

    // Four fillet radii, to see where a crease in the middle of a fillet comes in.
    let radii = [0.0f64, 12.0, 26.0, 40.0];
    for (tile, radius) in radii.into_iter().enumerate() {
        let (col, row) = (tile % 2, tile / 2);
        let mut graph = Graph::empty();
        let body = graph.add(Node::Sphere {
            centre: Place::At([lit(0.0); 3]),
            radius: lit(BALL_MM),
        });
        graph.add(Node::Pattern {
            options: Default::default(),
            input: body.into(),
            count: ParamRef::Literal(f64::from(COUNT)),
            kind: PatternKind::Radial {
                frame: Frame::at_along(Axis::Z, [lit(RING_MM), lit(0.0), lit(0.0)]),
            },
            join: true,
            fillet: if radius > 0.0 { Fillet::of(lit(radius)) } else { Fillet::hard() },
        });
        let program = generate(&doc, &graph).expect("generates");
        let values: Vec<f32> = program
            .params
            .iter()
            .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
            .collect();

        let renderer = match pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("no GPU available ({e})");
                return;
            }
        };
        renderer.set_params(&values);
        let mut shading = ShadingResources::new(renderer.device(), renderer.queue(), &maps)
            .expect("the lighting pipeline builds");

        let mut camera = Camera::framing(bounds, 1.0);
        let look = Look::studio(Material::default(), camera.forward, camera.right, camera.up);
        camera.occlusion = Some(Occlusion::for_scene(look.lights[0].direction, reach * 1.7));

        for frame in 0..8u32 {
            renderer
                .present(&mut shading, &camera, TILE, TILE, 512, &look, frame)
                .expect("the frame presents");
        }
        let image = renderer.read_display(&shading).expect("reads back");
        for y in 0..TILE {
            for x in 0..TILE {
                let i = ((y * TILE + x) * 4) as usize;
                sheet.put_pixel(
                    col as u32 * TILE + x,
                    row as u32 * TILE + y,
                    image::Rgba([image.rgba[i], image.rgba[i + 1], image.rgba[i + 2], 255]),
                );
            }
        }
    }

    std::fs::create_dir_all("target").ok();
    let path = "target/rounded_fold_look.png";
    sheet.save(path).expect("the sheet saves");
    println!("wrote {path}");
    println!("  fillet radii {radii:?}, reading left to right then down");
}
