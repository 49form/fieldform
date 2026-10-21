// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Renders the reference sphere sweep `shading-spec.md` §12 asks for, and the preset library.
//!
//! Six metallic values across, six roughness values down, one sphere each under the studio
//! environment — the image to hold beside the same grid from any glTF-conformant renderer. Not
//! a pixel test; a "does our GGX look like everyone else's GGX" test, which catches a wrong
//! Fresnel or a missing `1/π` in minutes where finding it later takes days.
//!
//! A last row shows every preset, so the library can be judged as a set.
//!
//! Writes `target/material_sweep.png` and prints how long the environment took to bake and each
//! frame took to present. Run with:
//! `cargo run --release -p fieldform-gpu --example material_sweep`

use fieldform_gpu::{
    generate, BakeSettings, Camera, EnvironmentMaps, Look, Occlusion, Renderer, ShadingResources,
    Studio,
};
use fieldform_scene::{Document, Graph, Material, Node, ParamRef};
use nalgebra::Point3;

const TILE: u32 = 160;
const RADIUS_MM: f64 = 40.0;

fn main() {
    let doc = Document::new();
    let mut graph = Graph::empty();
    graph.add(Node::Sphere {
        centre: fieldform_scene::Place::At([ParamRef::Literal(0.0), ParamRef::Literal(0.0), ParamRef::Literal(0.0)]),
        radius: ParamRef::Literal(RADIUS_MM),
    });
    let program = generate(&doc, &graph).expect("a sphere generates");

    let renderer = match pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("no GPU available ({e}); nothing else in the workspace needs one.");
            return;
        }
    };
    renderer.set_params(&[0.0, 0.0, 0.0, RADIUS_MM as f32]);

    let started = std::time::Instant::now();
    let maps = EnvironmentMaps::bake(&Studio::default(), BakeSettings::default());
    println!(
        "environment baked in {:.1} ms ({} specular levels from {}², irradiance {}², LUT {}²)",
        started.elapsed().as_secs_f64() * 1e3,
        maps.specular.len(),
        maps.specular[0].size,
        maps.irradiance.size,
        maps.lut_size,
    );
    let mut shading = ShadingResources::new(renderer.device(), renderer.queue(), &maps)
        .expect("the lighting pipeline builds");

    let bounds = (
        Point3::new(-RADIUS_MM, -RADIUS_MM, -RADIUS_MM),
        Point3::new(RADIUS_MM, RADIUS_MM, RADIUS_MM),
    );
    let mut camera = Camera::framing(bounds, 1.0);
    let base = Look::studio(Material::default(), camera.forward, camera.right, camera.up);
    camera.occlusion = Some(Occlusion::for_scene(base.lights[0].direction, RADIUS_MM * 1.7));

    let sweep = 6u32;
    let presets = Material::presets();
    let columns = sweep.max(presets.len() as u32);
    let rows = sweep + 1;
    let mut sheet = image::RgbaImage::from_pixel(columns * TILE, rows * TILE, image::Rgba([24, 26, 30, 255]));
    let mut frame_ms = Vec::new();

    let mut tile = |shading: &mut ShadingResources, col: u32, row: u32, material: Material| {
        let look = Look { material, ..base.clone() };
        let started = std::time::Instant::now();
        renderer
            .present(shading, &camera, TILE, TILE, 512, &look, 0)
            .expect("the frame presents");
        let image = renderer.read_display(shading).expect("the frame reads back");
        frame_ms.push(started.elapsed().as_secs_f64() * 1e3);
        for y in 0..TILE {
            for x in 0..TILE {
                let i = ((y * TILE + x) * 4) as usize;
                let px = image::Rgba([image.rgba[i], image.rgba[i + 1], image.rgba[i + 2], 255]);
                sheet.put_pixel(col * TILE + x, row * TILE + y, px);
            }
        }
    };

    // Rows: roughness 0 → 1. Columns: metallic 0 → 1. A warm base so the metal's tint shows.
    //
    // The paint kind, because it is the one kind whose metallic is free to vary — a plastic's is
    // held at zero and a metal's at one — and with no coat and no flake it is exactly the plain
    // glTF metallic-roughness model this sweep is a sweep of.
    for row in 0..sweep {
        for col in 0..sweep {
            let material = Material {
                kind: fieldform_scene::Kind::Paint,
                base_color: [0.62, 0.09, 0.05],
                metallic: col as f64 / (sweep - 1) as f64,
                roughness: row as f64 / (sweep - 1) as f64,
                clearcoat: 0.0,
                clearcoat_roughness: 0.0,
                anisotropy: 0.0,
                flake: 0.0,
            };
            tile(&mut shading, col, row, material);
        }
    }
    for (col, preset) in presets.iter().enumerate() {
        tile(&mut shading, col as u32, sweep, Material::from(preset.material));
    }

    std::fs::create_dir_all("target").ok();
    let path = "target/material_sweep.png";
    sheet.save(path).expect("the sheet saves");
    let mean = frame_ms.iter().sum::<f64>() / frame_ms.len() as f64;
    println!(
        "{} tiles of {TILE}² presented and read back in {mean:.2} ms each (the readback is most \
         of it); wrote {path}",
        frame_ms.len()
    );
    println!(
        "last row, left to right: {}",
        presets.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")
    );
}
