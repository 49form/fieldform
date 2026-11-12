// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The traced-reflection acceptance scenes (backlog item 54, the shading notes §15.11), as
//! stills written to a folder -- and, with `--compare <other folder>`, held byte for byte
//! against an earlier run.
//!
//! Two uses. **The pure-refactor gate:** run it before and after the march loop's extraction,
//! with reflections off, and every picture must be the same bytes. **The acceptance:** run it
//! with `--reflections` against the run without, and read the table: a convex body is
//! bit-identical with reflections on and off and its reflected rays all miss, two spheres
//! each reflect the other, a grazing chrome plate hits nothing of itself, the lattice's rays
//! that ran out of budget are counted, and a matte part dispatches nothing.
//!
//! ```text
//! cargo run -p fieldform-gpu --example reflection_scenes -- <out dir> [--reflections] [--compare <dir>]
//! ```

use fieldform_gpu::{
    generate, halton_jitter, Camera, EnvironmentChoice, Look, Occlusion, Reflections, Renderer,
    Sections, ShadingResources,
};
use fieldform_scene::{
    BooleanOp, CellMap, Continuity, Document, Fillet, Frame, Graph, LatticeCell, LatticePhase,
    Material, Node, ParamRef,
};
use nalgebra::{Point3, Vector3};
use std::path::{Path, PathBuf};

const WIDTH: u32 = 480;
const HEIGHT: u32 = 320;
const FRAMES: u32 = 8;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn at(x: f64, y: f64, z: f64) -> fieldform_scene::Place {
    fieldform_scene::Place::At([lit(x), lit(y), lit(z)])
}

fn sphere(graph: &mut Graph, centre: (f64, f64, f64), radius: f64) -> fieldform_scene::NodeId {
    graph.add(Node::Sphere { centre: at(centre.0, centre.1, centre.2), radius: lit(radius) })
}

fn cuboid(graph: &mut Graph, centre: (f64, f64, f64), size: (f64, f64, f64), edge: f64) -> fieldform_scene::NodeId {
    graph.add(Node::Cuboid {
        centre: at(centre.0, centre.1, centre.2),
        size: [lit(size.0), lit(size.1), lit(size.2)],
        turn: fieldform_scene::no_turn(),
        edge: if edge > 0.0 {
            Fillet { radius: lit(edge), continuity: Continuity::G1, second: None }
        } else {
            Fillet::hard()
        },
    })
}

/// A scene: its graph, the material it is drawn in, where the camera stands, and a section.
struct Scene {
    name: &'static str,
    graph: Graph,
    material: Material,
    bounds: (Point3<f64>, Point3<f64>),
    /// `None` frames the bounds; `Some` looks from here at the origin.
    eye: Option<Point3<f64>>,
    sections: Vec<(Vector3<f64>, f64)>,
}

fn chrome() -> Material {
    Material::preset("Chrome").expect("the chrome preset")
}

fn scenes() -> Vec<Scene> {
    let box_of = |h: f64| (Point3::new(-h, -h, -h), Point3::new(h, h, h));
    let mut out = Vec::new();

    // A sphere alone, then a rounded box alone: each convex, so a reflected ray never hits
    // anything, and the picture with reflections on is the picture with them off. One body a
    // scene, because two convex bodies are not a convex scene -- the first version put the two
    // side by side and the box's rounded edge reflected the sphere, 152 pixels of it.
    let mut g = Graph::empty();
    sphere(&mut g, (0.0, 0.0, 0.0), 18.0);
    out.push(Scene { name: "sphere", graph: g, material: chrome(), bounds: box_of(20.0), eye: None, sections: Vec::new() });
    let mut g = Graph::empty();
    cuboid(&mut g, (0.0, 0.0, 0.0), (30.0, 30.0, 30.0), 6.0);
    out.push(Scene { name: "box", graph: g, material: chrome(), bounds: box_of(18.0), eye: None, sections: Vec::new() });

    // Two chrome spheres facing each other: each shows the other once.
    let mut g = Graph::empty();
    sphere(&mut g, (-30.0, 0.0, 0.0), 20.0);
    sphere(&mut g, (30.0, 0.0, 0.0), 20.0);
    out.push(Scene { name: "mutual", graph: g, material: chrome(), bounds: box_of(55.0), eye: None, sections: Vec::new() });

    // A gyroid cube in chrome, reflecting itself.
    let mut g = Graph::empty();
    let block = cuboid(&mut g, (0.0, 0.0, 0.0), (60.0, 60.0, 60.0), 0.0);
    let lattice = g.add(Node::Lattice {
        cell: LatticeCell::Gyroid,
        frame: Frame::default(),
        map: CellMap::Rectangular { cell: lit(20.0) },
        phase: LatticePhase::Sheet { thickness: lit(2.0), offset: lit(0.0) },
    });
    g.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![block.into(), lattice.into()] });
    out.push(Scene { name: "lattice", graph: g, material: chrome(), bounds: box_of(32.0), eye: None, sections: Vec::new() });

    // A flat chrome plate seen at a grazing angle.
    let mut g = Graph::empty();
    cuboid(&mut g, (0.0, 0.0, 0.0), (120.0, 80.0, 2.0), 0.0);
    out.push(Scene {
        name: "plate",
        graph: g,
        material: chrome(),
        bounds: (Point3::new(-60.0, -40.0, -1.0), Point3::new(60.0, 40.0, 1.0)),
        eye: Some(Point3::new(0.0, -140.0, 9.0)),
        sections: Vec::new(),
    });

    // The two spheres again, the right one cut in half by a section through its middle: what the
    // left one reflects is the kept half and its flat cap, never the half the section removed.
    // `a_reflected_ray_finds_the_sphere_the_arithmetic_says` holds every reflected hit to the
    // kept side; here the picture is the record.
    let mut g = Graph::empty();
    sphere(&mut g, (-30.0, 0.0, 0.0), 20.0);
    sphere(&mut g, (30.0, 0.0, 0.0), 20.0);
    out.push(Scene {
        name: "section",
        graph: g,
        material: chrome(),
        bounds: box_of(55.0),
        eye: None,
        sections: vec![(Vector3::x(), 30.0)],
    });

    // The two spheres in a glossy finish, roughness a fifth: the reflected direction is a sample
    // of the lobe, seeded by pixel and frame, and the eight frames integrate it.
    let mut g = Graph::empty();
    sphere(&mut g, (-30.0, 0.0, 0.0), 20.0);
    sphere(&mut g, (30.0, 0.0, 0.0), 20.0);
    let glossy = Material { roughness: 0.2, ..chrome() };
    out.push(Scene { name: "glossy", graph: g, material: glossy, bounds: box_of(55.0), eye: None, sections: Vec::new() });

    // Car paint on a concave form: the coat reflects the neighbouring wall. The wall stands on
    // the far side of the floor from the camera (`Camera::framing` looks from +x, +y, +z), so the
    // floor's mirror directions run into it.
    let mut g = Graph::empty();
    let floor = cuboid(&mut g, (0.0, 0.0, -20.0), (80.0, 80.0, 10.0), 2.0);
    let wall = cuboid(&mut g, (0.0, -35.0, 10.0), (80.0, 10.0, 60.0), 2.0);
    g.add(Node::Boolean { op: BooleanOp::Union, inputs: vec![floor.into(), wall.into()] });
    out.push(Scene {
        name: "paint",
        graph: g,
        material: Material::preset("Car paint").expect("the car-paint preset"),
        bounds: box_of(45.0),
        eye: None,
        sections: Vec::new(),
    });

    // A matte clay sphere beside a wall: nothing is dispatched for it.
    let mut g = Graph::empty();
    sphere(&mut g, (-20.0, 0.0, 0.0), 18.0);
    cuboid(&mut g, (25.0, 0.0, 0.0), (10.0, 50.0, 50.0), 0.0);
    out.push(Scene { name: "matte", graph: g, material: Material::default(), bounds: box_of(40.0), eye: None, sections: Vec::new() });
    out
}

/// One still of `scene`, `FRAMES` jittered frames averaged, with the still tier's occlusion
/// and -- when asked -- reflections.
fn still(scene: &Scene, reflections: bool) -> Result<Still, String> {
    let doc = Document::new();
    let program = generate(&doc, &scene.graph).map_err(|e| format!("{}: {e}", scene.name))?;
    let renderer = pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0)).map_err(|e| e.to_string())?;
    let values: Vec<f32> = program
        .params
        .iter()
        .map(|p| match p {
            ParamRef::Literal(v) => *v as f32,
            _ => unreachable!("literals only"),
        })
        .collect();
    renderer.set_params(&values);
    let aspect = f64::from(WIDTH) / f64::from(HEIGHT);
    let mut camera = match scene.eye {
        None => Camera::framing(scene.bounds, aspect),
        Some(eye) => {
            let centre = nalgebra::center(&scene.bounds.0, &scene.bounds.1);
            let radius = (scene.bounds.1 - scene.bounds.0).norm() * 0.5;
            Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), 45.0, aspect)
        }
    };
    camera.sections = Sections::of(&scene.sections);
    let maps = EnvironmentChoice::Studio.bake();
    let mut shading = ShadingResources::new(renderer.device(), renderer.queue(), &maps).map_err(|e| e.to_string())?;
    let look = Look::for_environment(EnvironmentChoice::Studio, scene.material.clone(), camera.forward, camera.right, camera.up);
    camera.occlusion = Some(Occlusion::for_scene(look.lights[0].direction, camera.scene_radius));
    // The per-frame decision the viewport makes, from the material alone: a matte part traces
    // nothing, and the table says so.
    camera.reflections = if reflections { Reflections::for_material(&scene.material, FRAMES) } else { None };
    let traced = camera.reflections.is_some();
    for frame in 0..FRAMES {
        camera.jitter = halton_jitter(frame);
        renderer
            .present(&mut shading, &camera, WIDTH, HEIGHT, 512, &look, frame)
            .map_err(|e| e.to_string())?;
    }
    renderer.device().poll(wgpu::PollType::wait_indefinitely()).map_err(|_| "the device did not finish".to_string())?;
    let image = renderer.read_display(&shading).map_err(|e| e.to_string())?;
    // What the last frame's reflected rays found, where any were traced: hits, misses, and the
    // rays that ran out of budget -- the count the budget is judged by.
    let rays = match renderer.read_reflections(&shading).map_err(|e| e.to_string())? {
        Some(found) if traced => {
            let mut rays = Rays::default();
            for word in found {
                if word[3] >= 0.0 {
                    rays.hit += 1;
                } else if word[3] == -1.0 {
                    rays.missed += 1;
                } else if word[3] == -2.0 {
                    rays.exhausted += 1;
                }
            }
            Some(rays)
        }
        _ => None,
    };
    // What the reflected march costs this scene: one still frame's march on the card's own
    // clock, with the request and without, medians of five after one thrown away. The lighting
    // pass is left out of the number, since the reflected hit's second `shade_surface` is its
    // own small cost and the march is the one the budget is about.
    let march_ms = |camera: &Camera, shading: &mut ShadingResources| -> Option<f64> {
        let mut costs = Vec::new();
        for i in 0..6 {
            let ticket = renderer.present(shading, camera, WIDTH, HEIGHT, 512, &look, 0).ok()?;
            renderer.device().poll(wgpu::PollType::wait_indefinitely()).ok()?;
            for (done, cost) in shading.costs() {
                if done == ticket && i > 0 {
                    costs.push(f64::from(cost.march_ms?));
                }
            }
        }
        costs.sort_by(f64::total_cmp);
        costs.get(costs.len() / 2).copied()
    };
    let (with, without) = if traced {
        let with = march_ms(&camera, &mut shading);
        camera.reflections = None;
        let without = march_ms(&camera, &mut shading);
        (with, without)
    } else {
        (None, None)
    };
    Ok(Still { width: image.width, height: image.height, rgba: image.rgba, rays, march_ms: without.zip(with) })
}

/// One scene's still and, where reflections were traced, what the last frame's rays found.
struct Still {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    rays: Option<Rays>,
    /// One still frame's march without and with the reflected march, where one was traced.
    march_ms: Option<(f64, f64)>,
}

#[derive(Default)]
struct Rays {
    hit: usize,
    missed: usize,
    exhausted: usize,
}

fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) {
    image::RgbaImage::from_raw(width, height, rgba.to_vec())
        .expect("the picture read back is the right size")
        .save(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

fn differing(a: &[u8], b: &[u8]) -> (usize, u8) {
    let mut pixels = 0;
    let mut worst = 0u8;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let d = pa.iter().zip(pb).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
        if d > 0 {
            pixels += 1;
            worst = worst.max(d);
        }
    }
    (pixels, worst)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(out) = args.first().map(PathBuf::from) else {
        eprintln!("usage: reflection_scenes <out dir> [--reflections] [--compare <dir>]");
        std::process::exit(2);
    };
    let reflections = args.iter().any(|a| a == "--reflections");
    let compare = args.iter().position(|a| a == "--compare").and_then(|i| args.get(i + 1)).map(PathBuf::from);
    std::fs::create_dir_all(&out).expect("the output folder");
    let mut failed = false;
    println!("| scene | traced | reflected rays in the last frame: hit / missed / exhausted | one still frame's march, without → with | against the other run: pixels differing, worst |");
    println!("|---|---|---|---|---|");
    for scene in scenes() {
        let still = match still(&scene, reflections) {
            Ok(r) => r,
            Err(why) => {
                println!("| {} | — | — | — | did not render: {why} |", scene.name);
                failed = true;
                continue;
            }
        };
        write_png(&out.join(format!("{}.png", scene.name)), still.width, still.height, &still.rgba);
        let against = compare.as_ref().and_then(|dir| image::open(dir.join(format!("{}.png", scene.name))).ok());
        let verdict = match against {
            Some(other) => {
                let other = other.to_rgba8();
                let (pixels, worst) = differing(&still.rgba, other.as_raw());
                format!("{pixels} pixels, worst {worst}")
            }
            None => "—".to_string(),
        };
        let rays = match &still.rays {
            Some(r) => format!("{} / {} / {}", r.hit, r.missed, r.exhausted),
            None => "none dispatched".to_string(),
        };
        let cost = match still.march_ms {
            Some((without, with)) => format!("{without:.1} → {with:.1} ms"),
            None => "—".to_string(),
        };
        println!("| {} | {} | {rays} | {cost} | {verdict} |", scene.name, if still.rays.is_some() { "yes" } else { "no" });
    }
    if failed {
        std::process::exit(1);
    }
}
