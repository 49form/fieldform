// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The same shapes marched at the origin and up to ten kilometres from it, each hit held to the
//! CPU's `f64` field.
//!
//! Geometry is `f64` and the march is `f32`. An `f32` carries 24 bits, so a coordinate of
//! 10^7 mm is held to the nearest millimetre and one of 10^5 mm to the nearest 8 µm: a part built
//! far from the origin is drawn at whatever precision is left over there, unless the shader
//! works relative to something near the part. This asks how much is left, scene by scene, with
//! the same yardstick `gpu_matches_cpu` uses: for every pixel the GPU says it hit, the CPU's
//! distance from that point to the surface.
//!
//! A lattice is the case to watch apart from the rest. Its cell is periodic in the *absolute*
//! coordinate when its frame sits at the world origin, so its argument there is the distance from
//! the origin in cells, and the phase of a sine of 10^6 radians is nothing at all in `f32`.
//!
//! `cargo run --release -p fieldform-gpu --example far_from_origin`

use fieldform_field::{GraphField, ScalarField};
use fieldform_gpu::{generate, Camera, Renderer};
use fieldform_scene::{
    BooleanOp, CellMap, Continuity, Document, Fillet, Frame, Graph, LatticeCell, LatticePhase, Node, ParamRef, Place,
};
use nalgebra::{Point3, Vector3};

const WIDTH: u32 = 480;
const HEIGHT: u32 = 320;

fn lit(v: f64) -> ParamRef {
    ParamRef::Literal(v)
}

fn at(p: Point3<f64>) -> Place {
    Place::At([lit(p.x), lit(p.y), lit(p.z)])
}

fn cuboid(graph: &mut Graph, centre: Point3<f64>, size: [f64; 3]) -> fieldform_scene::NodeId {
    graph.add(Node::Cuboid {
        edge: Default::default(),
        centre: at(centre),
        size: [lit(size[0]), lit(size[1]), lit(size[2])],
        turn: fieldform_scene::no_turn(),
    })
}

#[derive(Clone, Copy)]
enum Scene {
    Ball,
    Torus,
    Fillet,
    GyroidFramedAtTheOrigin,
    GyroidFramedAtThePart,
}

impl Scene {
    fn label(self) -> &'static str {
        match self {
            Scene::Ball => "ball r 20",
            Scene::Torus => "torus 30 / 5",
            Scene::Fillet => "wall on a slab, G1 fillet 4",
            Scene::GyroidFramedAtTheOrigin => "gyroid, frame at the world origin",
            Scene::GyroidFramedAtThePart => "gyroid, frame at the part",
        }
    }

    /// The scene with its centre at `c`, and the radius of a ball that holds it.
    fn build(self, c: Point3<f64>) -> (Graph, f64) {
        let mut graph = Graph::empty();
        let radius = match self {
            Scene::Ball => {
                graph.add(Node::Sphere { centre: at(c), radius: lit(20.0) });
                20.0
            }
            Scene::Torus => {
                graph.add(Node::Torus {
                    centre: at(c),
                    major_radius: lit(30.0),
                    minor_radius: lit(5.0),
                    turn: fieldform_scene::no_turn(),
                });
                35.0
            }
            Scene::Fillet => {
                let slab = cuboid(&mut graph, c + Vector3::new(0.0, 0.0, -5.0), [40.0, 40.0, 10.0]);
                let wall = cuboid(&mut graph, c + Vector3::new(-10.0, 0.0, 7.5), [10.0, 40.0, 25.0]);
                graph.add(Node::Blend {
                    op: BooleanOp::Union,
                    inputs: vec![slab.into(), wall.into()],
                    fillet: Fillet { radius: lit(4.0), continuity: Continuity::G1, second: None },
                });
                35.0
            }
            Scene::GyroidFramedAtTheOrigin | Scene::GyroidFramedAtThePart => {
                let frame = match self {
                    Scene::GyroidFramedAtThePart => Frame::At {
                        origin: [lit(c.x), lit(c.y), lit(c.z)],
                        axis: [lit(0.0), lit(0.0), lit(1.0)],
                    },
                    _ => Frame::default(),
                };
                let lattice = graph.add(Node::Lattice {
                    frame,
                    cell: LatticeCell::Gyroid,
                    map: CellMap::Rectangular { cell: lit(20.0) },
                    phase: LatticePhase::sheet(lit(2.0)),
                });
                let block = cuboid(&mut graph, c, [60.0, 60.0, 60.0]);
                graph.add(Node::Boolean { op: BooleanOp::Intersection, inputs: vec![block.into(), lattice.into()] });
                52.0
            }
        };
        (graph, radius)
    }
}

/// Every hit's distance from the CPU's surface, sorted.
fn residuals(cpu: &GraphField, camera: &Camera, hits: &[fieldform_gpu::Hit]) -> Vec<f64> {
    let mut out: Vec<f64> = hits
        .iter()
        .enumerate()
        .filter(|(_, hit)| hit.is_hit())
        .map(|(i, hit)| {
            let (px, py) = ((i as u32 % WIDTH) as f64 + 0.5, (i as u32 / WIDTH) as f64 + 0.5);
            let u = (px / f64::from(WIDTH) * 2.0 - 1.0) * camera.half_height * camera.aspect;
            let v = (1.0 - py / f64::from(HEIGHT) * 2.0) * camera.half_height;
            let ray = (camera.forward + camera.right * u + camera.up * v).normalize();
            cpu.distance(camera.eye + ray * hit.depth).abs()
        })
        .collect();
    out.sort_by(f64::total_cmp);
    out
}

fn main() {
    // Along a diagonal, so no axis is spared.
    let way = Vector3::new(1.0, -0.7, 0.4).normalize();
    let offsets = [0.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7];
    println!("{WIDTH} x {HEIGHT}, the camera at a fixed place relative to the part. Distances in mm;");
    println!("`f32` spacing is the gap between neighbouring `f32` values at that distance.\n");
    println!("| scene | from the origin | f32 spacing | hits | median | 99% | worst |");
    println!("|---|---|---|---|---|---|---|");
    let doc = Document::new();
    for scene in [Scene::Ball, Scene::Torus, Scene::Fillet, Scene::GyroidFramedAtTheOrigin, Scene::GyroidFramedAtThePart] {
        for &offset in &offsets {
            let centre = Point3::from(way * offset);
            let (graph, radius) = scene.build(centre);
            let cpu = GraphField::compile_graph(&doc, &graph, 1e-4).expect("the field compiles");
            let program = generate(&doc, &graph).expect("the shader generates");
            let renderer = match pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0)) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("no GPU available ({e})");
                    return;
                }
            };
            let values: Vec<f32> = program
                .params
                .iter()
                .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
                .collect();
            renderer.set_params(&values);
            let eye = centre + Vector3::new(1.0, 0.8, 0.6).normalize() * radius * 3.2;
            let camera = Camera::looking_at(
                eye,
                centre,
                Vector3::z(),
                (centre, radius),
                40.0,
                f64::from(WIDTH) / f64::from(HEIGHT),
            );
            let hits = renderer.render(&camera, WIDTH, HEIGHT, 1024).expect("renders");
            let r = residuals(&cpu, &camera, &hits);
            let spacing = f64::from((offset as f32).max(1.0)) * f64::from(f32::EPSILON);
            let pick = |q: f64| if r.is_empty() { f64::NAN } else { r[((r.len() - 1) as f64 * q) as usize] };
            println!(
                "| {} | {offset:.0e} | {spacing:.1e} | {} | {:.2e} | {:.2e} | {:.2e} |",
                scene.label(),
                r.len(),
                pick(0.5),
                pick(0.99),
                pick(1.0),
            );
        }
    }

    // Across scales, at the origin: a ball from ten micrometres to ten metres. The march stops at
    // a distance proportional to the scene's radius, so the error should be the same fraction of
    // the ball at every size, and any size where it is not is a constant in millimetres somewhere.
    println!("\n| ball radius | hits | worst | worst / radius |");
    println!("|---|---|---|---|");
    for radius in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0] {
        let mut graph = Graph::empty();
        graph.add(Node::Sphere { centre: at(Point3::origin()), radius: lit(radius) });
        let cpu = GraphField::compile_graph(&doc, &graph, 1e-4).expect("the field compiles");
        let program = generate(&doc, &graph).expect("the shader generates");
        let renderer = pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0)).expect("a GPU");
        let values: Vec<f32> = program
            .params
            .iter()
            .map(|p| p.resolve(&|id| doc.parameter_value(id)).unwrap_or(0.0) as f32)
            .collect();
        renderer.set_params(&values);
        let centre = Point3::origin();
        let eye = centre + Vector3::new(1.0, 0.8, 0.6).normalize() * radius * 3.2;
        let camera = Camera::looking_at(eye, centre, Vector3::z(), (centre, radius), 40.0, f64::from(WIDTH) / f64::from(HEIGHT));
        let hits = renderer.render(&camera, WIDTH, HEIGHT, 1024).expect("renders");
        let r = residuals(&cpu, &camera, &hits);
        let worst = r.last().copied().unwrap_or(f64::NAN);
        println!("| {radius} | {} | {worst:.2e} | {:.2e} |", r.len(), worst / radius);
    }
}
