// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Runs the shader's own skewed-box arithmetic on the device and holds it against the field's.
//!
//! **Why a probe and not only `gpu_matches_cpu`.** That example asks how far the CPU field says a
//! *hit point* is from the surface, so it only ever visits points on the surface — and a box's
//! hard cases are exactly the ones a ray sees least of: the region beside an edge, the cone of
//! points whose nearest surface point is a corner, the interior near a face's rim. This takes the
//! shader's functions out of `PRELUDE` by name (so what is measured is the text the tracer runs)
//! and evaluates value *and* normal at a few thousand points chosen to land in those regions.
//!
//! **The normal is held to a property, not to the CPU's answer.** Inside a box the nearest face is
//! tied along a medial surface, and either tied face is a correct normal there, so comparing
//! directions would call a correct answer wrong. What must hold of any correct normal `n` at
//! distance `d` is that `p − n·d` lies on the surface — the foot of the perpendicular — and that is
//! asked of the CPU field. Outside a convex shape the nearest point is unique, so the same test
//! pins the direction there.
//!
//! Four boxes, one of which is the control: **an axis-aligned one, whose answer is `Cuboid`'s**, so
//! a probe that disagreed with every box could not be mistaken for one that found something.
//!
//! ```bash
//! cargo run --release -p fieldform-gpu --example skewed_box_probe
//! ```

use fieldform_field::{ScalarField, SkewedBox};
use nalgebra::{Point3, Vector3};
use wgpu::util::DeviceExt;

/// The text of the prelude from one marker up to another, so a probe runs the shipped source.
fn slice_of(from: &str, to: &str) -> String {
    let prelude = fieldform_gpu::PRELUDE;
    let start = prelude.find(from).expect("the prelude declares it");
    let end = prelude.find(to).expect("the prelude has the end marker");
    prelude[start..end].to_string()
}

const ENTRY: &str = r#"
@group(0) @binding(0) var<storage, read> pts: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cfg: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&pts) { return; }
    let p = pts[i].xyz;
    let c = cfg[0].xyz;
    let e0 = cfg[1].xyz;
    let e1 = cfg[2].xyz;
    let e2 = cfg[3].xyz;
    out[i] = vec4<f32>(
        skewed_box_distance(p, c, e0, e1, e2),
        skewed_box_normal(p, c, e0, e1, e2),
    );
}
"#;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Gpu {
    /// Dispatch the probe over `inputs` storage buffers and read back `count` `vec4<f32>`s.
    fn run(&self, source: String, inputs: &[&[u8]], count: usize) -> Vec<[f32; 4]> {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skewed box probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let buffers: Vec<wgpu::Buffer> = inputs
            .iter()
            .map(|bytes| {
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .collect();
        let bytes = (count * 16) as u64;
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let mut entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: buffers.len() as u32,
            resource: out.as_entire_binding(),
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &read, 0, bytes);
        self.queue.submit([encoder.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the queue drains");
        bytemuck::cast_slice(&read.slice(..).get_mapped_range()).to_vec()
    }
}

/// A small deterministic generator, so a failing run can be run again.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn between(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }

    fn direction(&mut self) -> Vector3<f64> {
        loop {
            let v = Vector3::new(
                self.between(-1.0, 1.0),
                self.between(-1.0, 1.0),
                self.between(-1.0, 1.0),
            );
            if v.norm() > 0.1 && v.norm() <= 1.0 {
                return v.normalize();
            }
        }
    }
}

struct Case {
    name: &'static str,
    centre: Point3<f64>,
    edges: [Vector3<f64>; 3],
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "axis-aligned (the control)",
            centre: Point3::new(5.0, -3.0, 8.0),
            edges: [Vector3::new(40.0, 0.0, 0.0), Vector3::new(0.0, 10.0, 0.0), Vector3::new(0.0, 0.0, 6.0)],
        },
        Case {
            name: "leaning, right-handed",
            centre: Point3::new(-20.0, 10.0, 0.0),
            edges: [Vector3::new(50.0, 0.0, 0.0), Vector3::new(20.0, 40.0, 0.0), Vector3::new(15.0, 10.0, 35.0)],
        },
        Case {
            // A negative determinant: the shader divides by `det`, and a sign taken from the wrong
            // place flips every inside normal and nothing else.
            name: "leaning, left-handed",
            centre: Point3::new(30.0, 20.0, 5.0),
            edges: [Vector3::new(30.0, 10.0, 5.0), Vector3::new(10.0, 0.0, 35.0), Vector3::new(-10.0, 30.0, 5.0)],
        },
        Case {
            // Nearly flat: the third edge only just leaves the plane of the other two, so every
            // slab is thin and a rim candidate is the answer for most outside points.
            name: "sliver",
            centre: Point3::new(0.0, 0.0, 0.0),
            edges: [Vector3::new(60.0, 0.0, 0.0), Vector3::new(25.0, 45.0, 0.0), Vector3::new(10.0, 10.0, 4.0)],
        },
    ]
}

/// Points over the box's neighbourhood, plus points built to land on its faces' rims, its edges
/// and its corners — the regions a uniform scatter hardly ever reaches.
fn sample_points(b: &Case, rng: &mut Lcg) -> Vec<Point3<f64>> {
    let scale = b.edges.iter().map(Vector3::norm).fold(0.0, f64::max);
    let mut out = Vec::new();
    for _ in 0..3000 {
        out.push(
            b.centre
                + b.edges[0] * rng.between(-1.2, 1.2)
                + b.edges[1] * rng.between(-1.2, 1.2)
                + b.edges[2] * rng.between(-1.2, 1.2),
        );
    }
    for _ in 0..3000 {
        // A point on the surface, then pushed off it a little in a random direction.
        let face = (rng.next() * 3.0) as usize % 3;
        let mut t = [rng.between(-0.5, 0.5), rng.between(-0.5, 0.5), rng.between(-0.5, 0.5)];
        t[face] = if rng.next() < 0.5 { -0.5 } else { 0.5 };
        // Half of them on an edge or a corner, by pinning a second (and third) coordinate.
        if rng.next() < 0.5 {
            t[(face + 1) % 3] = if rng.next() < 0.5 { -0.5 } else { 0.5 };
        }
        if rng.next() < 0.3 {
            t[(face + 2) % 3] = if rng.next() < 0.5 { -0.5 } else { 0.5 };
        }
        let on = b.centre + b.edges[0] * t[0] + b.edges[1] * t[1] + b.edges[2] * t[2];
        let by = rng.between(0.0005, 0.3) * scale * if rng.next() < 0.5 { -1.0 } else { 1.0 };
        out.push(on + rng.direction() * by);
    }
    out
}

fn main() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("an adapter");
    println!("adapter: {}\n", adapter.get_info().name);
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("a device");
    let gpu = Gpu { device, queue };

    let source = format!(
        "{}{ENTRY}",
        slice_of("// A box whose faces are not square", "// The round solids.")
    );
    let mut failed = false;
    let mut rng = Lcg(0x5EED_CAFE);
    for case in cases() {
        let field = SkewedBox::new(case.centre, case.edges).expect("the case has volume");
        let scale = case.edges.iter().map(Vector3::norm).fold(0.0, f64::max);
        let points = sample_points(&case, &mut rng);

        let pts: Vec<[f32; 4]> =
            points.iter().map(|p| [p.x as f32, p.y as f32, p.z as f32, 0.0]).collect();
        let v = |x: Vector3<f64>| [x.x as f32, x.y as f32, x.z as f32, 0.0];
        let cfg = [v(case.centre.coords), v(case.edges[0]), v(case.edges[1]), v(case.edges[2])];
        let got = gpu.run(
            source.clone(),
            &[bytemuck::cast_slice(&pts), bytemuck::cast_slice(&cfg)],
            points.len(),
        );

        // `f32` over millimetre coordinates in the tens: a millionth of the box's size is the
        // arithmetic's own noise, and a thousandth would hide a wrong candidate in a thin box.
        let tolerance = 1e-4 * scale;
        let (mut worst_value, mut worst_foot) = (0.0f64, 0.0f64);
        let (mut bad_value, mut bad_foot, mut outside, mut inside) = (0usize, 0usize, 0usize, 0usize);
        let mut first_bad: Option<(Point3<f64>, f64, f64)> = None;
        for (p, g) in points.iter().zip(&got) {
            let (d_gpu, n_gpu) = (g[0] as f64, Vector3::new(g[1] as f64, g[2] as f64, g[3] as f64));
            let d_cpu = field.distance(*p);
            if d_cpu > 0.0 {
                outside += 1;
            } else {
                inside += 1;
            }
            let e = (d_gpu - d_cpu).abs();
            worst_value = worst_value.max(e);
            if e > tolerance {
                bad_value += 1;
                first_bad.get_or_insert((*p, d_gpu, d_cpu));
            }
            // The foot of the perpendicular by the shader's own normal and the shader's own
            // distance must land on the surface the CPU describes.
            let foot = *p - n_gpu * d_gpu;
            let off = field.distance(foot).abs();
            worst_foot = worst_foot.max(off);
            if off > tolerance * 4.0 || (n_gpu.norm() - 1.0).abs() > 1e-3 {
                bad_foot += 1;
            }
        }
        println!(
            "{:<28} {} points ({} outside, {} inside)\n  value  worst {:.2e} mm, {} over {:.1e}\n  normal worst foot {:.2e} mm, {} over {:.1e}",
            case.name,
            points.len(),
            outside,
            inside,
            worst_value,
            bad_value,
            tolerance,
            worst_foot,
            bad_foot,
            tolerance * 4.0,
        );
        if let Some((p, gpu_d, cpu_d)) = first_bad {
            println!("  first disagreement at {p:?}: shader {gpu_d}, field {cpu_d}");
        }
        failed |= bad_value > 0 || bad_foot > 0;
    }
    if failed {
        eprintln!("\nthe shader's skewed box disagrees with the field's");
        std::process::exit(1);
    }
    println!("\nthe shader's skewed box agrees with the field's");
}
