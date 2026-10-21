// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Runs the shader's own Voronoi arithmetic on the device and holds it against the field's.
//!
//! **`voronoi-lattice-spec.md` §4.1 asks for the first half of this, and says why**: the two
//! copies of the hash have to agree bit for bit at every jitter, and `gpu_matches_cpu` *will*
//! catch a mismatch — but it catches it as a completely rearranged foam in one awkward scene,
//! which is a miserable way to find a hash bug.
//!
//! The second half is the part the spec did not ask for and the build needed: the whole
//! `(min(d_k, ℓ_R) − d₁)/2` pass, one level up from the hash. An agreeing hash can still be
//! built into a disagreeing arrangement — by a ring searched in a different order, a clamp
//! taken at a different ceiling, a wrap on the wrong axis — and when `gpu_matches_cpu` reported
//! 34 mm with the hash provably identical, this is what said where.
//!
//! Both halves take the shader's functions out of `PRELUDE` by name rather than keeping a third
//! copy, so what is measured is the text the lattice actually runs.
//!
//! ```bash
//! cargo run --release -p fieldform-gpu --example voronoi_probe
//! ```

use wgpu::util::DeviceExt;

/// The text of the prelude from one marker up to another, so a probe runs the shipped source.
fn slice_of(from: &str, to: &str) -> String {
    let prelude = fieldform_gpu::PRELUDE;
    let start = prelude.find(from).expect("the prelude declares it");
    let end = prelude.find(to).expect("the prelude has the end marker");
    prelude[start..end].to_string()
}

const HASH_ENTRY: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&input) { return; }
    let row = input[i];
    let c = vec3<i32>(bitcast<i32>(row.x), bitcast<i32>(row.y), bitcast<i32>(row.z));
    output[i] = vec4<u32>(voronoi_hash3(c, row.w), 0u);
}
"#;

const MEASURE_ENTRY: &str = r#"
@group(0) @binding(0) var<storage, read> pts: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cfg: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&pts) { return; }
    let row = pts[i];
    let c = cfg[i];
    let m = voronoi_measure(c.z, row.xyz, row.w, c.x, f32(c.y));
    out[i] = vec4<f32>(m.value, m.gradient);
}
"#;

const CASES: [([i32; 3], u32); 9] = [
    ([0, 0, 0], 0),
    ([1, 0, 0], 0),
    ([0, 1, 0], 0),
    ([0, 0, 1], 0),
    ([-1, -1, -1], 0),
    ([0, 0, 0], 1),
    ([0, 0, 0], 0x9E37_79B9),
    ([7, -3, 11], 12345),
    ([i32::MAX, i32::MIN, 0], u32::MAX),
];

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Gpu {
    /// Dispatch one entry point over `inputs` storage buffers and read back `count` `vec4`s.
    fn run(&self, source: String, inputs: &[&[u8]], count: usize) -> Vec<[u32; 4]> {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voronoi probe"),
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

    hash(&gpu);
    measure(&gpu);
}

/// The hash itself, over the table the field's copy is pinned to and a few hundred more —
/// including large and negative indices, which is where a multiply overflows.
fn hash(gpu: &Gpu) {
    let mut cases: Vec<([i32; 3], u32)> = CASES.to_vec();
    for i in 0..300i32 {
        cases.push(([i - 150, 7 * i - 900, 17 - i], (i as u32).wrapping_mul(2_654_435_761)));
    }
    let input: Vec<[u32; 4]> = cases
        .iter()
        .map(|(c, seed)| [c[0] as u32, c[1] as u32, c[2] as u32, *seed])
        .collect();
    let source = format!("{}{HASH_ENTRY}", slice_of("fn voronoi_hash3(", "// The three nearest seeds"));
    let got = gpu.run(source, &[bytemuck::cast_slice(&input)], cases.len());

    let mut parted = 0usize;
    for (i, (c, seed)) in cases.iter().enumerate() {
        let want = fieldform_field::voronoi_hash3(*c, *seed);
        let have = [got[i][0], got[i][1], got[i][2]];
        if want != have {
            if parted < 6 {
                println!("  {c:?} seed {seed}\n    field:  {want:?}\n    shader: {have:?}");
            }
            parted += 1;
        }
    }
    println!(
        "the hash: {} of {} cases parted",
        parted,
        cases.len()
    );
}

/// The whole ring-and-clamp pass, in cell units, at the jitters and wraps the scenes use.
fn measure(gpu: &Gpu) {
    use fieldform_field::StochasticKind;
    let kinds = [StochasticKind::VoronoiFoam, StochasticKind::VoronoiBeam];
    let mut points: Vec<[f32; 4]> = Vec::new();
    let mut cfg: Vec<[u32; 4]> = Vec::new();
    let mut want: Vec<(usize, f64, u32, u32, [f64; 3])> = Vec::new();
    for (k, _kind) in kinds.iter().enumerate() {
        for (jitter, seed, period) in
            [
                (0.0f64, 0u32, 0u32),
                (0.3, 0, 0),
                (1.0, 0, 0),
                (1.0, 0x5EED_1234u32, 0),
                (0.0, 0, 24),
                (0.3, 0, 24),
                (1.0, 0, 24),
                (1.0, 0x5EED_1234u32, 24),
                // Out of range on purpose: a jitter is an ordinary parameter, so an expression
                // can evaluate to anything, and both sides have to clamp it to the same thing.
                (1.7, 0x5EED_1234u32, 24),
                (-0.4, 0x5EED_1234u32, 24),
            ]
        {
            for i in 0..1200 {
                let u = [
                    4.0 + 0.031_7 * i as f64,
                    -11.5 + 0.023_1 * i as f64,
                    -3.0 + 0.045_3 * i as f64,
                ];
                points.push([u[0] as f32, u[1] as f32, u[2] as f32, jitter as f32]);
                // The shader reads the order statistic as a number, so the probe picks it the
                // way `lattice_distance` does: one for the foam's walls, two for its edges.
                cfg.push([seed, period, 1 + k as u32, 0]);
                want.push((k, jitter, seed, period, u));
            }
        }
    }
    let source = format!(
        "{}{MEASURE_ENTRY}",
        slice_of("fn voronoi_hash3(", "// Which axis a map closes on")
    );
    let raw = gpu.run(
        source,
        &[bytemuck::cast_slice(&points), bytemuck::cast_slice(&cfg)],
        points.len(),
    );
    let got: Vec<f32> = raw.iter().map(|r| f32::from_bits(r[0])).collect();

    println!("\nthe measure, in cell units");
    println!("  kind            jitter   seed         wrap   samples   apart   mean |diff|   worst |diff|");
    let mut i = 0;
    while i < want.len() {
        let (k, jitter, seed, period, _) = want[i];
        let (mut sum, mut worst, mut count, mut off) = (0.0f64, 0.0f64, 0usize, 0usize);
        let mut worst_at = [0.0f64; 3];
        while i < want.len() && want[i].0 == k && want[i].1 == jitter && want[i].2 == seed {
            let u = want[i].4;
            let periods =
                if period > 0 { [None, Some(period), None] } else { [None, None, None] };
            let (truth, _) = fieldform_field::voronoi_measure(
                kinds[k],
                nalgebra::Point3::new(u[0], u[1], u[2]),
                jitter,
                seed,
                periods,
            );
            let d = (truth - got[i] as f64).abs();
            if d > 1e-4 { off += 1; }
            sum += d;
            if d > worst {
                worst = d;
                worst_at = u;
            }
            count += 1;
            i += 1;
        }
        println!(
            "  {:<14}  {jitter:>5.2}   {seed:#010x}   {period:>4}   {count:>7}   {off:>5}   {:>11.2e}   {worst:>11.2e}",
            kinds[k].label(),
            sum / count as f64,
        );
        if worst > 1e-4 {
            println!("      worst at {worst_at:?}");
        }
    }
}
