// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Renders the app's composed field three ways, to separate three things that the meshed
//! viewport confounds: the field itself, how a normal is recovered from it, and what
//! voxelising it costs.
//!
//! Nothing here meshes. Every image is sphere-traced straight at the field, so no Surface
//! Nets grid, no vertex normals, and no rasteriser sit between the field and the pixel. That
//! is the point: the composed viewport has been reading as faceted, and the open question is
//! whether the facets are the mesher's or the field's. An image made without a mesher answers
//! it.
//!
//!   direct-analytic.png   the exact field, shaded with `ScalarField::normal`
//!                         — ground truth for what this field can look like.
//!   direct-fd.png         the exact field, shaded by central-differencing `distance`
//!                         — the old normal path, with the mesher removed. If this alone is
//!                         faceted, normal recovery was the whole problem.
//!   voxel-trilinear.png   the field baked to a grid, then trilinearly interpolated and
//!                         shaded from that grid's gradient — what a GPU voxel display
//!                         actually shows, since a 3D texture is exactly this.
//!
//! They are also tiled left-to-right in that order into `comparison.png`, because faceting
//! this subtle is judged by looking from one to the next, not at one alone.
//!
//! Run with:
//! `cargo run --release -p fieldform-field --example raymarch_reference [-- <resolution>]`

use fieldform_cage::Cage;
use fieldform_field::{FieldMesh, GraphField, ScalarField, mesh_field};
use fieldform_scene::{BooleanOp, CageId, Curve, CurveId, Document, Graph, Node, ParamRef};
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

/// The app's default: a 1 m cube cage, meshed on a 64-cell grid.
const CAGE_SIZE_MM: f64 = 1000.0;
const DEFAULT_MESH_RESOLUTION: usize = 64;
/// Matching the pipe radius the app was last driven with, so the seam this produces is the
/// same seam the ragged screenshots show.
const PIPE_RADIUS_MM: f64 = 111.0;

fn main() {
    let width: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(600);
    let height = width;
    let subtract = std::env::args().any(|a| a == "--subtract");
    // The viewport's cell count, overridable because the rim's raggedness is measured in cells
    // and the only way to see how much of it is resolution is to change the resolution.
    let args: Vec<String> = std::env::args().collect();
    let mesh_resolution = args
        .iter()
        .position(|a| a == "--cells")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MESH_RESOLUTION);

    // --- the same field the app composes, built the same way ---
    let mut doc = Document::new();
    let cage_id = CageId::new();
    doc.cages.insert(cage_id, Cage::cube_of(CAGE_SIZE_MM).into());
    let mut graph = Graph::from_cage(cage_id);

    if subtract {
        // A curve running clean through the cage, so the channel's rim crosses the sampling
        // grid at every angle rather than lining up with it.
        let curve_id = CurveId::new();
        let s = CAGE_SIZE_MM;
        let curve = Curve::polyline(
            curve_id,
            "probe",
            vec![
                Point3::new(-s, -0.18 * s, -0.10 * s),
                Point3::new(-0.3 * s, -0.05 * s, 0.02 * s),
                Point3::new(0.3 * s, 0.10 * s, 0.05 * s),
                Point3::new(s, 0.22 * s, 0.12 * s),
            ],
            false,
        )
        .expect("valid polyline");
        doc.curves.insert(curve_id, curve);

        let cage_node = graph.roots()[0];
        let pipe_node = graph.add(Node::Pipe {
            curve: curve_id,
            radius: ParamRef::Literal(PIPE_RADIUS_MM),
        });
        graph.add(Node::Boolean {
            op: BooleanOp::Difference,
            inputs: vec![cage_node.into(), pipe_node.into()],
        });
    }

    let bounds = cage_bounds(doc.cage(cage_id).unwrap());
    let extent = bounds.1 - bounds.0;
    let cell = extent.x.max(extent.y).max(extent.z) / mesh_resolution as f64;
    let tolerance = (cell * 0.5).max(1e-4);
    let field = GraphField::compile_graph(&doc, &graph, tolerance).expect("valid graph");

    let tag = if subtract { "subtract" } else { "cube" };
    println!("scene: {tag}");
    println!("cage: {CAGE_SIZE_MM} mm cube");
    println!("mesh cell at resolution {mesh_resolution}: {cell:.1} mm");
    println!("field tolerance: {tolerance:.1} mm");
    println!("rendering {width}x{height}\n");

    let camera = Camera::framing(bounds, width, height);

    // A half-cell step, matching what `mesh_field` uses on its fallback path.
    let h = cell * 0.5;

    let analytic = render(
        &format!("{tag}-analytic.png"),
        &camera,
        width,
        height,
        bounds,
        |p| field.distance(p),
        |p| {
            field
                .normal(p)
                .unwrap_or_else(|| central_difference(&field, p, h))
        },
    );

    let finite_difference = render(
        &format!("{tag}-fd.png"),
        &camera,
        width,
        height,
        bounds,
        |p| field.distance(p),
        |p| central_difference(&field, p, h),
    );

    // --- the voxel path ---
    //
    // Baked at the same resolution the mesher samples at, because that is the honest
    // comparison: both spend one grid's worth of field evaluations, and the question is what
    // each does with them. A GPU 3D texture would hold exactly this array and interpolate it
    // exactly this way, so what this image shows is what that display would show.
    let grid = VoxelGrid::bake(&field, bounds, mesh_resolution);
    println!(
        "\nbaked {0}x{0}x{0} voxels ({1:.1} MB as f32)",
        mesh_resolution + 1,
        (grid.values.len() * 4) as f64 / 1e6
    );

    let voxel = render(
        &format!("{tag}-voxel.png"),
        &camera,
        width,
        height,
        bounds,
        |p| grid.sample(p),
        |p| grid.gradient(p),
    );

    tile(
        &format!("{tag}-comparison.png"),
        &[&analytic, &finite_difference, &voxel],
        width,
        height,
    );

    // --- what the app actually draws ---
    //
    // Everything above bypasses `mesh_field` in order to interrogate the field. This does not:
    // it meshes exactly as the viewport does and rasterises the triangles, under the same
    // camera and the same light. Set against the analytic raymarch it isolates the mesher's
    // own contribution — anything present here and absent there was introduced by meshing.
    let start = std::time::Instant::now();
    let mesh = mesh_field(&field, bounds, mesh_resolution);
    let meshed_in = start.elapsed().as_secs_f64();
    println!(
        "\nmeshed {} triangles at resolution {mesh_resolution} in {meshed_in:.2}s",
        mesh.triangle_count()
    );

    report_mesh_quality(&mesh, &field, cell);

    let meshed = rasterise(&format!("{tag}-meshed.png"), &camera, width, height, &mesh);
    tile(
        &format!("{tag}-field-vs-mesh.png"),
        &[&analytic, &meshed],
        width,
        height,
    );
}

/// Separates the two ways a meshed surface can disagree with its field, because they have
/// different fixes and a picture confuses them.
///
/// *Topology*: every undirected edge of a closed surface belongs to exactly two triangles.
/// Anything else is a hole or a fold, and reads as a dark bite out of the surface.
///
/// *Position*: a vertex should lie on the surface, so `|distance|` at it should be near zero.
/// A vertex far off the surface is placed wrong even though the mesh around it is sound, and
/// reads as a lump or a notch. Reported against the cell size, since that is the scale the
/// mesher is entitled to be wrong by.
fn report_mesh_quality(mesh: &FieldMesh, field: &GraphField, cell: f64) {
    use std::collections::HashMap;

    let mut edge_uses: HashMap<(u32, u32), u32> = HashMap::new();
    for t in &mesh.triangles {
        for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *edge_uses
                .entry(if a < b { (a, b) } else { (b, a) })
                .or_insert(0) += 1;
        }
    }
    let boundary = edge_uses.values().filter(|&&c| c == 1).count();
    let folded = edge_uses.values().filter(|&&c| c > 2).count();

    let offsets: Vec<f64> = mesh
        .positions
        .par_iter()
        .map(|p| field.distance(*p).abs())
        .collect();
    let worst = offsets.iter().cloned().fold(0.0f64, f64::max);
    let mean = offsets.iter().sum::<f64>() / offsets.len().max(1) as f64;
    let far = offsets.iter().filter(|d| **d > cell * 0.5).count();

    println!("  topology: {boundary} boundary edges, {folded} folded edges (both should be 0)");
    println!(
        "  position: worst {worst:.2} mm, mean {mean:.2} mm, {far} of {} vertices beyond half a \
         cell ({:.1} mm)",
        offsets.len(),
        cell * 0.5
    );
}

/// Draws `mesh` with a z-buffer, matching [`render`]'s camera and lighting so the two can be
/// set side by side and the only difference between them is the meshing.
fn rasterise(
    name: &str,
    camera: &Camera,
    width: usize,
    height: usize,
    mesh: &FieldMesh,
) -> Vec<[u8; 3]> {
    const BACKGROUND: [u8; 3] = [24, 26, 30];
    let light = (camera.forward * -1.0 + camera.up * 0.6 + camera.right * 0.35).normalize();

    let mut pixels = vec![BACKGROUND; width * height];
    let mut depth = vec![f64::INFINITY; width * height];

    // Project once; every triangle referencing a vertex reuses the result.
    let projected: Vec<Option<(f64, f64, f64)>> = mesh
        .positions
        .iter()
        .map(|p| {
            let v = p - camera.eye;
            let z = v.dot(&camera.forward);
            if z <= 1e-9 {
                return None; // behind the eye: no meaningful projection
            }
            let sx = (v.dot(&camera.right) / z) / (camera.half_h * camera.aspect);
            let sy = (v.dot(&camera.up) / z) / camera.half_h;
            Some((
                (sx + 1.0) * 0.5 * width as f64,
                (1.0 - sy) * 0.5 * height as f64,
                z,
            ))
        })
        .collect();

    for tri in &mesh.triangles {
        let (Some(a), Some(b), Some(c)) = (
            projected[tri[0] as usize],
            projected[tri[1] as usize],
            projected[tri[2] as usize],
        ) else {
            continue;
        };

        let area = (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1);
        if area.abs() < 1e-12 {
            continue;
        }

        let min_x = a.0.min(b.0).min(c.0).floor().max(0.0) as usize;
        let max_x = (a.0.max(b.0).max(c.0).ceil() as usize).min(width.saturating_sub(1));
        let min_y = a.1.min(b.1).min(c.1).floor().max(0.0) as usize;
        let max_y = (a.1.max(b.1).max(c.1).ceil() as usize).min(height.saturating_sub(1));

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let w0 = ((b.0 - a.0) * (py - a.1) - (px - a.0) * (b.1 - a.1)) / area;
                let w1 = ((px - a.0) * (c.1 - a.1) - (c.0 - a.0) * (py - a.1)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                // Each weight belongs to the vertex *opposite* the edge it was built from:
                // `w0` came from edge a-b so it weights c, `w1` from a-c so it weights b, and
                // the remainder weights a.
                let z = w2 * a.2 + w1 * b.2 + w0 * c.2;
                let slot = y * width + x;
                if z >= depth[slot] {
                    continue;
                }
                depth[slot] = z;

                let n = (mesh.normals[tri[0] as usize] * w2
                    + mesh.normals[tri[1] as usize] * w1
                    + mesh.normals[tri[2] as usize] * w0)
                    .try_normalize(1e-12)
                    .unwrap_or_else(Vector3::z);
                let shade = 0.15 + 0.85 * n.dot(&light).max(0.0);
                let v = (shade.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
                pixels[slot] = [v, v, v];
            }
        }
    }

    let mut buf = Vec::with_capacity(width * height * 3);
    for p in &pixels {
        buf.extend_from_slice(p);
    }
    image::save_buffer(
        name,
        &buf,
        width as u32,
        height as u32,
        image::ColorType::Rgb8,
    )
    .expect("writable path");
    println!("{name:<22} rasterised");
    pixels
}

/// Tiles the renders side by side, separated by a hairline so adjacent dark backgrounds do
/// not read as one image.
fn tile(name: &str, images: &[&[[u8; 3]]], width: usize, height: usize) {
    const GAP: usize = 2;
    const DIVIDER: [u8; 3] = [70, 74, 82];
    let total = images.len() * width + (images.len() - 1) * GAP;

    let mut buf = Vec::with_capacity(total * height * 3);
    for row in 0..height {
        for (i, image) in images.iter().enumerate() {
            if i > 0 {
                for _ in 0..GAP {
                    buf.extend_from_slice(&DIVIDER);
                }
            }
            for px in &image[row * width..(row + 1) * width] {
                buf.extend_from_slice(px);
            }
        }
    }
    image::save_buffer(
        name,
        &buf,
        total as u32,
        height as u32,
        image::ColorType::Rgb8,
    )
    .expect("writable path");
    println!("\n{name:<22} analytic | finite difference | voxel, left to right");
}

/// Central-differences a field's distance to recover a normal — the path `mesh_field` used
/// before `ScalarField::normal` existed, reproduced here so it can be seen in isolation.
fn central_difference(field: &GraphField, p: Point3<f64>, h: f64) -> Vector3<f64> {
    let d = |o: Vector3<f64>| field.distance(p + o);
    Vector3::new(
        d(Vector3::new(h, 0.0, 0.0)) - d(Vector3::new(-h, 0.0, 0.0)),
        d(Vector3::new(0.0, h, 0.0)) - d(Vector3::new(0.0, -h, 0.0)),
        d(Vector3::new(0.0, 0.0, h)) - d(Vector3::new(0.0, 0.0, -h)),
    )
    .try_normalize(1e-18)
    .unwrap_or_else(Vector3::z)
}

/// The field sampled once onto a uniform grid, then answered by interpolation — a CPU stand-in
/// for a `texture_3d<f32>` and its hardware trilinear filter.
struct VoxelGrid {
    lo: Point3<f64>,
    cell: Vector3<f64>,
    n: usize,
    values: Vec<f32>,
}

impl VoxelGrid {
    fn bake(field: &GraphField, bounds: (Point3<f64>, Point3<f64>), resolution: usize) -> Self {
        let n = resolution + 1;
        let extent = bounds.1 - bounds.0;
        let cell = Vector3::new(
            extent.x / resolution as f64,
            extent.y / resolution as f64,
            extent.z / resolution as f64,
        );
        let values = (0..n * n * n)
            .into_par_iter()
            .map(|i| {
                let x = i % n;
                let y = (i / n) % n;
                let z = i / (n * n);
                let p = bounds.0
                    + Vector3::new(x as f64 * cell.x, y as f64 * cell.y, z as f64 * cell.z);
                field.distance(p) as f32
            })
            .collect();
        Self {
            lo: bounds.0,
            cell,
            n,
            values,
        }
    }

    fn at(&self, x: usize, y: usize, z: usize) -> f64 {
        let x = x.min(self.n - 1);
        let y = y.min(self.n - 1);
        let z = z.min(self.n - 1);
        self.values[z * self.n * self.n + y * self.n + x] as f64
    }

    /// Trilinear interpolation, matching what a GPU sampler does between texels.
    fn sample(&self, p: Point3<f64>) -> f64 {
        let max = (self.n - 1) as f64;
        let g = Vector3::new(
            ((p.x - self.lo.x) / self.cell.x).clamp(0.0, max),
            ((p.y - self.lo.y) / self.cell.y).clamp(0.0, max),
            ((p.z - self.lo.z) / self.cell.z).clamp(0.0, max),
        );
        let (i, j, k) = (g.x.floor(), g.y.floor(), g.z.floor());
        let (tx, ty, tz) = (g.x - i, g.y - j, g.z - k);
        let (i, j, k) = (i as usize, j as usize, k as usize);

        let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
        let c00 = lerp(self.at(i, j, k), self.at(i + 1, j, k), tx);
        let c10 = lerp(self.at(i, j + 1, k), self.at(i + 1, j + 1, k), tx);
        let c01 = lerp(self.at(i, j, k + 1), self.at(i + 1, j, k + 1), tx);
        let c11 = lerp(self.at(i, j + 1, k + 1), self.at(i + 1, j + 1, k + 1), tx);
        lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz)
    }

    /// The interpolated field's own gradient. A voxel display has no analytic normal available
    /// — the analytic one was a property of the field, and baking threw the field away — so
    /// differencing the texture is the only normal it can have.
    fn gradient(&self, p: Point3<f64>) -> Vector3<f64> {
        let h = self.cell * 0.5;
        Vector3::new(
            self.sample(p + Vector3::new(h.x, 0.0, 0.0))
                - self.sample(p - Vector3::new(h.x, 0.0, 0.0)),
            self.sample(p + Vector3::new(0.0, h.y, 0.0))
                - self.sample(p - Vector3::new(0.0, h.y, 0.0)),
            self.sample(p + Vector3::new(0.0, 0.0, h.z))
                - self.sample(p - Vector3::new(0.0, 0.0, h.z)),
        )
        .try_normalize(1e-18)
        .unwrap_or_else(Vector3::z)
    }
}

struct Camera {
    eye: Point3<f64>,
    forward: Vector3<f64>,
    right: Vector3<f64>,
    up: Vector3<f64>,
    /// Half the image plane's height at unit distance, from the vertical field of view.
    half_h: f64,
    aspect: f64,
}

impl Camera {
    /// A three-quarter view backed off far enough to hold the whole model, matching the kind
    /// of framing the viewport opens with. The angle is deliberately off-axis: faceting shows
    /// worst where a face is seen at a glance, and a head-on view hides it.
    fn framing(bounds: (Point3<f64>, Point3<f64>), width: usize, height: usize) -> Self {
        let center = Point3::from((bounds.0.coords + bounds.1.coords) * 0.5);
        let radius = (bounds.1 - bounds.0).norm() * 0.5;
        let fov_y: f64 = 45f64.to_radians();
        let half_h = (fov_y * 0.5).tan();
        let dir = Vector3::new(1.0, 0.7, 0.55).normalize();
        let eye = center + dir * (radius / half_h) * 1.05;

        let forward = (center - eye).normalize();
        let right = forward.cross(&Vector3::z()).normalize();
        let up = right.cross(&forward);
        Self {
            eye,
            forward,
            right,
            up,
            half_h,
            aspect: width as f64 / height as f64,
        }
    }

    fn ray(&self, px: usize, py: usize, width: usize, height: usize) -> Vector3<f64> {
        let u = (px as f64 + 0.5) / width as f64 * 2.0 - 1.0;
        let v = 1.0 - (py as f64 + 0.5) / height as f64 * 2.0;
        (self.forward + self.right * (u * self.half_h * self.aspect) + self.up * (v * self.half_h))
            .normalize()
    }
}

/// Sphere-traces `distance` and shades each hit with `normal`, writing a PNG and handing the
/// pixels back for [`tile`].
///
/// `distance` and `normal` are separate arguments rather than one field because separating
/// them is the whole experiment: two of the three images here trace the identical surface and
/// differ only in how they answer "which way is this facing".
fn render(
    name: &str,
    camera: &Camera,
    width: usize,
    height: usize,
    bounds: (Point3<f64>, Point3<f64>),
    distance: impl Fn(Point3<f64>) -> f64 + Sync,
    normal: impl Fn(Point3<f64>) -> Vector3<f64> + Sync,
) -> Vec<[u8; 3]> {
    let start = std::time::Instant::now();
    const BACKGROUND: [u8; 3] = [24, 26, 30];
    // A fixed light in camera space, so all three images are lit identically and any
    // difference between them is the surface's, not the lighting's.
    let light = (camera.forward * -1.0 + camera.up * 0.6 + camera.right * 0.35).normalize();

    let pixels: Vec<[u8; 3]> = (0..width * height)
        .into_par_iter()
        .map(|i| {
            let dir = camera.ray(i % width, i / width, width, height);
            match trace(camera.eye, dir, bounds, &distance) {
                Some(hit) => {
                    let n = normal(hit);
                    // Lambert plus a little ambient: flat enough to read shape, with no
                    // specular highlight that could either hide or invent a crease.
                    let lambert = n.dot(&light).max(0.0);
                    let shade = 0.15 + 0.85 * lambert;
                    let c = (shade.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
                    [c, c, c]
                }
                None => BACKGROUND,
            }
        })
        .collect();

    let mut buf = Vec::with_capacity(width * height * 3);
    for p in &pixels {
        buf.extend_from_slice(p);
    }
    image::save_buffer(
        name,
        &buf,
        width as u32,
        height as u32,
        image::ColorType::Rgb8,
    )
    .expect("writable path");

    let hits = pixels.iter().filter(|p| **p != BACKGROUND).count();
    println!(
        "{name:<22} {:>6.2}s  {hits} surface pixels",
        start.elapsed().as_secs_f64()
    );
    pixels
}

/// Standard sphere tracing, clipped to `bounds` so rays that miss terminate immediately.
///
/// The step is scaled below the reported distance because a difference or intersection is a
/// max of two distances, which can overestimate how far it is safe to move; a full step there
/// would tunnel through a thin wall.
fn trace(
    eye: Point3<f64>,
    dir: Vector3<f64>,
    bounds: (Point3<f64>, Point3<f64>),
    distance: &(impl Fn(Point3<f64>) -> f64 + Sync),
) -> Option<Point3<f64>> {
    let (t_near, t_far) = slab_intersect(eye, dir, bounds)?;
    let epsilon = (bounds.1 - bounds.0).norm() * 1e-5;

    let mut t = t_near.max(0.0);
    for _ in 0..512 {
        if t > t_far {
            return None;
        }
        let p = eye + dir * t;
        let d = distance(p);
        if d < epsilon {
            return Some(p);
        }
        t += (d * 0.8).max(epsilon);
    }
    None
}

/// Ray/AABB slab test, returning the entry and exit parameters along the ray.
fn slab_intersect(
    eye: Point3<f64>,
    dir: Vector3<f64>,
    bounds: (Point3<f64>, Point3<f64>),
) -> Option<(f64, f64)> {
    let mut t_near = f64::NEG_INFINITY;
    let mut t_far = f64::INFINITY;
    for axis in 0..3 {
        let o = eye[axis];
        let d = dir[axis];
        let (lo, hi) = (bounds.0[axis], bounds.1[axis]);
        if d.abs() < 1e-12 {
            if o < lo || o > hi {
                return None;
            }
            continue;
        }
        let (mut t0, mut t1) = ((lo - o) / d, (hi - o) / d);
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t_near = t_near.max(t0);
        t_far = t_far.min(t1);
        if t_near > t_far {
            return None;
        }
    }
    Some((t_near, t_far))
}

/// The box the app would sample: the cage's own extent, with a margin so the surface is not
/// flush against the boundary.
fn cage_bounds(cage: &Cage) -> (Point3<f64>, Point3<f64>) {
    let mut lo = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut hi = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    for v in cage.vertex_ids() {
        let p = cage.vertex_position(v);
        lo = Point3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Point3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    let margin = (hi - lo).norm() * 0.02;
    (
        lo - Vector3::new(margin, margin, margin),
        hi + Vector3::new(margin, margin, margin),
    )
}
