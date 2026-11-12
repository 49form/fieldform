// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The still's reflected march (backlog item 54) finds what the arithmetic says it should.
//!
//! Two chrome spheres, framed as the examples frame theirs. For every pixel the primary march
//! hit, the reflected ray is rebuilt on the CPU from the same inputs the shader rebuilds it from
//! -- the pixel's ray, the hit's depth and normal, the same offset off the surface -- and
//! intersected with the two spheres in closed form. Where the march reports a hit, the
//! arithmetic must put one within a few hit thresholds of it; where it reports a miss, the
//! arithmetic must miss; no ray may run out of budget; and the comparison is not vacuous, since
//! each sphere reflects a patch of the other. A ray that grazes the other sphere by less than
//! the tolerance, or meets it at a glancing angle where a threshold along the ray is many along
//! the normal, is counted and not judged.
//!
//! Then the second sphere is cut in half by a section through its centre, and every reflected
//! hit must lie on the kept side: a reflection of material the section removed is the fault the
//! march going through `scene()` rather than `field()` exists to prevent.
//!
//! No adapter is a skip, said out loud, as in `the_shaded_pass_survives_every_lighting_path`;
//! anything else the adapter refuses is a failure.

use std::io::Write;

use fieldform_gpu::{
    cpu_march::pixel_ray, generate, Camera, EnvironmentChoice, Look, Reflections, RenderError,
    Renderer, Sections, ShadingResources,
};
use fieldform_scene::{Document, Graph, Material, Node, ParamRef, Place};
use nalgebra::{Point3, Vector3};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const RADIUS: f32 = 20.0;
const CENTRES: [[f32; 3]; 2] = [[-30.0, 0.0, 0.0], [30.0, 0.0, 0.0]];
/// `REFLECT_OFFSET` in `camera.wgsl`: how many hit thresholds off the surface a reflected ray
/// starts.
const OFFSET: f32 = 4.0;
/// A hit is judged only where the ray meets the sphere at least this steeply: the march stops
/// within a threshold of the surface *in distance*, which along a glancing ray is many
/// thresholds.
const STEEP_ENOUGH: f32 = 0.2;

fn spheres() -> Graph {
    let mut g = Graph::empty();
    for c in CENTRES {
        let lit = |v: f32| ParamRef::Literal(f64::from(v));
        g.add(Node::Sphere { centre: Place::At([lit(c[0]), lit(c[1]), lit(c[2])]), radius: lit(RADIUS) });
    }
    g
}

/// The nearest point along `origin + dir * s`, `s > 0`, on either sphere: its distance and how
/// steeply the ray meets the surface there. With it, how near the ray comes to the surface of a
/// sphere it misses -- the margin a miss is judged by.
fn analytic(origin: Vector3<f32>, dir: Vector3<f32>) -> (Option<(f32, f32)>, f32) {
    let mut nearest: Option<(f32, f32)> = None;
    let mut margin = f32::INFINITY;
    for c in CENTRES {
        let oc = origin - Vector3::new(c[0], c[1], c[2]);
        let b = oc.dot(&dir);
        let cc = oc.dot(&oc) - RADIUS * RADIUS;
        let disc = b * b - cc;
        if disc < 0.0 {
            // The closest approach is at `s = -b`; only ahead of the origin does it count.
            if -b > 0.0 {
                margin = margin.min((cc - b * b).max(0.0).sqrt() - RADIUS);
            }
            continue;
        }
        let root = disc.sqrt();
        for s in [-b - root, -b + root] {
            if s <= 0.0 {
                continue;
            }
            let at = origin + dir * s;
            let n = (at - Vector3::new(c[0], c[1], c[2])).normalize();
            let steepness = dir.dot(&n).abs();
            if nearest.is_none_or(|(best, _)| s < best) {
                nearest = Some((s, steepness));
            }
            break;
        }
    }
    (nearest, margin)
}

#[derive(Default)]
struct Verdict {
    hit_pixels: usize,
    judged: usize,
    agreed: usize,
    /// Of the agreed, those where both the march and the arithmetic hit the other sphere.
    agreed_hits: usize,
    disagreed: usize,
    grazing: usize,
    exhausted: usize,
    worst_mm: f32,
    past_section: usize,
}

fn run(section: Option<f64>) -> Result<Verdict, RenderError> {
    let doc = Document::new();
    let program = generate(&doc, &spheres()).map_err(|e| RenderError::Shader(e.to_string()))?;
    let renderer = pollster::block_on(Renderer::new(&program, &[], &[], &[], 32, 1.0))?;
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
    let bounds = (Point3::new(-55.0, -55.0, -55.0), Point3::new(55.0, 55.0, 55.0));
    let mut camera = Camera::framing(bounds, aspect);
    if let Some(x) = section {
        camera.sections = Sections::of(&[(Vector3::x(), x)]);
    }
    let chrome = Material::preset("Chrome").expect("the chrome preset");
    let maps = EnvironmentChoice::Studio.bake();
    let mut shading = ShadingResources::new(renderer.device(), renderer.queue(), &maps)?;
    let look = Look::for_environment(EnvironmentChoice::Studio, chrome.clone(), camera.forward, camera.right, camera.up);
    camera.reflections = Reflections::for_material(&chrome, 1);
    assert!(camera.reflections.is_some(), "chrome is a mirror, which a one-frame still traces");
    renderer.present(&mut shading, &camera, WIDTH, HEIGHT, 512, &look, 0)?;
    renderer.device().poll(wgpu::PollType::wait_indefinitely()).map_err(|_| RenderError::Readback)?;
    let found = renderer.read_reflections(&shading)?.expect("a still that traced reflections has a buffer to read");
    // The same rays and the same march, read back as hits: `render` and `present` share the
    // pipeline and the uniform, and the reflected march reads nothing the primary writes back.
    camera.reflections = None;
    let hits = renderer.render(&camera, WIDTH, HEIGHT, 512)?;
    assert_eq!(hits.len(), found.len(), "one reflected word per pixel");

    let epsilon = camera.epsilon as f32;
    let tolerance = epsilon * 16.0;
    let mut verdict = Verdict::default();
    for (i, hit) in hits.iter().enumerate() {
        if !hit.is_hit() {
            continue;
        }
        verdict.hit_pixels += 1;
        let word = found[i];
        if word[3] == -2.0 {
            verdict.exhausted += 1;
        }
        let (x, y) = (i as u32 % WIDTH, i as u32 / WIDTH);
        let (o, d) = pixel_ray(&camera, (WIDTH, HEIGHT), x, y);
        let p = o + d * hit.depth as f32;
        let n = Vector3::new(hit.normal.x as f32, hit.normal.y as f32, hit.normal.z as f32).normalize();
        let r = d - n * (2.0 * d.dot(&n));
        let origin = p + n * (OFFSET * epsilon);
        if let Some(x) = section {
            if word[3] >= 0.0 {
                let q = origin + r * word[3];
                if f64::from(q.x) > x + f64::from(tolerance) {
                    verdict.past_section += 1;
                }
            }
            // The closed form below knows nothing of the cap, so the sectioned run judges only
            // which side its hits lie on.
            continue;
        }
        let (expected, margin) = analytic(origin, r);
        match (word[3] >= 0.0, expected) {
            (true, Some((s, steepness))) => {
                if steepness < STEEP_ENOUGH {
                    verdict.grazing += 1;
                    continue;
                }
                verdict.judged += 1;
                let off = (word[3] - s).abs();
                if off <= tolerance {
                    verdict.agreed += 1;
                    verdict.agreed_hits += 1;
                } else {
                    verdict.disagreed += 1;
                }
                verdict.worst_mm = verdict.worst_mm.max(off);
            }
            (false, None) => {
                if margin < tolerance {
                    verdict.grazing += 1;
                    continue;
                }
                verdict.judged += 1;
                verdict.agreed += 1;
            }
            (true, None) => {
                if margin.abs() < tolerance {
                    verdict.grazing += 1;
                    continue;
                }
                verdict.judged += 1;
                verdict.disagreed += 1;
            }
            (false, Some((_, steepness))) => {
                if steepness < STEEP_ENOUGH {
                    verdict.grazing += 1;
                    continue;
                }
                verdict.judged += 1;
                verdict.disagreed += 1;
            }
        }
    }
    println!(
        "{} by {}, hit threshold {epsilon:.2e} mm, tolerance {tolerance:.2e} mm: {} pixels hit, {} judged, \
         {} agreed ({} of them reflected hits), {} disagreed, {} grazing and not judged, {} exhausted, \
         worst {:.2e} mm off, {} reflected hits past the section",
        WIDTH, HEIGHT, verdict.hit_pixels, verdict.judged, verdict.agreed, verdict.agreed_hits,
        verdict.disagreed, verdict.grazing, verdict.exhausted, verdict.worst_mm, verdict.past_section,
    );
    Ok(verdict)
}

#[test]
fn a_reflected_ray_finds_the_sphere_the_arithmetic_says() {
    let whole = match run(None) {
        Ok(v) => v,
        Err(RenderError::NoAdapter) => {
            let _ = writeln!(
                std::io::stderr(),
                "SKIPPED a_reflected_ray_finds_the_sphere_the_arithmetic_says: no GPU adapter on \
                 this machine, so the reflected march was not exercised"
            );
            return;
        }
        Err(e) => panic!("the renderer would not start on an adapter that exists: {e}"),
    };
    assert!(whole.hit_pixels > 1000, "the spheres fill some of the frame: {} pixels hit", whole.hit_pixels);
    assert!(whole.exhausted == 0, "{} reflected rays ran out of budget on two spheres", whole.exhausted);
    // A patch of each sphere shows the other, so the agreement is about hits and not only
    // about misses.
    assert!(whole.judged > whole.hit_pixels / 2, "{} of {} judged", whole.judged, whole.hit_pixels);
    assert!(whole.agreed_hits >= 20, "only {} reflected hits agreed with the arithmetic", whole.agreed_hits);
    assert!(
        whole.disagreed * 200 <= whole.judged,
        "{} of {} judged pixels disagree with the arithmetic (worst {:.3} mm)",
        whole.disagreed,
        whole.judged,
        whole.worst_mm
    );

    let cut = run(Some(30.0)).expect("the sectioned run, on the adapter the whole one ran on");
    assert_eq!(cut.past_section, 0, "reflected hits on the side the section removed");
}
