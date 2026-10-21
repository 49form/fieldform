// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The light a part sits in, prepared for image-based lighting.
//!
//! Image-based lighting reads the environment twice per pixel: once blurred by the surface's
//! roughness for the specular reflection, once integrated against a cosine for the diffuse
//! response. Both integrals are independent of the model and of the camera, so they are done
//! once here, on the CPU, and the shader only looks them up — the split-sum approximation
//! (Karis, *Real Shading in Unreal Engine 4*, 2013), with the BRDF's own integral tabulated in
//! [`EnvironmentMaps::brdf_lut`] so the shader needs no Monte Carlo at all.
//!
//! # Why the environment is generated rather than shipped
//!
//! `shading-spec.md` §6.3 says to bake a fixed set of studio environments offline and ship
//! them. The [`Studio`] here is that environment, only described analytically — a graded dome
//! with a few soft panels in it — so there is no asset to license, provenance to record (§11),
//! or file to lose, and the bake takes tens of milliseconds at startup on the development
//! machine. A photographed HDRI is a different [`Environment`] behind the same trait, and the
//! prefilter does not care which it is given.
//!
//! # The cubemap convention
//!
//! Texel-to-direction follows the WebGPU cube face layout (+X, −X, +Y, −Y, +Z, −Z, with the
//! face-local axes the specification gives them), so what is baked for a direction here is
//! what `textureSample` returns for the same direction on the GPU. World up is +Z, as it is
//! everywhere in fieldform.

use nalgebra::Vector3;
use rayon::prelude::*;
use std::f32::consts::PI;

/// Where light comes from: radiance per direction, in linear RGB.
pub trait Environment: Sync {
    /// Radiance arriving *from* the unit direction `dir`, in world space.
    fn radiance(&self, dir: Vector3<f32>) -> [f32; 3];
}

/// The same radiance in every direction. What a white furnace test is run inside.
///
/// Every furnace test is in this file, and nothing lights a part with a uniform sky, so only
/// the tests make one and only the tests compile it.
#[cfg(test)]
struct Uniform(f32);

#[cfg(test)]
impl Environment for Uniform {
    fn radiance(&self, _dir: Vector3<f32>) -> [f32; 3] {
        [self.0; 3]
    }
}

/// One large soft light panel in the dome.
#[derive(Clone, Copy, Debug)]
pub struct Panel {
    /// The direction to the panel's centre, normalised on use.
    pub direction: Vector3<f32>,
    /// Angular radius in radians over which the panel is at full strength.
    pub radius: f32,
    /// How far past the radius it fades to nothing, in radians. A softbox has a soft edge, and
    /// a hard one would draw a crisp rectangle in every chrome reflection.
    pub feather: f32,
    /// Linear RGB radiance at full strength.
    pub color: [f32; 3],
}

/// A photographic studio: a graded neutral dome with soft panels for the key, fill and rim.
///
/// The dome is brighter and cooler overhead, neutral at the horizon and darker and warmer
/// underneath, which is what a matte grey part needs to read as a solid: an environment that
/// is the same in every direction lights every normal identically and flattens the shape. The
/// panels are what a polished surface shows — without them chrome reflects a featureless
/// gradient and looks like plastic.
#[derive(Clone, Debug)]
pub struct Studio {
    pub zenith: [f32; 3],
    pub horizon: [f32; 3],
    pub floor: [f32; 3],
    pub panels: Vec<Panel>,
}

impl Default for Studio {
    fn default() -> Self {
        Self {
            zenith: [0.62, 0.66, 0.74],
            horizon: [0.42, 0.42, 0.43],
            floor: [0.16, 0.15, 0.14],
            panels: vec![
                // Key: high, front-left of a camera looking along +Y from −Y.
                Panel {
                    direction: Vector3::new(-0.55, -0.45, 0.75),
                    radius: 0.30,
                    feather: 0.18,
                    color: [3.2, 3.1, 2.9],
                },
                // Fill: lower, right, dimmer and larger.
                Panel {
                    direction: Vector3::new(0.8, -0.35, 0.25),
                    radius: 0.45,
                    feather: 0.30,
                    color: [1.1, 1.15, 1.25],
                },
                // Rim: behind and above, so a silhouette catches an edge of light.
                Panel {
                    direction: Vector3::new(0.2, 0.9, 0.55),
                    radius: 0.22,
                    feather: 0.15,
                    color: [2.4, 2.4, 2.6],
                },
            ],
        }
    }
}

impl Environment for Studio {
    fn radiance(&self, dir: Vector3<f32>) -> [f32; 3] {
        // Up is +Z. The dome is two blends meeting at the horizon, eased so the horizon line
        // is not a crease in every reflection.
        let up = dir.z.clamp(-1.0, 1.0);
        let mut out = if up >= 0.0 {
            lerp3(self.horizon, self.zenith, smoothstep(up))
        } else {
            lerp3(self.horizon, self.floor, smoothstep(-up))
        };
        for panel in &self.panels {
            let centre = panel.direction.normalize();
            let angle = dir.dot(&centre).clamp(-1.0, 1.0).acos();
            let t = ((panel.radius + panel.feather - angle) / panel.feather).clamp(0.0, 1.0);
            let weight = smoothstep(t);
            for (o, c) in out.iter_mut().zip(panel.color) {
                *o += c * weight;
            }
        }
        out
    }
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One level of a cubemap: six square faces of RGBA texels, +X, −X, +Y, −Y, +Z, −Z.
#[derive(Clone, Debug)]
pub struct CubeLevel {
    pub size: u32,
    pub faces: Vec<Vec<[f32; 4]>>,
}

impl CubeLevel {
    fn filled(size: u32, f: impl Fn(usize, u32, u32) -> [f32; 4] + Sync) -> Self {
        let n = size as usize;
        let faces = (0..6)
            .map(|face| {
                (0..n * n)
                    .into_par_iter()
                    .map(|k| f(face, (k % n) as u32, (k / n) as u32))
                    .collect()
            })
            .collect();
        Self { size, faces }
    }

    /// Box-filtered to half the size, for the source chain the prefilter samples from.
    fn halved(&self) -> Self {
        let size = (self.size / 2).max(1);
        Self::filled(size, |face, i, j| {
            let mut sum = [0.0f32; 4];
            for (di, dj) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let s = self.texel(face, (i * 2 + di).min(self.size - 1), (j * 2 + dj).min(self.size - 1));
                for (a, b) in sum.iter_mut().zip(s) {
                    *a += b * 0.25;
                }
            }
            sum
        })
    }

    fn texel(&self, face: usize, i: u32, j: u32) -> [f32; 4] {
        self.faces[face][(j * self.size + i) as usize]
    }

    /// Bilinear within the face the direction lands on. Seams are not filtered across, which
    /// is a small error for a smooth environment and none at all once the GPU's own sampler
    /// filters the result.
    fn sample(&self, dir: Vector3<f32>) -> [f32; 3] {
        let (face, s, t) = face_uv(dir);
        let fs = s * self.size as f32 - 0.5;
        let ft = t * self.size as f32 - 0.5;
        let i0 = fs.floor();
        let j0 = ft.floor();
        let fi = fs - i0;
        let fj = ft - j0;
        let clamp = |v: f32| (v.max(0.0) as u32).min(self.size - 1);
        let (i0, i1, j0, j1) = (clamp(i0), clamp(i0 + 1.0), clamp(j0), clamp(j0 + 1.0));
        let mut out = [0.0f32; 3];
        for (i, j, w) in [
            (i0, j0, (1.0 - fi) * (1.0 - fj)),
            (i1, j0, fi * (1.0 - fj)),
            (i0, j1, (1.0 - fi) * fj),
            (i1, j1, fi * fj),
        ] {
            let texel = self.texel(face, i, j);
            for (o, c) in out.iter_mut().zip(texel) {
                *o += c * w;
            }
        }
        out
    }
}

/// The direction through the centre of texel `(i, j)` of `face`, at `size` texels a side.
///
/// `j` counts rows from the top of the face image, which is `t = 0`.
pub(crate) fn texel_direction(face: usize, i: u32, j: u32, size: u32) -> Vector3<f32> {
    let s = (i as f32 + 0.5) / size as f32;
    let t = (j as f32 + 0.5) / size as f32;
    face_direction(face, s, t)
}

/// The WebGPU cube face layout, inverted: face and `(s, t)` in `[0, 1]²` to a direction.
fn face_direction(face: usize, s: f32, t: f32) -> Vector3<f32> {
    let sc = 2.0 * s - 1.0;
    let tc = 2.0 * t - 1.0;
    let v = match face {
        0 => Vector3::new(1.0, -tc, -sc),
        1 => Vector3::new(-1.0, -tc, sc),
        2 => Vector3::new(sc, 1.0, tc),
        3 => Vector3::new(sc, -1.0, -tc),
        4 => Vector3::new(sc, -tc, 1.0),
        _ => Vector3::new(-sc, -tc, -1.0),
    };
    v.normalize()
}

/// The WebGPU cube face layout: which face a direction lands on, and where in it.
fn face_uv(dir: Vector3<f32>) -> (usize, f32, f32) {
    let (x, y, z) = (dir.x, dir.y, dir.z);
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
    let (face, sc, tc, ma) = if ax >= ay && ax >= az {
        if x > 0.0 { (0, -z, -y, ax) } else { (1, z, -y, ax) }
    } else if ay >= az {
        if y > 0.0 { (2, x, z, ay) } else { (3, x, -z, ay) }
    } else if z > 0.0 {
        (4, x, -y, az)
    } else {
        (5, -x, -y, az)
    };
    (face, 0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0))
}

/// Solid angle of texel `(i, j)` on a cube face `size` texels a side.
///
/// Exact rather than `4π / (6 size²)`: a corner texel subtends under half what a central one
/// does, and an irradiance integral that weighted them equally would be brightest wherever the
/// environment happened to put light near a corner.
pub(crate) fn texel_solid_angle(i: u32, j: u32, size: u32) -> f32 {
    let inv = 2.0 / size as f32;
    let x0 = i as f32 * inv - 1.0;
    let y0 = j as f32 * inv - 1.0;
    let x1 = x0 + inv;
    let y1 = y0 + inv;
    area_element(x0, y0) - area_element(x0, y1) - area_element(x1, y0) + area_element(x1, y1)
}

fn area_element(x: f32, y: f32) -> f32 {
    (x * y).atan2((x * x + y * y + 1.0).sqrt())
}

/// Everything the shader looks up, baked from one [`Environment`].
#[derive(Clone, Debug)]
pub struct EnvironmentMaps {
    /// The environment blurred by increasing roughness: level `m` is prefiltered for a
    /// perceptual roughness of `m / (levels − 1)`, with level 0 the environment itself.
    pub specular: Vec<CubeLevel>,
    /// Diffuse irradiance divided by π, so `albedo × lookup` is the Lambert response.
    pub irradiance: CubeLevel,
    /// The split-sum BRDF integral, `lut[roughness][n·v] = (scale, bias)` such that the
    /// specular response to a unit environment is `F0 × scale + bias`. Square, `lut_size`
    /// a side, with `n·v` along the row.
    pub brdf_lut: Vec<[f32; 2]>,
    pub lut_size: u32,
}

/// How finely the maps are made.
///
/// The defaults are `shading-spec.md` §9.1's: a 128² base with six roughness levels is ample for
/// the studio's soft panels, and bakes in about 70 ms. An environment with hard edges asks for
/// [`BakeSettings::sharp`] instead — see [`EnvironmentChoice::settings`].
#[derive(Clone, Copy, Debug)]
pub struct BakeSettings {
    pub specular_size: u32,
    pub specular_levels: u32,
    pub irradiance_size: u32,
    pub lut_size: u32,
    /// GGX samples per prefiltered texel, and per BRDF table entry.
    pub samples: u32,
}

impl Default for BakeSettings {
    fn default() -> Self {
        Self {
            specular_size: 128,
            specular_levels: 6,
            irradiance_size: 32,
            lut_size: 64,
            samples: 128,
        }
    }
}

impl BakeSettings {
    /// Twice the base resolution, for an environment a mirror shows the edges of. A zebra band
    /// reflected off a 128² face shows every texel of it; at 256² a band is sixteen texels
    /// wide and reads as a line. Costs about three times the bake — 224 ms against 70 on the
    /// development machine — because the first prefiltered level doubles in each direction.
    pub(crate) fn sharp() -> Self {
        Self { specular_size: 256, ..Self::default() }
    }
}

impl EnvironmentMaps {
    pub fn bake(env: &dyn Environment, settings: BakeSettings) -> Self {
        // The environment rasterised once at twice the output size, then box-filtered down.
        // The prefilter reads whichever level matches each sample's solid angle — filtered
        // importance sampling — so a small bright panel is averaged rather than hit or missed
        // by chance, which at 128 samples is the difference between a soft highlight and a
        // sparkle of fireflies across every rough metal.
        let source_size = (settings.specular_size * 2).max(8);
        let mut source = vec![CubeLevel::filled(source_size, |face, i, j| {
            let [r, g, b] = env.radiance(texel_direction(face, i, j, source_size));
            [r, g, b, 1.0]
        })];
        while source.last().unwrap().size > 1 {
            let next = source.last().unwrap().halved();
            source.push(next);
        }

        let levels = settings.specular_levels.max(2);
        let specular = (0..levels)
            .map(|level| {
                let size = (settings.specular_size >> level).max(1);
                let roughness = level as f32 / (levels - 1) as f32;
                if level == 0 {
                    // A mirror shows the environment itself.
                    CubeLevel::filled(size, |face, i, j| {
                        let [r, g, b] = sample_chain(&source, texel_direction(face, i, j, size), 0.0);
                        [r, g, b, 1.0]
                    })
                } else {
                    CubeLevel::filled(size, |face, i, j| {
                        let n = texel_direction(face, i, j, size);
                        let [r, g, b] = prefilter(&source, n, roughness, settings.samples);
                        [r, g, b, 1.0]
                    })
                }
            })
            .collect();

        // Integrated over every texel of a coarse source level rather than sampled: 16² × 6
        // texels against a cosine is exact for that level and has no noise to hide.
        let coarse = source
            .iter()
            .find(|l| l.size <= 16)
            .unwrap_or_else(|| source.last().unwrap());
        let cells: Vec<(Vector3<f32>, f32, [f32; 4])> = (0..6)
            .flat_map(|face| {
                (0..coarse.size).flat_map(move |j| {
                    (0..coarse.size).map(move |i| (face, i, j))
                })
            })
            .map(|(face, i, j)| {
                (
                    texel_direction(face, i, j, coarse.size),
                    texel_solid_angle(i, j, coarse.size),
                    coarse.texel(face, i, j),
                )
            })
            .collect();
        let irradiance = CubeLevel::filled(settings.irradiance_size, |face, i, j| {
            let n = texel_direction(face, i, j, settings.irradiance_size);
            let mut sum = [0.0f32; 3];
            for (dir, omega, radiance) in &cells {
                let cos = n.dot(dir);
                if cos > 0.0 {
                    for (s, c) in sum.iter_mut().zip(radiance) {
                        *s += c * cos * omega;
                    }
                }
            }
            [sum[0] / PI, sum[1] / PI, sum[2] / PI, 1.0]
        });

        let lut_size = settings.lut_size.max(2);
        let brdf_lut = (0..lut_size * lut_size)
            .into_par_iter()
            .map(|k| {
                let i = k % lut_size;
                let j = k / lut_size;
                let n_dot_v = ((i as f32 + 0.5) / lut_size as f32).max(1e-3);
                let roughness = (j as f32 + 0.5) / lut_size as f32;
                integrate_brdf(n_dot_v, roughness, settings.samples * 4)
            })
            .collect();

        Self { specular, irradiance, brdf_lut, lut_size }
    }

    /// The split-sum table entry for a view angle and roughness, bilinearly interpolated — what
    /// the shader's sampler does, for tests that want to check a number.
    ///
    /// The shader samples the table itself, so the furnace tests here are the only readers, and
    /// only the tests compile it.
    #[cfg(test)]
    fn brdf(&self, n_dot_v: f32, roughness: f32) -> [f32; 2] {
        let n = self.lut_size as f32;
        let fx = (n_dot_v.clamp(0.0, 1.0) * n - 0.5).clamp(0.0, n - 1.0);
        let fy = (roughness.clamp(0.0, 1.0) * n - 0.5).clamp(0.0, n - 1.0);
        let (x0, y0) = (fx.floor() as u32, fy.floor() as u32);
        let (x1, y1) = ((x0 + 1).min(self.lut_size - 1), (y0 + 1).min(self.lut_size - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let at = |x: u32, y: u32| self.brdf_lut[(y * self.lut_size + x) as usize];
        let mut out = [0.0f32; 2];
        for (x, y, w) in [
            (x0, y0, (1.0 - tx) * (1.0 - ty)),
            (x1, y0, tx * (1.0 - ty)),
            (x0, y1, (1.0 - tx) * ty),
            (x1, y1, tx * ty),
        ] {
            let v = at(x, y);
            out[0] += v[0] * w;
            out[1] += v[1] * w;
        }
        out
    }
}

/// Trilinear lookup into a box-filtered chain: bilinear in the face, linear between levels.
fn sample_chain(chain: &[CubeLevel], dir: Vector3<f32>, level: f32) -> [f32; 3] {
    let level = level.clamp(0.0, (chain.len() - 1) as f32);
    let lo = level.floor() as usize;
    let hi = (lo + 1).min(chain.len() - 1);
    let t = level - lo as f32;
    let a = chain[lo].sample(dir);
    if t <= 0.0 || lo == hi {
        return a;
    }
    let b = chain[hi].sample(dir);
    lerp3(a, b, t)
}

/// The environment seen in a reflection of roughness `roughness` about `n`, with the view
/// taken along `n` — the split-sum's one approximation, which is what lets the result be a
/// function of direction alone.
fn prefilter(chain: &[CubeLevel], n: Vector3<f32>, roughness: f32, samples: u32) -> [f32; 3] {
    let alpha = (roughness * roughness).max(1e-3);
    let (tangent, bitangent) = tangent_frame(n);
    let source_size = chain[0].size as f32;
    let texel_solid_angle = 4.0 * PI / (6.0 * source_size * source_size);
    let mut sum = [0.0f32; 3];
    let mut weight = 0.0f32;
    for i in 0..samples {
        let (u, v) = hammersley(i, samples);
        let h_local = importance_sample_ggx(u, v, alpha);
        let h = tangent * h_local.x + bitangent * h_local.y + n * h_local.z;
        // With v = n, the reflected direction and the half vector are related by a single dot.
        let n_dot_h = h_local.z;
        let l = h * (2.0 * n_dot_h) - n;
        let n_dot_l = n.dot(&l);
        if n_dot_l <= 0.0 {
            continue;
        }
        // Filtered importance sampling (Křivánek & Colbert): the sample's own footprint picks
        // the source level, so a sparse sample set averages the environment instead of dotting
        // it. `pdf = D / 4` because v = n makes n·h and v·h the same number.
        let pdf = d_ggx(n_dot_h, alpha) * 0.25;
        let sample_solid_angle = 1.0 / (samples as f32 * pdf + 1e-6);
        let level = (0.5 * (sample_solid_angle / texel_solid_angle).log2() + 1.0).max(0.0);
        let c = sample_chain(chain, l, level);
        for (s, x) in sum.iter_mut().zip(c) {
            *s += x * n_dot_l;
        }
        weight += n_dot_l;
    }
    if weight <= 0.0 {
        return chain[0].sample(n);
    }
    [sum[0] / weight, sum[1] / weight, sum[2] / weight]
}

/// The split-sum's second term: how much of a unit environment a GGX lobe reflects, split
/// into what scales `F0` and what does not (Karis 2013, eq. for `EnvBRDF`).
///
/// Height-correlated Smith visibility, as the shader uses, so the table and the analytic lights
/// agree about the same surface. At zero roughness the two entries sum to one exactly: a
/// mirror reflects everything, and `F0 × 1 + 0` at normal incidence is `F0`.
fn integrate_brdf(n_dot_v: f32, roughness: f32, samples: u32) -> [f32; 2] {
    let alpha = (roughness * roughness).max(1e-4);
    let v = Vector3::new((1.0 - n_dot_v * n_dot_v).max(0.0).sqrt(), 0.0, n_dot_v);
    let mut a = 0.0f32;
    let mut b = 0.0f32;
    for i in 0..samples {
        let (u, w) = hammersley(i, samples);
        let h = importance_sample_ggx(u, w, alpha);
        let v_dot_h = v.dot(&h);
        let l = h * (2.0 * v_dot_h) - v;
        let n_dot_l = l.z;
        let n_dot_h = h.z.max(0.0);
        let v_dot_h = v_dot_h.max(0.0);
        if n_dot_l <= 0.0 || n_dot_h <= 0.0 {
            continue;
        }
        // G_vis = V × 4 × n·l × v·h / n·h, with V the height-correlated Smith term.
        let vis = v_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);
        let g_vis = vis * 4.0 * n_dot_l * v_dot_h / n_dot_h;
        let fc = (1.0 - v_dot_h).powi(5);
        a += (1.0 - fc) * g_vis;
        b += fc * g_vis;
    }
    [a / samples as f32, b / samples as f32]
}

fn d_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d).max(1e-12)
}

fn v_smith_ggx_correlated(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ggx_v = n_dot_l * (n_dot_v * n_dot_v * (1.0 - a2) + a2).sqrt();
    let ggx_l = n_dot_v * (n_dot_l * n_dot_l * (1.0 - a2) + a2).sqrt();
    0.5 / (ggx_v + ggx_l).max(1e-12)
}

/// A half vector distributed as GGX about +Z, from two uniform numbers.
fn importance_sample_ggx(u: f32, v: f32, alpha: f32) -> Vector3<f32> {
    let phi = 2.0 * PI * u;
    let cos_theta = ((1.0 - v) / (1.0 + (alpha * alpha - 1.0) * v)).max(0.0).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    Vector3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

/// The Hammersley point set: evenly spread, deterministic, and the same for every texel so the
/// prefiltered map has structured rather than random error — which blurs away, where noise
/// would not.
fn hammersley(i: u32, n: u32) -> (f32, f32) {
    let mut bits = i;
    bits = bits.rotate_right(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    let radical_inverse = bits as f32 * 2.328_306_4e-10;
    ((i as f32 + 0.5) / n as f32, radical_inverse)
}

fn tangent_frame(n: Vector3<f32>) -> (Vector3<f32>, Vector3<f32>) {
    let up = if n.z.abs() < 0.999 { Vector3::z() } else { Vector3::x() };
    let tangent = up.cross(&n).normalize();
    let bitangent = n.cross(&tangent);
    (tangent, bitangent)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quick() -> BakeSettings {
        BakeSettings {
            specular_size: 16,
            specular_levels: 4,
            irradiance_size: 8,
            lut_size: 16,
            samples: 64,
        }
    }

    #[test]
    fn a_texel_direction_lands_back_on_its_own_texel() {
        let size = 8;
        for face in 0..6 {
            for j in 0..size {
                for i in 0..size {
                    let dir = texel_direction(face, i, j, size);
                    let (f, s, t) = face_uv(dir);
                    assert_eq!(f, face, "face {face} texel ({i}, {j}) went to face {f}");
                    let (ii, jj) = ((s * size as f32) as u32, (t * size as f32) as u32);
                    assert_eq!((ii, jj), (i, j), "face {face}");
                }
            }
        }
    }

    #[test]
    fn the_solid_angles_of_a_cube_add_up_to_the_whole_sphere() {
        let size = 32;
        let total: f32 = (0..6)
            .flat_map(|_| (0..size).flat_map(move |j| (0..size).map(move |i| (i, j))))
            .map(|(i, j)| texel_solid_angle(i, j, size))
            .sum();
        assert!((total - 4.0 * PI).abs() < 1e-3, "{total}");
    }

    /// The Lambert response to a uniform environment is the environment: E/π = L when L is
    /// constant, since ∫cos over the hemisphere is π.
    #[test]
    fn the_irradiance_of_a_uniform_environment_is_that_radiance() {
        let maps = EnvironmentMaps::bake(&Uniform(0.7), quick());
        for face in &maps.irradiance.faces {
            for texel in face {
                assert!((texel[0] - 0.7).abs() < 0.02, "{texel:?}");
            }
        }
    }

    /// Every roughness level of a uniform environment is still uniform: blurring a constant
    /// changes nothing, so any variation here is the prefilter's own error.
    #[test]
    fn a_uniform_environment_prefilters_to_itself_at_every_roughness() {
        let maps = EnvironmentMaps::bake(&Uniform(1.0), quick());
        for (level, cube) in maps.specular.iter().enumerate() {
            for face in &cube.faces {
                for texel in face {
                    assert!((texel[0] - 1.0).abs() < 1e-3, "level {level}: {texel:?}");
                }
            }
        }
    }

    #[test]
    fn the_mirror_level_is_the_environment_itself() {
        let studio = Studio::default();
        let maps = EnvironmentMaps::bake(&studio, quick());
        let level0 = &maps.specular[0];
        let mut worst = 0.0f32;
        for face in 0..6 {
            for j in 0..level0.size {
                for i in 0..level0.size {
                    let dir = texel_direction(face, i, j, level0.size);
                    let expected = studio.radiance(dir);
                    let got = level0.texel(face, i, j);
                    worst = worst.max((got[0] - expected[0]).abs());
                }
            }
        }
        // Box-filtered from a map twice the size, so the value is the texel's average rather
        // than its centre — a small difference on a smooth dome, and none on the flat parts.
        assert!(worst < 0.25, "worst difference {worst}");
    }

    /// Rougher is blurrier: the spread of values across a level can only fall as the roughness
    /// rises, since each level is a wider average of the same environment.
    #[test]
    fn each_roughness_level_varies_less_than_the_one_before() {
        let maps = EnvironmentMaps::bake(&Studio::default(), quick());
        let spread = |cube: &CubeLevel| {
            let values: Vec<f32> = cube.faces.iter().flatten().map(|t| t[0]).collect();
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32
        };
        let spreads: Vec<f32> = maps.specular.iter().map(spread).collect();
        for pair in spreads.windows(2) {
            assert!(pair[1] <= pair[0] * 1.02, "{spreads:?}");
        }
    }

    /// A mirror reflects everything: the two table entries sum to one at zero roughness. At
    /// any roughness they sum to at most one, since single-scatter GGX loses energy and never
    /// gains it.
    #[test]
    fn the_brdf_table_conserves_energy_and_is_complete_for_a_mirror() {
        let maps = EnvironmentMaps::bake(&Uniform(1.0), quick());
        let [a, b] = maps.brdf(1.0, 0.0);
        assert!((a + b - 1.0).abs() < 0.02, "scale {a} + bias {b}");
        for entry in &maps.brdf_lut {
            assert!(entry[0] + entry[1] <= 1.01, "{entry:?}");
            assert!(entry[0] >= 0.0 && entry[1] >= 0.0, "{entry:?}");
        }
        // Rougher surfaces lose more to multiple scattering the table does not count.
        let [a1, b1] = maps.brdf(1.0, 1.0);
        assert!(a1 + b1 < a + b, "{} vs {}", a1 + b1, a + b);
    }

    /// The studio is lit from above: the irradiance map must be brighter looking up than down,
    /// or a part would shade upside down.
    #[test]
    fn the_studio_lights_from_above() {
        let maps = EnvironmentMaps::bake(&Studio::default(), quick());
        let up = maps.irradiance.sample(Vector3::z());
        let down = maps.irradiance.sample(-Vector3::z());
        assert!(up[0] > down[0] * 1.5, "up {up:?} down {down:?}");
    }
}

/// Zebra stripes, as surface-analysis tools draw them: black and white bands stacked from the floor to the
/// zenith. Reflected off a mirror finish they trace the surface's curvature, and any kink,
/// wobble or flat spot in a band is a kink, wobble or flat spot in the surface — which is why a
/// surfacing tool has them and why they are useless on anything but chrome.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Zebra {
    /// Bands from nadir to zenith.
    pub stripes: u32,
}

impl Default for Zebra {
    fn default() -> Self {
        Self { stripes: 32 }
    }
}

impl Environment for Zebra {
    fn radiance(&self, dir: Vector3<f32>) -> [f32; 3] {
        // By elevation, so the bands are lines of latitude and a reflection of them reads the
        // surface's slope against the vertical.
        let elevation = dir.z.clamp(-1.0, 1.0).asin();
        let t = (elevation / PI + 0.5) * self.stripes as f32;
        let band = t.floor();
        let phase = t - band;
        let white = (band as i32).rem_euclid(2) == 0;
        // The edge is softened over a sliver of the band. Rasterised hard it would alias into
        // stair-steps, and a stair-step in a zebra band is exactly what a surface defect looks
        // like — the one thing this environment must never invent.
        let to_edge = phase.min(1.0 - phase);
        let sharpness = smoothstep((to_edge / 0.06).min(1.0));
        let value = if white { 1.0 } else { 0.0 };
        [0.5 + (value - 0.5) * sharpness; 3]
    }
}

/// A car showroom: rows of long light strips in a bright ceiling, grey walls, and a dark
/// polished floor. What a painted body shows in its reflections is the ceiling's strips
/// foreshortening along its curves, which is how a showroom flatters a car and how a designer
/// judges a highlight line.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Showroom;

impl Environment for Showroom {
    fn radiance(&self, dir: Vector3<f32>) -> [f32; 3] {
        let z = dir.z.clamp(-1.0, 1.0);
        let wall = [0.30, 0.31, 0.33];
        if z <= 0.0 {
            // Polished floor: dark underfoot, reflecting the walls toward the horizon.
            return lerp3([0.22, 0.22, 0.24], [0.09, 0.09, 0.10], smoothstep(-z));
        }
        // The ceiling as a plane one unit up, so its strips foreshorten toward the horizon the
        // way a real ceiling's do. Strips run along Y, a unit apart, and stop at the walls.
        let u = dir.x / z.max(1e-3);
        let v = dir.y / z.max(1e-3);
        let inside = u.abs() < 4.0 && v.abs() < 4.0;
        let across = (u + 0.5).rem_euclid(1.0) - 0.5;
        let strip = if inside {
            smoothstep(((0.11 - across.abs()) / 0.02 + 0.5).clamp(0.0, 1.0))
        } else {
            0.0
        };
        let ceiling = lerp3([0.62, 0.62, 0.66], [5.0, 5.1, 5.6], strip);
        if z > 0.35 {
            ceiling
        } else if z > 0.2 {
            // Where the ceiling meets the walls.
            lerp3(wall, ceiling, smoothstep((z - 0.2) / 0.15))
        } else {
            wall
        }
    }
}

/// The environments the viewport offers, as `View > Environment` lists them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EnvironmentChoice {
    Studio,
    Showroom,
    Zebra,
    /// Each pixel's normal painted as a colour, lighting nothing (backlog item 116).
    ///
    /// **Not a room, which is why it sits oddly in this enum and still belongs here.** The
    /// picture is made in two passes — the march finds a position and a normal per pixel, the
    /// lighting pass lights them — and this shows the first alone. So a mark that is in the
    /// lighting (a shadow hugging a rim, a highlight) vanishes, and a mark that is in the
    /// geometry (a hit drawn proud, a wrong normal, item 115's fat bake) stays. That is the
    /// whole value: one switch that says which half of the pipeline a doubt belongs to, with no
    /// instrument to run. It fixes nothing by itself.
    ///
    /// It is in this enum because this is the one list that already picks the look, in `View` and
    /// at `environment name=normals`, and a second control for one flag would be a second place
    /// to look. What it costs is `environment()` and `settings()` answering for something that is
    /// never sampled; see those.
    Normals,
}

impl EnvironmentChoice {
    pub const ALL: [EnvironmentChoice; 4] = [
        EnvironmentChoice::Studio,
        EnvironmentChoice::Showroom,
        EnvironmentChoice::Zebra,
        EnvironmentChoice::Normals,
    ];

    pub fn label(self) -> &'static str {
        match self {
            EnvironmentChoice::Studio => "Studio",
            EnvironmentChoice::Showroom => "Showroom",
            EnvironmentChoice::Zebra => "Zebra stripes",
            EnvironmentChoice::Normals => "Normals",
        }
    }

    /// Whether this choice paints normals instead of lighting the surface.
    pub fn paints_normals(self) -> bool {
        matches!(self, EnvironmentChoice::Normals)
    }

    pub fn hint(self) -> &'static str {
        match self {
            EnvironmentChoice::Studio => {
                "A graded dome with three soft panels and three lamps: the default."
            }
            EnvironmentChoice::Showroom => {
                "Rows of ceiling strips over a polished floor, lit by the room alone."
            }
            EnvironmentChoice::Zebra => {
                "Black and white bands for reading surface continuity. They show in a mirror \
                 finish: set the material to Chrome to see them."
            }
            EnvironmentChoice::Normals => {
                "Each pixel's normal as a colour, lit by nothing: a mark that survives is in the \
                 geometry, one that vanishes was in the lighting."
            }
        }
    }

    /// The choice called `name`, matched without regard to case or spaces.
    pub fn from_name(name: &str) -> Option<Self> {
        let fold = |s: &str| -> String {
            s.chars()
                .filter(|c| !c.is_whitespace() && *c != '_' && *c != '-')
                .flat_map(char::to_lowercase)
                .collect()
        };
        let wanted = fold(name);
        Self::ALL.into_iter().find(|c| {
            fold(c.label()) == wanted || fold(&format!("{c:?}")) == wanted
        })
    }

    /// Whether the studio's three lamps are lit under this environment. The others are lit
    /// by the room alone: a lamp's highlight would sit on top of the very strips a zebra
    /// exists to show, and a showroom is its ceiling.
    pub(crate) fn has_lamps(self) -> bool {
        matches!(self, EnvironmentChoice::Studio)
    }

    /// Whether occlusion means anything here. Not under zebra, where darkening a crease would
    /// read as a ripple in the reflection being inspected.
    /// Not under normals either, where there is no lighting for it to darken — which is where
    /// this view's being *cheaper* than a lit one comes from, together with the lighting it skips.
    pub fn wants_occlusion(self) -> bool {
        !matches!(self, EnvironmentChoice::Zebra | EnvironmentChoice::Normals)
    }

    pub(crate) fn environment(self) -> Box<dyn Environment> {
        match self {
            EnvironmentChoice::Studio => Box::new(Studio::default()),
            EnvironmentChoice::Showroom => Box::new(Showroom),
            EnvironmentChoice::Zebra => Box::new(Zebra::default()),
            // **Never sampled, and still baked.** The lighting pass returns the normal before it
            // reads a map, so what this answers reaches nothing; answering *something* keeps
            // `bake` total and the bind group the shape every other choice gives it, rather than
            // threading an `Option` through the renderer for a view that wants no room at all.
            // The studio's, so the maps are the ones the cache most likely already holds.
            EnvironmentChoice::Normals => Box::new(Studio::default()),
        }
    }

    /// How finely this environment is baked: sharp where it has edges a mirror would show.
    pub(crate) fn settings(self) -> BakeSettings {
        match self {
            EnvironmentChoice::Studio => BakeSettings::default(),
            EnvironmentChoice::Showroom | EnvironmentChoice::Zebra => BakeSettings::sharp(),
            // As the studio's, matching the maps above: nothing samples them.
            EnvironmentChoice::Normals => BakeSettings::default(),
        }
    }

    /// The maps for this environment.
    pub fn bake(self) -> EnvironmentMaps {
        EnvironmentMaps::bake(&*self.environment(), self.settings())
    }
}

#[cfg(test)]
mod choice_tests {
    use super::*;

    #[test]
    fn every_choice_is_found_by_its_label_and_its_name() {
        for choice in EnvironmentChoice::ALL {
            assert_eq!(EnvironmentChoice::from_name(choice.label()), Some(choice));
            assert_eq!(EnvironmentChoice::from_name(&format!("{choice:?}")), Some(choice));
        }
        assert_eq!(EnvironmentChoice::from_name("zebrastripes"), Some(EnvironmentChoice::Zebra));
        assert_eq!(EnvironmentChoice::from_name("aquarium"), None);
    }

    /// Straight up and straight down land in opposite bands' worth of nothing in between: the
    /// zebra is black or white almost everywhere, and grey only along an edge.
    #[test]
    fn the_zebra_is_black_and_white_with_soft_edges() {
        let zebra = Zebra::default();
        let mut extremes = 0;
        let mut samples = 0;
        for i in 0..2000 {
            let elevation = (i as f32 / 2000.0 - 0.5) * PI * 0.999;
            let dir = Vector3::new(elevation.cos(), 0.0, elevation.sin());
            let v = zebra.radiance(dir)[0];
            samples += 1;
            if v < 0.01 || v > 0.99 {
                extremes += 1;
            }
        }
        assert!(extremes as f32 / samples as f32 > 0.8, "{extremes} of {samples}");
        // Bands alternate: two elevations a band apart differ.
        let band = PI / zebra.stripes as f32;
        let a = zebra.radiance(Vector3::new((0.5 * band).cos(), 0.0, (0.5 * band).sin()))[0];
        let b = zebra.radiance(Vector3::new((1.5 * band).cos(), 0.0, (1.5 * band).sin()))[0];
        assert!((a - b).abs() > 0.9, "{a} vs {b}");
    }

    /// The showroom is lit from its ceiling: much brighter looking up than down, and the
    /// strips brighter than the ceiling between them.
    #[test]
    fn the_showroom_is_lit_from_its_ceiling() {
        let up = Showroom.radiance(Vector3::z())[0];
        let down = Showroom.radiance(-Vector3::z())[0];
        assert!(up > down * 3.0, "up {up} down {down}");
        // Straight up is the middle of a strip (u = 0); a quarter-unit across is between strips.
        let between = Showroom.radiance(Vector3::new(0.5, 0.0, 1.0).normalize())[0];
        assert!(up > between * 4.0, "strip {up} between {between}");
    }
}
