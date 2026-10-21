// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The flake in a metallic paint, as a tiled normal map.
//!
//! Filament and other real-time car-paint models all do the same
//! thing: the flakes are a **normal map on the base layer**, under a clear coat that keeps the
//! geometric normal. Nothing about the coat is perturbed, which is the whole look of the material
//! — a smooth lacquer over a glittering base — and it is what the first attempt here got wrong by
//! tilting the base by twelve degrees and letting that tilt reach the diffuse and the Fresnel.
//!
//! Mipmapping does the distance behaviour for free: averaged over a pixel the tilts cancel, so
//! distant paint is a smooth metallic sheen and near paint sparkles, with no distance logic in
//! the shader. Rump et al. (*Photo-realistic rendering of metallic car paint from image-based
//! measurements*, 2008) measured flake orientation as roughly Gaussian about the surface with a
//! spread of a few degrees, which is what the tilts below are drawn from.
//!
//! Generated rather than shipped, like the environments: a tile of small flat facets with random
//! tilts is a page of arithmetic, and a texture on disk would be an asset to license and lose.
//! The map is sampled on the three world planes in `shade.wgsl`, since a field carries no UVs.

use std::f32::consts::PI;

/// World size of one repeat of the tile, in millimetres.
///
/// The shader keeps its own copy and nothing here scales by it, so this one exists to be held
/// to that copy: only the test that does so reads it, and only the tests compile it.
#[cfg(test)]
const TILE_MM: f32 = 16.0;

/// The tangent-plane offset one full encoded unit stands for: a byte of 255 is a tilt of
/// `atan(TILT_RANGE)`, about 27°, well past anything the generator draws.
pub(crate) const TILT_RANGE: f32 = 0.5;

/// A tiled normal map holding flake tilts, with its mip chain.
///
/// Each texel stores the flake's tangent-plane offset `(x, y)` — the tangent of its tilt along
/// each in-plane axis — encoded as bytes in the red and green channels. Blue and alpha are
/// unused, and kept so the texture is a plain `Rgba8Unorm`.
#[derive(Clone, Debug)]
pub struct FlakeMap {
    /// Texels a side, at level 0.
    pub size: u32,
    /// Level 0 first, each level half the size of the one before, down to a single texel.
    pub levels: Vec<Vec<[u8; 4]>>,
}

impl FlakeMap {
    /// A `size`² tile of facets `cell` texels across, tilted by a Gaussian spread of
    /// `sigma_degrees`, from `seed`.
    ///
    /// Each facet is one flat flake: every texel in the cell shares its tilt, with a one-texel
    /// margin of untilted binder around it, so a flake reads as a facet rather than as noise.
    /// A few cells are left empty, since paint is not wall-to-wall flake.
    pub(crate) fn generate(size: u32, cell: u32, sigma_degrees: f32, seed: u64) -> Self {
        let size = size.max(2).next_power_of_two();
        let cell = cell.clamp(2, size);
        let cells = size / cell;
        let sigma = sigma_degrees.to_radians();

        let mut level0 = vec![encode([0.0, 0.0]); (size * size) as usize];
        for cy in 0..cells {
            for cx in 0..cells {
                let mut rng = Rng::new(seed ^ ((u64::from(cy) << 32) | u64::from(cx)));
                // Roughly one cell in eight is binder with no flake in it.
                if rng.uniform() < 0.12 {
                    continue;
                }
                let direction = rng.uniform() * 2.0 * PI;
                // Half-normal about the surface, clamped well inside what the bytes can hold.
                let tilt = (rng.gaussian().abs() * sigma).min(TILT_RANGE.atan() * 0.8);
                let offset = [tilt.tan() * direction.cos(), tilt.tan() * direction.sin()];
                let texel = encode(offset);
                // The margin: the last texel row and column of the cell stay flat.
                for j in 0..cell.saturating_sub(1) {
                    for i in 0..cell.saturating_sub(1) {
                        let x = cx * cell + i;
                        let y = cy * cell + j;
                        level0[(y * size + x) as usize] = texel;
                    }
                }
            }
        }

        // The chain: offsets averaged two by two, which is how a normal map's mips are built
        // and what makes the flake fade with distance rather than alias.
        let mut levels = vec![level0];
        let mut current = size;
        while current > 1 {
            let next = current / 2;
            let prev = levels.last().unwrap();
            let mut level = Vec::with_capacity((next * next) as usize);
            for j in 0..next {
                for i in 0..next {
                    let mut sum = [0.0f32; 2];
                    for (di, dj) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let o = decode(prev[((j * 2 + dj) * current + i * 2 + di) as usize]);
                        sum[0] += o[0] * 0.25;
                        sum[1] += o[1] * 0.25;
                    }
                    level.push(encode(sum));
                }
            }
            levels.push(level);
            current = next;
        }
        Self { size, levels }
    }

    /// The tilt stored at a texel of a level, as a tangent-plane offset.
    ///
    /// The shader decodes the texels itself, so only the tests read a tilt back, and only the
    /// tests compile this.
    #[cfg(test)]
    fn offset(&self, level: usize, i: u32, j: u32) -> [f32; 2] {
        let size = self.size >> level;
        decode(self.levels[level][(j * size + i) as usize])
    }
}

fn encode(offset: [f32; 2]) -> [u8; 4] {
    let byte = |v: f32| ((v / TILT_RANGE).clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0;
    [byte(offset[0]).round() as u8, byte(offset[1]).round() as u8, 128, 255]
}

fn decode(texel: [u8; 4]) -> [f32; 2] {
    let value = |b: u8| (b as f32 / 255.0 * 2.0 - 1.0) * TILT_RANGE;
    [value(texel[0]), value(texel[1])]
}

/// A small deterministic generator, so the same seed always makes the same flake and the
/// crate does not take a dependency for eight lines.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03)
    }

    /// SplitMix64.
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal, by Box–Muller.
    fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map() -> FlakeMap {
        FlakeMap::generate(256, 4, 6.0, 7)
    }

    /// Flakes lean every which way and no way in particular: the mean tilt over the tile is
    /// nothing, so the map adds no bias to the surface it is laid on.
    #[test]
    fn the_flakes_have_no_preferred_direction() {
        let m = map();
        let n = (m.size * m.size) as f32;
        let mut mean = [0.0f32; 2];
        for j in 0..m.size {
            for i in 0..m.size {
                let o = m.offset(0, i, j);
                mean[0] += o[0] / n;
                mean[1] += o[1] / n;
            }
        }
        assert!(mean[0].abs() < 0.01 && mean[1].abs() < 0.01, "{mean:?}");
    }

    /// A few degrees, as measured paint is — not the twelve the hash version tilted by.
    #[test]
    fn the_flakes_tilt_by_a_few_degrees() {
        let m = map();
        let mut sum_sq = 0.0f32;
        let mut count = 0usize;
        for j in 0..m.size {
            for i in 0..m.size {
                let o = m.offset(0, i, j);
                let tilt = (o[0] * o[0] + o[1] * o[1]).sqrt().atan().to_degrees();
                if tilt > 0.0 {
                    sum_sq += tilt * tilt;
                    count += 1;
                }
            }
        }
        let rms = (sum_sq / count as f32).sqrt();
        assert!((3.0..=9.0).contains(&rms), "rms tilt {rms}°");
    }

    /// The last level is one texel: the whole tile averaged, which is flat. That is the distance
    /// behaviour — far paint has no flake to show.
    #[test]
    fn the_coarsest_level_is_flat() {
        let m = map();
        assert_eq!(m.levels.last().unwrap().len(), 1);
        let o = m.offset(m.levels.len() - 1, 0, 0);
        assert!(o[0].abs() < 0.01 && o[1].abs() < 0.01, "{o:?}");
    }

    #[test]
    fn the_same_seed_makes_the_same_flake() {
        assert_eq!(map().levels[0], FlakeMap::generate(256, 4, 6.0, 7).levels[0]);
        assert_ne!(map().levels[0], FlakeMap::generate(256, 4, 6.0, 8).levels[0]);
    }

    /// `shade.wgsl` repeats both numbers rather than being handed them, so nothing but this
    /// holds the two copies together. A tile size that drifted would stretch every flake across
    /// the part; a tilt range that drifted would decode every texel to the wrong angle. Neither
    /// fails anything else, and both look like a material choice rather than a fault.
    #[test]
    fn the_shader_reads_the_flakes_at_the_size_and_encoding_they_were_made_at() {
        let shader = crate::shading::SHADE;
        let tile = format!("const FLAKE_TILE_MM: f32 = {TILE_MM:?};");
        let tilt = format!("const FLAKE_TILT_RANGE: f32 = {TILT_RANGE:?};");
        assert!(shader.contains(&tile), "shade.wgsl no longer says `{tile}`");
        assert!(shader.contains(&tilt), "shade.wgsl no longer says `{tilt}`");
    }
}
