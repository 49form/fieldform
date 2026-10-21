// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The beam cells: each wireframe's strut distance, where its struts meet, and its fill tables.

use nalgebra::Point3;
#[cfg(doc)] use super::{r3_point, CellKind, TpmsKind};

/// Which wireframe a beam cell is built from.
///
/// A beam cell is not a trig sum: it is a set of struts, and its field is the distance to them
/// less their radius. That one difference is why it needs a type of its own rather than another
/// [`TpmsKind`] variant — see [`CellKind`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BeamKind {
    /// Beams along the three axes through each lattice point.
    Cubic,
    /// Body-centred cubic: beams from each corner of the cell to its centre.
    Bcc,
    /// Face-centred cubic: the cube's edges plus both diagonals on every face.
    Fcc,
    /// The octet truss: an octahedron on the six face centres, each face capped by a tetrahedron
    /// to a cube corner.
    Octet,
    /// Diamond cubic: two interpenetrating face-centred sublattices, every site bonded
    /// tetrahedrally to four others.
    Diamond,
}

impl BeamKind {
    pub const ALL: [BeamKind; 5] = [
        BeamKind::Cubic,
        BeamKind::Bcc,
        BeamKind::Fcc,
        BeamKind::Octet,
        BeamKind::Diamond,
    ];

    pub fn label(self) -> &'static str {
        match self {
            BeamKind::Cubic => "Cubic",
            BeamKind::Bcc => "BCC",
            BeamKind::Fcc => "FCC",
            BeamKind::Octet => "Octet",
            BeamKind::Diamond => "Diamond",
        }
    }

    /// Distance from a point in unit-cube coordinates to the strut axes, in those same units.
    ///
    /// **Symmetry folding is what makes this cheap, and it is exact rather than an
    /// approximation.** The honest evaluation is a minimum over every strut and its 26
    /// neighbouring images, which is far too much for a per-pixel march. Every cell here is
    /// symmetric under reflection about each mid-plane, and reflecting a point across a plane the
    /// strut set is symmetric in leaves its distance to that set unchanged — so folding into one
    /// octant is free, and inside that octant a single representative strut is the nearest one.
    ///
    /// The argument that the representative is nearest: any strut outside the octant has a
    /// reflection inside it, at the same distance from the folded point, and the only strut
    /// inside is the representative. So the minimum is attained there.
    pub fn axis_distance(self, u: Point3<f64>) -> f64 {
        // **Diamond is the exception and does not fold at all.** Reflecting it about a
        // mid-plane sends its second sublattice to a place that is not a diamond site — the
        // structure has glide symmetry rather than the cell's own mirrors — so the argument
        // above does not hold for it and applying the fold anyway would be quietly wrong over
        // most of the cell. It has a reduction of its own; see [`Self::diamond_distance`].
        if let BeamKind::Diamond = self {
            return Self::diamond_distance(u);
        }
        // Into [0, ½] on each axis: `frac` puts the point in its own cell, and the fold about
        // ½ reflects the far half onto the near one.
        let fold = |t: f64| {
            let f = t - t.floor();
            0.5 - (f - 0.5).abs()
        };
        let f = Point3::new(fold(u.x), fold(u.y), fold(u.z));
        match self {
            // The nearest axis line runs through the origin of the folded octant, so the
            // distance to it is the distance in the other two coordinates.
            BeamKind::Cubic => f
                .y
                .hypot(f.z)
                .min(f.z.hypot(f.x))
                .min(f.x.hypot(f.y)),
            // Every strut runs corner to centre, and the only one inside the folded octant is
            // the main diagonal.
            BeamKind::Bcc => {
                point_to_segment(f, Point3::origin(), Point3::new(0.5, 0.5, 0.5))
            }
            // Three face diagonals and three cube edges, all from the folded origin. **Derived
            // rather than reasoned out**: the unfolded set was stated from the crystallographic
            // definition, a dense grid of the octant was minimised over all 27 images, and these
            // are the segments that were ever the nearest. `the_folded_strut_distance_is_the_
            // honest_minimum_over_every_image` is what says the derivation was complete.
            BeamKind::Fcc => {
                let o = Point3::origin();
                [
                    Point3::new(0.0, 1.0, 1.0),
                    Point3::new(1.0, 0.0, 1.0),
                    Point3::new(1.0, 1.0, 0.0),
                    Point3::new(1.0, 0.0, 0.0),
                    Point3::new(0.0, 1.0, 0.0),
                    Point3::new(0.0, 0.0, 1.0),
                ]
                .into_iter()
                .map(|end| point_to_segment(f, o, end))
                .fold(f64::INFINITY, f64::min)
            }
            // Three struts from the folded corner to the face centres around it, and the three
            // octahedron edges joining those face centres. Derived the same way.
            BeamKind::Octet => {
                let a = Point3::new(0.5, 0.5, 0.0);
                let b = Point3::new(0.5, 0.0, 0.5);
                let c = Point3::new(0.0, 0.5, 0.5);
                let o = Point3::origin();
                [(o, a), (o, b), (o, c), (a, b), (a, c), (b, c)]
                    .into_iter()
                    .map(|(s, e)| point_to_segment(f, s, e))
                    .fold(f64::INFINITY, f64::min)
            }
            // Returned above, before the fold that would have been wrong for it.
            BeamKind::Diamond => unreachable!("diamond does not fold"),
        }
    }
}

/// Diamond's struts, in the wedge its own symmetry reduces to.
///
/// **Derived, not written out by hand.** The bonds were stated from the crystallographic
/// definition, a dense grid of the wedge was minimised over every strut in all 27 images, and
/// these are the segments that were ever nearest or within a whisker of it — the whisker because a
/// near-tie at a region boundary would otherwise leave a hole. Twenty-four of them, against six
/// for FCC and octet, which is what losing the mirror fold costs.
/// `the_folded_strut_distance_is_the_honest_minimum_over_every_image` is what says the list is
/// complete; if a segment is missing, that test reports where.
const DIAMOND_STRUTS: [(Point3<f64>, Point3<f64>); 24] = [
    (Point3::new(0.00, 0.00, 0.00), Point3::new(0.25, 0.25, 0.25)),
    (Point3::new(0.00, 0.00, 1.00), Point3::new(-0.25, 0.25, 0.75)),
    (Point3::new(0.00, 0.00, 1.00), Point3::new(0.25, -0.25, 0.75)),
    (Point3::new(0.00, 0.00, 1.00), Point3::new(0.25, 0.25, 1.25)),
    (Point3::new(0.00, 0.50, 0.50), Point3::new(-0.25, 0.25, 0.75)),
    (Point3::new(0.00, 0.50, 0.50), Point3::new(0.25, 0.25, 0.25)),
    (Point3::new(0.00, 0.50, 0.50), Point3::new(0.25, 0.75, 0.75)),
    (Point3::new(0.00, 1.00, 1.00), Point3::new(0.25, 0.75, 0.75)),
    (Point3::new(0.50, 0.00, 0.50), Point3::new(0.25, -0.25, 0.75)),
    (Point3::new(0.50, 0.00, 0.50), Point3::new(0.25, 0.25, 0.25)),
    (Point3::new(0.50, 0.00, 0.50), Point3::new(0.75, 0.25, 0.75)),
    (Point3::new(0.50, 0.50, 0.00), Point3::new(0.25, 0.25, 0.25)),
    (Point3::new(0.50, 0.50, 1.00), Point3::new(0.25, 0.25, 1.25)),
    (Point3::new(0.50, 0.50, 1.00), Point3::new(0.25, 0.75, 0.75)),
    (Point3::new(0.50, 0.50, 1.00), Point3::new(0.75, 0.25, 0.75)),
    (Point3::new(0.50, 0.50, 1.00), Point3::new(0.75, 0.75, 1.25)),
    (Point3::new(0.50, 1.00, 0.50), Point3::new(0.25, 0.75, 0.75)),
    (Point3::new(0.50, 1.00, 0.50), Point3::new(0.75, 0.75, 0.25)),
    (Point3::new(0.50, 1.00, 0.50), Point3::new(0.75, 1.25, 0.75)),
    (Point3::new(1.00, 0.50, 0.50), Point3::new(0.75, 0.25, 0.75)),
    (Point3::new(1.00, 0.50, 0.50), Point3::new(0.75, 0.75, 0.25)),
    (Point3::new(1.00, 1.00, 1.00), Point3::new(0.75, 0.75, 1.25)),
    (Point3::new(1.00, 1.00, 1.00), Point3::new(0.75, 1.25, 0.75)),
    (Point3::new(1.00, 1.00, 1.00), Point3::new(1.25, 0.75, 0.75)),
];

/// Distance from a point to a line segment.
///
/// Written here rather than reaching for [`crate::Capsule`] because that answers a *signed*
/// distance to a thickened, possibly tapered solid in world space, and what a strut set needs is
/// the unsigned distance to a bare axis in cell space, taken a few times per sample. The shapes
/// of the two questions differ more than the three lines they share.
pub(crate) fn point_to_segment(p: Point3<f64>, a: Point3<f64>, b: Point3<f64>) -> f64 {
    let ab = b - a;
    let denominator = ab.dot(&ab);
    if denominator < 1e-18 {
        return (p - a).norm();
    }
    let t = ((p - a).dot(&ab) / denominator).clamp(0.0, 1.0);
    (p - (a + ab * t)).norm()
}

impl BeamKind {
    /// Diamond's distance, through the one symmetry it does have.
    ///
    /// It is not mirror-symmetric about the cell's mid-planes, so the octant fold every other
    /// cell uses is invalid here. What it *is* invariant under is permuting the coordinates —
    /// both of its sublattices are permutation-closed sets — so wrapping into the cell and then
    /// sorting the three coordinates reduces the query to a sixth of the cell, which is where
    /// [`DIAMOND_STRUTS`] lives.
    fn diamond_distance(u: Point3<f64>) -> f64 {
        let wrap = |t: f64| t - t.floor();
        let mut q = [wrap(u.x), wrap(u.y), wrap(u.z)];
        q.sort_by(f64::total_cmp);
        let p = Point3::new(q[0], q[1], q[2]);
        DIAMOND_STRUTS
            .iter()
            .map(|(a, b)| point_to_segment(p, *a, *b))
            .fold(f64::INFINITY, f64::min)
    }

    /// Where the struts meet their neighbours: half the smallest gap between two struts that
    /// neither share a node nor cross, in cell units.
    ///
    /// **Exact, and the end of every strut table.** Below it each strut overlaps only the struts
    /// it shares a node with, so the material is a union of capsules round nodes that do not
    /// touch; above it two struts with no node in common fuse, and the cell is a different
    /// shape that no table here describes and no source measured. The cubic's closed form makes
    /// the point sharply: `3πr² − 8√2r³` is exact up to exactly `1/2` and wrong past it -- it
    /// peaks at `π/(4√2) ≈ 0.555` and falls to 0.74 at 0.7, where the cell is full.
    ///
    /// | cell | merge radius | the pair that meets |
    /// |---|---|---|
    /// | Cubic | 1/2 | two parallel edges a cell apart |
    /// | BCC | √2/4 ≈ 0.354 | half-diagonals from neighbouring centres to different corners |
    /// | FCC | √3/6 ≈ 0.289 | face diagonals on two faces that meet at a cube edge |
    /// | Octet | 1/4 | a corner strut and an octahedron edge of the cell beside it |
    /// | Diamond | √3/8 ≈ 0.217 | two parallel bonds that share no atom |
    ///
    /// FCC's two diagonals on one face cross at its centre, and that crossing is a node rather
    /// than a meeting -- they overlap from the first micron, as struts at a node do.
    /// `the_merge_radius_is_half_the_nearest_gap_between_struts_that_do_not_meet` recomputes
    /// every row from the unfolded strut list by closed-form segment distances.
    pub fn merge_radius(self) -> f64 {
        match self {
            BeamKind::Cubic => 0.5,
            BeamKind::Bcc => 2f64.sqrt() / 4.0,
            BeamKind::Fcc => 3f64.sqrt() / 6.0,
            BeamKind::Octet => 0.25,
            BeamKind::Diamond => 3f64.sqrt() / 8.0,
        }
    }

    /// Strut length per cell volume, in cell units: what a thin strut's fill is `π r²` times.
    ///
    /// Exact, counted from the crystallography. Cubic: three edges. BCC: eight half body
    /// diagonals, `8·√3/2`. FCC: three edges and six face diagonals (each face shared by two
    /// cells), `3 + 6√2`. Octet: an FCC lattice's 24 nearest-neighbour bonds of `√2/2`.
    /// Diamond: sixteen bonds of `√3/4`.
    pub fn strut_length_per_cell(self) -> f64 {
        match self {
            BeamKind::Cubic => 3.0,
            BeamKind::Bcc => 4.0 * 3f64.sqrt(),
            BeamKind::Fcc => 3.0 + 6.0 * 2f64.sqrt(),
            BeamKind::Octet => 12.0 * 2f64.sqrt(),
            BeamKind::Diamond => 4.0 * 3f64.sqrt(),
        }
    }

    /// The fraction of a cubic cell the struts fill, against the radius over the cell.
    ///
    /// Exactly what `Lattice::distance` calls material: everything within `r` of a strut axis.
    /// **The cubic is closed form**, `3πr² − 8√2r³` -- three cylinders a cell long, less the
    /// three bicylinders they share pairwise at a node (`16r³/3` each), plus the tricylinder all
    /// three share (`8(2 − √2)r³`) -- and exact up to exactly the merge radius. **The rest are
    /// measured** by `beam_profile.rs`, as `ρ/r²`: that coefficient is `π` times the strut length
    /// per cell, less a node term nearly linear in `r`, so 1% rows interpolate near-exactly where
    /// interpolating `ρ` itself would be 100% wrong halfway to the first row. Row 0 is the exact
    /// `π·ℓ`. `None` past [`Self::merge_radius`], where the struts have met their neighbours.
    pub fn strut_density(self, radius_over_cell: f64) -> Option<f64> {
        let r = radius_over_cell;
        let merge = self.merge_radius();
        if !(r.is_finite() && r >= 0.0) || r > merge + 1e-12 {
            return None;
        }
        let rows: &[f64] = match self {
            BeamKind::Cubic => {
                return Some(3.0 * std::f64::consts::PI * r * r - 8.0 * 2f64.sqrt() * r * r * r)
            }
            BeamKind::Bcc => &STRUT_DENSITY_BCC,
            BeamKind::Fcc => &STRUT_DENSITY_FCC,
            BeamKind::Octet => &STRUT_DENSITY_OCTET,
            BeamKind::Diamond => &STRUT_DENSITY_DIAMOND,
        };
        // Row `k` is at `k/100`, except the last, which is at the merge radius itself wherever
        // that does not fall on a hundredth.
        let last = rows.len() - 1;
        let at = |k: usize| if k == last { merge } else { k as f64 / 100.0 };
        let i = ((r * 100.0).floor() as usize).min(last - 1);
        let t = ((r - at(i)) / (at(i + 1) - at(i))).clamp(0.0, 1.0);
        Some((rows[i] + (rows[i + 1] - rows[i]) * t) * r * r)
    }
}

/// A beam cell's fill over the radius-to-cell ratio `r`, as `ρ/r²`, at `r` = 0, 0.01, ... and
/// a last row at the cell's own [`BeamKind::merge_radius`] wherever that is not a hundredth.
/// Row 0 is the exact `π` times [`BeamKind::strut_length_per_cell`]. Read by
/// [`BeamKind::strut_density`]; the cubic cell's closed form needs no table.
///
/// **Measured by `beam_profile.rs`**: the fraction of 2²⁴ points of [`r3_point`] whose
/// `axis_distance` is at most `r`. Its error bound is the larger of two columns. The control --
/// the cubic cell measured the same way against `3πr² − 8√2r³` -- is 0.047% at the first
/// row and at most 0.042% from 0.02. The 2²² prefix against the whole is, at the first row and
/// from 0.02: BCC 0.193% and 0.033%, FCC 0.125% and 0.047%, octet 0.097% and 0.014%, diamond
/// 0.312% and 0.070%. Rows 1 and 2 carried linearly to `r = 0` land within 0.12% of `π·ℓ`
/// for every cell, and the field's own sign on 2²⁰ points of a 10 mm lattice matches the axis
/// distance point for point at the first, middle and last rows.
///
/// **Re-run it and paste the four arrays back after touching `axis_distance` or a strut list.**
/// `the_strut_density_table_is_the_fraction_the_field_fills` fails first.
const STRUT_DENSITY_BCC: [f64; 37] = [
    21.76559, 21.36171, 20.98352, 20.58764, 20.19949, 19.80515, 19.41376, 19.02262,
    18.62993, 18.23773, 17.84679, 17.45433, 17.06245, 16.67039, 16.27877, 15.88669,
    15.49483, 15.10317, 14.71095, 14.31912, 13.92734, 13.53534, 13.14335, 12.75143,
    12.35961, 11.96753, 11.57574, 11.18372, 10.79187, 10.40003, 10.00796, 9.61607,
    9.22424, 8.83228, 8.44042, 8.04842, 7.90921,
];
const STRUT_DENSITY_FCC: [f64; 30] = [
    36.08208, 35.17985, 34.26567, 33.34416, 32.42776, 31.51424, 30.60228, 29.69010,
    28.77599, 27.86435, 26.95014, 26.03769, 25.12420, 24.21136, 23.29790, 22.38507,
    21.47166, 20.55869, 19.64558, 18.73236, 17.81916, 16.90662, 16.01233, 15.15488,
    14.34025, 13.56918, 12.84116, 12.15439, 11.50668, 10.97467,
];
const STRUT_DENSITY_OCTET: [f64; 26] = [
    53.31460, 51.77975, 50.23375, 48.68057, 47.13591, 45.58928, 44.04455, 42.50014,
    40.95334, 39.40871, 37.86291, 36.31863, 34.77310, 33.22836, 31.68312, 30.13846,
    28.59309, 27.04819, 25.50276, 23.95789, 22.41274, 20.87452, 19.39897, 18.01936,
    16.74395, 15.57375,
];
const STRUT_DENSITY_DIAMOND: [f64; 23] = [
    21.76559, 21.37661, 20.98173, 20.58811, 20.19949, 19.80662, 19.41391, 19.02145,
    18.62913, 18.23873, 17.84627, 17.45446, 17.06268, 16.67075, 16.27902, 15.88692,
    15.49466, 15.10288, 14.71137, 14.31917, 13.92734, 13.53527, 13.28041,
];
