// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What to tell a B-rep translator, in the translator's own terms.
//!
//! Split from `step.rs` for a reason that was measured rather than tidy. `step.rs` links
//! OpenCASCADE — 29.4 MB of statically linked CAD kernel — and the application needs these
//! settings to draw its import dialog. While they lived beside the reader, naming them was
//! enough to pull the kernel into any binary that did, so the application could not be made
//! lean no matter where the import itself ran.
//!
//! Nothing here knows anything about OCCT. They are distances, an angle and two switches.
//!
//! # Why these are numbers and not three named levels
//!
//! They were Coarse/Normal/Fine and a repair strength, which is what a first pass should offer.
//! Real files argue with a preset: a part imports open and the fix is more sewing, not finer
//! facets, and a preset that moves both cannot say so. So the settings are the translator's own
//! parameters under the translator's own names, and the presets stay as **buttons that fill
//! them in** rather than as a state of their own — one description of the settings, not two.

/// Everything the STEP reader is told, beyond the file itself.
///
/// Every field is something the file genuinely does not say. A STEP carries geometry and units;
/// it has no opinion on how finely to facet it, or how hard to try to close a shape its
/// exporter left open.
#[derive(Copy, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepOptions {
    /// **Linear deflection**, in millimetres: the greatest distance a facet may sit from the
    /// true surface it stands in for.
    ///
    /// The quantity every CAD translator uses, and absolute rather than a fraction of the part
    /// — so it means the same on a watch component and a car body. A fraction of the bounding
    /// box does not: it makes a 0.1 mm fillet on a 300 mm bracket coarser than the fillet
    /// itself, which is how this importer originally crashed.
    pub linear_deflection_mm: f64,

    /// **Angular deflection**, in degrees: how far a facet's normal may turn from its
    /// neighbour's.
    ///
    /// The criterion linear deflection does not bound, and the one a curved face is judged by,
    /// because faceting is a shading artefact and shading is the normal. The two are related
    /// through the radius — a facet subtending angle `t` on radius `r` sits `r(1 - cos(t/2))`
    /// below the arc — so the same chord height allows a far larger angle on a small radius.
    /// 0.05 mm on a 50 mm radius turns 5.1 degrees; on a 5 mm fillet it turns 16.
    ///
    /// **OCCT's own default is 28.6 degrees** (0.5 rad), which is what every import here used
    /// before this was exposed, and what a user sees as faceting.
    ///
    /// Measured by `fieldform-io/examples/angular_sweep.rs` on a real solid, as the **95th
    /// percentile** crease between neighbouring facets — not the largest, since a solid has
    /// genuinely sharp edges and the maximum measures the part rather than the mesh:
    ///
    /// | linear | angular | triangles | 95th percentile crease | time |
    /// |---|---|---|---|---|
    /// | 0.1 | 28.6° (OCCT default) | 7 958 | 22.7° | 0.47 s |
    /// | 0.1 | 11.5° | 19 156 | 14.6° | 0.26 s |
    /// | 0.05 | 28.6° | 16 402 | 17.7° | 0.80 s |
    /// | 0.05 | 11.5° | 25 022 | **13.0°** | 0.32 s |
    /// | 0.01 | 28.6° | 82 732 | 7.8° | 3.20 s |
    /// | 0.01 | 5.7° | 118 358 | **6.6°** | 1.18 s |
    ///
    /// Note the last two rows: at 0.01 mm the linear criterion is already tight enough that the
    /// angular one barely binds. It earns its keep at the coarse end.
    ///
    /// **Below about 3 degrees it stops helping and starts hurting**: 1.1° produced 2.3 million
    /// triangles whose creases were *worse* than the coarsest setting, the mesher having begun
    /// to emit slivers whose normals are noise.
    pub angular_deflection_deg: f64,

    /// **Sewing tolerance**, in millimetres, or zero to take the file at its word.
    ///
    /// Exporters routinely emit faces that meet geometrically but are not joined
    /// topologically, and OCCT reads that faithfully — so a part that is obviously one solid
    /// arrives as a bag of faces with an open edge along every seam. Sewing is what closes
    /// them, and it is why a CAD import dialog has this control at all.
    ///
    /// **Both directions are wrong.** Too small and nothing is joined; too large and genuinely
    /// separate faces are welded into each other, quietly changing the part. A wall thinner
    /// than this can disappear.
    pub sew_tolerance_mm: f64,

    /// Run OCCT's general repair pass before sewing: wires out of order, edges that do not meet
    /// their curves, faces facing the wrong way.
    ///
    /// Separate from the sewing tolerance because it fixes a different kind of damage. A file
    /// can need one, the other, both or neither, and a single "healing strength" could not say
    /// which — which is the argument for exposing them apart.
    pub shape_fix: bool,

    /// Mesh the faces in parallel.
    ///
    /// A shape's faces tessellate independently, so this is close to free on a large part. Off
    /// only if a file is ever found that OCCT cannot mesh threaded — the reason it is a switch
    /// rather than always-on is that it is the first thing worth turning off when a specific
    /// file misbehaves.
    pub parallel: bool,
}

impl Default for StepOptions {
    fn default() -> Self {
        Self::normal()
    }
}

impl StepOptions {
    /// Fast enough to look at, visibly faceted on small radii.
    ///
    /// Keeps OCCT's own angular default, so this is the behaviour every import had before the
    /// angular criterion was exposed.
    pub fn coarse() -> Self {
        Self {
            linear_deflection_mm: 0.1,
            angular_deflection_deg: 28.6,
            sew_tolerance_mm: 0.01,
            shape_fix: false,
            parallel: true,
        }
    }

    /// What most translators default to.
    pub fn normal() -> Self {
        Self {
            linear_deflection_mm: 0.05,
            angular_deflection_deg: 11.5,
            sew_tolerance_mm: 0.01,
            shape_fix: false,
            parallel: true,
        }
    }

    /// Manufacturing-grade, and expensive in triangles: a 6-solid assembly goes from 25 MB of
    /// mesh to 178 MB, and from about 32 s to 76 s.
    pub fn fine() -> Self {
        Self {
            linear_deflection_mm: 0.01,
            angular_deflection_deg: 5.7,
            sew_tolerance_mm: 0.01,
            shape_fix: false,
            parallel: true,
        }
    }

    /// For a part that arrives open under everything else: repair first, then sew ten times
    /// further.
    ///
    /// The setting to reach for when the report says a solid is not closed, and the one most
    /// likely to lose a thin wall.
    pub fn repair() -> Self {
        Self {
            sew_tolerance_mm: 0.1,
            shape_fix: true,
            ..Self::normal()
        }
    }

    /// The presets, as name and value, for a row of buttons.
    pub fn presets() -> [(&'static str, StepOptions); 4] {
        [
            ("Coarse", Self::coarse()),
            ("Normal", Self::normal()),
            ("Fine", Self::fine()),
            ("Repair", Self::repair()),
        ]
    }

    /// Which preset this matches, if any — so a dialog can show one as current without storing
    /// a second copy of the answer.
    pub fn preset_name(&self) -> Option<&'static str> {
        Self::presets()
            .into_iter()
            .find(|(_, o)| o == self)
            .map(|(name, _)| name)
    }

    /// The angular criterion in radians, which is what the mesher takes.
    pub fn angular_deflection_rad(&self) -> f64 {
        self.angular_deflection_deg.to_radians()
    }

    /// Clamped to what the mesher can actually deliver.
    ///
    /// Both ends are measured rather than chosen: below about 3 degrees the mesher emits
    /// slivers whose creases are worse than the coarsest setting, and a linear deflection of
    /// zero asks for infinite triangles.
    pub fn sane(mut self) -> Self {
        self.linear_deflection_mm = self.linear_deflection_mm.clamp(0.001, 10.0);
        self.angular_deflection_deg = self.angular_deflection_deg.clamp(3.0, 60.0);
        self.sew_tolerance_mm = self.sew_tolerance_mm.max(0.0);
        self
    }

    /// A one-line description, for the import report.
    pub fn summary(&self) -> String {
        let sew = if self.sew_tolerance_mm > 0.0 {
            format!("sew {:.3} mm", self.sew_tolerance_mm)
        } else {
            "no sewing".to_string()
        };
        let fix = if self.shape_fix { ", repaired" } else { "" };
        format!(
            "{:.3} mm / {:.1}\u{b0}, {sew}{fix}",
            self.linear_deflection_mm, self.angular_deflection_deg,
        )
    }
}
