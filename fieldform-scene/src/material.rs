// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What a part looks like in the viewport.
//!
//! One material per part, never per face — D1 in `commercial-readiness.md`: a machined bracket
//! is aluminium, the whole part is, and hanging appearance off cage faces would contradict the
//! thesis that the cage is scaffolding.
//!
//! The channels are the glTF 2.0 metallic-roughness model plus its clearcoat and anisotropy
//! extensions, adopted as the *data model* rather than designed here (`shading-spec.md` §6.1):
//! a material then round-trips to every DCC and CAD tool without a mapping layer, the maths can
//! be checked against open reference implementations, and "which materials do we support" is a
//! citation instead of an open-ended conversation.
//!
//! **A material has a kind, and the kind decides which channels mean anything.** Clay does not
//! have flake; chrome has no clearcoat; a lacquer has no brush direction. So a [`Kind`] names
//! what sort of thing the material is — plastic, metal, metallic paint, glass — and fixes every
//! channel that sort of thing does not have. The panel shows only the channels the kind has, and
//! [`Material::normalised`] holds the rest at the kind's value, so the kind and the channels
//! cannot disagree: the kind is authoritative and the channels follow it.
//!
//! **Nothing here says which preset a material came from.** The panel offers named presets —
//! chrome, car paint, rubber — but a preset is a way of *setting* the channels, and the channels
//! are the truth. Storing the preset's name beside them would be two descriptions of one
//! appearance, free to disagree the moment a slider moves; the label is derived instead, by
//! [`Material::preset_name`], and a material that matches no preset is called custom.
//!
//! What a part is *made of* for manufacturing — an alloy name, a density for mass properties —
//! is a different fact and is not here yet. It belongs on this struct when something reads it.

use serde::{Deserialize, Serialize};

/// What sort of thing a material is. Decides which channels it has — see the module notes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// A dielectric with a coloured surface: clay, moulded plastic, rubber, powder coat.
    Plastic,
    /// A conductor. Its colour is its reflectance, and it may be brushed.
    Metal,
    /// A metallic base under a clear lacquer: car paint.
    Paint,
    /// A tinted dielectric read by its reflection alone — no refraction (shading-spec §8).
    Glass,
}

impl Kind {
    /// Every kind. Only the test that walks each kind's channels lists them, so only the tests
    /// compile this (decision 1.4).
    #[cfg(test)]
    const ALL: [Kind; 4] = [Kind::Plastic, Kind::Metal, Kind::Paint, Kind::Glass];

    /// The channels a material of this kind has, in the order the panel shows them. Every
    /// other channel is fixed by the kind — see [`Material::normalised`].
    pub fn channels(self) -> &'static [Channel] {
        match self {
            Kind::Plastic => &[Channel::Roughness],
            Kind::Metal => &[Channel::Roughness, Channel::Anisotropy],
            Kind::Paint => &[
                Channel::Metallic,
                Channel::Roughness,
                Channel::Clearcoat,
                Channel::ClearcoatRoughness,
                Channel::Flake,
            ],
            Kind::Glass => &[Channel::Roughness],
        }
    }

    /// What the base colour *is* for this kind, which is what the panel should call it.
    pub fn color_label(self) -> &'static str {
        match self {
            Kind::Plastic | Kind::Paint => "Colour:",
            Kind::Metal => "Reflectance:",
            Kind::Glass => "Tint:",
        }
    }
}

/// One numeric channel of a material, for the panel to name and bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    Metallic,
    Roughness,
    Clearcoat,
    ClearcoatRoughness,
    Anisotropy,
    Flake,
}

impl Channel {
    pub fn label(self) -> &'static str {
        match self {
            Channel::Metallic => "Metallic base:",
            Channel::Roughness => "Roughness:",
            Channel::Clearcoat => "Clearcoat:",
            Channel::ClearcoatRoughness => "Coat roughness:",
            Channel::Anisotropy => "Brush:",
            Channel::Flake => "Flake:",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Channel::Metallic => {
                "How metallic the paint under the lacquer is: 0 a solid colour, 1 all flake."
            }
            Channel::Roughness => "0 is a mirror, 1 is fully matte.",
            Channel::Clearcoat => "How much clear lacquer sits over the base. Its own reflection.",
            Channel::ClearcoatRoughness => "How polished the lacquer is.",
            Channel::Anisotropy => {
                "Stretches the highlight along the brush direction, which is world Z laid on \
                 the surface. Negative stretches it across instead. Zero is unbrushed."
            }
            Channel::Flake => {
                "Tilted flakes in the base, under the clearcoat, as a normal map: sparkle up \
                 close, a sheen from afar. The lacquer over them stays smooth."
            }
        }
    }

    pub fn range(self) -> std::ops::RangeInclusive<f64> {
        match self {
            Channel::Anisotropy => -1.0..=1.0,
            _ => 0.0..=1.0,
        }
    }

    pub fn get(self, m: &Material) -> f64 {
        match self {
            Channel::Metallic => m.metallic,
            Channel::Roughness => m.roughness,
            Channel::Clearcoat => m.clearcoat,
            Channel::ClearcoatRoughness => m.clearcoat_roughness,
            Channel::Anisotropy => m.anisotropy,
            Channel::Flake => m.flake,
        }
    }

    pub fn set(self, m: &mut Material, value: f64) {
        match self {
            Channel::Metallic => m.metallic = value,
            Channel::Roughness => m.roughness = value,
            Channel::Clearcoat => m.clearcoat = value,
            Channel::ClearcoatRoughness => m.clearcoat_roughness = value,
            Channel::Anisotropy => m.anisotropy = value,
            Channel::Flake => m.flake = value,
        }
    }
}

/// The appearance of one part, in the viewport's shading model.
///
/// Every field is a plain number rather than a `ParamRef`. A radius is *geometry* and belongs in
/// the parametric layer where another block can be driven by it; a roughness drives nothing but
/// pixels, and routing it through the parameter table would put a display choice in the same
/// namespace as the numbers that define the part.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Material {
    /// What sort of thing this is, and so which of the channels below mean anything.
    pub kind: Kind,
    /// Linear-light RGB in `[0, 1]`, **not** display-encoded. For a metal this is its specular
    /// colour (its reflectance at normal incidence); for a dielectric it is the diffuse albedo;
    /// for glass, its tint.
    ///
    /// Linear because that is what the shader multiplies, and a value that had to be decoded
    /// first would be decoded in two places — the panel's colour picker and the shader — and
    /// one of them would forget.
    pub base_color: [f64; 3],
    /// `0` is a dielectric, `1` a metal. Fixed by the kind for everything but paint, where a
    /// fraction is how metallic the base under the lacquer is.
    pub metallic: f64,
    /// Perceptual roughness in `[0, 1]`: `0` is a mirror, `1` is fully diffuse. The shader
    /// squares it for the microfacet distribution, which is the glTF convention.
    pub roughness: f64,
    /// Weight of a clear lacquer layer over the base, `[0, 1]`. This is what makes car paint
    /// car paint: a second, near-mirror reflection whose Fresnel rim the base lobe alone
    /// cannot produce. Paint only.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub clearcoat: f64,
    /// Roughness of that lacquer layer. Near zero for a polished finish. Paint only.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub clearcoat_roughness: f64,
    /// How much the highlight is stretched along the surface's brush direction, `[-1, 1]`.
    /// Positive stretches along the tangent, negative across it. Zero is isotropic. Metal only.
    ///
    /// The brush direction itself is the world Z axis projected onto the surface — a field has
    /// no UV to carry one, and machining marks on a turned part do follow its axis.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub anisotropy: f64,
    /// Strength of the flake in the base lobe, `[0, 1]`: how far the base normal follows the
    /// flake normal map under the clearcoat. The coat itself is never perturbed. Paint only.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub flake: f64,
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

/// A named starting point for a material. See [`Material::presets`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Preset {
    pub name: &'static str,
    /// What it is, in one line — shown where the preset is picked.
    pub hint: &'static str,
    pub material: MaterialValues,
}

/// The numeric part of a [`Material`], constructible in a `const` so the preset table can be.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialValues {
    pub kind: Kind,
    pub base_color: [f64; 3],
    pub metallic: f64,
    pub roughness: f64,
    pub clearcoat: f64,
    pub clearcoat_roughness: f64,
    pub anisotropy: f64,
    pub flake: f64,
}

impl From<MaterialValues> for Material {
    fn from(v: MaterialValues) -> Self {
        Self {
            kind: v.kind,
            base_color: v.base_color,
            metallic: v.metallic,
            roughness: v.roughness,
            clearcoat: v.clearcoat,
            clearcoat_roughness: v.clearcoat_roughness,
            anisotropy: v.anisotropy,
            flake: v.flake,
        }
    }
}

const fn plastic(base_color: [f64; 3], roughness: f64) -> MaterialValues {
    MaterialValues {
        kind: Kind::Plastic,
        base_color,
        metallic: 0.0,
        roughness,
        clearcoat: 0.0,
        clearcoat_roughness: 0.0,
        anisotropy: 0.0,
        flake: 0.0,
    }
}

const fn metal(base_color: [f64; 3], roughness: f64, anisotropy: f64) -> MaterialValues {
    MaterialValues {
        kind: Kind::Metal,
        base_color,
        metallic: 1.0,
        roughness,
        clearcoat: 0.0,
        clearcoat_roughness: 0.0,
        anisotropy,
        flake: 0.0,
    }
}

/// The library a designer picks from, in the order the panel offers it.
///
/// Reflectances for the metals are measured values (aluminium, chromium) rather than picked
/// colours, so "chrome" is chrome and not a guess at silver. The dielectrics are chosen for
/// how they read under a studio dome: rubber is nearly black and rough, ABS is a light grey with
/// a soft sheen, and the default clay is the terracotta of the stuff itself — warm and matte and
/// unsaturated enough that a part's shape is still what you see.
pub(crate) const PRESETS: &[Preset] = &[
    Preset {
        name: "Clay",
        hint: "Matte terracotta, like the modelling clay. The default, so the shape is what \
               you look at.",
        // **Picked on screen, not on a swatch**, which for a matte body is the only way that
        // works: the studio dome's diffuse light lifts it several shades and pulls it toward
        // neutral, most of all in blue. This is linear, as every colour here is — sRGB
        // (139, 77, 47), a deep terracotta — and it *renders* as (162, 112, 89), the warm
        // brown of the clay a form is worked in. Chosen at the swatch value instead, it came
        // back on screen as a pale pinkish tan. `fieldform-gpu/examples/material_sweep.rs`
        // draws the row this was read from.
        //
        // Mid rather than dark once it is on screen, because the default finish has to show a
        // form rather than flatter it, and a dark body throws away the shading range that says
        // where the surface turns.
        material: plastic([0.26, 0.075, 0.028], 0.65),
    },
    Preset {
        name: "Painted metal",
        hint: "A powder-coat finish: coloured, slightly glossy, no metallic sheen.",
        material: plastic([0.06, 0.14, 0.45], 0.35),
    },
    Preset {
        name: "Brushed aluminium",
        hint: "Aluminium with machining marks: the highlight stretches along the brush.",
        material: metal([0.91, 0.92, 0.92], 0.4, 0.7),
    },
    Preset {
        name: "Chrome",
        hint: "Polished chromium. A mirror, so what you see is mostly the environment.",
        material: metal([0.55, 0.556, 0.554], 0.02, 0.0),
    },
    Preset {
        name: "ABS plastic",
        hint: "Light grey injection-moulded plastic with a soft sheen.",
        material: plastic([0.7, 0.7, 0.7], 0.25),
    },
    Preset {
        name: "Rubber",
        hint: "Near-black and rough, with almost no highlight.",
        material: plastic([0.02, 0.02, 0.02], 0.85),
    },
    Preset {
        name: "Car paint",
        hint: "Metallic red under a polished clearcoat, with flake in the base.",
        material: MaterialValues {
            kind: Kind::Paint,
            base_color: [0.5, 0.005, 0.02],
            // A metallic base smooth enough for its flakes to catch the light: a rough base
            // spreads each flake's reflection so wide that no flake shows.
            metallic: 0.6,
            roughness: 0.22,
            clearcoat: 1.0,
            clearcoat_roughness: 0.03,
            anisotropy: 0.0,
            flake: 0.8,
        },
    },
    Preset {
        name: "Glass",
        hint: "Dark tinted glass: reflection and Fresnel only, no refraction (shading-spec §8).",
        material: MaterialValues {
            kind: Kind::Glass,
            base_color: [0.02, 0.03, 0.03],
            metallic: 0.0,
            roughness: 0.02,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            anisotropy: 0.0,
            flake: 0.0,
        },
    },
];

impl Default for Material {
    /// The first preset, which is the neutral one. `Part::material` being `None` means this.
    fn default() -> Self {
        Material::from(PRESETS[0].material)
    }
}

impl Material {
    /// The named presets, in the order the panel offers them.
    pub fn presets() -> &'static [Preset] {
        PRESETS
    }

    /// The preset called `name`, matched without regard to case or spaces so a console can
    /// say `brushedaluminium` and mean the same thing the menu does.
    pub fn preset(name: &str) -> Option<Material> {
        let wanted = fold(name);
        PRESETS
            .iter()
            .find(|p| fold(p.name) == wanted)
            .map(|p| Material::from(p.material))
    }

    /// Which preset these channels are, if they are one exactly — derived rather than stored,
    /// so a slider moved off a preset cannot leave its name behind claiming otherwise.
    pub fn preset_name(&self) -> Option<&'static str> {
        PRESETS
            .iter()
            .find(|p| Material::from(p.material) == *self)
            .map(|p| p.name)
    }

    /// Every channel clamped to the range the shader gives meaning to.
    ///
    /// A roughness of 1.3 or a colour of -0.2 is not a material anyone asked for, and the
    /// shader would either normalise it away or produce a NaN; refusing it at the type is
    /// cheaper than checking for it in every reader.
    pub(crate) fn clamped(mut self) -> Self {
        for c in &mut self.base_color {
            *c = c.clamp(0.0, 1.0);
        }
        self.metallic = self.metallic.clamp(0.0, 1.0);
        self.roughness = self.roughness.clamp(0.0, 1.0);
        self.clearcoat = self.clearcoat.clamp(0.0, 1.0);
        self.clearcoat_roughness = self.clearcoat_roughness.clamp(0.0, 1.0);
        self.anisotropy = self.anisotropy.clamp(-1.0, 1.0);
        self.flake = self.flake.clamp(0.0, 1.0);
        self
    }

    /// Clamped, with every channel the kind does not have held at the kind's value.
    ///
    /// This is what keeps the kind and the channels one description: a plastic's `metallic` is
    /// always `0` and a metal's always `1`, not because anyone remembered to set them but
    /// because nothing that stores a material skips this. The shader reads the channels alone
    /// and never needs to know the kind.
    pub fn normalised(mut self) -> Self {
        match self.kind {
            Kind::Plastic | Kind::Glass => {
                self.metallic = 0.0;
                self.clearcoat = 0.0;
                self.clearcoat_roughness = 0.0;
                self.anisotropy = 0.0;
                self.flake = 0.0;
            }
            Kind::Metal => {
                self.metallic = 1.0;
                self.clearcoat = 0.0;
                self.clearcoat_roughness = 0.0;
                self.flake = 0.0;
            }
            Kind::Paint => {
                self.anisotropy = 0.0;
            }
        }
        self.clamped()
    }

    /// The same material as another kind of thing.
    ///
    /// Becoming paint gains a lacquer, since paint with none is not the thing anyone switched
    /// to paint for; everything else the new kind lacks is fixed by [`Self::normalised`].
    ///
    /// The appearance panel does not offer the kind, only obeys it: it comes from the preset,
    /// and the note where that panel is built says why. So only the tests change a kind, and
    /// only the tests compile this; it becomes public again if the kind is ever exposed.
    #[cfg(test)]
    fn with_kind(mut self, kind: Kind) -> Self {
        if kind == Kind::Paint && self.kind != Kind::Paint && self.clearcoat == 0.0 {
            self.clearcoat = 1.0;
            self.clearcoat_roughness = 0.03;
        }
        self.kind = kind;
        self.normalised()
    }
}

fn fold(name: &str) -> String {
    name.chars()
        .filter(|c| !c.is_whitespace() && *c != '_' && *c != '-')
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_has_a_distinct_name_and_distinct_channels() {
        for (i, a) in PRESETS.iter().enumerate() {
            for b in &PRESETS[i + 1..] {
                assert_ne!(fold(a.name), fold(b.name), "{} and {}", a.name, b.name);
                assert_ne!(a.material, b.material, "{} and {}", a.name, b.name);
            }
        }
    }

    #[test]
    fn every_preset_is_recognised_as_itself() {
        for preset in PRESETS {
            let material = Material::from(preset.material);
            assert_eq!(material.preset_name(), Some(preset.name));
            assert_eq!(Material::preset(preset.name), Some(material));
        }
    }

    #[test]
    fn a_preset_is_found_however_it_is_spelled() {
        assert_eq!(
            Material::preset("brushed_aluminium"),
            Material::preset("Brushed Aluminium")
        );
        assert_eq!(Material::preset("carpaint"), Material::preset("Car paint"));
        assert_eq!(Material::preset("unobtainium"), None);
    }

    #[test]
    fn a_material_moved_off_its_preset_is_no_longer_called_by_its_name() {
        let mut chrome = Material::preset("Chrome").unwrap();
        chrome.roughness += 0.1;
        assert_eq!(chrome.preset_name(), None);
    }

    /// A preset is what its kind says it is: nothing in the table sets a channel its kind
    /// fixes, so normalising a preset changes nothing about it.
    #[test]
    fn every_preset_is_already_normalised_for_its_kind() {
        for preset in PRESETS {
            let material = Material::from(preset.material);
            assert_eq!(material.clone().normalised(), material, "{}", preset.name);
        }
    }

    /// The panel shows a kind's channels and nothing else, so a channel the kind fixes must
    /// not be one the kind lists — or a slider would appear for a number the kind overwrites.
    #[test]
    fn a_kind_never_lists_a_channel_it_fixes() {
        for kind in Kind::ALL {
            let mut probe = Material { kind, ..Material::default() };
            for channel in [
                Channel::Metallic,
                Channel::Roughness,
                Channel::Clearcoat,
                Channel::ClearcoatRoughness,
                Channel::Anisotropy,
                Channel::Flake,
            ] {
                channel.set(&mut probe, 0.37);
            }
            let normalised = probe.clone().normalised();
            for channel in [
                Channel::Metallic,
                Channel::Roughness,
                Channel::Clearcoat,
                Channel::ClearcoatRoughness,
                Channel::Anisotropy,
                Channel::Flake,
            ] {
                let listed = kind.channels().contains(&channel);
                let survives = channel.get(&normalised) == 0.37;
                assert_eq!(listed, survives, "{kind:?} {channel:?}");
            }
        }
    }

    #[test]
    fn clay_has_no_flake_and_chrome_has_no_lacquer() {
        assert!(!Kind::Plastic.channels().contains(&Channel::Flake));
        assert!(!Kind::Metal.channels().contains(&Channel::Clearcoat));
        assert!(Kind::Paint.channels().contains(&Channel::Flake));
    }

    #[test]
    fn switching_kind_fixes_what_the_new_kind_has_no_say_over() {
        let clay = Material::default();
        let as_metal = clay.clone().with_kind(Kind::Metal);
        assert_eq!(as_metal.metallic, 1.0);
        assert_eq!(as_metal.base_color, clay.base_color);
        let as_paint = clay.with_kind(Kind::Paint);
        assert_eq!(as_paint.clearcoat, 1.0, "paint arrives lacquered");
        let back = as_paint.with_kind(Kind::Plastic);
        assert_eq!(back.clearcoat, 0.0);
        assert_eq!(back.flake, 0.0);
    }

    /// The optional channels are left out when zero, so a plain material writes what it uses
    /// and a file written before a channel existed reads as if it were zero.
    #[test]
    fn unused_channels_are_not_written_and_missing_ones_read_as_zero() {
        let json = serde_json::to_string(&Material::default()).unwrap();
        assert!(!json.contains("clearcoat"), "{json}");
        assert!(!json.contains("anisotropy"), "{json}");
        assert!(!json.contains("flake"), "{json}");
        assert!(json.contains(r#""kind":"plastic""#), "{json}");
        let back: Material = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Material::default());

        let paint = Material::preset("Car paint").unwrap();
        let json = serde_json::to_string(&paint).unwrap();
        assert!(json.contains("clearcoat") && json.contains("flake"), "{json}");
        let back: Material = serde_json::from_str(&json).unwrap();
        assert_eq!(back, paint);
    }

    /// D3's third row: a part-owned object with a field this build does not know is refused,
    /// not read with the field dropped — reading it and saving would write the loss back.
    #[test]
    fn a_channel_this_build_does_not_know_is_refused_rather_than_dropped() {
        let json = r#"{"kind":"plastic","base_color":[0.5,0.5,0.5],"metallic":0.0,"roughness":0.5,"sheen":0.3}"#;
        assert!(serde_json::from_str::<Material>(json).is_err());
    }
}
