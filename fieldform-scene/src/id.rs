// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Stable identities for the things that need them.
//!
//! # Why random, not sequential
//!
//! A counter is prettier in a file, but two people adding a part on separate branches both
//! get the same number, and merging them silently collides. Random 64-bit ids cannot collide
//! in practice, cost nothing but slightly uglier files, and keep semantic merge possible
//! without deciding now whether it will ever be built. Fidget's own `Var::new()` takes this
//! exact approach for the same reason.
//!
//! # Why only these things
//!
//! Stable ids are for the *few* things — parts, cages, curves, parameters, nodes. Tens to
//! hundreds per document. The *many* things (vertices, faces: hundreds of thousands) keep
//! plain indices, because nothing outside the mesh ever needs to name one durably. That is
//! what makes stable identity essentially free here.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

macro_rules! stable_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(u64);

        /// Serialized as the same hex string `Display` writes, never as a bare number.
        ///
        /// A random 64-bit id exceeds what JSON's number type is safe for: every reader
        /// built on IEEE doubles — which is every JavaScript one, and so most tools that
        /// would ever inspect this format — silently rounds anything past 2^53. An id that
        /// survives a round trip through a third-party viewer only most of the time is worse
        /// than one that is obviously a string.
        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.collect_str(self)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let text = <std::borrow::Cow<'de, str>>::deserialize(d)?;
                Self::from_hex(&text).ok_or_else(|| {
                    serde::de::Error::custom(format!(
                        "{:?} is not a 16-digit hex id",
                        text.as_ref()
                    ))
                })
            }
        }

        impl $name {
            /// Mints a fresh, effectively-unique id.
            pub fn new() -> Self {
                Self(fresh_u64())
            }

            /// Reconstructs an id from its raw value — for deserialization, never for
            /// inventing one.
            pub fn from_raw(raw: u64) -> Self {
                Self(raw)
            }

            pub fn raw(self) -> u64 {
                self.0
            }

            /// A second id for the *same thing*, the same every time it is asked for.
            ///
            /// **Derived rather than minted, because a copy that stands in for a block is not
            /// a new block.** The viewport draws a consumed block by adding a second,
            /// unconsumed reference to its subtree, and the graph's roots are sorted by id —
            /// so a freshly minted id put that copy in a different position in the drawn list
            /// on every frame. Everything keyed on the position then moved with it: the index
            /// the shader is given to outline the selection pointed at a different body each
            /// frame, and the still's key changed, so the accumulation restarted forever
            /// instead of settling. Reported as the selection outline jittering, and only when
            /// a block was being drawn beside the bodies — which is exactly when a stand-in
            /// exists.
            ///
            /// An involution, so it is its own inverse and two sources cannot land on one
            /// stand-in. Where it sorts is not the point and cannot be chosen; that it sorts
            /// to the *same place twice* is the whole point.
            pub fn stand_in(self) -> Self {
                // "STANDIN" as bytes, which is arbitrary and only has to be fixed.
                Self(self.0 ^ 0x5354_414e_4449_4e00)
            }

            /// Parses the hex form produced by `Display`.
            ///
            /// Scripts and config files need to name an id, and hex is how these print. A
            /// raw integer would work too, but random 64-bit values exceed `i64` and would
            /// arrive as negative numbers through JSON — confusing to read and easy to
            /// mangle. Hex round-trips unambiguously.
            pub fn from_hex(s: &str) -> Option<Self> {
                let raw = u64::from_str_radix(s.trim(), 16).ok()?;
                (raw != 0).then_some(Self(raw))
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{:016x}", self.0)
            }
        }
    };
}

stable_id!(PartId, "Identifies a part. Durable: PLM systems hold these for years.");
stable_id!(CageId, "Identifies a control cage within a document.");
stable_id!(CurveId, "Identifies a curve within a document.");
stable_id!(
    SourceId,
    "A file an import came from, so what it produced can be found again.\n\
     \n\
     The link that makes an import a *block* rather than a one-way door: every mesh a file\n\
     produced carries this, so the block list can group them under one row, a re-import can\n\
     find what to replace, and the settings the file was read with have somewhere to live."
);

stable_id!(MeshId, "Identifies an imported triangle mesh within a document.");
stable_id!(
    QuadMeshId,
    "Identifies an open quad mesh within a document -- a surface to follow, not a solid."
);
stable_id!(ParamId, "Identifies a named parameter within a document.");
stable_id!(
    RelationId,
    "Identifies a relation between two curves (`profile-group-spec.md` §5): one curve \
     following another, one way."
);
stable_id!(
    TessellationId,
    "Identifies a tessellation block: a body of a part, turned into triangles."
);
stable_id!(ExportId, "Identifies an export block: a file written from a tessellation.");
stable_id!(NodeId, "Identifies a node inside one part's composition graph.");

/// Draws a fresh 64-bit value.
///
/// Zero is reserved as a never-issued sentinel so a defaulted or zeroed struct can't be
/// mistaken for a real id.
fn fresh_u64() -> u64 {
    loop {
        let v = rand::random::<u64>();
        if v != 0 {
            return v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ids_do_not_collide() {
        // The property the whole choice rests on. 100k draws is far more than a document
        // will hold, and a collision here would mean the generator is broken.
        let mut seen = HashSet::new();
        for _ in 0..100_000 {
            assert!(seen.insert(PartId::new()), "duplicate id generated");
        }
    }

    #[test]
    fn ids_are_never_zero() {
        for _ in 0..1000 {
            assert_ne!(CageId::new().raw(), 0);
        }
    }

    #[test]
    fn ids_of_different_kinds_do_not_interchange() {
        // Compile-time property, asserted here as documentation: a CageId cannot be passed
        // where a PartId is expected. If these were bare u64s that mistake would compile.
        let part = PartId::new();
        let cage = CageId::new();
        assert_ne!(part.raw(), cage.raw());
    }

    #[test]
    fn ids_round_trip_through_their_hex_form() {
        // How a script or config file names an id.
        for _ in 0..1000 {
            let id = CageId::new();
            let text = id.to_string();
            assert_eq!(CageId::from_hex(&text), Some(id), "hex round trip: {text}");
        }
        assert_eq!(CageId::from_hex("not hex"), None);
        assert_eq!(CageId::from_hex("0"), None, "zero is the reserved sentinel");
    }

    #[test]
    fn ids_round_trip_through_json() {
        let id = PartId::new();
        let json = serde_json::to_string(&id).unwrap();
        let back: PartId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn ids_serialize_as_hex_strings_rather_than_numbers() {
        // The property, not the formatting preference: a bare 64-bit number loses its low
        // bits in any reader backed by doubles, and the loss is silent.
        let id = PartId::from_raw(0xfedc_ba98_7654_3210);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"fedcba9876543210\"");
        assert_eq!(serde_json::from_str::<PartId>(&json).unwrap(), id);
    }

    #[test]
    fn a_malformed_id_is_refused_rather_than_defaulted() {
        // A file naming an id this build cannot parse is a file whose graph would be wired
        // to the wrong things. Failing here is the only honest option.
        assert!(serde_json::from_str::<PartId>("\"nonsense\"").is_err());
        assert!(serde_json::from_str::<PartId>("\"0\"").is_err(), "zero is the sentinel");
        assert!(serde_json::from_str::<PartId>("12345").is_err(), "not a number anymore");
    }

    #[test]
    fn ids_are_usable_as_json_map_keys() {
        // How every collection in the document is written: keyed by id.
        let id = CageId::new();
        let map: std::collections::HashMap<CageId, u32> = [(id, 7)].into_iter().collect();
        let json = serde_json::to_string(&map).unwrap();
        assert_eq!(json, format!("{{\"{id}\":7}}"));
        let back: std::collections::HashMap<CageId, u32> = serde_json::from_str(&json).unwrap();
        assert_eq!(back[&id], 7);
    }
}
