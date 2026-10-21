// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The block list's rows, and the panels of its blocks.
//!
//! `list.rs` is the rows: what a row is, how rows nest, the drop gutter between them and the
//! dropdown a picker can be typed into. Every other module here is one block panel, lifted out
//! as `node_block_controls` sheds its larger arms; see `lattice.rs` for why the arms move at
//! all. The smaller arms stay in `app/block_controls.rs`.

pub(crate) mod attract;
pub(crate) mod boolean;
pub(crate) mod extrude;
pub(crate) mod lattice;
pub(crate) mod list;
pub(crate) mod loft;
pub(crate) mod morph;
/// Not a block panel: the operand row every block panel draws its slots with.
pub(crate) mod operands;
pub(crate) mod pattern;
