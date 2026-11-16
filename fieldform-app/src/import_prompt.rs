// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! What an import asks before it reads a file, and what it reports having made.

use crate::*;

/// Which translator a file needs, and so which options it has.
///
/// Read from the extension rather than asked for. The file already says what it is, and a
/// format picker that can disagree with the file is a second description of one fact — the
/// shape of bug this codebase keeps meeting. What the dialog asks for is the part the file
/// genuinely does *not* carry.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ImportFormat {
    Step,
    Obj,
    Iges,
}

impl ImportFormat {
    pub(crate) fn of(path: &std::path::Path) -> Option<Self> {
        match path.extension()?.to_string_lossy().to_lowercase().as_str() {
            "stp" | "step" => Some(Self::Step),
            "obj" => Some(Self::Obj),
            "igs" | "iges" => Some(Self::Iges),
            _ => None,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Step => "STEP",
            Self::Obj => "Wavefront OBJ",
            Self::Iges => "IGES",
        }
    }
}

/// A file chosen for import, waiting on its options.
pub(crate) struct ImportPrompt {
    pub(crate) path: std::path::PathBuf,
    pub(crate) format: ImportFormat,
}

/// The importer's file name, without the platform's executable suffix.
///
/// A separate program rather than this one re-invoked with a flag, and the reason is size as
/// much as safety. OpenCASCADE is statically linked and measures 29.4 MB; while the worker was
/// this binary under another name, every copy of the application carried a CAD kernel most
/// sessions never open. Apart, `fieldform` is ~16 MB and the importer is the component that can
/// be shipped separately, fetched on demand, or left out for anyone who never opens a STEP
/// file.
///
/// The safety half stands unchanged: OCCT reports some failures by *aborting*, which no
/// `catch_unwind` can catch, so the import has to be something allowed to die on its own.
pub(crate) const IMPORT_WORKER: &str = "fieldform-import";

/// Where the importer is, if it is anywhere.
///
/// Beside this executable, which is where a build puts it and where an installer would. Not
/// searched for on `PATH` deliberately: picking up a `fieldform-import` from elsewhere on the
/// system would run some other build's importer against this build's `mesh_wire` format.
pub(crate) fn import_worker_path() -> Option<std::path::PathBuf> {
    let beside = std::env::current_exe()
        .ok()?
        .parent()?
        .join(format!("{IMPORT_WORKER}{}", std::env::consts::EXE_SUFFIX));
    beside.is_file().then_some(beside)
}

/// An import running in another process, and what the application needs to finish it.
pub(crate) struct PendingImport {
    pub(crate) name: String,
    /// The file being read, kept so the import it produces can name and re-read it.
    pub(crate) path: std::path::PathBuf,
    /// The import this is a new revision of, when it is one.
    ///
    /// A re-import replaces what an earlier read produced *in place* rather than adding a
    /// second copy beside it, which is the whole point of an import being a block: point it at
    /// a newer file and everything built on it re-evaluates.
    pub(crate) replacing: Option<fieldform_scene::id::SourceId>,
    /// Where the worker will write the mesh. Removed when the import finishes, however it
    /// finishes — a cancelled or crashed import must not leave a file behind.
    pub(crate) output: std::path::PathBuf,
    pub(crate) child: std::process::Child,
    pub(crate) started: std::time::Instant,
}

/// What an OBJ is taken to be, which the file itself cannot say.
///
/// **Three readings of the same triangles and quads, and they are not variations on each
/// other.** A cage is scaffolding for a subdivision surface and has to be closed; a model is
/// finished geometry to reference; a surface is an *open* sheet that nothing draws at all and
/// only a conformal lattice names. One control, because it is one question -- what is this
/// file -- and a second checkbox beside the first would let two answers be given at once.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ObjReading {
    Cage,
    Model,
    Surface,
}

impl ObjReading {
    /// In the order they are offered, with what each one means. The strings live here rather
    /// than in the panel so the list and its explanations cannot come apart.
    pub(crate) const ALL: [(ObjReading, &'static str, &'static str); 3] = [
        (
            ObjReading::Cage,
            "Subdivision cage",
            "The file is a control mesh and the body is the smooth surface it describes, \
             which you can then push and pull. Needs the file to be closed, manifold and all \
             quads.",
        ),
        (
            ObjReading::Model,
            "Model",
            "The triangles are the geometry. Takes any OBJ, including one with holes in it, \
             and the result is referenced rather than edited.",
        ),
        (
            ObjReading::Surface,
            "Surface to follow",
            "An open quad sheet -- a panel or a shell -- that a conformal lattice lays its \
             cells over. Nothing draws it and it is not a body; it is geometry for another \
             block to reference. Quads, but open where a cage may not be.",
        ),
    ];
}

/// What an import brought in, said under its name — a count per kind, and nothing for a kind
/// it did not bring.
pub(crate) fn import_contents(bodies: usize, surfaces: usize, curves: usize) -> String {
    let plural = |n: usize, one: &str, many: &str| format!("{n} {}", if n == 1 { one } else { many });
    let mut parts = Vec::new();
    if bodies > 0 {
        parts.push(plural(bodies, "body", "bodies"));
    }
    if surfaces > 0 {
        parts.push(plural(surfaces, "surface", "surfaces"));
    }
    if curves > 0 {
        parts.push(plural(curves, "curve", "curves"));
    }
    if parts.is_empty() {
        "nothing left from this import".to_string()
    } else {
        parts.join(", ")
    }
}

/// What an import that did not make meshes produced, for [`App::record_import`].
#[derive(Default)]
pub(crate) struct ImportMade {
    pub(crate) cages: Vec<CageId>,
    pub(crate) surfaces: Vec<fieldform_scene::QuadMeshId>,
    pub(crate) curves: Vec<CurveId>,
}

pub(crate) fn made_cages(cages: Vec<CageId>) -> ImportMade {
    ImportMade { cages, ..Default::default() }
}

pub(crate) fn made_surfaces(surfaces: Vec<fieldform_scene::QuadMeshId>) -> ImportMade {
    ImportMade { surfaces, ..Default::default() }
}

pub(crate) fn made_curves(curves: Vec<CurveId>) -> ImportMade {
    ImportMade { curves, ..Default::default() }
}

/// The STEP reader's own parameters, wherever they are being edited.
///
/// Shared between the import dialog and an import block, because they are the same settings:
/// the dialog says how to read a file, and the block says how one *was* read and re-reads it.
/// Two copies of these controls would drift, and the pair that drifted would disagree about
/// what a document had been built from.
pub(crate) fn step_option_controls(ui: &mut egui::Ui, o: &mut fieldform_io::StepOptions) {
        // Presets first, as buttons that *fill in* the fields below rather than
        // as a mode of their own. A preset that were its own state would be a
        // second description of the settings, and the two would disagree the
        // moment a number was nudged.
        ui.horizontal(|ui| {
            ui.label("Preset:");
            let current = o.preset_name();
            for (name, preset) in fieldform_io::StepOptions::presets() {
                if ui.selectable_label(current == Some(name), name).clicked() {
                    *o = preset;
                }
            }
        })
        .response
        .on_hover_text(
            "Fills in the settings below. Change any of them and no preset is \
             shown, because none of them describes what you have.",
        );
        ui.separator();

        let o = &mut *o;
        ui.horizontal(|ui| {
            ui.label("Linear deflection:");
            ui.add(
                egui::DragValue::new(&mut o.linear_deflection_mm)
                    .speed(0.005).max_decimals(4)
                    .range(0.001..=10.0)
                    .suffix(" mm"),
            );
        })
        .response
        .on_hover_text(
            "How far a facet may sit from the true surface. Finer costs \
             triangles and time.",
        );
        ui.horizontal(|ui| {
            ui.label("Angular deflection:");
            ui.add(
                egui::DragValue::new(&mut o.angular_deflection_deg)
                    .speed(0.25).max_decimals(4)
                    .range(3.0..=60.0)
                    .suffix("\u{b0}"),
            );
        })
        .response
        .on_hover_text(
            "How far a facet's normal may turn from its neighbour's. This is \
             what faceting on a curved face actually is — OCCT's own \
             default is 28.6\u{b0}. Below about 3\u{b0} the mesher starts \
             emitting slivers and the result gets worse, not better.",
        );
        ui.horizontal(|ui| {
            ui.label("Sewing tolerance:");
            ui.add(
                egui::DragValue::new(&mut o.sew_tolerance_mm)
                    .speed(0.005).max_decimals(4)
                    .range(0.0..=5.0)
                    .suffix(" mm"),
            );
        })
        .response
        .on_hover_text(
            "Joins faces that meet but are not joined, which is what makes an \
             exported part a solid. Zero takes the file at its word. Too large \
             welds genuinely separate faces together and can lose a thin wall.",
        );
        ui.checkbox(&mut o.shape_fix, "Repair before sewing")
            .on_hover_text(
                "Runs OCCT's general repair pass first — wires out of \
                 order, edges that do not meet their curves, faces the wrong \
                 way round. A different kind of damage from an unsewn seam.",
            );
        ui.checkbox(&mut o.parallel, "Mesh faces in parallel")
            .on_hover_text(
                "Faces tessellate independently, so this is close to free. The \
                 first thing to turn off if one particular file misbehaves.",
            );
}
