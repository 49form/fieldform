// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The STEP importer, as a program of its own.
//!
//! # Why this is a separate binary and not a flag on the application
//!
//! Two reasons, and only the first was true when the worker was introduced.
//!
//! **It must be able to die.** OpenCASCADE reads files it did not write and reports some
//! failures by *aborting*. `catch_unwind` cannot catch an abort — nothing in-process can — so a
//! thread that hits one takes the window and the unsaved document with it. A separate process
//! turns that into an exit code, and the worst case becomes an error message about one file.
//!
//! **It carries OpenCASCADE so the application does not.** OCCT is statically linked and
//! measures 29.4 MB of an executable. While the worker was the application re-running itself
//! with a flag, the application had to contain it: 45.8 MB, of which 64% was a CAD kernel most
//! sessions never touch. Split out, `fieldform` is ~16 MB and this is ~29 MB, and the heavy
//! half is a component that can be shipped separately, fetched on demand, or left out entirely
//! for anyone who never opens a STEP file.
//!
//! Nothing about the protocol changed in the split. The two halves talk the way they already
//! did: positional arguments in, a `mesh_wire` file out, counts on stdout, trouble on stderr,
//! and the exit code carrying whether any of it worked.

/// Reads a STEP file and writes its solids to `output` as a `mesh_wire` payload.
///
/// Arguments are `key=value`, which this interface earned when it grew past two settings: five
/// positional numbers are five chances to transpose a pair silently, and a wrong pair that
/// looks reasonable is the failure this whole import path keeps guarding against.
fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // `key=value`, not positional. There are five settings now and they are all numbers or
    // switches, so a positional list would be five chances to transpose two of them silently —
    // and the pair that looks reasonable and is wrong is exactly what this format prevents.
    let mut fields: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    let mut input: Option<&str> = None;
    let mut output: Option<&str> = None;
    for a in &args {
        match a.split_once('=') {
            Some(("in", v)) => input = Some(v),
            Some(("out", v)) => output = Some(v),
            Some((k, v)) => {
                fields.insert(k, v);
            }
            None => {}
        }
    }
    let (Some(input), Some(output)) = (input, output) else {
        eprintln!(
            "usage: fieldform-import in=<file> out=<file> [linear=mm] [angular=deg] \
             [sew=mm] [fix=true|false] [parallel=true|false]"
        );
        return std::process::ExitCode::from(2);
    };

    // Each falls back to the default rather than refusing. A value this cannot parse means the
    // caller and this program disagree about the protocol, and importing at the default is a
    // better answer to that than importing nothing.
    let base = fieldform_io::StepOptions::default();
    let num = |k: &str, fallback: f64| {
        fields.get(k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(fallback)
    };
    let flag = |k: &str, fallback: bool| {
        fields.get(k).and_then(|v| v.parse::<bool>().ok()).unwrap_or(fallback)
    };
    let options = fieldform_io::StepOptions {
        linear_deflection_mm: num("linear", base.linear_deflection_mm),
        angular_deflection_deg: num("angular", base.angular_deflection_deg),
        sew_tolerance_mm: num("sew", base.sew_tolerance_mm),
        shape_fix: flag("fix", base.shape_fix),
        parallel: flag("parallel", base.parallel),
    };

    match fieldform_io::import_step_with(input.as_ref(), "imported", options) {
        Ok((meshes, curves, report)) => {
            let Ok(file) = std::fs::File::create(output) else {
                eprintln!("could not write the result");
                return std::process::ExitCode::from(3);
            };
            let mut file = std::io::BufWriter::new(file);
            if fieldform_io::mesh_wire::write_payload(&mut file, &meshes, &curves).is_err() {
                eprintln!("could not write the result");
                return std::process::ExitCode::from(3);
            }
            // The report goes back on stdout: it is small, and a second file to clean up for a
            // handful of counts is not worth it.
            println!(
                "{} {} {}",
                report.solids_found,
                report.reversed,
                report.unconverted.values().sum::<usize>(),
            );
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::from(1)
        }
    }
}
