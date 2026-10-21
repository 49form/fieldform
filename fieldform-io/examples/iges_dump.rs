// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! Reports what an IGES file contains, as fieldform sees it.
//!
//! A diagnostic rather than a test: point it at a file from any CAD system and it says what
//! came through, what did not, and the actual geometry of each curve — so a file that imports
//! wrongly can be diagnosed without a debugger.
//!
//! Run with: `cargo run --release -p fieldform-io --example iges_dump -- <file.igs>`

use std::path::PathBuf;

fn main() {
    let Some(path) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: iges_dump <file.igs>");
        std::process::exit(2);
    };

    let import = match fieldform_io::import_iges(&path) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            std::process::exit(1);
        }
    };

    println!("{}", path.display());
    println!("{}", import.summary());
    println!("units scale to mm: {}", import.scale_to_mm);
    println!();

    for c in &import.curves {
        let (lo, hi) = c.domain();
        let start = c.evaluate(lo);
        let end = c.evaluate(hi);
        // 0.01 mm is finer than any display need and far finer than most machining, so the
        // length here is a real measurement rather than a chord-count artefact.
        let length = c.length(0.01);

        println!("{}", c.name);
        println!("  degree      {}", c.degree());
        println!(
            "  control     {} points{}",
            c.control_points().len(),
            if c.is_rational() {
                " (rational)"
            } else {
                " (polynomial)"
            }
        );
        println!("  domain      [{lo}, {hi}]");
        println!("  start       ({:.4}, {:.4}, {:.4}) mm", start.x, start.y, start.z);
        println!("  end         ({:.4}, {:.4}, {:.4}) mm", end.x, end.y, end.z);
        println!("  end-to-end  {:.4} mm", (end - start).norm());
        println!("  arc length  {length:.4} mm");
        println!("  closed      {}", c.is_closed());
        println!(
            "  polyline    {} segments at 0.01 mm tolerance",
            c.tessellate(0.01).len() - 1
        );
        println!();
    }
}
