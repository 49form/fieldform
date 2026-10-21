// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! What the march does, counted: steps per ray, and how wide one reading of the field is.
//!
//! Only under the `march-stats` cargo feature. The investigation at `482ef38` found the march
//! to be over 99% of every frame and could not say whether that was many steps or dear steps;
//! this is the instrument that says. Nothing here is on any frame's path — a capture is asked
//! for, traced by a copy of the primary march that counts (`march_stats.wgsl`), and read back.

use std::collections::BTreeMap;
use std::fmt::Write;

/// Why a ray stopped, in the order the shader numbers them.
pub(crate) const EXITS: [&str; 4] = [
    "missed the scene's bounding sphere",
    "converged on a surface",
    "passed the far plane",
    "ran out of steps",
];

/// One captured frame: for every pixel, how many passes of the primary march's loop its ray
/// made and why it stopped.
pub struct MarchCapture {
    pub width: u32,
    pub height: u32,
    /// The step budget the frame was traced with — 512 for a field that reports true distance,
    /// more where a divided field bought itself steps (`traced/mod.rs`, `STEP_BUDGET`).
    pub cap: u32,
    pub steps: Vec<u32>,
    pub exits: Vec<u32>,
    /// Calls of a pattern's sub-field along each ray's counted readings.
    pub calls: Vec<u32>,
}

/// Mean, median, 95th and 99th percentile and maximum of a sorted list.
fn spread(sorted: &[u32]) -> Option<(f64, u32, u32, u32, u32)> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    let at = |q: f64| sorted[(((n - 1) as f64) * q).round() as usize];
    let mean = sorted.iter().map(|&s| f64::from(s)).sum::<f64>() / n as f64;
    Some((mean, at(0.5), at(0.95), at(0.99), sorted[n - 1]))
}

impl MarchCapture {
    /// The steps of every ray that stopped for reason `class`, sorted.
    pub fn class(&self, class: u32) -> Vec<u32> {
        let mut out: Vec<u32> = self
            .steps
            .iter()
            .zip(&self.exits)
            .filter(|(_, e)| **e == class)
            .map(|(s, _)| *s)
            .collect();
        out.sort_unstable();
        out
    }

    /// The capture as the tables the report wants. Classes are kept apart throughout: a scene
    /// averaging forty steps because most rays miss is a different problem from one averaging
    /// forty with every ray converging, and one mean would hide both.
    pub fn report(&self) -> String {
        let total = self.steps.len().max(1);
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{} × {} = {} rays, step cap {}\n",
            self.width,
            self.height,
            self.steps.len(),
            self.cap
        );
        let _ = writeln!(out, "| exit | rays | share | mean | median | p95 | p99 | max |");
        let _ = writeln!(out, "|---|---|---|---|---|---|---|---|");
        for (class, name) in EXITS.iter().enumerate() {
            let steps = self.class(class as u32);
            let share = 100.0 * steps.len() as f64 / total as f64;
            match spread(&steps) {
                Some((mean, median, p95, p99, max)) if class > 0 => {
                    let _ = writeln!(
                        out,
                        "| {class} {name} | {} | {share:.2}% | {mean:.1} | {median} | {p95} | {p99} | {max} |",
                        steps.len()
                    );
                }
                _ => {
                    let _ = writeln!(out, "| {class} {name} | {} | {share:.2}% | — | — | — | — | — |", steps.len());
                }
            }
        }
        let entered: Vec<u32> = {
            let mut all: Vec<u32> =
                self.steps.iter().zip(&self.exits).filter(|(_, e)| **e != 0).map(|(s, _)| *s).collect();
            all.sort_unstable();
            all
        };
        let field_reads: u64 = self.steps.iter().map(|&s| u64::from(s)).sum();
        let _ = writeln!(
            out,
            "\n{} field readings in the primary march, {:.1} a ray over every ray and {:.1} over the rays that entered the loop.",
            field_reads,
            field_reads as f64 / total as f64,
            field_reads as f64 / entered.len().max(1) as f64,
        );
        let sub_calls: u64 = self.calls.iter().map(|&c| u64::from(c)).sum();
        let _ = writeln!(
            out,
            "\n{} calls of a pattern's sub-field in those readings: **{:.2} a reading**.",
            sub_calls,
            sub_calls as f64 / field_reads.max(1) as f64,
        );
        for class in [1u32, 3] {
            let steps = self.class(class);
            if steps.is_empty() {
                continue;
            }
            let _ = writeln!(out, "\nSteps of the rays that {} , in buckets of 16:\n", EXITS[class as usize]);
            let _ = writeln!(out, "| steps | rays | share of the class |");
            let _ = writeln!(out, "|---|---|---|");
            let mut buckets: BTreeMap<u32, usize> = BTreeMap::new();
            for &s in &steps {
                // Everything past 512 in one bucket, so a raised cap does not lengthen the table.
                let b = if s > 512 { 512 } else { (s.saturating_sub(1) / 16) * 16 };
                *buckets.entry(b).or_default() += 1;
            }
            for (b, n) in buckets {
                let label = if b >= 512 { "over 512".to_string() } else { format!("{}–{}", b + 1, b + 16) };
                let _ = writeln!(out, "| {label} | {n} | {:.2}% |", 100.0 * n as f64 / steps.len() as f64);
            }
        }
        out
    }
}

/// What one generated value function is made of.
pub(crate) struct FunctionCounts {
    pub name: String,
    pub lines: usize,
    /// Binary `+ - * /` as the generator spaces them.
    pub operators: usize,
    /// Every call site, constructors and builtins included.
    pub calls: usize,
    /// Call sites of the prelude's own distance functions, by name: each is one primitive
    /// evaluated per reading of this function.
    pub primitives: BTreeMap<String, usize>,
    /// Calls into a pattern's subtree function, which a loop makes once per copy visited.
    pub pattern_calls: BTreeMap<String, usize>,
}

/// The identifier before each `(` in a line of generated source.
fn call_names(line: &str) -> Vec<&str> {
    let bytes = line.as_bytes();
    let mut out = Vec::new();
    for (i, &b) in bytes.iter().enumerate() {
        if b != b'(' {
            continue;
        }
        let mut start = i;
        while start > 0 && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_') {
            start -= 1;
        }
        if start < i && !bytes[start].is_ascii_digit() {
            out.push(&line[start..i]);
        }
    }
    out
}

/// Counts for every value-pass function the generator emitted: one `body_N` per body and one
/// `pattern_sub_N` per patterned subtree. The normal pass's twins are left out — they are the
/// same composition again with a gradient beside each value, and run once per hit pixel rather
/// than once per step.
pub(crate) fn function_counts(source: &str) -> Vec<FunctionCounts> {
    let mut out = Vec::new();
    let mut current: Option<FunctionCounts> = None;
    for line in source.lines() {
        if let Some(rest) = line.strip_prefix("fn ") {
            let name = rest.split('(').next().unwrap_or("").to_string();
            let value_pass = (name.starts_with("body_") && !name.starts_with("body_normal_"))
                || name.starts_with("pattern_sub_");
            current = value_pass.then(|| FunctionCounts {
                name,
                lines: 0,
                operators: 0,
                calls: 0,
                primitives: BTreeMap::new(),
                pattern_calls: BTreeMap::new(),
            });
            continue;
        }
        if line.starts_with('}') {
            if let Some(done) = current.take() {
                out.push(done);
            }
            continue;
        }
        let Some(f) = current.as_mut() else { continue };
        let code = line.split("//").next().unwrap_or("");
        if code.trim().is_empty() {
            continue;
        }
        f.lines += 1;
        f.operators += [" + ", " - ", " * ", " / "].iter().map(|op| code.matches(op).count()).sum::<usize>();
        for name in call_names(code) {
            if matches!(name, "if" | "for" | "while" | "switch" | "return") {
                continue;
            }
            f.calls += 1;
            if name.starts_with("pattern_sub_") {
                *f.pattern_calls.entry(name.to_string()).or_default() += 1;
            } else if (name.ends_with("_distance") && name != "box_distance") || name == "sample_baked" {
                *f.primitives.entry(name.to_string()).or_default() += 1;
            }
        }
    }
    out
}

/// The static counts as a table, for the report.
pub fn function_report(source: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "| function | lines | operators | call sites | primitives evaluated | pattern calls |");
    let _ = writeln!(out, "|---|---|---|---|---|---|");
    for f in function_counts(source) {
        let list = |m: &BTreeMap<String, usize>| {
            if m.is_empty() {
                "—".to_string()
            } else {
                m.iter().map(|(k, n)| format!("{n} × `{k}`")).collect::<Vec<_>>().join(", ")
            }
        };
        let _ = writeln!(
            out,
            "| `{}` | {} | {} | {} | {} | {} |",
            f.name,
            f.lines,
            f.operators,
            f.calls,
            list(&f.primitives),
            list(&f.pattern_calls)
        );
    }
    let _ = writeln!(out, "\nWhole source: {} characters, {} lines.", source.len(), source.lines().count());
    out
}

/// How many copies of a pattern one reading of the field visits, read off the loops the
/// generator emitted, with their bounds resolved against the parameter values of the capture.
///
/// Two forms exist (`PatternLoop::walk` in codegen.rs). A window wide enough to cover every
/// instance walks the instances: `for (var s = i32(params[first]); s < i32(params[count]); …)`,
/// a count fixed by two parameters. Otherwise it walks a window about the instance nearest the
/// sample, `for (var s = -r; s <= r; …)` with `r = i32(params[reach])`, and drops the steps
/// whose instance falls outside `[first, count)` — so the visits are at most `2r + 1` and fewer
/// near the ends of a row, which is a count that **varies with where the sample is**.
pub fn pattern_widths(source: &str, params: &[f32]) -> String {
    let value = |text: &str| -> Option<f32> {
        let inner = text.split("params[").nth(1)?.split(']').next()?;
        params.get(inner.parse::<usize>().ok()?).copied()
    };
    let mut out = String::new();
    for line in source.lines() {
        if let Some(said) = line.trim().strip_prefix("// pattern ") {
            let _ = writeln!(out, "- the generator says of pattern {said}");
        }
    }
    let mut function = String::new();
    let mut reach: BTreeMap<String, f32> = BTreeMap::new();
    let lines: Vec<&str> = source.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if let Some(rest) = line.strip_prefix("fn ") {
            function = rest.split('(').next().unwrap_or("").to_string();
        }
        if function.starts_with("body_normal_") || function.starts_with("pattern_subn_") || function == "field_normal" {
            continue;
        }
        let code = line.trim();
        // `let r3 = i32(params[17]);`
        if let Some(rest) = code.strip_prefix("let r") {
            if let (Some(tag), Some(v)) = (rest.split(' ').next(), value(code)) {
                if code.contains("= i32(params[") {
                    reach.insert(tag.to_string(), v);
                }
            }
        }
        // The generator's own loops are `s` and a number; the march's `step` loop is not one.
        let is_pattern_loop = code
            .strip_prefix("for (var s")
            .and_then(|rest| rest.chars().next())
            .is_some_and(|c| c.is_ascii_digit());
        if !is_pattern_loop {
            continue;
        }
        if code.contains("= i32(params[") {
            // The whole ring or row, by index.
            let mut bounds = code.split("params[").skip(1).filter_map(|t| {
                params.get(t.split(']').next()?.parse::<usize>().ok()?).copied()
            });
            if let (Some(first), Some(count)) = (bounds.next(), bounds.next()) {
                let _ = writeln!(
                    out,
                    "- `{function}`: walks every instance, `{first:.0} ≤ s < {count:.0}` — **{:.0} copies a reading, fixed** by two parameters, the same wherever the sample is.",
                    count - first
                );
            }
        } else if code.contains("< pc") || code.contains("* pc") {
            // The pruned walk, nearest instance first. How many it calls is no longer something
            // the source can say: it is every instance whose floor is under the running minimum,
            // which is what the capture's sub-field calls a reading measures.
            let count = lines[..i].iter().rev().take(3).find_map(|l| l.contains("let pc").then(|| value(l)).flatten());
            let _ = writeln!(
                out,
                "- `{function}`: walks every instance nearest first, and calls the sub-field only where its floor is under the running minimum — **at most {} copies a reading, varying with the sample's position**; the capture counts how many.",
                count.map_or("?".into(), |v| format!("{v:.0}")),
            );
        } else if let Some(tag) = code.strip_prefix("for (var s").and_then(|r| r.split(' ').next()) {
            let r = reach.get(tag).copied().unwrap_or(f32::NAN);
            // The drop test sits on one of the next lines.
            let drop = lines[i + 1..(i + 5).min(lines.len())].iter().find(|l| l.contains("continue"));
            let (first, count) = drop
                .map(|l| {
                    let mut b = l.split("params[").skip(1).filter_map(|t| {
                        params.get(t.split(']').next()?.parse::<usize>().ok()?).copied()
                    });
                    (b.next(), b.next())
                })
                .unwrap_or((None, None));
            let _ = writeln!(
                out,
                "- `{function}`: a window of `-{r:.0} ≤ s ≤ {r:.0}` about the nearest instance — **up to {:.0} copies a reading, varying with the sample's position**: steps whose instance falls outside `[{}, {})` are dropped, so a sample near an end of the row visits fewer.",
                2.0 * r + 1.0,
                first.map_or("?".into(), |v| format!("{v:.0}")),
                count.map_or("?".into(), |v| format!("{v:.0}")),
            );
        }
    }
    if out.is_empty() {
        out.push_str("- no pattern loop in this source: one reading of the field is one pass over each body.\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Classes are counted apart, and a ray that never entered the loop has no steps to average.
    #[test]
    fn a_capture_keeps_its_exit_classes_apart() {
        let capture = MarchCapture {
            width: 4,
            height: 2,
            cap: 512,
            steps: vec![0, 0, 10, 20, 30, 512, 7, 9],
            exits: vec![0, 0, 1, 1, 1, 3, 2, 2],
            calls: vec![0, 0, 20, 40, 60, 1024, 14, 18],
        };
        assert_eq!(capture.class(1), vec![10, 20, 30]);
        assert_eq!(capture.class(3), vec![512]);
        assert_eq!(spread(&capture.class(1)), Some((20.0, 20, 30, 30, 30)));
        let report = capture.report();
        assert!(report.contains("| 1 converged on a surface | 3 | 37.50% | 20.0 | 20 | 30 | 30 | 30 |"), "{report}");
        assert!(report.contains("| 0 missed the scene's bounding sphere | 2 | 25.00% | — |"), "{report}");
        assert!(report.contains("| 3 ran out of steps | 1 | 12.50% | 512.0 |"), "{report}");
        // 1176 calls over 588 readings.
        assert!(report.contains("**2.00 a reading**"), "{report}");
    }

    /// The counts read a body function and leave its normal twin out.
    #[test]
    fn the_static_counts_read_value_functions_and_not_their_normal_twins() {
        let source = "fn body_0(p: vec3<f32>) -> f32 {\n    let p0 = p;\n    let n1 = sphere_distance(p0, vec3<f32>(params[0], params[1], params[2]), params[3]);\n    let n2 = n1 - params[4] * 2.0;\n    return n2;\n}\n\nfn body_normal_0(p: vec3<f32>) -> vec4<f32> {\n    let v1 = sphere_distance(p, vec3<f32>(0.0), 1.0);\n    return vec4<f32>(v1);\n}\n";
        let counts = function_counts(source);
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[0].name, "body_0");
        assert_eq!(counts[0].lines, 4);
        assert_eq!(counts[0].operators, 2);
        assert_eq!(counts[0].primitives.get("sphere_distance"), Some(&1));
    }
}
