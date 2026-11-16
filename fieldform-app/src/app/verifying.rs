// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Checking a part against what was promised of it, from the panel and from the console.

use crate::*;

impl App {
    /// Everything a verify's answer depends on, in an order that does not vary: the blocks, the
    /// marks, and every number a block might read. Compared, never parsed.
    pub(crate) fn verification_fingerprint(&self) -> String {
        let graph = self.part_graph();
        let mut nodes: Vec<(NodeId, &Node)> = graph.nodes().collect();
        nodes.sort_by_key(|(id, _)| id.raw());
        format!("{nodes:?}|{:?}|{:?}", graph.critical(), named_values(self))
    }

    /// The last verify, if it still describes the part on screen.
    pub(crate) fn current_verification(&self) -> Option<&Verification> {
        let found = self.verification.as_ref()?;
        (found.part == self.active_part && found.fingerprint == self.verification_fingerprint())
            .then_some(found)
    }

    /// What to say beside a blend's radius about the surfaces it may be moving, if anything.
    ///
    /// Nothing at all for a part with no critical marks. With marks: the blame, if a current
    /// verify lays one on this blend; or that nobody has looked, which is not the same as fine.
    pub(crate) fn critical_note(&self, blend: NodeId) -> Option<(bool, String, String)> {
        if self.part_graph().critical().is_empty() {
            return None;
        }
        let Some(found) = self.current_verification() else {
            return Some((
                false,
                "unverified".to_string(),
                "This part has critical surfaces and has changed since they were last measured \
                 — or never was. Type `verify` in the console."
                    .to_string(),
            ));
        };
        let lines: Vec<String> = found
            .reports
            .iter()
            .filter(|r| r.blames(blend))
            .flat_map(|r| {
                let this = self.body_label(r.node);
                r.causes.iter().filter(|c| c.blend == blend).map(move |c| {
                    format!("moves \u{2018}{this}\u{2019} by up to {:.3} mm", c.movement)
                })
            })
            .collect();
        (!lines.is_empty()).then(|| {
            (
                true,
                lines.join("; "),
                format!(
                    "Measured by `verify` on {}: with only this fillet rounded, the critical \
                     surface sits that far from where the hard boolean puts it. Reduce the \
                     radius, or change what it joins.",
                    found.when
                ),
            )
        })
    }

    /// `critical node=<name> [tol=<mm>] [clearance=<mm>]`, `critical node=<name> -off`, or
    /// `critical` alone to list the marks.
    pub(crate) fn critical_from_console(&mut self, args: &Args) {
        let Some(wanted) = args.get("node").map(|v| match v.as_str() {
            Some(text) => text.to_string(),
            // A block called `7` parses as a number on the way in; it is still a name.
            None => v.as_f64().map(|n| n.to_string()).unwrap_or_default(),
        }) else {
            let marks = self.part_graph().critical().to_vec();
            if marks.is_empty() {
                self.say(
                    "No critical surfaces. Say `critical node=<block> tol=0.01` to mark one, \
                     then `verify`.",
                );
            }
            for mark in marks {
                self.say(format!(
                    "critical node={:?} tol={} clearance={}",
                    self.body_label(mark.node),
                    mark.tolerance,
                    self.resolve(mark.clearance),
                ));
            }
            return;
        };
        let Some(node) = self.node_named(&wanted) else {
            self.warn(format!("No block called {wanted:?} in this part"));
            return;
        };
        let mut graph = self.part_graph().clone();
        let off = args.get("off").is_some_and(|v| !matches!(v, Value::Bool(false)));
        if off {
            if !graph.unmark_critical(node) {
                self.warn(format!("{wanted:?} is not marked critical"));
                return;
            }
            self.set_part_graph(graph);
            self.mark_changed();
            // A mark tapers every fillet near it, so taking one off changes the shape drawn.
            self.rebuild_render_resources();
            self.say(format!("{wanted:?} is no longer critical; fillets near it are whole again"));
            return;
        }
        // Said again, a mark keeps what it was not told to change.
        let mut mark = graph
            .critical()
            .iter()
            .find(|m| m.node == node)
            .cloned()
            .unwrap_or_else(|| fieldform_scene::Critical::new(node));
        if let Some(tol) = args.get("tol").or(args.get("tolerance")).and_then(|v| v.as_f64()) {
            if !(tol > 0.0 && tol.is_finite()) {
                self.warn("A tolerance is a positive number of millimetres");
                return;
            }
            mark.tolerance = tol;
        }
        if let Some(c) = args.get("clearance").and_then(|v| v.as_f64()) {
            mark.clearance = ParamRef::Literal(c.max(0.0));
        }
        let (tol, clearance) = (mark.tolerance, self.resolve(mark.clearance));
        graph.mark_critical(mark);
        self.set_part_graph(graph);
        self.mark_changed();
        self.rebuild_render_resources();
        self.say(format!(
            "{wanted:?} is critical: every fillet now thins to nothing {}, and held means within \
             {tol} mm. Unverified until `verify` runs.",
            if clearance > 0.0 {
                format!("{clearance} mm short of it")
            } else {
                "at it".to_string()
            },
        ));
    }

    /// Sweeps the part's field for the fastest it changes and says whether that is what it
    /// promises. A detector rather than a proof: a grid sees a jump only if it is wider than the
    /// grid's own step, which is why the step is said with the answer.
    pub(crate) fn verify_the_promise(&mut self) {
        // The third of the three statements about the part, asking the one rule the other two
        // ask (`no_statement_while_a_slot_is_empty`). Asked of the *part* rather than of the
        // display graph, which is the point: the pruning is what would hide the hole from the
        // sweep.
        if let Some(why) = crate::app::no_statement_while_a_slot_is_empty(
            self.part_graph(),
            |id| self.body_label(id),
            "say whether the field keeps its promise",
        ) {
            self.warn(why);
            return;
        }
        let graph = self.display_graph();
        if graph.roots().is_empty() {
            return;
        }
        let Some((lo, hi)) = fieldform_field::part_extent(&self.doc, &graph).as_box() else {
            self.say("The part reaches to infinity, so there is no box to sweep for a promise.");
            return;
        };
        let tolerance = ((hi - lo).norm() * 1e-3).max(1e-4);
        let field = match GraphField::compile_graph(&self.doc, &graph, tolerance) {
            Ok(field) => field,
            Err(e) => {
                let why = self.explain_compile(&e);
                self.warn(format!("Could not verify the field: {why}"));
                return;
            }
        };
        // A margin, so a surface lying on the box's own face is swept from both sides.
        let margin = (hi - lo) * 0.02;
        let started = std::time::Instant::now();
        let found =
            fieldform_field::find_overstep(&field, (lo - margin, hi + margin), PROMISE_SAMPLES);
        let took = started.elapsed().as_secs_f64();
        let spacing = (hi + margin - (lo - margin)).norm() / PROMISE_SAMPLES as f64;
        match found {
            Some(over) => {
                let at = over.at;
                let body = field
                    .body_at(at, over.moved.max(spacing))
                    .map(|node| format!(" in \u{2018}{}\u{2019}", self.body_label(node)))
                    .unwrap_or_default();
                // **A step and a slope are the same fault and read differently.** Where the
                // refinement closed to nothing and the value still moved, a rate is a number
                // with eleven digits and no meaning; what is worth saying is that the value
                // steps at all. Where it settled at a finite span, the rate is the thing.
                let what = if over.across <= spacing * 1e-4 {
                    format!("steps {:.3} mm across nothing at all", over.moved)
                } else {
                    format!(
                        "changes {:.2} mm per mm ({:.3} mm across {:.3} mm)",
                        over.rate, over.moved, over.across
                    )
                };
                self.warn(format!(
                    "The field {what} at ({:.1}, {:.1}, {:.1}){body}, where it promises {:.2} \
                     per mm. A mesh of it has holes there and rays step through it: the mesher \
                     cuts cells away by that promise. Looked in {took:.1} s.",
                    at.x, at.y, at.z, over.allowed,
                ));
            }
            None => self.say(format!(
                "The field keeps its promise: nothing changes faster than it says, from a grid \
                 of {spacing:.1} mm in {took:.1} s. A fault between two of those points would \
                 not be seen.",
            )),
        }
    }

    /// `verify`: measures every critical surface of the active part and says what it found, in
    /// the sentences the panel and a transcript share.
    pub(crate) fn verify_from_console(&mut self) {
        // **The field's own promise first, because nothing else is worth checking if it is
        // broken.** Every consumer believes `lipschitz`: the mesher prunes a cell by it and the
        // tracer steps by it. A part that breaks it meshes with holes nowhere near anything that
        // looks wrong — which is how a conformal lattice on a spiked surface failed, at 2.86 mm
        // per millimetre where a distance promises 1.0. It takes no marks and applies to every
        // part, so it runs whether or not anything is critical.
        self.verify_the_promise();
        if self.part_graph().critical().is_empty() {
            self.say(
                "Nothing more to verify: no surface is marked critical. `critical node=<block>`.",
            );
            return;
        }
        let started = std::time::Instant::now();
        let reports =
            match fieldform_field::verify_critical(&self.doc, self.part_graph(), 1e-3) {
                Ok(reports) => reports,
                Err(e) => {
                    // Through `explain_compile`, because this refusal reaches a person like
                    // every other one: an empty slot's own message carries the holed block's
                    // id, and a raw id names nothing anybody can go and look at.
                    let why = self.explain_compile(&e);
                    self.warn(format!("Could not verify: {why}"));
                    return;
                }
            };
        let when = utc_date(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        );
        let mut lines: Vec<(bool, String)> = Vec::new();
        for report in &reports {
            let fine = report.verdict() == fieldform_field::Verdict::Held;
            for sentence in report.sentences(&|id| self.body_label(id)) {
                lines.push((fine, sentence));
            }
        }
        let held = reports
            .iter()
            .filter(|r| r.verdict() == fieldform_field::Verdict::Held)
            .count();
        for (fine, sentence) in lines {
            if fine { self.say(sentence) } else { self.warn(sentence) }
        }
        // What the marks are costing: the same surfaces with the protection taken off. Said
        // only when it is something, and as information — the part on screen is the one above.
        if let Ok(bare) = fieldform_field::verify_unprotected(&self.doc, self.part_graph(), 1e-3) {
            for report in bare.iter().filter(|r| r.verdict() == fieldform_field::Verdict::Moved) {
                for cause in &report.causes {
                    self.say(format!(
                        "Unprotected, blend \u{2018}{}\u{2019} would move \u{2018}{}\u{2019} by up \
                         to {:.3} mm; its fillet is given up there instead.",
                        self.body_label(cause.blend),
                        self.body_label(report.node),
                        cause.movement,
                    ));
                }
            }
        }
        self.say(format!(
            "Verified {when}: {held} of {} held, in {:.1} s. Any edit makes this stale.",
            reports.len(),
            started.elapsed().as_secs_f64(),
        ));
        self.verification = Some(Verification {
            part: self.active_part,
            fingerprint: self.verification_fingerprint(),
            when,
            reports,
        });
    }
}
