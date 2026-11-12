// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The console: the line typed, what it says back, and the scripts and checks run through it.

use crate::*;

impl App {
    /// The console, along the bottom of the panel.
    ///
    /// `backlog.md` describes this as a transparent overlay at the bottom-left of the
    /// viewport; it is here instead, under the blocks, because that is where it was asked for
    /// and because a command that adds a block reads better directly beneath the list it adds
    /// to.
    pub(crate) fn console(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label(egui::RichText::new("Console").weak());

        // **The input first, then the log fills what is left.** Laid out bottom-up so the entry
        // field keeps its place against the panel's lower edge while the log grows and shrinks
        // with the drag; a top-down layout would put a fixed-height log above a field that
        // wandered.
        let response = ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.console_input)
                    .id(CONSOLE_INPUT_ID.with("console"))
                    .desired_width(f32::INFINITY)
                    .hint_text("command")
                    .font(egui::TextStyle::Monospace),
            );

            // **Clamped to what is actually left.** `auto_shrink` false keeps the log filling
            // the panel, but filling is not the same as fitting: dragged to its smallest the
            // panel has almost nothing left once the heading and the entry field have taken
            // theirs, and a scroll area asked to fill a space smaller than one line of text
            // spilled over the field instead of scrolling inside what it had.
            //
            // **And the floor clamped too**, which the cap alone did not do: a scroll area is never
            // shorter than egui's 64-point minimum unless told otherwise, so with the panel at its
            // smallest the log kept its 64 points and was drawn over the entry field, the lines
            // and the typing on top of each other (reported 2026-11-07).
            let room = ui.available_height().max(0.0);
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .auto_shrink([false, false])
                .max_height(room)
                .min_scrolled_height(room)
                .id_salt("console_log")
                .show(ui, |ui| {
                    ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                        // **Wrapped here, though the panel truncates everywhere else.** The
                        // panel's rule exists because a label carrying a live number — a curve's
                        // length while a control point is dragged — would move the divider once
                        // per digit. A console line is written once and never changes, so it
                        // cannot do that, and truncating it loses the end of the sentence, which
                        // for a log is most of the point of having kept it.
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                        for line in &self.console_log {
                            let shown = if line.said > 1 {
                                format!("{}  \u{00d7} {}", line.text, line.said)
                            } else {
                                line.text.clone()
                            };
                            let text = egui::RichText::new(shown).monospace();
                            if line.ok {
                                ui.label(text.weak());
                            } else {
                                ui.label(text.color(ui.visuals().warn_fg_color));
                            }
                        }
                        // The one line that is rewritten rather than added to.
                        if let Some(doing) = &self.console_progress {
                            ui.label(
                                egui::RichText::new(doing)
                                    .monospace()
                                    .color(ui.visuals().strong_text_color()),
                            );
                        }
                    });
                });
            response
        })
        .inner;
        // The arrows walk the history only while the field has the keyboard, so they still
        // scroll the panel when it does not.
        if response.has_focus() {
            let (up, down) = ui.input(|i| {
                (i.key_pressed(egui::Key::ArrowUp), i.key_pressed(egui::Key::ArrowDown))
            });
            if up || down {
                self.recall_history(up);
            }
        }
        if std::mem::take(&mut self.console_complete) && response.has_focus() {
            self.complete_command(ui.ctx(), response.id);
        }
        if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            let line = std::mem::take(&mut self.console_input);
            self.run_command(&line);
            response.request_focus();
        }
    }

    /// Records what a stage of this frame cost, for the slow-frame line.
    ///
    /// **Because a hang has to be attributable before it can be fixed.** "Half a second after
    /// moving a face" was reported and the two obvious candidates — the compile the picking
    /// needs and the cage's own pipelines — measured 2 ms and 0.4 ms
    /// (`fieldform-field/examples/cage_edit_cost.rs`, `fieldform-gpu/examples/cage_pipeline_cost.rs`),
    /// so the answer was in neither. The frame's own clock already exists; this says which part
    /// of it was slow, in the document the person actually hit it in.
    pub(crate) fn note_stage(&mut self, what: &'static str, started: std::time::Instant) {
        let ms = started.elapsed().as_secs_f32() * 1e3;
        // Only what is worth reading. A stage under a millisecond is not the hang.
        if ms >= 1.0 {
            self.stages.push((what, ms));
        }
    }

    /// Reports something that happened, into the console.
    ///
    /// **There is one place operations report, and it is the console.** This used to be a label
    /// beside the view-mode button, which had two problems the user named: it held exactly one
    /// message, so anything said while you were looking elsewhere was gone, and the console
    /// underneath it was already the place the same operations reported to when they were *typed*
    /// rather than clicked. `run_command` cleared the status line, dispatched, and copied whatever
    /// the operation had said into the log — so a menu item and a console command already produced
    /// the same sentence, and only one of them kept it.
    ///
    /// So the label is gone and this is the only reporting path. What an operation says is a
    /// record now, not a flash.
    pub(crate) fn say(&mut self, line: impl Into<String>) {
        self.report(true, line.into());
    }

    /// Reports something that went wrong, or that was refused. Same place, marked.
    pub(crate) fn warn(&mut self, line: impl Into<String>) {
        self.report(false, line.into());
    }

    /// **A line said twice in a row is counted, not repeated, and the log is capped.**
    ///
    /// Both of those exist because of one bug, which is worth recording rather than just fixing.
    /// The status line this replaced was a single slot, so a report made *every frame* simply
    /// overwrote itself and cost nothing. Appending to a log instead turns the same call into an
    /// unbounded stream — `draw_traced` re-reported an un-compilable part sixty times a second,
    /// and the console filled with one sentence.
    ///
    /// That particular call now reports only when the error changes. These two are the general
    /// answer underneath it: a per-frame report degrades into `× 4132` on one line instead of
    /// four thousand lines, and the log cannot grow without bound however badly a caller behaves.
    /// A safety net is worth having precisely because the next such caller will be written by
    /// someone who has forgotten this one.
    fn report(&mut self, ok: bool, text: String) {
        if let Some(capture) = self.line_capture.as_mut() {
            capture.push((ok, text.clone()));
        }
        self.console_reports += 1;
        if !ok {
            self.console_warnings += 1;
        }
        match self.console_log.last_mut() {
            Some(last) if last.ok == ok && last.text == text => {
                last.said += 1;
                // Not written to the file again either — the count is the news, and the file is
                // read by something that would rather see one line than four thousand.
                return;
            }
            _ => {}
        }
        self.console_log.push(ConsoleLine { ok, text: text.clone(), said: 1 });
        if self.console_log.len() > CONSOLE_LINES {
            let over = self.console_log.len() - CONSOLE_LINES;
            self.console_log.drain(..over);
        }
        self.mirror_to_file(ok, &text);
    }

    /// Appends one line to the console's file, and keeps that file from growing without end.
    ///
    /// **Why a file rather than stdout.** The same reason `install_crash_log` writes one: a
    /// windowed application launched from Explorer has no console attached, so anything printed
    /// goes nowhere anybody can read. A file is readable however the application was started,
    /// which is the point — it is there to be read back after something went wrong, by the user
    /// or by whoever they are asking for help.
    ///
    /// Truncated at startup, so what is in it is this session and not an archaeology of every
    /// session before it. Capped as well, because a session can be long.
    fn mirror_to_file(&mut self, ok: bool, text: &str) {
        use std::io::Write;
        let Some(path) = self.console_file.clone() else { return };

        // Rewritten from what is in memory once it grows past the cap. The in-memory log is
        // itself capped, so this terminates rather than trading one unbounded thing for another.
        if self.console_file_bytes > CONSOLE_FILE_BYTES {
            let mut whole = String::new();
            for line in &self.console_log {
                whole.push_str(&format!(
                    "{} {}\n",
                    if line.ok { "  " } else { "!!" },
                    one_line(&line.text)
                ));
            }
            if std::fs::write(&path, &whole).is_ok() {
                self.console_file_bytes = whole.len() as u64;
            }
            return;
        }

        let entry = format!(
            "[{:7.2}s] {} {}\n",
            self.started.elapsed().as_secs_f64(),
            if ok { "  " } else { "!!" },
            one_line(text),
        );
        if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            if file.write_all(entry.as_bytes()).is_ok() {
                self.console_file_bytes += entry.len() as u64;
            }
        }
    }

    /// Completes the command being typed, on Tab.
    ///
    /// **Only the first word.** Everything after it is `key=value`, where the keys differ per
    /// operation and are not written down anywhere this could read — so completing there would
    /// have to guess, and a completion that guesses is worse than none: you stop reading what
    /// it inserted.
    ///
    /// One match fills it in and adds a space. Several fill in as much as they share and list
    /// the candidates, which is the behaviour every shell has and the reason it is worth
    /// matching: the common prefix is progress even when the answer is still ambiguous, and
    /// seeing the alternatives is how you choose. None does nothing, silently — a console that
    /// scolded you for pressing Tab would be teaching you not to.
    fn complete_command(&mut self, ctx: &egui::Context, id: egui::Id) {
        let typed = self.console_input.clone();
        // Only while the first word is still being typed. Once there is a space, the command is
        // settled and Tab has nothing to say about the arguments.
        if typed.trim().is_empty() || typed.contains(char::is_whitespace) {
            return;
        }
        let prefix = typed.to_lowercase();
        let matches: Vec<String> = console_commands()
            .into_iter()
            .map(|(word, _)| word.to_string())
            .filter(|word| word.starts_with(&prefix))
            .collect();
        let Some(first) = matches.first() else { return };

        // As much as every match agrees on, which is the whole word when there is only one.
        let mut shared = first.clone();
        for word in &matches[1..] {
            let keep = shared
                .chars()
                .zip(word.chars())
                .take_while(|(a, b)| a == b)
                .count();
            shared.truncate(keep);
        }
        if matches.len() == 1 {
            shared.push(' ');
        } else if shared.len() == prefix.len() {
            // Nothing to add, so say what the choices are rather than appearing to do nothing.
            self.say(matches.join(", "));
        }
        if shared.len() > prefix.len() || matches.len() == 1 {
            self.console_input = shared;
            // The caret follows the text. Without this it stays where it was and the next
            // keystroke lands in the middle of the word that was just completed.
            let at = self.console_input.chars().count();
            if let Some(mut state) = egui::TextEdit::load_state(ctx, id) {
                state
                    .cursor
                    .set_char_range(Some(egui::text::CCursorRange::one(egui::text::CCursor::new(at))));
                state.store(ctx, id);
            }
        }
    }

    /// Walks back and forward through what has been typed.
    fn recall_history(&mut self, backwards: bool) {
        if self.console_history.is_empty() {
            return;
        }
        let last = self.console_history.len() - 1;
        self.console_history_at = match (self.console_history_at, backwards) {
            (None, true) => Some(last),
            (Some(i), true) => Some(i.saturating_sub(1)),
            (Some(i), false) if i < last => Some(i + 1),
            (Some(_), false) => None,
            (None, false) => None,
        };
        self.console_input = self
            .console_history_at
            .and_then(|i| self.console_history.get(i).cloned())
            .unwrap_or_default();
    }

    /// Runs one console line.
    ///
    /// A word, then `key=value` pairs. The word is looked up as a friendly name first and as a
    /// raw operation name second, so `sphere` and `create.sphere` both work and a script
    /// written against the operation names needs no translation.
    pub(crate) fn run_command(&mut self, line: &str) {
        // Whatever the GUI changed before this line is written first, so the record keeps the
        // order things happened in; and what this line changes is not echoed, since the line is
        // its own record (`interaction-spec.md` §6).
        // A typed line is a command like any other, so a session pending is kept first (§3.4).
        self.commit_session();
        self.end_join_preview();
        self.echo_settle();
        self.run_command_unechoed(line);
        self.echo_rebase();
    }

    /// Runs one console line and says what it did, as data.
    ///
    /// **The same `run_command`, with the words it reports caught as they are made**, so what a
    /// caller reads here is what the console printed and the two cannot differ
    /// (`mcp-server-spec.md` §6.1). The console, the log and `--run` are untouched.
    #[cfg_attr(not(test), allow(dead_code))] // read by the agent host, which a later phase adds
    pub(crate) fn run_line(&mut self, line: &str) -> outcome::LineOutcome {
        self.line_capture = Some(Vec::new());
        self.line_error = None;
        let before = outcome::BlockSnapshot::of(&self.doc);
        self.run_command(line);
        let caught = self.line_capture.take().unwrap_or_default();
        let error = self.line_error.take();
        let mut out = outcome::LineOutcome::from_reports(line, caught, error);
        out.blocks = before.changes_to(&outcome::BlockSnapshot::of(&self.doc), &self.doc);
        out
    }

    pub(crate) fn run_command_unechoed(&mut self, line: &str) {
        let line = line.trim();
        if line.is_empty() {
            return;
        }
        self.console_history.push(line.to_string());
        self.console_history_at = None;
        self.say(format!("> {line}"));

        let mut words = console_words(line).into_iter();
        let name = words.next().map(|(w, _)| w).unwrap_or_default().to_lowercase();
        // The two built-ins, known by the operation the registry gives their words, so a second
        // spelling of either would reach them too.
        let builtin = console_commands().into_iter().find(|(w, _)| *w == name).map(|(_, op)| op);
        if builtin == Some("console.help") {
            // `help sphere`: what one verb reads, from the same table that refuses what it
            // does not (`console_keys::VERBS`).
            if let Some((verb, _)) = words.next() {
                let verb = verb.to_lowercase();
                let op = console_commands()
                    .into_iter()
                    .find(|(word, _)| *word == verb)
                    .map(|(_, op)| op.to_string())
                    .unwrap_or(verb.clone());
                match console_keys::help_for(&verb, &op) {
                    Some(line) => {
                        self.say(line);
                        // The catalogue's worked line, when it has one (`examples.rs`).
                        if let Some(example) = examples::example_of(&verb) {
                            self.say(format!("e.g. {example}"));
                        }
                    }
                    None => self.say(format!("`{verb}` is not in the key table — `help` lists every command")),
                }
                return;
            }
            let names = console_commands()
                .into_iter()
                .map(|(word, _)| word)
                .collect::<Vec<_>>()
                .join(", ");
            self.say(names);
            self.say("Any operation name works too, and `key=value` after it becomes arguments.");
            return;
        }
        if builtin == Some("console.clear") {
            self.console_log.clear();
            return;
        }
        // `marchstats`, or `marchstats px=240`: a capture of steps per ray. Only in a build
        // with the `march-stats` feature, which is a measuring build and not a release.
        #[cfg(feature = "march-stats")]
        if name == "marchstats" {
            let sizes: Vec<u32> = words
                .filter_map(|(w, _)| w.strip_prefix("px=").and_then(|v| v.parse().ok()))
                .collect();
            self.capture_march_stats(if sizes.is_empty() { &[240, 1080] } else { &sizes });
            return;
        }

        // **`curve3` is `curve degree=3`.** A degree is the one number worth putting in the word
        // itself: it is what you change most often about a curve, and it is what you are already
        // thinking about when you reach for the console rather than the menu. The long form still
        // works, and this is only a way of writing it -- so there is no second command to keep in
        // step, and `help` still lists one `curve`.
        let (name, suffixed) = split_degree_suffix(&name);
        // The words after the verb, read by the one function that knows the grammar: a quoted
        // word is a name, a dash-word is a flag, and the rest is key=value.
        let (mut args, mut naming) = match console_arguments(words) {
            Ok(read) => read,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        // A degree written out wins over the one in the word, as it did when the word's was
        // set first and the argument's over it.
        if let Some(degree) = suffixed.filter(|_| args.get("degree").is_none()) {
            args = args.with("degree", parse_value(&degree.to_string()));
        }

        let op = console_commands()
            .into_iter()
            .find(|(word, _)| *word == name)
            .map(|(_, op)| op.to_string())
            .unwrap_or(name.clone());
        // **Which `name` this is, asked of the key table rather than of a list here.** For the
        // verbs that read one -- `param`, `number`, `environment`, `variable` and `inline` -- the
        // name is the argument, so a quoted word becomes it; for every other verb `name=` is
        // another way of writing `sphere 'ball'`, so it becomes the naming.
        if console_keys::reads_its_own_name(&op) {
            if let Some(wanted) = naming.take() {
                args = args.with("name", Value::Str(wanted));
            }
        } else if let Some(Value::Str(wanted)) = args.get("name").cloned() {
            naming = Some(wanted);
            args = args.without("name");
        } else if let Some(other) = args.get("name").cloned() {
            // A number or a flag written as `name=`: still a name, read as it was typed.
            naming = Some(match other {
                Value::Int(i) => i.to_string(),
                Value::Float(f) => f.to_string(),
                Value::Bool(b) => b.to_string(),
                other => format!("{other:?}"),
            });
            args = args.without("name");
        }
        // **A key the verb does not read is refused by name, with what it does read**, before
        // anything is made (`console_keys`). `cube radius=5` used to make a cube and say "Added
        // a cube", which is the same silence a wrong value falls back into. Only on the typed
        // path: a menu passes no keys, and a verb that has not declared its keys is passed
        // through as it always was.
        match self.agent_guard_paths(&op, args) {
            Ok(guarded) => args = guarded,
            Err(error) => {
                self.line_error = Some(error.clone());
                self.warn(error.message);
                return;
            }
        }
        if let Some(why) = console_keys::refusal(&name, &op, &args) {
            self.line_error = Some(outcome::unknown_key_error(&name, &op, &args, &why));
            self.warn(why);
            return;
        }

        // **Whether the operation said anything, asked of the log itself.** This used to clear
        // the status line, dispatch, and copy whatever was left there — which worked, and was the
        // only path by which anything an operation said was kept. Now that every operation reports
        // straight into the log, the question left is just whether this one did, and the log is
        // the thing that knows.
        let said_before = self.console_reports;
        // What was selected before, so the name lands on what this command *made* rather than
        // on whatever happened to be selected when a command made nothing.
        let was_body = self.selected_body;
        let was_curve = self.selected_curve;
        self.dispatch(&op, &args);
        if let Some(wanted) = naming {
            let made_body = self.selected_body.filter(|id| Some(*id) != was_body);
            let made_curve = self.selected_curve.filter(|id| Some(*id) != was_curve);
            match (made_body, made_curve) {
                // A label, like any name now (`block-ownership-spec.md` §6): the block can be
                // typed by it, and is adopted by the first block that reads it all the same.
                (Some(id), _) => {
                    if self.set_body_name(id, wanted.trim()) {
                        self.say(format!("Named \u{201c}{}\u{201d}", wanted.trim()));
                    }
                }
                (None, Some(id)) => {
                    self.rename_curve(id, wanted.trim());
                    self.say(format!("Named \u{201c}{}\u{201d}", wanted.trim()));
                }
                // Nothing new to name. Said rather than ignored: a name typed at a command that
                // makes nothing has gone nowhere, and silence would look like it worked.
                (None, None) => self.warn(format!(
                    "{:?} made nothing to name — a name goes on a command that creates \
                     something",
                    name
                )),
            }
        }
        // `console_reports` rather than the log's length, because a repeat collapses into the
        // line above it and leaves the length alone -- running the same refusal twice would
        // otherwise look like an operation that said nothing.
        if self.console_reports > said_before {
            return;
        }
        if op.contains('.') {
            self.say(format!("{op} ran"));
        } else {
            self.line_error = Some(outcome::unknown_verb_error(&name));
            self.warn(format!("{name:?} is not a command — type `help` for the list"));
        }
    }

    /// The part's bodies as one line: "Cage + Pipe", or "Subtract" once they are joined.
    ///
    /// Bodies rather than every node in order, because that is what the part *is* — what a
    /// joined body is made of belongs to the graph view, not to a summary.
    pub(crate) fn graph_summary(&self) -> String {
        let graph = self.part_graph();
        if graph.evaluation_order().is_none() {
            return "graph has a cycle".to_string();
        }
        // A point is never a body (`Node::is_body`), so a loop's joints are not counted here
        // and need no folding out: "Point + Point + Point" over a held triangle was the clutter
        // the profile row exists to remove.
        //
        // Read in the same order the list below shows, so the summary and the blocks agree, and
        // from the readings the list keeps: this is drawn every frame, twice, and searching the
        // bodies for every block made it quadratic in the part.
        let list = self.list_readings();
        let bodies: Vec<NodeId> =
            list.listed.iter().copied().filter(|id| list.nesting.is_root(*id)).collect();
        if bodies.is_empty() {
            return "no bodies".to_string();
        }
        bodies
            .iter()
            .map(|id| self.body_label(*id))
            .collect::<Vec<_>>()
            .join(" + ")
    }

    /// An `expect` that could not be measured at all — recorded, not merely warned.
    ///
    /// **A refused expect used to vanish from the tally.** `curves.ffc` says `expect target=bar`
    /// and the run reported "6 of 6 expects passed" with exit 0, while the seventh had never run:
    /// the sweep that would have made `bar` was refused, so the name did not resolve, so the
    /// expect was refused, so nothing counted it. A suite that can drop a check silently is worse
    /// than one that fails, because the failure it hides is indistinguishable from a pass.
    ///
    /// An unresolved record is what [`SCRIPT_EXIT_CODE`] already counts as a failed benchmark,
    /// so the run says `6 of 7` and exits non-zero. It is deliberately not a `Fail`: nothing was
    /// measured, and reporting a measurement that did not happen as a wrong answer would be its
    /// own lie.
    fn expect_unresolved(&mut self, what: String, why: String) {
        let said = format!("expect {what}: {why}");
        self.warn(said.clone());
        self.expect_results.push(ExpectRecord {
            what,
            wanted: String::new(),
            got: String::new(),
            tol: String::new(),
            verdict: expect::Verdict::Unresolved,
            said,
        });
    }

    /// `expect …`: one measurement asserted (`expect.rs`). A pass is said; a fail or an
    /// unresolved one is warned, which is what stops a script.
    pub(crate) fn expect_command(&mut self, args: &Args) {
        let asked = match expect::read(args, &|t| self.lookup(t)) {
            Ok(asked) => asked,
            Err(why) => {
                // Named by whatever the line asked for, since nothing resolved: the target's own
                // text is what the reader of the report has to go on.
                let what = args
                    .get("target")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| "the line".to_string());
                self.expect_unresolved(what, why);
                return;
            }
        };
        let label = |app: &Self, id: NodeId| app.body_name(id).map(str::to_string).unwrap_or_else(|| app.body_label(id));
        let record = match asked {
            expect::Expect::Approx { target, wanted } => {
                let got = fieldform_field::approximate_blends(&self.doc, self.part_graph()).contains(&target);
                let verdict = if got == wanted { expect::Verdict::Pass } else { expect::Verdict::Fail };
                let what = label(self, target);
                ExpectRecord {
                    what: format!("{what} approx"),
                    wanted: wanted.to_string(),
                    got: got.to_string(),
                    tol: String::new(),
                    verdict,
                    said: format!("expect {what} approx: {} — wanted {wanted}, got {got}", verdict.word().to_uppercase()),
                }
            }
            expect::Expect::Measure { of, quantity, wanted } => {
                let what = match of {
                    expect::Of::One(id) => label(self, id),
                    expect::Of::Diff(a, b) => format!("{} − {}", label(self, a), label(self, b)),
                    expect::Of::Ratio(a, b) => format!("{} / {}", label(self, a), label(self, b)),
                };
                // Each block compiled once, alone with what it reads, and measured within its own
                // extent; the first grid from the first block's size.
                let mut fields: std::collections::HashMap<NodeId, Result<(GraphField, (Point3<f64>, Point3<f64>)), String>> =
                    Default::default();
                let doc = &self.doc;
                let graph = self.part_graph().clone();
                let prepare = |id: NodeId| -> Result<(GraphField, (Point3<f64>, Point3<f64>)), String> {
                    let alone = expect::graph_of(&graph, id);
                    let bounds = fieldform_field::part_extent(doc, &alone)
                        .as_box()
                        .ok_or_else(|| "it has no bounds to measure within — an unbounded block cannot be measured".to_string())?;
                    let field = GraphField::compile_graph(doc, &alone, crate::MEASURE_TOLERANCE_MM).map_err(|e| e.to_string())?;
                    Ok((field, bounds))
                };
                let first = match of {
                    expect::Of::One(id) | expect::Of::Diff(id, _) | expect::Of::Ratio(id, _) => id,
                };
                let every: Vec<NodeId> = match of {
                    expect::Of::One(id) => vec![id],
                    expect::Of::Diff(a, b) | expect::Of::Ratio(a, b) => vec![a, b],
                };
                for id in &every {
                    match prepare(*id) {
                        Ok(prepared) => {
                            fields.insert(*id, Ok(prepared));
                        }
                        Err(why) => {
                            self.expect_unresolved(what, why);
                            return;
                        }
                    }
                }
                let boxes: Vec<(Point3<f64>, Point3<f64>)> =
                    every.iter().filter_map(|id| fields.get(id).and_then(|f| f.as_ref().ok()).map(|(_, b)| *b)).collect();
                let e = boxes[0].1 - boxes[0].0;
                // The smallest feature the block asks for bounds the first grid, so a sheet
                // thinner than two cells is not missed at both levels (§3.3); and the grid is
                // coarsened until its second level fits one measurement of every block in it.
                let features = expect::smallest_feature(&self.doc, &graph, first);
                let fits = |h: f64| {
                    boxes.iter().all(|b| fieldform_field::anchored_corners(*b, h) <= fieldform_field::ANCHORED_CORNERS_MAX)
                };
                let h0 = match expect::first_spacing_that_fits(e.x.max(e.y).max(e.z), features, &fits) {
                    Ok(h0) => h0,
                    Err(why) => {
                        self.expect_unresolved(what, why);
                        return;
                    }
                };
                // A thickness is walked only when one was asked for: it costs a walk inward from
                // every vertex, and the number it is held to is the threshold that bounds it.
                let thickness_threshold = match (quantity.is_thickness(), &wanted) {
                    (true, expect::Wanted::Above { v, .. }) => Some(*v),
                    _ => None,
                };
                let mut measure = |id: NodeId, h: f64| -> Result<fieldform_field::Measured, String> {
                    let entry = fields.entry(id).or_insert_with(|| prepare(id));
                    let (field, bounds) = entry.as_ref().map_err(|e| e.clone())?;
                    let read = match thickness_threshold {
                        Some(t) => fieldform_field::MassProperties::anchored_with_thickness(field, *bounds, h, t),
                        None => fieldform_field::MassProperties::anchored(field, *bounds, h),
                    };
                    match read {
                        fieldform_field::Anchored::Measured(m) => Ok(m),
                        fieldform_field::Anchored::Empty => Err("it has no surface to measure".to_string()),
                        fieldform_field::Anchored::TooFine { corners } => Err(format!(
                            "a grid of {h} mm would be {corners} corners, past what one measurement holds"
                        )),
                    }
                };
                let started = std::time::Instant::now();
                let outcome = expect::decide(of, quantity, &wanted, h0, &mut measure);
                let secs = started.elapsed().as_secs_f64();
                let nums = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",");
                let (wanted_said, tol_said) = match &wanted {
                    expect::Wanted::Near { values, tol } => (nums(values), format!("{tol}")),
                    expect::Wanted::Above { v, strict } => (format!("{}{v}", if *strict { ">" } else { ">=" }), String::new()),
                };
                ExpectRecord {
                    what: format!("{what} {}", quantity.word()),
                    wanted: wanted_said,
                    got: nums(&outcome.got),
                    tol: tol_said,
                    verdict: outcome.verdict,
                    said: format!("{} ({secs:.2}s)", expect::line(&what, quantity, &wanted, &outcome)),
                }
            }
        };
        if record.verdict == expect::Verdict::Pass {
            self.say(record.said.clone());
        } else {
            self.warn(record.said.clone());
        }
        self.expect_results.push(record);
    }

    /// `snap file="ref.png" [view=iso] [quality=still width=5000 ss=2 bg=transparent …]`: the
    /// finished still, written as a PNG or a JPEG. `snap block=<name>` runs a Save Image block's
    /// settings. The keys are the Save Image block's own, so what the panel can ask for a line
    /// can say; `console_keys::VERBS` is the one list of them.
    ///
    /// **Without `quality=still` it is the viewport's own picture**, read from the texture it
    /// drew, and `view=` moves the one camera to take it: the behaviour every script written
    /// before the block existed relies on. `quality=still` draws a picture of its own at any
    /// size, beside the viewport, and `view=` then aims a copy of the camera so nobody's view
    /// is moved. **`size=` is refused by name**, as it always was: the displayed picture is the
    /// window's size, and a size of its own is `width=`/`height=` on a Still.
    pub(crate) fn snap_command(&mut self, args: &Args) {
        if args.get("size").is_some() {
            self.warn(
                "`snap size=` is not read — the displayed picture is the window's size; a size of \
                 its own is `quality=still width=…`",
            );
            return;
        }
        if let Some(name) = args.get("block").map(|v| v.text()) {
            match self.image_block_named(name.trim()) {
                Some(id) => self.run_image_block(id, false),
                None => self.warn(format!("there is no Save Image block called {name:?}")),
            }
            return;
        }
        let Some(file) = args.get("file").map(|v| v.text()) else {
            self.warn("`snap` wants `file=` — `snap file=\"ref.png\"`, or `block=` to run a Save Image block");
            return;
        };
        let mut request = image_export::ImageRequest::displayed(std::path::PathBuf::from(&file));
        if let Err(why) = image_export::read_image_keys(&mut request, args) {
            self.warn(why);
            return;
        }
        if let Some(ext) = request.path.extension().and_then(|e| e.to_str())
            && args.get("format").is_none()
            && let Some(f) = fieldform_scene::ImageFormat::from_extension(ext)
        {
            request.format = f;
        }
        request.line = image_export::snap_line(&request);
        if let Err(why) = self.begin_image(request) {
            self.warn(why);
        }
    }

    /// `run file=… [-keepgoing]`: arms a script for the next frame. See [`Script`].
    pub(crate) fn run_script_command(&mut self, args: &Args) {
        if self.script_running {
            self.warn("`run` inside a script is refused — a script that runs scripts never ends");
            return;
        }
        let Some(path) = args.get("file").and_then(|v| v.as_str()) else {
            self.warn("`run` wants `file=` — `run file=\"C:/tests/scene.ffc\"`");
            return;
        };
        match Script::read_one_or_folder(
            std::path::Path::new(path),
            console_keys::flag(args, "keepgoing"),
            false,
        ) {
            Ok(script) => self.script = Some(script),
            Err(why) => self.warn(why),
        }
    }

    /// Runs the armed script's lines, one frame after it was armed. See [`Script`].
    pub(crate) fn drive_script(&mut self, ctx: &egui::Context) {
        let Some(script) = self.script.as_mut() else { return };
        script.frames += 1;
        if script.frames < 2 {
            ctx.request_repaint();
            return;
        }
        // A snapshot in flight holds the run: the next line may aim the camera somewhere else,
        // and two snaps would then write one picture twice.
        if self.pending_snap.is_some() {
            ctx.request_repaint();
            return;
        }
        let Some(mut script) = self.script.take() else { return };
        let name = benchmark_name(&script.path);
        // Said once, however many frames the run takes: a snapshot sends it round again.
        let mut tally = match self.script_tally.take() {
            Some(tally) => tally,
            None => {
                self.say(format!("Running {name}: {} lines", script.lines.len()));
                ScriptTally { started: Some(std::time::Instant::now()), ..Default::default() }
            }
        };
        self.script_running = true;
        let started = tally.started.unwrap_or_else(std::time::Instant::now);
        while script.next < script.lines.len() {
            let (n, line) = script.lines[script.next].clone();
            script.next += 1;
            // Every line is its own undo step: a file's lines are not one gesture.
            self.gesture = None;
            self.say(format!("{name}:{n}"));
            let warned = self.console_warnings;
            let expected = self.expect_results.len();
            self.run_command(&line);
            // **What the line moved is named and said before the next line runs**, as the next
            // frame does after a click (`block-ownership-spec.md` §4.2). Every line runs in this
            // one frame, so left to the frame a block one line made a variable had no name a later
            // line could type, its line came after the run's summary, and a folder's next `new`
            // threw it away unsaid. Said, never warned, so it cannot stop the run.
            self.drain_ownership_changes(false);
            tally.expects.extend((expected..self.expect_results.len()).map(|i| (n, i)));
            if self.console_warnings > warned && !script.keep_going {
                tally.stopped = Some(n);
                break;
            }
            // **A snapshot stops the run here**, not at the next frame's start: the line after it
            // may aim the camera somewhere else, and the picture would then be of the wrong view.
            // The script keeps its place, so it carries on from the next line once the file is
            // written.
            if self.pending_snap.is_some() {
                self.script_running = false;
                self.script_tally = Some(tally);
                self.script = Some(script);
                ctx.request_repaint();
                return;
            }
        }
        self.script_running = false;
        let (stopped, expects) = (tally.stopped, tally.expects);
        let not_passed = expects
            .iter()
            .filter(|(_, i)| self.expect_results[*i].verdict != expect::Verdict::Pass)
            .count();
        let passed = expects.len() - not_passed;
        match stopped {
            Some(n) => self.warn(format!(
                "{name}:{n}: stopped at the first refusal — `run file=… -keepgoing` runs to the end"
            )),
            None => self.say(format!(
                "{name}: ran {} lines; {passed} of {} expects passed",
                script.lines.len(),
                expects.len()
            )),
        }
        // This script's entry, kept so a folder's report covers every one of them.
        script.done.push(serde_json::json!({
            "name": name,
            "passed": stopped.is_none() && not_passed == 0,
            "refused_at": stopped,
            "ms": started.elapsed().as_millis() as u64,
            "expects": expects.iter().map(|(line, i)| {
                let r = &self.expect_results[*i];
                serde_json::json!({
                    "line": line,
                    "what": r.what,
                    "wanted": r.wanted,
                    "got": r.got,
                    "tol": r.tol,
                    "verdict": r.verdict.word(),
                    "pass": r.verdict == expect::Verdict::Pass,
                })
            }).collect::<Vec<_>>(),
        }));

        // **A folder's next benchmark starts from a new document**, so one cannot leave a body
        // standing in the next, and the run carries on next frame — `new` has a dialog to refuse
        // if anything is unsaved, and a script's own document is never worth keeping.
        if !script.then.is_empty() {
            let next = script.then.remove(0);
            match Script::read(&next, script.keep_going, script.exit_after) {
                Ok(following) => {
                    self.mark_saved();
                    self.dispatch("file.new", &Args::new());
                    self.script = Some(Script {
                        report: script.report.clone(),
                        then: std::mem::take(&mut script.then),
                        done: std::mem::take(&mut script.done),
                        ..following
                    });
                    ctx.request_repaint();
                    return;
                }
                Err(why) => self.warn(why),
            }
        }

        // The report the instructions describe: one entry a script, however many a folder held.
        if let Some(path) = &script.report {
            let report = serde_json::json!({ "benchmarks": script.done });
            match serde_json::to_string_pretty(&report)
                .map_err(|e| e.to_string())
                .and_then(|text| std::fs::write(path, text).map_err(|e| e.to_string()))
            {
                Ok(()) => self.say(format!("Report written to {}", path.display())),
                Err(why) => self.warn(format!("The report could not be written to {}: {why}", path.display())),
            }
        }
        if script.exit_after {
            // The number of failed benchmarks, which is what `--exit` is defined to answer; a
            // folder of ten with two bad ones exits 2.
            let failed = script
                .done
                .iter()
                .filter(|b| b.get("passed").and_then(serde_json::Value::as_bool) != Some(true))
                .count();
            SCRIPT_EXIT_CODE.store(failed.min(i32::MAX as usize) as i32, std::sync::atomic::Ordering::SeqCst);
            // Closing without the unsaved-changes question: a run never writes over a document
            // anyone opened, and `--exit` was asked for.
            self.closing = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}
