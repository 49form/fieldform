// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Making and changing blocks: the create commands, the operations, and the pickers they share.

use crate::*;

impl App {
    /// Puts a cage body back, or makes a new one.
    ///
    /// Deleting a cage body leaves the cage itself in the document — nothing is dropped — so
    /// the first thing this does is look for one no node is drawing and give it a body again.
    /// That makes deleting the cage recoverable, which is what it was missing.
    pub(crate) fn add_cage_body(&mut self) {
        let drawn: Vec<CageId> = self.part_graph().referenced_cages();
        let orphan = self
            .doc
            .cages
            .keys()
            .copied()
            .find(|id| !drawn.contains(id));
        let cage = match orphan {
            Some(id) => id,
            None => self.doc.add_cage(Cage::cube_of(DEFAULT_BODY_MM)),
        };
        let mut graph = self.part_graph().clone();
        let id = graph.add(Node::Cage(cage));
        self.set_part_graph(graph);
        self.active_cage = cage;
        self.select_body(Some(id));
        self.clear_hidden(id);
        self.mark_changed();
        self.rebuild_render_resources();
        let (centre, radius) = cage_bounds(self.cage());
        self.camera.frame(centre, radius);
        self.say("Added a cage");
    }

    /// Keeps the active cage -- the one the cage tools, Tab and the cage mode act on -- pointed at
    /// a cage that is there to be worked on: the selected cage block's, else the one it already
    /// is while its block is in the part and visible, else the first visible cage block's, else
    /// the first cage block's. It used to be set only when a cage was made, so with two cages
    /// it stayed on the second after that was hidden ("the cage block is hidden") and after it
    /// was deleted ("this part has no cage block"), and Tab would not enter the first (the
    /// owner, 2026-10-24).
    pub(crate) fn resync_active_cage(&mut self) {
        let graph = self.part_graph();
        let cages: Vec<(NodeId, CageId)> =
            graph.nodes().filter_map(|(id, n)| if let Node::Cage(c) = n { Some((id, *c)) } else { None }).collect();
        if cages.is_empty() {
            return;
        }
        let visible = |app: &App, node: NodeId| app.is_visible(node) && app.body_containing(node).is_none_or(|b| app.is_visible(b));
        let selected = self.selected_body.and_then(|body| {
            cages.iter().copied().find(|&(node, _)| node == body || self.body_containing(node) == Some(body))
        });
        let current_ok = cages.iter().any(|&(node, c)| c == self.active_cage && visible(self, node));
        // A hidden cage is not there to be worked on, selected or not: the eye moves the tools.
        let chosen = selected
            .filter(|&(node, _)| visible(self, node))
            .map(|(_, c)| c)
            .or(if current_ok { Some(self.active_cage) } else { None })
            .or_else(|| cages.iter().find(|&&(node, _)| visible(self, node)).map(|&(_, c)| c))
            .or(selected.map(|(_, c)| c))
            .unwrap_or(cages[0].1);
        if chosen != self.active_cage {
            self.active_cage = chosen;
            // The control mesh drawn and picked is this cage's, so what is drawn moves with it.
            self.rebuild_render_resources();
        }
    }

    /// A named plane at `origin`, facing `along`, ready for something to be aimed at it.
    ///
    /// **Because picking is only half of aiming**, which is the same gap `point_block_at` was
    /// written to close for positions: the aim controls listed the planes that already existed
    /// and offered nothing else, so the only way to get a facing out of a block and into
    /// something reusable was to make a plane elsewhere, read three numbers off, and come back.
    /// Reported as "if the doc is empty with just a plane you can't declare turn/angle a
    /// variable".
    ///
    /// **A variable on creation, with a name, where `Pattern` and `Mirror` stopped naming what
    /// they consume.** Those two were writing a name onto a block the person had made for its own
    /// sake, which is the app answering a question that is theirs. This is the other case: the
    /// gesture that reaches here exists *only* to produce something for other blocks to be aimed
    /// at, so being a variable is that gesture said out loud rather than a decision taken for
    /// anybody. A plane nobody points at is a plane nobody asked for.
    ///
    /// **Made one, not named into one** (`block-ownership-spec.md` §14 Q4). The name used to be
    /// what made it a variable, when the list was laid out by names; it is a label now, so the
    /// plane is stored as a variable outright and keeps the name it always got. The first block
    /// aimed at it then refers to it rather than adopting it. The name is editable in the block
    /// like any other.
    fn plane_block(&mut self, origin: Point3<f64>, along: Rotation3<f64>) -> NodeId {
        let (x, y, z) = along.euler_angles();
        let mut graph = self.part_graph().clone();
        let id = graph.add(Node::Plane {
            origin: fieldform_scene::Place::At([
                ParamRef::Literal(origin.x),
                ParamRef::Literal(origin.y),
                ParamRef::Literal(origin.z),
            ]),
            flipped: false,
            turn: Turn::By([
                ParamRef::Literal(x.to_degrees()),
                ParamRef::Literal(y.to_degrees()),
                ParamRef::Literal(z.to_degrees()),
            ]),
        });
        self.set_part_graph(graph);
        let name = self.make_it_a_variable(id);
        self.say(format!(
            "\u{201c}{name}\u{201d} is a plane — turn it and everything aimed at it \
             follows"
        ));
        id
    }

    /// A plane facing the way a block's own frame points, for the aim controls.
    pub(crate) fn plane_facing(&mut self, frame: fieldform_scene::Frame) -> NodeId {
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        let (origin, rotation) =
            fieldform_field::resolve_frame(&self.doc, self.part_graph(), &frame, &numbers)
                .unwrap_or((Vector3::zeros(), Rotation3::identity()));
        self.plane_block(Point3::from(origin), rotation)
    }

    /// A plane facing the way a block's own turn points, at that block's own place.
    fn plane_facing_turn(&mut self, id: NodeId, turn: &Turn) -> NodeId {
        let rotation = self.aim_of(turn);
        let at = self
            .part_graph()
            .node(id)
            .and_then(|n| n.places().first().map(|p| (*p).clone()))
            .map(|place| self.resolve_place(&place))
            .unwrap_or_else(Point3::origin);
        self.plane_block(at, rotation)
    }

    /// A point at a position, for anything that needs one to sit on.
    ///
    /// **Because picking is only half of naming.** The V beside a position listed the points
    /// that already existed and offered nothing else, so the only way to get a position out of
    /// a block and into something reusable was to make a point somewhere else, read the three
    /// numbers off, type them in, and then come back. Said plainly: "by clicking on it you
    /// should also have the option to declare what u have now as a variable - currently u can
    /// only pick".
    ///
    /// A variable on creation, with a name, for the reason [`Self::plane_block`] gives at length:
    /// the gesture that asks for one is already the declaration, since a point exists to be
    /// referred to.
    pub(crate) fn point_block_at(&mut self, at: Point3<f64>) -> NodeId {
        let mut graph = self.part_graph().clone();
        let id = graph.add(Node::Point {
            at: [
                ParamRef::Literal(at.x),
                ParamRef::Literal(at.y),
                ParamRef::Literal(at.z),
            ],
        });
        self.set_part_graph(graph);
        let name = self.make_it_a_variable(id);
        self.say(format!(
            "\u{201c}{name}\u{201d} is a point at {:.2}, {:.2}, {:.2} — move it and \
             everything sitting on it follows",
            at.x, at.y, at.z
        ));
        id
    }

    /// Makes a block just made a variable, named as `unique_variable_name` names one, and
    /// answers what it is called: the plane and the point above, which exist to be referred to.
    /// One undo step of its own after the one that made the block, as its naming always was.
    fn make_it_a_variable(&mut self, id: NodeId) -> String {
        let block = BlockKey::Node(id);
        if let Some(cmd) = make_variable(&self.doc, block, || self.unique_variable_name(id)) {
            self.set_ownership(cmd);
        }
        self.body_label(id)
    }

    /// The node a console argument names, by its variable name or by the label its row shows.
    ///
    /// Both, because the console is asked in whichever of the two the person can see: a named
    /// block wears its name in the list, an unnamed one wears "Sphere" or "Ramp".
    pub(crate) fn node_named(&self, name: &str) -> Option<NodeId> {
        let graph = self.part_graph();
        let mut ids: Vec<NodeId> = graph.nodes().map(|(id, _)| id).collect();
        ids.sort_by_key(|id| id.raw());
        ids.iter()
            .copied()
            .find(|id| self.body_name(*id).is_some_and(|n| n.eq_ignore_ascii_case(name)))
            .or_else(|| {
                ids.iter().copied().find(|id| self.body_label(*id).eq_ignore_ascii_case(name))
            })
    }

    /// What a bare word in a console value names: a parameter, a block, both, or nothing
    /// (`console_keys::Lookup`). Both is refused by every reader rather than guessed.
    pub(crate) fn lookup(&self, text: &str) -> console_keys::Lookup {
        match (self.doc.parameter_named(text), self.node_named(text)) {
            (Some(p), None) => console_keys::Lookup::Parameter(p),
            (None, Some(b)) => console_keys::Lookup::Block(b),
            (Some(_), Some(_)) => console_keys::Lookup::Both,
            (None, None) => console_keys::Lookup::Nothing,
        }
    }

    /// `target=` read by name for a modifier, said and `Err` when it names nothing; `Ok(None)`
    /// where it was not given, so the caller falls back to the selection as it always has.
    ///
    /// **`target=none` is the empty slot asked for out loud** (`console_keys::NOTHING`), which is
    /// why this answers an `Operand` rather than a block: a line that says `none` has said *do
    /// not* guess, and a caller that fell back to the selection would be doing something other
    /// than what it was told.
    fn named_target(
        &mut self,
        word: &str,
        args: &Args,
    ) -> Result<Option<fieldform_scene::Operand>, ()> {
        match console_keys::operand_named(word, "target", args, &|t| self.lookup(t)) {
            Ok(named) => Ok(named),
            Err(why) => {
                self.warn(why);
                Err(())
            }
        }
    }

    /// Which block a modifier is made over: the one the line named, or -- where it named none --
    /// the selection, or the first body, or no block at all.
    ///
    /// **An empty slot rather than a refusal, which is what lets a script build a part
    /// top-down**: `shell` in a part with nothing in it makes the shell and leaves its slot
    /// waiting, the same state the Create menu leaves a block in (`notebook-spec.md` §7.2). The
    /// verbs used to refuse with "Needs a body to shell", which made the block unreachable until
    /// something existed to put in it -- so a script had to be written bottom-up even where the
    /// part is easier to describe the other way round.
    ///
    /// Free of `self` so a test can hold the rule without a window and a device, as
    /// [`Self::refusal_of`] is.
    fn target_operand(
        named: Option<fieldform_scene::Operand>,
        selected: Option<NodeId>,
        bodies: &[NodeId],
    ) -> fieldform_scene::Operand {
        // What the line said wins, the empty slot included: `target=none` is a choice, not a
        // failure to find anything.
        if let Some(asked) = named {
            return asked;
        }
        selected
            .filter(|id| bodies.contains(id))
            .or_else(|| bodies.first().copied())
            .map_or_else(fieldform_scene::Operand::empty, fieldform_scene::Operand::filled)
    }

    /// `move`, `turn` and `scale`: one placement block over the target, or the placement the
    /// target already is, edited in place (`console_keys::placed`).
    pub(crate) fn apply_placing(&mut self, placing: console_keys::Placing, args: &Args) {
        let word = placing.word();
        let Ok(named) = self.named_target(word, args) else { return };
        // **`target=none` is refused here rather than guessed past.** A placement folds into or
        // wraps the chain already over a block, so there is no placement to make without one --
        // and a line that asked for an empty slot and was handed the selection instead would be
        // the console doing something other than what it was told. The other modifiers take it
        // and leave the slot waiting ([`Self::target_operand`]); this one says why it cannot.
        if named.is_some_and(|slot| slot.is_empty()) {
            self.warn(format!(
                "`{word} target=none` has nothing to {word} \u{2014} a placement holds a block"
            ));
            return;
        }
        let bodies = self.doc.listed_bodies(self.part_graph());
        let Some(target) = named
            .and_then(|slot| slot.source())
            .or(self.selected_body.filter(|id| bodies.contains(id)))
            .or(bodies.first().copied())
        else {
            self.warn(format!("Needs a body to {word}"));
            return;
        };
        // **The outermost placement already over the target is the one edited or wrapped.**
        // `target=part` names the body, and after one `move` the body sits inside a placement
        // the user never named; a second verb that wrapped the body again would place it twice
        // -- the first run's saved document held three copies of one part. So the chain of
        // placements over the target is followed to its end, and that end is what `placed`
        // folds into or wraps, whether it was made here or by the manipulator.
        let outermost = {
            let graph = self.part_graph();
            let mut at = target;
            loop {
                let over = graph
                    .nodes()
                    .filter(|(_, n)| matches!(n, Node::Transform { input, .. } if input.source() == Some(at)))
                    .map(|(id, _)| id)
                    .max_by_key(|id| id.raw());
                match over {
                    Some(id) => at = id,
                    None => break,
                }
            }
            (at != target || matches!(graph.node(at), Some(Node::Transform { .. })))
                .then(|| graph.node(at).cloned().map(|node| (at, node)))
                .flatten()
        };
        let (node, edits) = match console_keys::placed(
            placing,
            target,
            outermost.as_ref().map(|(id, node)| (*id, node)),
            args,
            &|t| self.lookup(t),
        ) {
            Ok(placed) => placed,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let mut graph = self.part_graph().clone();
        let wrapped = outermost.as_ref().map(|(id, _)| *id).unwrap_or(target);
        let id = match edits {
            Some(id) => {
                if let Some(slot) = graph.node_mut(id) {
                    *slot = node;
                }
                id
            }
            None => graph.add(node),
        };
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        self.set_part_graph(graph);
        if edits.is_none() {
            self.clear_hidden(wrapped);
        }
        self.select_body(Some(id));
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        let label = self.body_label(target);
        self.say(match placing {
            console_keys::Placing::Move => format!("Moved {label}"),
            console_keys::Placing::Turn => format!("Turned {label}"),
            console_keys::Placing::Scale => format!("Scaled {label}"),
        });
    }

    /// What a ramp reads: one list holding every block and every curve it could be.
    ///
    /// **One question, one control.** This was two — an operand picker and a curve picker
    /// beside it — so "what does this read" was asked twice and answered in whichever of them
    /// happened to hold it. A point, a plane, a pipe and a curve are all answers to the same
    /// question, so they belong in the same list.
    ///
    /// **A curve is not a node, so a ramp cannot name one.** `ParamRef::Field` names a graph
    /// node and a curve lives in the document — the same arrangement an extrude's profile has.
    /// `ramp-decision.md` §7.1 gives the route: `Node::Pipe`, whose field *is* distance to the
    /// curve, less its radius. So picking a curve here builds that pipe at the smallest radius
    /// the panel allows and points the ramp at it, which is distance to the curve to within a
    /// micron.
    ///
    /// A pipe rather than teaching `Node::Ramp` about curves, because the pipe is a block: it
    /// appears in the list, its radius can be opened out to make the band start away from the
    /// line, and it is the same thing anybody would have built by hand. A second spelling of
    /// "distance to this curve" is what D9 is about.
    ///
    /// Nothing is an entry rather than an absence: a ramp arrives unwired, so the list has to be
    /// able to say so and to go back to it.
    pub(crate) fn pick_field_to_read(&mut self, ui: &mut egui::Ui, ramp: NodeId, current: Option<NodeId>) {
        // The same blocks any operand picker offers, minus anything that reads this ramp —
        // choosing one of those would close a loop.
        let blocks = self.operand_choices(ramp);
        let mut curves: Vec<(CurveId, String)> =
            self.doc.curves.iter().map(|(id, c)| (*id, c.name.clone())).collect();
        curves.sort_by_key(|(id, _)| id.raw());

        let showing = match current {
            Some(id) => self.body_label(id),
            None => "nothing yet".to_string(),
        };
        let mut chosen: Option<Option<NodeId>> = None;
        let mut wanted_curve: Option<CurveId> = None;
        ui.horizontal(|ui| {
            row_label(ui, "Reads:");
            egui::ComboBox::from_id_salt((ramp, "ramp_input"))
                .selected_text(showing)
                .show_ui(ui, |ui| {
                    if ui.selectable_label(current.is_none(), "nothing yet").clicked()
                        && current.is_some()
                    {
                        chosen = Some(None);
                    }
                    if let Some(picked) = filtered_choices(ui, &blocks, current)
                        && current != Some(picked)
                    {
                        chosen = Some(Some(picked));
                    }
                    // Curves under a heading rather than mixed in, because picking one *builds*
                    // a block where picking a block only points at it. The heading stays when
                    // there are none, so what a ramp can read is readable from the ramp.
                    ui.separator();
                    ui.label(egui::RichText::new("distance to a curve").weak());
                    if curves.is_empty() {
                        ui.add_enabled(false, egui::Button::new("no line yet").frame(false))
                            .on_disabled_hover_text(
                                "Draw a line and it appears here — picking one builds a \
                                 pipe along it for this to read.",
                            );
                    }
                    for (cid, name) in &curves {
                        if ui.selectable_label(false, name).clicked() {
                            wanted_curve = Some(*cid);
                        }
                    }
                });
        })
        .response
        .on_hover_text(
            "The field this ramp reads. A point gives a radial falloff, a plane a gradient \
             along an axis, any body its own distance. Picking a curve builds the pipe that \
             measures distance to it, at its smallest radius — which is the line itself.",
        );

        if chosen.is_none() && wanted_curve.is_none() {
            return;
        }
        let mut graph = self.part_graph().clone();
        let read = match wanted_curve {
            Some(curve) => Some(graph.add(Node::Pipe {
                curve,
                radius: ParamRef::Literal(*POSITIVE_MM.start()),
            })),
            None => chosen.flatten(),
        };
        if let Some(Node::Ramp { input, .. }) = graph.node_mut(ramp) {
            // Nothing chosen empties the slot rather than leaving the last choice in it: "read
            // nothing" is a state a ramp is allowed to be in, and the one it is born in.
            *input = read.map_or_else(fieldform_scene::Operand::empty, fieldform_scene::Operand::filled);
        }
        // A ramp reading the body it drives is a cycle, and a cyclic graph does not compile.
        // Refused before it is committed, so the answer is a message rather than a blank
        // viewport and a saved file that opens to the same error.
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        self.set_part_graph(graph);
        self.mark_changed();
        self.rebuild_render_resources();
        self.say(match read {
            Some(id) => format!("Ramp reads {}", self.body_label(id)),
            None => "Ramp reads nothing — pick a block or a curve".to_string(),
        });
    }

    /// A field already in the part that a ramp could read, or `None` when there is none.
    ///
    /// **Because a driver should be found before it is built.** Asked for directly: a ramp and
    /// a point "should only be added when there is nothing to use in the scene". A point at the
    /// origin arriving beside every ramp is litter when the part already holds one, and the
    /// second is indistinguishable from the first in the list.
    ///
    /// A point first, then a plane: `ramp-decision.md` §7.1's own order, and the two whose
    /// reading is obvious before you have seen it — distance from a point is a radial falloff,
    /// and a plane's field *is* a signed coordinate. Anything else in the graph is a body, and
    /// reading a body to drive that same body is the cycle this has to avoid rather than the
    /// answer.
    ///
    /// `avoid` is the block being driven: nothing that consumes it may be read, or the graph
    /// closes a loop and stops compiling. The newest of each kind, which is the one just made.
    pub(crate) fn field_to_read(&self, avoid: Option<NodeId>) -> Option<NodeId> {
        let graph = self.part_graph();
        // Everything downstream of the consumer, which is everything that cannot be read.
        let mut loops_back: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        let mut walk: Vec<NodeId> = avoid.into_iter().collect();
        loops_back.extend(walk.iter().copied());
        while let Some(here) = walk.pop() {
            for (other, node) in graph.nodes() {
                if node.inputs().contains(&here) && loops_back.insert(other) {
                    walk.push(other);
                }
            }
        }
        let newest = |wanted: fn(&Node) -> bool| {
            let mut found: Vec<NodeId> = graph
                .nodes()
                .filter(|(id, n)| wanted(n) && !loops_back.contains(id))
                .map(|(id, _)| id)
                .collect();
            found.sort_by_key(|id| (self.doc.created.get(id).copied().unwrap_or(0), id.raw()));
            found.last().copied()
        };
        newest(|n| matches!(n, Node::Point { .. }))
            .or_else(|| newest(|n| matches!(n, Node::Plane { .. })))
    }

    /// A ramp nothing reads, which is the one to reach for before building another.
    ///
    /// **Because the switch and the delete rule have to be one design.** Turning "varies through
    /// space" off leaves the ramp standing — `ramp-decision.md` §7.2 makes it a block, and
    /// deleting a visible block as a side effect of a toggle is what the delete cascade was
    /// already changed to stop doing. But building a fresh one every time it is turned back on
    /// then piles up ramps and points that nothing reads, which is what was reported. Reaching
    /// for the idle one first makes the round trip land back where it started.
    ///
    /// Only an *idle* one. A ramp already driving something is not spare: taking it would
    /// silently rewire whatever was using it, and wanting a second, differently-shaped ramp is
    /// the ordinary case rather than the odd one.
    ///
    /// The newest, because that is the one just switched off.
    pub(crate) fn spare_ramp(&self) -> Option<NodeId> {
        let graph = self.part_graph();
        let consumed = graph.consumed();
        let mut idle: Vec<NodeId> = graph
            .nodes()
            .filter(|(id, n)| matches!(n, Node::Ramp { .. }) && !consumed.contains(id))
            .map(|(id, _)| id)
            .collect();
        idle.sort_by_key(|id| (self.doc.created.get(id).copied().unwrap_or(0), id.raw()));
        idle.last().copied()
    }

    /// Every ramp in the part, for a socket to choose between.
    fn ramps_available(&self) -> Vec<(NodeId, String)> {
        let graph = self.part_graph();
        let mut found: Vec<NodeId> = graph
            .nodes()
            .filter(|(_, n)| matches!(n, Node::Ramp { .. }))
            .map(|(id, _)| id)
            .collect();
        found.sort_by_key(|id| (self.doc.created.get(id).copied().unwrap_or(0), id.raw()));
        found.into_iter().map(|id| (id, self.body_label(id))).collect()
    }

    /// Which ramp drives this number, and which others it could use instead.
    ///
    /// **One picker for every driveable socket** (`ramp-decision.md` §7), so the second socket
    /// cannot grow a different way of choosing — the mistake D9 is about, and the one the
    /// row-embedded ramp would have made unavoidable by giving each row a ramp of its own.
    ///
    /// Offered only where §7 marks the socket driveable, which today is `Offset.distance`,
    /// `Morph.t`, `Pipe.radius`, a lattice phase's own number and, since
    /// `round-chamfer-spec.md` §9.1, `Blend.radius`. `driven_refusal` stays the backstop underneath: a hand-edited file naming a
    /// driver on a sphere's radius is refused by name whether or not any panel offered it.
    ///
    /// Returns the choice when it changed — `Some(None)` for "back to one number", and
    /// `Some(Some(ramp))` for a ramp to wire. A ramp already driving something else is offered
    /// exactly like any other: one ramp driving two sockets is the arrangement `ParamRef::Field`
    /// was built for, and it is the case worth having.
    pub(crate) fn pick_driver(
        &mut self,
        ui: &mut egui::Ui,
        salt: &str,
        current: Option<NodeId>,
    ) -> Option<Option<NodeId>> {
        let ramps = self.ramps_available();
        // **Shown even with no ramp in the document**, saying so, rather than the row vanishing:
        // what a block can do should be readable from the block, and a control that is not there
        // cannot say why. Create > Ramp is what fills it.
        if ramps.is_empty() {
            ui.horizontal(|ui| {
                row_label(ui, "Driven by:");
                ui.add_enabled(
                    false,
                    egui::Button::new(egui::RichText::new("no ramp yet").weak()),
                )
                .on_disabled_hover_text(
                    "A ramp is what makes this vary through space. Create > Ramp adds one, and \
                     it appears here.",
                );
            });
            return None;
        }
        let mut chosen = None;
        ui.horizontal(|ui| {
            row_label(ui, "Driven by:");
            // **This list says which ramp, and nothing else.** It used to carry "One number" as
            // an entry as well, which is the same answer the button beside it already gives —
            // one idea with two controls, and the two could disagree about which was current.
            // The button turns the driver off; this chooses between drivers.
            // **A field, with the off switch where it already was.** Turning the driver off is the
            // "One number" button beside this, and a × here would be a second control for the same
            // answer — the duplication the comment above removed from the list. So this only
            // chooses *which* ramp, and with nothing driving it is empty and says what to do.
            let offered = self.block_choices(&ramps);
            let shown = current
                .map_or_else(String::new, |id| self.shown_label(&offered, id, "missing"));
            if let Some(picked) = self.reference_field_in(
                ui,
                (salt, "driver"),
                current,
                &shown,
                &offered,
                RefKind::Ramp,
                DropsFrom::block(|n| n),
            ) && current != Some(picked)
            {
                chosen = Some(Some(picked));
            }
        });
        // The ramp itself, under the row that names it (§7.1, §17): a graded wall's ramp is a
        // block like any other, and a number driven by one is where it is read.
        if let Some(ramp) = current {
            self.host_nested_block(ui, ramp);
        }
        chosen
    }

    /// Points a driveable socket at a ramp that already exists.
    ///
    /// The other half of [`Self::pick_driver`]: the switch makes a ramp and wires it, this
    /// wires one that is already there. Both write the same `ParamRef::Field`, which is why a
    /// ramp reached either way is the same ramp.
    pub(crate) fn wire_driver(&mut self, holder: NodeId, slot: usize, ramp: NodeId) {
        let mut graph = self.part_graph().clone();
        if let Some(node) = graph.node_mut(holder)
            && let Some(reference) = node.param_refs_mut().get_mut(slot)
        {
            **reference = ParamRef::Field(ramp);
        }
        // Checked before it is committed: a ramp reading the very body this drives is a cycle,
        // and a cyclic graph does not compile. Refusing here leaves the socket as it was rather
        // than committing a document that opens to an error naming a node.
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        self.set_part_graph(graph);
        self.mark_changed();
        self.rebuild_render_resources();
        self.say(format!("Driven by {}", self.body_label(ramp)));
    }

    /// Places a ramp: a block that reads one field and gives back a number.
    ///
    /// **A block rather than something hidden inside the row it drives** — `ramp-decision.md`
    /// §7.2. One that lives in a row cannot be selected, cannot be reused by a second socket,
    /// and cannot be found once the row is collapsed, which makes it a second kind of ramp.
    /// There is one kind, and this and the "varies through space" switch both produce it.
    ///
    /// Whatever `ramp in=0,60 out=0,1 shape=G1 input=plane` said is read before anything is
    /// added, so a refused key leaves no half-made block behind.
    pub(crate) fn add_ramp_with(&mut self, args: &Args) {
        // **Nothing else is created.** It used to reach for a field to read and build a point
        // when it found none, so placing a ramp silently added a block nobody asked for — and a
        // point at the origin is rarely the one wanted anyway. A ramp with nothing to read is a
        // legal state now that `Node::Ramp::input` says so, which is what `ramp-decision.md`
        // §7.2 describes: a root that draws nothing until it is told what to read.
        let node = Node::Ramp {
            input: fieldform_scene::Operand::empty(),
            in_min: ParamRef::Literal(0.0),
            in_max: ParamRef::Literal(10.0),
            out_min: ParamRef::Literal(0.0),
            out_max: ParamRef::Literal(1.0),
            shape: RampShape::Linear,
        };
        let node = match console_keys::ramp_with_keys(node, args, &|t| self.lookup(t)) {
            Ok(node) => node,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let mut graph = self.part_graph().clone();
        let ramp = graph.add(node);
        self.set_part_graph(graph);
        self.select_body(Some(ramp));
        self.mark_changed();
        self.rebuild_render_resources();
        self.say(
            "Added a ramp — pick what it Reads in its block, then pick the ramp under \
             Amount on an offset or a morph.",
        );
    }

    /// Adds a primitive as a body, at a size you can predict.
    ///
    /// This used to be a fraction of whatever was already in the scene, on the reasoning that a
    /// fixed size is invisible beside a metre cage. What that actually produced was a sphere
    /// whose size depended on what happened to be visible when you asked for it — 50 mm, then
    /// 19 mm because the first sphere was now the whole scene. A number you can predict beats a
    /// number that is scaled to something you were not thinking about, and both are one drag of
    /// the radius from right.
    ///
    /// Whatever was typed after its word -- `sphere radius=10 centre=40,0,0` -- is read before
    /// anything is added, so a refused key leaves no default block behind.
    pub(crate) fn add_primitive_with(&mut self, kind: PrimitiveKind, args: &Args) {
        let centre = [ParamRef::Literal(0.0); 3];
        let node = kind.node(centre, DEFAULT_BODY_MM);
        let node = match console_keys::primitive_with_keys(node, &kind.word(), args, &|t| self.lookup(t)) {
            Ok(node) => node,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        // A plane told where to be and which way to face is not then turned to the drawing
        // plane: the keys said it.
        let placed_by_keys = args.get("origin").is_some() || args.get("turn").is_some();
        let mut graph = self.part_graph().clone();
        let id = graph.add(node);

        // **A plane made while drawing on one arrives facing the same way, and is one block.**
        // This used to wrap the plane in a `Node::Transform` to turn it, because back then a
        // plane had no facing of its own -- and two rows appeared for one thing: "Plane" and
        // "Placed (Plane)". Reported as exactly that, as a bug, which it is: the second row is
        // a block nobody asked for, and aiming anything at the *inner* plane then read the
        // facing it had before it was turned.
        //
        // `Node::Plane` carries its own `turn` now, so the facing goes where the facing lives.
        // `Graph::with_turns_expanded` still rewrites it into that same placement for
        // evaluation -- one description of what a turn is, read by both paths -- but the
        // document holds one block and the list shows one row.
        //
        // Only a plane. A sphere does not care which way it faces, and deciding that a cylinder
        // should stand on the drawing plane rather than across it is a guess about intent that
        // this does not need to make. A plane has exactly one thing to say and the drawing plane
        // has already said it.
        if kind == PrimitiveKind::Plane
            && !placed_by_keys
            && let Some((origin, rotation)) = self
                .drawing_frame()
                .and_then(|f| {
                    let numbers = |r: &ParamRef| Some(self.resolve(*r));
                    fieldform_field::resolve_frame(&self.doc, self.part_graph(), &f, &numbers).ok()
                })
        {
            // Intrinsic XYZ Euler angles in degrees, which is what a turn stores and what the
            // panel shows, so the numbers a person reads back are the ones they could have
            // typed.
            let (x, y, z) = rotation.euler_angles();
            if let Some(Node::Plane { origin: at, turn, .. }) = graph.node_mut(id) {
                *at = fieldform_scene::Place::At([
                    ParamRef::Literal(origin.x),
                    ParamRef::Literal(origin.y),
                    ParamRef::Literal(origin.z),
                ]);
                *turn = Turn::By([
                    ParamRef::Literal(x.to_degrees()),
                    ParamRef::Literal(y.to_degrees()),
                    ParamRef::Literal(z.to_degrees()),
                ]);
            }
        }

        self.set_part_graph(graph);
        self.select_body(Some(id));
        // Same reasoning as a swept pipe: what was just made is nearly always one operand of
        // the next boolean.
        self.boolean_b = Some(id);
        if self.boolean_a.is_none() || self.boolean_a == Some(id) {
            self.boolean_a =
                self.doc.listed_bodies(self.part_graph()).into_iter().find(|b| *b != id);
        }
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        self.say(format!("Added a {}", kind.label().to_lowercase()));
    }

    /// Adds a lattice as a body, sized so a default part gets a handful of cells across it.
    ///
    /// A lattice alone is not a part — it is infinite, and the status line says so rather than
    /// leaving the viewport to explain why everything went solid. Intersecting it with a body
    /// is the whole operation, and that is a boolean the user makes next.
    /// `tpms` with its keys: the default gyroid on a rectangular map, then everything typed
    /// applied to it before it is added (`console_keys::lattice_with_keys`), so a refused key
    /// makes nothing. A menu passes no keys and gets the default.
    pub(crate) fn add_lattice_with(&mut self, args: &Args) {
        let cell = LatticeCell::Gyroid;
        let node = Node::Lattice {
            frame: fieldform_scene::Frame::default(),
            cell,
            map: CellMap::Rectangular { cell: ParamRef::Literal(DEFAULT_BODY_MM * 0.2) },
            // **Seeded from what the cell supports, not from a sheet.** A fifth of the default
            // body, so a new lattice reads as a pattern rather than as one enormous cell or a
            // grey haze -- but a beam cell has no wall to give that to, and handing it one is a
            // document the compile refuses the moment it is drawn. The panel already re-seeds
            // on the next frame, which turned a hard refusal into a flicker and a status line
            // that had already said the block was added.
            phase: seeded_phase(
                fieldform_field::CellKind::from(cell).supported_bandings()[0],
                DEFAULT_BODY_MM * 0.02,
            ),
        };
        let offered = |cell: &LatticeCell| fieldform_field::CellKind::from(*cell).supported_bandings();
        let seed = |banding: Banding| seeded_phase(banding, DEFAULT_BODY_MM * 0.02);
        let node = match console_keys::lattice_with_keys(node, args, &offered, &seed, &|t| self.lookup(t)) {
            Ok(node) => node,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        self.add_lattice_node(node);
    }

    fn add_lattice_node(&mut self, node: Node) {
        let Node::Lattice { cell, .. } = &node else { return };
        let cell = *cell;
        let mut graph = self.part_graph().clone();
        let id = graph.add(node);
        // A phase number typed as a block's name is `ParamRef::Field` of whatever that block
        // is, and only a ramp may drive it: refused here, before it is committed to a document
        // that cannot be drawn.
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        self.set_part_graph(graph);
        self.select_body(Some(id));
        // Nearly always one operand of the next boolean, the same as a new solid.
        self.boolean_b = Some(id);
        if self.boolean_a.is_none() || self.boolean_a == Some(id) {
            self.boolean_a =
                self.doc.listed_bodies(self.part_graph()).into_iter().find(|b| *b != id);
        }
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        self.say(format!(
            "Added a {} — intersect it with a body to infill",
            lattice_cell_label(cell).to_lowercase(),
        ));
    }

    /// How many blocks could be operands here if they were named, and are not offered.
    ///
    /// **Asked so the panel can say so**, at the moment somebody is looking down a list for a
    /// body that is not in it. Naming a block is what makes it reusable — that has been true and
    /// tested since the feature was built — and the only place it was said was a tooltip on the
    /// name field, which is the one place a person is *not* looking when the thing they want is
    /// missing from a dropdown. The owner asked for this feature by name while it already
    /// existed, which is as clear a measurement of an invisible affordance as there is.
    fn unnamed_blocks(&self, id: NodeId) -> usize {
        let graph = self.part_graph();
        let offered =
            operand_choices_in(graph, id, |n| self.doc.made_at(n));
        graph
            .nodes()
            .map(|(n, _)| n)
            .filter(|n| *n != id && !offered.contains(n))
            .count()
    }

    /// Says, on the picture itself, that the picture is not the document yet.
    ///
    /// **Because the viewport is honest about the *previous* shape and nothing else explains
    /// that.** A shader the driver has not compiled before costs seconds — measured at 3.2 s for
    /// the owner's wing by `pipeline_cost`, and paid once per combination of hidden bodies, since
    /// hiding takes a body out of the graph and a different graph is a different shader
    /// (`hide_variants`). The compile runs on a worker so the application stays live, which is
    /// right and is exactly what makes the stale picture confusing: everything responds, and the
    /// shape is the one from before the change.
    ///
    /// It was already said in the console. The owner asked for it where it cannot be missed:
    /// "we should probably have a very visible note that the shader is compiling".
    ///
    /// **After a beat, not immediately.** A shader the driver has seen before comes back from its
    /// own disk cache in about ten milliseconds, and a banner that flashed on every rebuild would
    /// be noise in front of the model. A quarter of a second is under the threshold where a wait
    /// starts to feel like a fault, so anything that appears is a wait worth knowing about.
    ///
    /// On the foreground layer, so it is over the traced image whatever order the viewport paints
    /// in, and it carries the seconds so a long compile reads as slow rather than as hung.
    pub(crate) fn compiling_notice(&self, ui: &egui::Ui, rect: egui::Rect) {
        const SHOW_AFTER: std::time::Duration = std::time::Duration::from_millis(250);
        let Some(waited) = self.traced.is_compiling().filter(|w| *w >= SHOW_AFTER) else {
            return;
        };
        let painter = ui.ctx().layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("compiling-notice"),
        ));
        let text = format!(
            "Compiling shader — {:.1} s. Showing the shape from before the change.",
            waited.as_secs_f32()
        );
        let galley =
            painter.layout_no_wrap(text, egui::FontId::proportional(13.0), egui::Color32::WHITE);
        let pad = egui::vec2(10.0, 6.0);
        let at = egui::pos2(rect.center().x - galley.size().x * 0.5, rect.top() + 12.0);
        let box_rect = egui::Rect::from_min_size(at - pad, galley.size() + pad * 2.0);
        // Amber rather than red: nothing is wrong, something is pending.
        painter.rect_filled(box_rect, 6.0, egui::Color32::from_rgba_unmultiplied(40, 32, 8, 225));
        painter.rect_stroke(
            box_rect,
            6.0,
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(200, 160, 60)),
            egui::StrokeKind::Inside,
        );
        painter.galley(at, galley, egui::Color32::WHITE);
    }

    /// The line under an operand list that says why a block might be missing from it.
    ///
    /// One place, because every block that takes a body shows it and they must say the same
    /// thing: the answer to "why is my body not in this list" cannot depend on which block you
    /// happen to be standing in.
    pub(crate) fn reuse_hint(&self, ui: &mut egui::Ui, id: NodeId) {
        let hidden = self.unnamed_blocks(id);
        if hidden == 0 {
            return;
        }
        ui.label(
            egui::RichText::new(format!(
                "{hidden} more block{} — name one to reuse it here",
                if hidden == 1 { "" } else { "s" }
            ))
            .weak(),
        )
        .on_hover_text(
            "These lists offer the bodies and any block you have named. Naming a block makes \
             it an operand for every other block, so one shape can be cut from two lattices \
             instead of being modelled twice — which is also how a body is combined with \
             something derived from itself. Clearing the name takes it back out of the lists.",
        );
    }

    /// The body a one-input block works on, offered the way a boolean offers its two.
    ///
    /// **Every block that takes a body lets you say which**, which is the owner\'s rule and was
    /// true of exactly one block before this. A shell, an offset, a twist, a mirror, a pattern
    /// and a placement all took whatever happened to be selected when they were made, and
    /// nothing afterwards could change it — so a shell on the wrong body meant deleting it and
    /// building it again, while a *boolean* on the wrong body was two clicks. One kind of input,
    /// two ways of saying it, and which one you got depended on which block you were in.
    ///
    /// The same `operand_choices` a boolean asks, so a named block is offered here too and the
    /// hint underneath reads the same everywhere.
    ///
    /// **A slot in and a slot out**, rather than a block either way: the block being drawn may be
    /// one whose slot is empty, and an empty slot picked past is the panel quietly putting
    /// something in it. The field itself offers no way back out -- that is §7.2's drag-out --
    /// so an empty slot stays empty until a block is chosen for it.
    ///
    /// `Self::operand_input` is the row a boolean's operands get, with the same slot and the
    /// in-panel way out; the standing debt is that the blocks reaching the panel through here
    /// draw the older row, and the repair is to forward this to that one. Said in both places so
    /// neither reads as the settled arrangement.
    pub(crate) fn pick_input(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        input: fieldform_scene::Operand,
    ) -> fieldform_scene::Operand {
        let choices = self.operand_choices(id);
        let mut chosen = input.source();
        self.pick_operand(ui, "Of:", "edit_input", id, &choices, &mut chosen);
        self.reuse_hint(ui, id);
        ui.separator();
        chosen.map_or(input, fieldform_scene::Operand::filled)
    }

    /// Where a place is, for the manipulator and the panels.
    ///
    /// Through `fieldform_field::resolve_place`, which the field and the shader also use, so the
    /// handle sits where the surface is rather than somewhere agreeing with it by coincidence.
    pub(crate) fn resolve_place(&self, place: &fieldform_scene::Place) -> Point3<f64> {
        let numbers = |r: &ParamRef| Some(self.resolve(*r));
        fieldform_field::resolve_place(&self.doc, self.part_graph(), place, &numbers)
            .unwrap_or_else(|_| Point3::origin())
    }

    /// The blocks a place can be taken from: a point, or another plane.
    ///
    /// Nothing that consumes this one, for the reason `operand_choices_in` gives — picking it
    /// would close a loop, and a loop here is a stack that does not come back rather than an
    /// error naming a node.
    fn place_sources(&self, id: NodeId) -> Vec<(NodeId, String)> {
        self.sources_for(id)
    }

    /// The blocks that can answer "where" and "which way": a point, a plane, or a placement
    /// holding one.
    ///
    /// **One list for places and for frames.** They were two, and the frame one had no cycle
    /// guard at all — so a mirror could be aimed at a plane built on the very body it was
    /// mirroring, which compiles to `Cyclic` and takes the viewport with it.
    ///
    /// A placement holding a plane counts, because a block that has been turned *is* a
    /// placement holding that block. Leaving those out is why aiming at a turned plane offered
    /// nothing to pick: the row said "Plane" and the list did not have it.
    pub(crate) fn sources_for(&self, id: NodeId) -> Vec<(NodeId, String)> {
        let graph = self.part_graph();
        let loops_back = graph.downstream_of(id);
        let bottoms_out = |mut n: NodeId| loop {
            match graph.node(n) {
                Some(Node::Point { .. }) | Some(Node::Plane { .. }) => return true,
                // A placement whose slot is empty holds nothing, so it answers neither "where"
                // nor "which way" and is not a place to aim at.
                Some(Node::Transform { input, .. }) => match input.source() {
                    Some(held) => n = held,
                    None => return false,
                },
                _ => return false,
            }
        };
        let mut out: Vec<(NodeId, String)> = graph
            .nodes()
            .filter(|(n, _)| *n != id && !loops_back.contains(n) && bottoms_out(*n))
            .map(|(n, _)| (n, self.body_label(n)))
            .collect();
        out.sort_by_key(|(n, _)| n.raw());
        out
    }

    /// Which way a block is turned, on one line: three angles, or what it is aimed at.
    ///
    /// **One V for the facing**, the way [`Self::place_fields`] has one for the position. A
    /// facing is one thing, and three angles each offering to be named put three marks on a row
    /// describing a single fact. Asked for in those words: "treat the turn the same way we treat
    /// the `at` coordinate — the whole XYZ matrix can become a V, instead of each X Y or Z being
    /// treated separately".
    ///
    /// And the V is where a block gets *aimed*, which is the other half of the same request. A
    /// cylinder could already be pointed at a line, because its [`Ends`] can name one; a cuboid,
    /// a torus and a plane could not, because three angles cannot be pointed at anything. Now
    /// they can be pointed at a plane or a line, and moving that re-aims them.
    ///
    /// A single angle can still be named and driven, and shows its own mark when it is. What it
    /// cannot be is the whole answer.
    pub(crate) fn turn_fields(&mut self, ui: &mut egui::Ui, id: NodeId, turn: Turn) -> Turn {
        let sources = self.sources_for(id);
        let lines = self.axis_candidates(id);
        let mut out = turn;
        // Set inside the menu and acted on outside it, because making a block edits the graph
        // and the menu is holding a borrow of the panel that draws it.
        let mut declare = false;
        ui.horizontal(|ui| {
            match out {
                Turn::By(angles) => {
                    out = Turn::By(self.xyz_row_naming(
                        ui,
                        "Angle:",
                        angles,
                        1.0,
                        DEGREES,
                        "\u{b0}",
                        Naming::OnlyWhenSet,
                    ));
                }
                _ => {
                    row_label(ui, "Angle:");
                    // Across the three number columns, so the V after it stays where it is in
                    // every other row.
                    let width = AXIS_FIELD_W * 3.0;
                    let (name, hint) = match out {
                        Turn::From { node } => (
                            self.body_label(node),
                            "Facing the same way as this block — turn it and this follows",
                        ),
                        Turn::Along { curve } => (
                            self.doc
                                .curves
                                .get(&curve)
                                .map_or_else(|| "a curve".to_string(), |c| c.name.clone()),
                            "Pointing along this line — drag its ends to aim this",
                        ),
                        Turn::Between { from, to } => (
                            // `\u{203a}` for the reason the Add menu's mark is one: the
                            // proportional face has no arrow, and an arrow it cannot draw is an
                            // empty box.
                            format!("{} \u{203a} {}", self.body_label(from), self.body_label(to)),
                            "Pointing from the first of these to the second — move \
                             either and this re-aims",
                        ),
                        Turn::Tangent { curve } => (
                            format!(
                                "across {}",
                                self.doc
                                    .curves
                                    .get(&curve)
                                    .map_or_else(|| "a curve".to_string(), |c| c.name.clone())
                            ),
                            "Square across the curve where it starts. These angles are read \
                             from the curve, not typed — redraw it and this follows.",
                        ),
                        Turn::By(_) => unreachable!("handled above"),
                        // The panel reads the document's own graph, which never holds one:
                        // `Turn::Absorbed` is made by flattening and lives only in the lowered
                        // graph a compiler reads.
                        Turn::Absorbed { .. } => {
                            unreachable!("a lowered graph never reaches the panel")
                        }
                    };
                    // The angles it currently comes to, so an aimed block still says where it
                    // is facing rather than only what it is reading.
                    let (x, y, z) = self.aim_of(&out).euler_angles();
                    ui.allocate_ui(egui::vec2(width, ui.spacing().interact_size.y), |ui| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(name).italics()).on_hover_text(hint);
                            ui.label(
                                egui::RichText::new(format!(
                                    "{:.1}, {:.1}, {:.1}\u{b0}",
                                    x.to_degrees(),
                                    y.to_degrees(),
                                    z.to_degrees()
                                ))
                                .weak(),
                            );
                        });
                    });
                }
            }

            let taken = !matches!(out, Turn::By(_));
            let mark = egui::RichText::new("V")
                .color(if taken { VARIABLE_ON } else { VARIABLE_OFF });
            let mark = if taken { mark.strong() } else { mark };
            ui.menu_button(mark, |ui| {
                if taken
                    && ui
                        .button("Type the angles instead")
                        .on_hover_text("Keeps the way it faces now, so nothing jumps")
                        .clicked()
                {
                    let (x, y, z) = self.aim_of(&out).euler_angles();
                    out = Turn::By([
                        ParamRef::Literal(x.to_degrees()),
                        ParamRef::Literal(y.to_degrees()),
                        ParamRef::Literal(z.to_degrees()),
                    ]);
                    ui.close();
                }
                // **Offered first, and offered whether or not anything exists to point at.**
                // This is the half that was missing, and the same half a position's V was
                // missing before "Make a point here": a facing typed here can become a block in
                // its own right without leaving the panel, which is what makes it reusable
                // rather than merely re-typeable. Reported as "if the document is empty with
                // just a plane you can't declare turn/angle a variable, which is incorrect".
                if !taken
                    && ui
                        .button("Make a plane facing this way")
                        .on_hover_text(
                            "Makes a named plane at this block, facing the way it faces, and \
                             aims this at it — so other blocks can be aimed at the same \
                             one and all of them turn together",
                        )
                        .clicked()
                {
                    declare = true;
                    ui.close();
                }
                // Both lists always, empty or not. A section that vanishes cannot say why.
                ui.separator();
                ui.label(egui::RichText::new("face the same way as").weak());
                if sources.is_empty() {
                    ui.add_enabled(false, egui::Button::new("no plane yet").frame(false))
                        .on_disabled_hover_text(
                            "Add a plane, or use the offer above to make one from this block's \
                             own facing.",
                        );
                }
                for (n, name) in &sources {
                    let here = matches!(out, Turn::From { node } if node == *n);
                    if ui.selectable_label(here, name).clicked() {
                        out = Turn::From { node: *n };
                        ui.close();
                    }
                }
                // **Between two of them**, which is how a facing is said when there is no line
                // to draw: "a vector ... read from geometry -- a line, or two points". Backlog
                // item 18, and the same `Frame::Between` the aimed blocks read, so two points
                // mean one thing across the application.
                ui.separator();
                ui.label(egui::RichText::new("point between").weak());
                if sources.len() < 2 {
                    ui.add_enabled(false, egui::Button::new("needs two blocks").frame(false))
                        .on_disabled_hover_text(
                            "Add a second point or plane and this becomes available: the facing \
                             then runs from the first to the second.",
                        );
                }
                for (n, name) in &sources {
                    if sources.len() < 2 {
                        break;
                    }
                    let here = matches!(out, Turn::Between { from, .. } if from == *n);
                    if ui
                        .selectable_label(here, format!("from {name}"))
                        .on_hover_text("The other end is whichever is picked below it")
                        .clicked()
                    {
                        // The second end is whatever it already was, or the first other block
                        // there is -- a facing between a block and itself has no direction.
                        let other = match out {
                            Turn::Between { to, .. } if to != *n => to,
                            _ => sources.iter().map(|(o, _)| *o).find(|o| o != n).unwrap_or(*n),
                        };
                        out = Turn::Between { from: *n, to: other };
                        ui.close();
                    }
                }
                if let Turn::Between { from, to } = out
                    && sources.len() >= 2
                {
                    ui.label(egui::RichText::new("toward").weak());
                    for (n, name) in &sources {
                        if *n == from {
                            continue;
                        }
                        if ui.selectable_label(to == *n, format!("toward {name}")).clicked() {
                            out = Turn::Between { from, to: *n };
                            ui.close();
                        }
                    }
                }
                ui.separator();
                ui.label(egui::RichText::new("point along").weak());
                if lines.is_empty() {
                    ui.add_enabled(false, egui::Button::new("no line yet").frame(false))
                        .on_disabled_hover_text(
                            "Draw a line and it appears here. This block's own +Z would then be \
                             turned onto the way that line runs.",
                        );
                }
                for (c, name) in &lines {
                    let here = matches!(out, Turn::Along { curve } if curve == *c);
                    if ui.selectable_label(here, name).clicked() {
                        out = Turn::Along { curve: *c };
                        ui.close();
                    }
                }

            })
            .response
            .on_hover_text(
                "Aim this at a plane or a line, so moving that re-aims this — several \
                 blocks aimed at one turn together, which is what copying the angles cannot do.",
            );
        });
        if declare {
            let plane = self.plane_facing_turn(id, &out);
            out = Turn::From { node: plane };
        }
        out
    }

    /// Where a block sits, on one line: three numbers, or the block it sits on.
    ///
    /// **One row and one mark, because it is one decision.** This used to be three rows -- a
    /// `Typed` / `On a block` pair of buttons, then the numbers, then a combo box naming the
    /// block -- for a value that is a position. Two of those rows existed only to say which of
    /// two things the third one was, which is what a mark on the value itself says without
    /// spending a line on it.
    ///
    /// The mark is the same green V a named value wears everywhere else, and means the same
    /// thing here: this value comes from somewhere with a name rather than from numbers typed
    /// in place. Dim is an offer, bright is a fact. It is offered even when the document holds
    /// no point to sit on, and the menu says so -- withholding a control cannot explain itself,
    /// which the revolve axis taught once already.
    ///
    /// Laid out as `label | X | Y | Z | V`, which is the curve point rows' layout, so a
    /// sphere's centre and a curve's start line up column for column down the panel.
    pub(crate) fn place_fields(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        label: &str,
        place: fieldform_scene::Place,
    ) -> fieldform_scene::Place {
        let sources = self.place_sources(id);
        let mut out = place;
        // Set inside the menu and acted on outside it, because making a block edits the graph
        // and the menu is holding a borrow of the panel that draws it.
        let mut declare = false;
        ui.horizontal(|ui| {
            match out {
                fieldform_scene::Place::At(xyz) => {
                    // One V for the position, at the end of the row, rather than one per
                    // coordinate. A single axis can still be driven -- and shows its mark when
                    // it is -- but naming happens for the place, which is the thing.
                    out = fieldform_scene::Place::At(self.xyz_row_naming(
                        ui,
                        label,
                        xyz,
                        0.5,
                        ANYWHERE_MM,
                        " mm",
                        Naming::OnlyWhenSet,
                    ));
                }
                _ => {
                    row_label(ui, label);
                    // Across the three number columns, so the V after it stays where it is in
                    // every other row.
                    let width = AXIS_FIELD_W * 3.0;
                    let at = self.resolve_place(&out);
                    let (name, hint) = match out {
                        fieldform_scene::Place::From { node } => (
                            self.body_label(node),
                            "Sitting on this block — move it and this follows",
                        ),
                        fieldform_scene::Place::StartOf { curve } => (
                            format!(
                                "start of {}",
                                self.doc
                                    .curves
                                    .get(&curve)
                                    .map_or_else(|| "a curve".to_string(), |c| c.name.clone())
                            ),
                            "Standing where this curve begins — drag that end and this \
                             follows it",
                        ),
                        fieldform_scene::Place::At(_) => unreachable!("handled above"),
                    };
                    ui.allocate_ui(egui::vec2(width, ui.spacing().interact_size.y), |ui| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(name).italics())
                                .on_hover_text(hint);
                            ui.label(
                                egui::RichText::new(format!(
                                    "{:.2}, {:.2}, {:.2}",
                                    at.x, at.y, at.z
                                ))
                                .weak(),
                            );
                        });
                    });
                }
            }

            let taken = !matches!(out, fieldform_scene::Place::At(_));
            let mark = egui::RichText::new("V")
                .color(if taken { VARIABLE_ON } else { VARIABLE_OFF });
            let mark = if taken { mark.strong() } else { mark };
            ui.menu_button(mark, |ui| {
                if taken
                    && ui
                        .button("Type the numbers instead")
                        .on_hover_text("Keeps where it is now, so nothing jumps")
                        .clicked()
                {
                    // Keeping where it is, so switching back does not send the block to the
                    // origin -- the mistake the frame picker made once and says so about.
                    let here = self.resolve_place(&out);
                    out = fieldform_scene::Place::At([
                        ParamRef::Literal(here.x),
                        ParamRef::Literal(here.y),
                        ParamRef::Literal(here.z),
                    ]);
                    ui.close();
                }
                // **Offered first, and offered whether or not anything exists to pick.** This
                // is the half that was missing: a position typed here can become a point block
                // in its own right without leaving the panel, which is what makes the numbers
                // reusable rather than merely re-typeable.
                if !taken
                    && ui
                        .button("Make a point here")
                        .on_hover_text(
                            "Makes a named point block at these coordinates and sits this on \
                             it, so other blocks can sit on the same one",
                        )
                        .clicked()
                {
                    declare = true;
                    ui.close();
                }
                ui.separator();
                ui.label(egui::RichText::new("sit on").weak());
                if sources.is_empty() {
                    ui.add_enabled(false, egui::Button::new("no point yet").frame(false))
                        .on_disabled_hover_text(
                            "Add a point or a plane, or use the offer above to make one here.",
                        );
                }
                // **And where a curve begins**, which is a position in the document as much as a
                // point block is -- and the one a plane standing at the start of a pipe wants,
                // since dragging that end should carry the plane with it.
                let lines = self.axis_candidates(id);
                ui.separator();
                ui.label(egui::RichText::new("stand where a curve starts").weak());
                if lines.is_empty() {
                    ui.add_enabled(false, egui::Button::new("no line yet").frame(false))
                        .on_disabled_hover_text(
                            "Draw a line and it appears here. This would then stand at its \
                             first point and follow it.",
                        );
                }
                for (c, name) in &lines {
                    let here = matches!(out, fieldform_scene::Place::StartOf { curve } if curve == *c);
                    if ui.selectable_label(here, format!("start of {name}")).clicked() {
                        out = fieldform_scene::Place::StartOf { curve: *c };
                        ui.close();
                    }
                }
                for (n, name) in &sources {
                    let here = matches!(out, fieldform_scene::Place::From { node } if node == *n);
                    if ui.selectable_label(here, name).clicked() {
                        out = fieldform_scene::Place::From { node: *n };
                        ui.close();
                    }
                }
            })
            .response
            .on_hover_text(
                "Sit this on a point or a plane, so moving that block moves this — \
                 several bodies on one point move together, which is what copying the numbers \
                 cannot do.",
            );
        });
        if declare {
            let here = self.resolve_place(&out);
            let point = self.point_block_at(here);
            out = fieldform_scene::Place::From { node: point };
        }
        out
    }

    /// How far a block with no volume of its own sits from the surface of another block.
    ///
    /// **The number that decides whether blending to a point does anything at all.** Below it
    /// the blend reads exactly nought at the point and changes nothing else -- no material, and
    /// the solid still stops where it did -- which looks like success if the only thing read is
    /// the value at the point. Above it the blend swallows the point and there is a shape.
    /// Measured in `fieldform-field/examples/boolean_with_nothing.rs`: bisecting for the
    /// smallest radius that puts material at the point, over gaps of 10, 20, 40 and 80 mm, gives
    /// a ratio of exactly 1.0000 every time. So the rule is simply "more than this".
    ///
    /// Compiled from the other block alone rather than read off the part, because the part's
    /// field is the union of every body and would answer with whichever surface happens to be
    /// nearest. Everything the block is not made of is removed by id, which `Graph::remove`
    /// allows and `remove_body` would refuse for a node something still consumes.
    ///
    /// Not cheap, and it does not need to be: it runs when an operand is picked, not per frame.
    pub(crate) fn gap_between(&self, empty: NodeId, solid: NodeId) -> Option<f64> {
        let at = match self.part_graph().node(empty)? {
            Node::Point { at } => self.resolve_point(at),
            // Only a point today. Anything else with no volume would want its own answer to
            // "where is it", and there is nothing else with no volume.
            _ => return None,
        };
        let graph = expect::graph_of(self.part_graph(), solid);
        let field = GraphField::compile_graph(&self.doc, &graph, 0.05).ok()?;
        Some(field.distance(at))
    }


    /// One row: a label and a dropdown of the bodies a block may take.
    ///
    /// Restored with master's single-input pickers, which need it. The boolean's own operands
    /// are a list now — rows that can be added to and removed — and a list is not this; what
    /// the two still share is `operand_choices`, so a named block is offered everywhere and the
    /// hint under both reads the same.
    ///
    /// `consumer` is the block whose slot this is, because what may go in one depends on what the
    /// block does with it -- see [`operand_refusal`]. One funnel for every operand field, so no
    /// block can start offering something the rest refuse.
    pub(crate) fn pick_operand(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        salt: &str,
        consumer: NodeId,
        choices: &[(NodeId, String)],
        current: &mut Option<NodeId>,
    ) {
        // **A field, not a dropdown — the same one a boolean's operands are.** It was left a
        // `ComboBox` when the rest were converted, and a panel with two spellings of "which
        // block" is what `reference_field` exists to end (item 72).
        //
        // **The `Option` is the slot, and there is still no "clear" to offer**, which is why this
        // takes no ×. A caller's slot may be empty -- `Operand` says so, and a block waiting to be
        // told what to read is a legal state -- so "none" is now both what an empty slot shows and
        // what shows when the input is not among the choices. Emptying one is §7.2's drag-out and
        // §6.5's *Remove contents* and nothing else, so this field fills a slot and never empties
        // it: picking nothing leaves what was there, empty or filled.
        let graph = self.part_graph().clone();
        let mut offered = self.block_choices(choices);
        // **What this block could do something with** is `operand_refusal`'s rule, and it goes with
        // the field: a refused block is not suggested, and typing its name or dropping it says why
        // instead of taking it — the dropdown greyed it with the reason, and this keeps the reason.
        for choice in &mut offered {
            choice.refusal = operand_refusal(&self.doc, &graph, consumer, choice.key);
        }
        let held = *current;
        let shown = held.map_or_else(|| "none".to_string(), |id| self.shown_label(&offered, id, "none"));
        ui.horizontal(|ui| {
            ui.label(label);
            if let Some(picked) = self.reference_field_in(
                ui,
                (salt, "operand"),
                held,
                &shown,
                &offered,
                RefKind::Block,
                DropsFrom::block(|n| n),
            ) {
                *current = Some(picked);
            }
        });
        // **And the block it reads, under the row that reads it** (`notebook-spec.md` §7.1, §17).
        // This serves the five blocks still on [`Self::pick_input`] -- twist, mirror, shell,
        // offset and a placement's transform -- and an attractor's body, which is why the owner
        // saw a shell's block listed at the bottom while a boolean's were in their rows.
        //
        // **The D9 debt the comment on `operand_input` names is still owed**: these rows are this
        // older field rather than the operand row, so they have no error state and no Clear. What
        // they have now is the containment, which is what was asked for.
        if let Some(read) = *current {
            self.host_nested_block(ui, read);
        }
    }

    /// The nodes that could stand in as an operand of `id`, with what to call each.
    ///
    /// The rule itself is [`operand_choices_in`], which is a free function so it can be tested
    /// against a graph without an application around it.
    pub(crate) fn operand_choices(&self, id: NodeId) -> Vec<(NodeId, String)> {
        let graph = self.part_graph().clone();
        operand_choices_in(&graph, id, |n| self.doc.made_at(n))
            .into_iter()
            .map(|n| (n, format!("{}{}", self.body_label(n), self.operand_aside(n))))
            .collect()
    }

    /// What to add after an operand's name when the operand is not a solid.
    ///
    /// **Because a list of names cannot say that two of its entries behave differently.**
    /// Reported as "Boolean shows planes (should only show sdfs)": a plane sits in the list
    /// reading exactly like a sphere, and choosing one intersects with a half-space -- which is
    /// what `Node::Plane` is *for* (trimming a part is the reason it exists and must keep
    /// working, see `encloses_material`), and is not what the row suggested. A point is the same
    /// surprise from the other side: it has no volume, so choosing one turns a hard boolean into
    /// a blend.
    ///
    /// Said rather than removed, because both are reachable on purpose. `is_reference_body` is
    /// the existing question for "a solid that fills the viewport", answered by extent rather
    /// than by kind, so a bare lattice is marked for the same reason a plane is and a confined
    /// one is not.
    fn operand_aside(&self, id: NodeId) -> &'static str {
        let hollow = self
            .part_graph()
            .node(id)
            .is_some_and(|n| !n.encloses_material());
        if hollow {
            return " — no volume, makes this a blend";
        }
        if self.is_reference_body(id) {
            return " — fills the view";
        }
        ""
    }

    /// Why a graph may not be committed, if it may not.
    ///
    /// **Every path that rewires the graph asks this, and none of them asks half of it.** The
    /// checks used to live inline in `replace_node`, so `apply_boolean` -- which builds a node
    /// rather than replacing one -- committed without any of them, and a filleted union
    /// against a lattice went straight into the document. The failure was invisible at the
    /// point it happened and total afterwards: nothing drawn, nothing exportable, and no
    /// indication of which click did it.
    ///
    /// Ordered so the message is the most specific one available: a cycle explains itself, the
    /// shader generator catches what it cannot lower, and the kernel catches what only it
    /// knows -- D8's rule that a blend needs true distances.
    pub(crate) fn refuse_reason(&self, graph: &Graph) -> Option<String> {
        Self::refusal_of(&self.doc, graph, &|e| self.explain_compile(e))
    }

    /// [`Self::refuse_reason`] without the `App`, with the kernel's refusal put into words by
    /// `explain`: free so a test can hold an edit to the same check, since an `App` needs a window
    /// and a device.
    ///
    /// **An empty slot is not something to refuse.** It is a legal, savable state -- a block
    /// waiting to be told what to read (`notebook-spec.md` §7.2) -- so the gate asks the two
    /// compilers about the part *without* the bodies that hold one, which is the same part the
    /// viewport draws: a holed body is pruned there and its siblings carry on, and a gate that
    /// refused the whole edit would have made the gesture that empties a slot impossible to
    /// perform. The precedent is next door in [`Self::delete_breaks`], which refuses only what
    /// makes a part that *builds* broken.
    ///
    /// **The cycle check is still asked of the whole graph**, holed blocks and all: a loop is a
    /// walk that does not come back rather than a block short of an input, and leaving one in
    /// because a slot above it is empty would commit a document that cannot be opened.
    pub(crate) fn refusal_of(
        doc: &Document,
        graph: &Graph,
        explain: &dyn Fn(&fieldform_field::CompileError) -> String,
    ) -> Option<String> {
        if graph.evaluation_order().is_none() {
            return Some("That would make the part depend on itself".into());
        }
        let compilers = |graph: &Graph| -> Option<String> {
            if let Err(e) = fieldform_gpu::generate(doc, graph) {
                return Some(format!("That combination cannot be drawn: {e}"));
            }
            if let Err(e) = GraphField::compile_graph(doc, graph, 1e-3) {
                return Some(explain(&e));
            }
            None
        };
        // Nothing holed, nothing to take out, and nothing cloned: this runs on every graph edit.
        // Of the document, so a body that is or reads an open cage is taken out here too: the
        // gate would otherwise refuse every edit while a cage is red, and editing never blocks.
        let holed = doc.incomplete_in(graph);
        if holed.is_empty() {
            return compilers(graph);
        }
        // With everything the body alone used, as `drawn_graph` takes a hidden body out, so the
        // graph asked about here is the graph on screen rather than one with the operands of a
        // pruned body loose in it as bodies of their own.
        let mut buildable = graph.clone();
        for body in graph.roots() {
            if holed.contains(&body) {
                buildable.remove_body_and_its_own(body);
            }
        }
        // Every body of the part is waiting to be told what to read, so there is nothing for
        // either compiler to be asked about -- a part in the making, not a part at fault. The
        // same exemption [`Self::fault_in`] makes for the delete that empties a part, and for
        // the same reason: the compilers refuse an empty part because there is nothing to draw.
        if buildable.roots().is_empty() {
            return None;
        }
        compilers(&buildable)
    }

    /// Writes a node back, refusing an edit that would make the graph impossible.
    ///
    /// Applied and then checked, rather than checked first: an operand change can close a
    /// loop in ways that are tedious to predict and trivial to detect, and `evaluation_order`
    /// is the same test evaluation itself uses. Graphs here are a handful of nodes.
    pub(crate) fn replace_node(&mut self, id: NodeId, node: Node) {
        let mut graph = self.part_graph().clone();
        // A change of *kind* counts, not only a change of inputs. Switching a join from Sharp
        // to Filleted turns a `Boolean` into a `Blend` with the same two operands, so an
        // inputs-only test called it unstructural and skipped every check -- and that one edit
        // is precisely the one D8 can refuse. Turning a radius still changes neither, so a
        // drag stays as cheap as it was.
        // The curves too, not only the node inputs. Switching an extrude's sweep from the
        // profile's normal to a line changes which curves the shader binds -- as real a change
        // of shape as swapping an operand -- and comparing only `inputs` called that
        // unstructural, so the edit skipped the refusal check and was committed even when it
        // could not compile.
        // And the kind of section a block is built on. Turning Wall on or off names the same
        // curve and leaves the block the same kind of block, so none of the three above sees
        // it — and it is the edit that decides whether an open curve can be built on at all,
        // and which polylines the shader binds for it.
        let walled = |n: &Node| n.outline().map(|o| o.is_wall());
        let structural = graph.node(id).map(|n| n.inputs()) != Some(node.inputs())
            || graph.node(id).map(|n| n.curves()) != Some(node.curves())
            || graph.node(id).map(std::mem::discriminant) != Some(std::mem::discriminant(&node))
            || graph.node(id).map(walled) != Some(walled(&node));
        let Some(slot) = graph.node_mut(id) else { return };
        let before = std::mem::replace(slot, node);
        // **The second half of §7.3's move, in this same edit** -- see `App::completing_a_move`.
        // The drop filled a slot of this node and said which block has to let go; doing it here is
        // what makes the pair one undo step, and doing it only when this node now reads that block
        // is what keeps an unrelated edit in the same frame from taking it.
        if let Some((from, moved)) = self.completing_a_move
            && graph.node(id).is_some_and(|n| n.operands().contains(&moved))
        {
            self.completing_a_move = None;
            if from == id {
                // **Out of a sibling slot of the same block**, where "every slot that names it"
                // would empty the one just filled as well. The destination is the slot that
                // changed, so the slots to empty are the ones that did not.
                let after = graph.node(id).map(fieldform_scene::Node::operand_slots).unwrap_or_default();
                let was = before.operand_slots();
                let leaving: Vec<usize> = after
                    .iter()
                    .enumerate()
                    .filter(|(i, slot)| {
                        slot.source() == Some(moved) && was.get(*i) == Some(slot)
                    })
                    .map(|(i, _)| i)
                    .collect();
                if let Some(node) = graph.node_mut(id) {
                    for i in leaving {
                        node.empty_operand_slot(i);
                    }
                }
            } else if let Some(node) = graph.node_mut(from) {
                node.stop_reading(moved);
            }
            // **Said, because half of what happened is off screen.** The slot that was filled is
            // under the pointer; the slot that emptied is in another row, possibly scrolled out
            // of view or inside a block that is collapsed. Sharing says nothing, which is
            // deliberate and older than this: a second reader is visible as a second row, and the
            // line that used to announce one went with the promotion it was announcing.
            let what = self.body_label(moved);
            let left = self.body_label(from);
            self.say(match from == id {
                true => format!("\u{201c}{what}\u{201d} moved to another slot of {left}"),
                false => format!(
                    "\u{201c}{what}\u{201d} moved out of {left}, which is waiting to be told what to read instead"
                ),
            });
        }
        // Refused here rather than accepted into a document that cannot be drawn. A blank
        // viewport with an error in the status bar is the worst of both: the edit is
        // committed, the model is gone, and nothing says which change did it.
        //
        // Only when the edit was structural, since a radius drag arrives once per frame and
        // compiling twice for it would double the work the traced view is already doing.
        if structural && let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        self.set_part_graph(graph);
        self.mark_changed();
        self.rebuild_render_resources();
    }

    /// Turns a closed curve into a solid, extruded or revolved.
    ///
    /// **The first operation that makes a solid out of a curve** rather than sweeping something
    /// along one, so it is also the first that can refuse for a reason about the curve itself.
    /// It picks the selected curve when one is selected and the first closed curve otherwise,
    /// and says which it took — a tool that silently chose among several would be a tool you
    /// have to check after every use.
    pub(crate) fn add_profile_solid(&mut self, revolved: bool) {
        // **Any curve of the loop will do, and there is no way to pick two.** That was the
        // confusing part: two curves meeting at both ends are a closed profile, and the answer
        // to "how do I select them both" is that you do not — the block names one and
        // `profile_loop` walks outward from it through ends that meet. This used to demand a
        // *closed* curve, which refused exactly the case the walk exists for.
        // Which curve, and why it may not be the selected one, is `profile_seed`'s to explain.
        // Checked before the block is made, so a loop that does not close says why *here*
        // rather than leaving a block that refuses to compile with nothing pointing at the gap.
        if self.doc.curves.is_empty() {
            self.warn("Needs a curve to build from — draw one first");
            return;
        }
        // The selected curve, or the first curve of a selected profile row — which is the
        // curve the row's walk started from, so the loop the block reads is the row.
        let in_hand = self.curve_in_hand();
        let as_a_wall = if revolved { WallBlock::Revolve } else { WallBlock::Extrude };
        // **The curve in hand decides which kind it is.** A curve that closes no loop has no
        // inside to fill, and since a wall is a thing a block can be built on, the one tool
        // builds that: a profile when the curve closes a loop, a wall when it does not. This
        // used to look past a selected open curve to a loop elsewhere, because a stray line
        // beside a finished profile used to refuse the profile; now the stray line is what is
        // extruded, as a wall, and the log says so. One entry in the menu for both, which is
        // how the owner asked for it: there is no reason to tell them apart before the block
        // exists, and its Wall checkbox turns either into the other.
        if let Some(id) = in_hand
            && fieldform_field::profile_loop(&self.doc, id, None).is_err()
        {
            self.add_wall_solid(as_a_wall, &Args::new());
            return;
        }
        let curve = match fieldform_field::profile_seed(&self.doc, in_hand) {
            Ok(id) => id,
            // No loop anywhere, and nothing in hand: the oldest open curve, as a wall.
            Err(_) => {
                self.add_wall_solid(as_a_wall, &Args::new());
                return;
            }
        };
        let name = self.doc.curves.get(&curve).map(|c| c.name.clone()).unwrap_or_default();

        // Extruded along Z and revolved about it, which is what a profile drawn on the ground
        // plane wants in each case. Aim it afterwards with the frame controls, the same ones
        // every other block has.
        let mut graph = self.part_graph().clone();
        let node = if revolved {
            // **An axis beside the profile, not through the world origin.** `Frame::default()`
            // is the world Z axis, so a profile drawn anywhere but around the origin straddled
            // it and the block was refused the instant it was made, for a choice nobody had
            // made. `axis_beside` puts the axis in the profile's own plane with the profile
            // resting against it; the frame controls change it, as they always did.
            let frame = fieldform_field::profile_loop(&self.doc, curve, None)
                .ok()
                .and_then(|found| fieldform_field::axis_beside(&found.points))
                .map(|(origin, dir)| fieldform_scene::Frame::At {
                    origin: [
                        ParamRef::Literal(origin.x),
                        ParamRef::Literal(origin.y),
                        ParamRef::Literal(origin.z),
                    ],
                    axis: [
                        ParamRef::Literal(dir.x),
                        ParamRef::Literal(dir.y),
                        ParamRef::Literal(dir.z),
                    ],
                })
                .unwrap_or_default();
            graph.add(Node::Revolve { outline: fieldform_scene::Outline::closed(curve), frame, angle: fieldform_scene::ParamRef::Literal(360.0), sides: fieldform_scene::AngleSides::One, rims: Default::default() })
        } else {
            graph.add(Node::Extrude {
                outline: fieldform_scene::Outline::closed(curve),
                // Straight out of the profile's own plane, which is exact. Leaning it is a
                // choice made afterwards in the block, and one that costs exactness.
                sweep: fieldform_scene::Sweep::Normal,
                height: ParamRef::Literal(DEFAULT_BODY_MM * 0.25),
                draft: ParamRef::Literal(0.0),
                start: ParamRef::Literal(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
                rims: Default::default(),
            })
        };
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        self.set_part_graph(graph);
        self.select_body(Some(node));
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        // **Named loudly when it is not the curve that was selected.** `profile_seed` falls
        // back past an explicit selection on purpose -- drawing a curve selects it, so a stray
        // line beside a finished loop would otherwise refuse the loop -- and its own comment
        // says that is safe "because the caller says which curve it took". It said it in the
        // same sentence it uses when nothing was substituted, which is not the same as saying
        // it: reported as "extrude does not seem to always recognize lines", which is what
        // building on a different curve looks like from the outside.
        let substituted = in_hand.filter(|had| *had != curve);
        match substituted {
            Some(had) => {
                let had = self
                    .doc
                    .curves
                    .get(&had)
                    .map(|c| c.name.clone())
                    .unwrap_or_else(|| "the selected curve".to_string());
                self.warn(format!(
                    "{} {name} instead: {had} is not part of a closed loop",
                    if revolved { "Revolved" } else { "Extruded" },
                ));
            }
            None => self.say(format!(
                "{} {name}",
                if revolved { "Revolved" } else { "Extruded" },
            )),
        }
    }

    /// A curve given a thickness and made a solid: what `extrude`, `revolve` and `sweep` do
    /// when the curve in hand closes no loop, and what `extrude --wall 2 --side left` asks for
    /// by name, with the numbers.
    ///
    /// The curve is the one in hand, and with none in hand the oldest that closes no loop —
    /// the thing a wall is for and nothing else can use — and it says which it took. The rest
    /// starts as an extrude, a revolve or a sweep of a profile starts, and is changed in the
    /// block: the side and the ends in the one editor all three share.
    pub(crate) fn add_wall_solid(&mut self, kind: WallBlock, args: &Args) {
        use fieldform_scene::{Outline, WallEnds, WallSide};
        if self.doc.curves.is_empty() {
            self.warn("Needs a curve to give a thickness to — draw one first");
            return;
        }
        let thickness = match wall_thickness_asked(args, self.wall_thickness) {
            Ok(thickness) => thickness,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let ends = match args.get("ends").map(|v| v.as_str().map(ends_from_word)) {
            None => WallEnds::Square,
            Some(Some(Ok(ends))) => ends,
            Some(Some(Err(why))) => {
                self.warn(why);
                return;
            }
            Some(None) => {
                self.warn("`ends` takes a word — square or round");
                return;
            }
        };
        let Some(curve) = self.wall_seed() else { return };
        let outline = Outline::Wall {
            curve,
            thickness: ParamRef::Literal(thickness),
            side: WallSide::Centre,
            ends,
        };
        // The curves the wall runs along, so that a path is not picked from among them and
        // an axis is put beside the whole of it.
        let chain = fieldform_field::outline_chain(&self.doc, curve, None).ok();

        let mut along: Option<CurveId> = None;
        let mut block = match kind {
            WallBlock::Extrude => Node::Extrude {
                outline,
                // Straight out of the curve's own plane, which is exact, as a profile starts.
                sweep: fieldform_scene::Sweep::Normal,
                height: ParamRef::Literal(DEFAULT_BODY_MM * 0.25),
                draft: ParamRef::Literal(0.0),
                start: ParamRef::Literal(0.0),
                sides: fieldform_scene::Sides::One,
                corners: Default::default(),
                rims: Default::default(),
            },
            WallBlock::Revolve => {
                // An axis beside the curve, in its own plane, for the reason a profile is
                // given one: the world's axis is an answer only for a curve drawn about it.
                // A straight line has no plane to put one in and keeps the world's.
                let frame = chain
                    .as_ref()
                    .and_then(|found| fieldform_field::axis_beside(&found.points))
                    .map(|(origin, dir)| fieldform_scene::Frame::At {
                        origin: [
                            ParamRef::Literal(origin.x),
                            ParamRef::Literal(origin.y),
                            ParamRef::Literal(origin.z),
                        ],
                        axis: [
                            ParamRef::Literal(dir.x),
                            ParamRef::Literal(dir.y),
                            ParamRef::Literal(dir.z),
                        ],
                    })
                    .unwrap_or_default();
                Node::Revolve {
                    outline,
                    frame,
                    angle: ParamRef::Literal(360.0),
                    sides: fieldform_scene::AngleSides::One,
                    rims: Default::default(),
                }
            }
            WallBlock::Sweep { spine: named } => {
                // The path: a curve the wall does not run along, open ones first, since a
                // closed one is more likely another section. A path drawn from the section's
                // own end is walked into its chain, so with nothing outside the chain the
                // path is taken from the chain's far end — and left out of the walk from
                // then on, which is what a sweep does with its spine.
                let in_chain: Vec<CurveId> =
                    chain.as_ref().map(|found| found.curves.clone()).unwrap_or_else(|| vec![curve]);
                let mut outside: Vec<CurveId> =
                    self.doc.curves.keys().copied().filter(|id| !in_chain.contains(id)).collect();
                outside.sort_by_key(|id| id.raw());
                let open = outside
                    .iter()
                    .copied()
                    .find(|id| self.doc.curves.get(id).is_some_and(|c| !c.is_closed()));
                let far_end = match in_chain.first() {
                    Some(first) if *first == curve => in_chain.last().copied(),
                    first => first.copied(),
                }
                .filter(|id| *id != curve);
                // A spine named -- `sweep curve= wall= spine=`, or `sweep curve= spine=` with no
                // closed profile to carry -- is built on from the start, for the reason
                // `add_sweep_body` builds on one: a pick of this arm's own first is a random curve,
                // refused or read for an instant before the named one is reached.
                let spine = match named.or(open).or(far_end).or_else(|| outside.first().copied()) {
                    Some(id) => id,
                    None => match self.add_line() {
                        Some(id) => id,
                        None => return,
                    },
                };
                along = Some(spine);
                Node::Sweep {
                    outline,
                    spine,
                    section: Default::default(),
                    law: Default::default(),
                    rims: Default::default(),
                }
            }
        };
        // The side last, because inside and outside are read from the block as it will be.
        if let Some(word) = args.get("side") {
            let asked = match word.as_str() {
                Some(word) => side_from_word(word, self.inside_of(&block, curve)),
                None => Err("`side` takes a word — centre, left or right".to_string()),
            };
            match (asked, block.outline_mut()) {
                (Ok(side), Some(Outline::Wall { side: slot, .. })) => *slot = side,
                (Err(why), _) => {
                    self.warn(why);
                    return;
                }
                _ => {}
            }
        }

        let mut graph = self.part_graph().clone();
        let node = graph.add(block);
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        let name = |id: CurveId| self.doc.curves.get(&id).map(|c| c.name.clone()).unwrap_or_default();
        let (curve_name, along_name) = (name(curve), along.map(name));
        self.wall_thickness = thickness;
        self.set_part_graph(graph);
        self.select_body(Some(node));
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        self.say(match along_name {
            Some(path) => format!(
                "{} {curve_name} as a wall {thickness} mm thick, along {path} — pick either \
                 curve in the block",
                kind.did(),
            ),
            None => format!(
                "{} {curve_name} as a wall {thickness} mm thick — its side and ends are in \
                 the block",
                kind.did(),
            ),
        });
    }

    /// What `extrude`'s arguments say about the depth of the block it just made: where the
    /// material begins, how deep, how drafted, and which way about the plane.
    ///
    /// `--start 5` begins the material five millimetres along the sweep; `--depth 30` and
    /// `--draft 5` are side one's; `--symmetric` is half each way; `--two 10 --draft2 3` is a
    /// second side of its own depth and draft. Applied to the selected block, which is the one
    /// the command left selected, and only if it is an extrude -- a refused command made none.
    pub(crate) fn apply_extrude_options(&mut self, args: &Args) {
        // **Read by the extrude's own reader** (`block_edit::extrude_with_keys`), which takes a
        // parameter's name wherever a number goes. This read each number with `as_f64`, which
        // took a literal and dropped a name without a word -- so `extrude height=Height` built
        // the default depth, and the echo of an extrude driven by a variable could not replay.
        self.apply_options_to_made(
            args,
            &["start", "depth", "draft", "symmetric", "two", "draft2", "corners", "rims", "rims_continuity", "rims_at", "rims_second"],
            |n| matches!(n, Node::Extrude { .. }),
        );
    }

    /// The keys among `keys` that `args` gives, set on the block the command just made and left
    /// selected, if it is the kind `is` asks for -- a refused command made none.
    fn apply_options_to_made(&mut self, args: &Args, keys: &[&str], is: impl Fn(&Node) -> bool) {
        let given = keys
            .iter()
            .filter_map(|k| args.get(k).map(|v| (*k, v.clone())))
            .fold(Args::new(), |a, (k, v)| a.with(k, v));
        if given.iter().next().is_none() {
            return;
        }
        let Some(id) = self.selected_body else { return };
        if !self.part_graph().node(id).is_some_and(&is) {
            return;
        }
        if let Err(why) = self.set_block_keys(id, &given) {
            self.warn(why);
        }
    }

    /// What `revolve`'s arguments say about the angle of the block it just made: `--angle 120`
    /// is how far round, `--symmetric` half each way, `--two 45` a second way of its own. Applied
    /// to the selected block, which is the one the command left selected, if it is a revolve.
    pub(crate) fn apply_revolve_options(&mut self, args: &Args) {
        // The revolve's own reader, which takes a parameter's name for an angle, and the axis
        // with it: `apply_revolve_axis` was a second pass over the same block.
        self.apply_options_to_made(
            args,
            &["angle", "symmetric", "two", "axis", "origin", "rims", "rims_continuity", "rims_at", "rims_second"],
            |n| matches!(n, Node::Revolve { .. }),
        );
    }

    /// What `sweep`'s arguments say about the law of the block it just made: `taper_scale=0.5`
    /// or `taper_draft=5` for how the section grows, `twist=90` for how far it turns, `law=g1`
    /// for how both are spread, `corners=sharp` for how a drafted corner is cut, and
    /// `extent=10,90` for the stretch of path it runs over. Applied to the selected block,
    /// which is the one the command left selected, if it is a sweep.
    ///
    /// **The keys are the law's own field names** (the project notes), with the words this branch was
    /// written with kept as aliases where they differ -- `taper-scale`, `draft`, `sharp`. The
    /// extent is one key rather than the `starts`/`ends` pair it was: `ends` is already a wall's
    /// key, meaning square or round, and `wall_wanted` reads any `ends` as asking for a wall, so
    /// a law spelt that way could not be reached on a sweep that is not one.
    ///
    /// **The two tapers are refused together**, as an extrude's `two` and `symmetric` are:
    /// they are two answers to how the section grows, and taking the last one typed would be
    /// guessing which the user meant.
    pub(crate) fn apply_sweep_options(&mut self, args: &Args) {
        self.apply_options_to_made(
            args,
            &["taper_scale", "taper-scale", "taper_draft", "twist", "law", "extent", "corners", "rims", "rims_continuity", "rims_at", "rims_second"],
            |n| matches!(n, Node::Sweep { .. }),
        );
    }

    /// The curve a wall is made on when a command is not told: the one in hand, or the oldest
    /// that closes no loop, or the oldest there is.
    ///
    /// An open curve first, because that is what a wall is for and what nothing else can be
    /// built on; a loop will do, since a wall round one is a tube.
    fn wall_seed(&mut self) -> Option<CurveId> {
        if let Some(id) = self.curve_in_hand() {
            return Some(id);
        }
        // `curve_in_hand` has just refreshed the rows this reads.
        let mut ids: Vec<CurveId> = self.doc.curves.keys().copied().collect();
        ids.sort_by_key(|id| (self.doc.curves_created.get(id).copied().unwrap_or(0), id.raw()));
        ids.iter()
            .copied()
            .find(|id| {
                self.profiles.key_of(*id).is_none()
                    && self.doc.curves.get(id).is_some_and(|c| !c.is_closed())
            })
            .or_else(|| ids.first().copied())
    }

    /// A profile carried along a curve -- `sweep`, and Create > Sweep > Sweep.
    ///
    /// The profile is found the way an extrude finds its own (`profile_seed`), and the path is
    /// `spine` when one is named, else the selected curve when that is not the profile, else the
    /// first curve outside the profile's loop -- open ones first, since a closed curve is more
    /// likely another profile -- else a fresh line, as a pipe makes one rather than refusing.
    /// Both are picked again in the block, so what matters here is that the block appears with
    /// something to show.
    ///
    /// **A named spine is built on from the start**, never pointed at after a pick of this
    /// function's own. That pick goes by id, which is random, so `sweep profile= spine=` built on
    /// a different curve each run: one the profile could not follow refused the line before the
    /// named spine was reached -- `curves.ffc` failed three runs in ten that way -- and one
    /// another block held was a second reader of it for an instant (refactor plan 2.22).
    pub(crate) fn add_sweep_body(&mut self, spine: Option<CurveId>) {
        if self.doc.curves.is_empty() {
            self.warn("Needs a closed profile and a curve to carry it along — draw them first");
            return;
        }
        let selected = self.curve_in_hand();
        let (profile, chosen) = match fieldform_field::profile_seed(&self.doc, selected) {
            Ok(id) => (id, None),
            // The selected curve may be the path rather than the profile: a line is what a
            // person draws to sweep along, and a line is exactly what cannot be a profile.
            // With no closed curve anywhere to carry, it is the section, given a thickness, and
            // swept along the spine named, if one was.
            Err(_) => match selected
                .and_then(|_| fieldform_field::profile_seed(&self.doc, None).ok())
            {
                Some(id) => (id, selected),
                None => {
                    self.add_wall_solid(WallBlock::Sweep { spine }, &Args::new());
                    return;
                }
            },
        };
        let in_loop: Vec<CurveId> = fieldform_field::profile_loop(&self.doc, profile, None)
            .map(|found| found.curves)
            .unwrap_or_else(|_| vec![profile]);
        let mut others: Vec<CurveId> =
            self.doc.curves.keys().copied().filter(|id| !in_loop.contains(id)).collect();
        others.sort_by_key(|id| id.raw());
        let open = others
            .iter()
            .copied()
            .find(|id| self.doc.curves.get(id).is_some_and(|c| !c.is_closed()));
        let spine = match spine.or(chosen).or(open).or_else(|| others.first().copied()) {
            Some(id) => id,
            None => match self.add_line() {
                Some(id) => id,
                None => return,
            },
        };
        let mut graph = self.part_graph().clone();
        let node = graph.add(Node::Sweep {
            outline: fieldform_scene::Outline::closed(profile),
            spine,
            section: Default::default(),
            law: Default::default(),
            rims: Default::default(),
        });
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        let name = |id: CurveId| self.doc.curves.get(&id).map(|c| c.name.clone()).unwrap_or_default();
        let (profile_name, spine_name) = (name(profile), name(spine));
        self.set_part_graph(graph);
        self.select_body(Some(node));
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        self.say(format!(
            "Swept {profile_name} along {spine_name} — pick either curve in the block"
        ));
    }

    /// Adds a loft through the sections `loft_asked` reads from `args`, and selects it.
    ///
    /// With nothing named it takes every closed curve and loop in the document in list order,
    /// which is a starting point and not a guess at intent: the block is where the order is
    /// fixed and sections are added or taken out, and the line said afterwards says so.
    pub(crate) fn add_loft_body(&mut self, args: &Args) {
        self.refresh_profiles();
        let (sections, spine, smooth) = match loft_asked(&self.doc, &self.profiles.loops, args) {
            Ok(asked) => asked,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        if sections.len() < 2 {
            self.warn(
                "Needs two or more closed curves to pass through — draw them first, or name \
                 them: loft sections=a,b,c",
            );
            return;
        }
        // Read as an edit reads it, so a reach driven by a parameter -- which the rims' V makes and
        // the echo writes -- builds the same loft from its line.
        let rims = match block_edit::rims_read("loft", args, fieldform_scene::Rims::hard(), &|t| self.lookup(t)) {
            Ok(asked) => asked.unwrap_or_default(),
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let mut graph = self.part_graph().clone();
        let node = graph.add(Node::Loft {
            sections: sections.iter().copied().map(fieldform_scene::Outline::closed).collect(),
            spine,
            smooth,
            rims,
        });
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        let along = match spine.and_then(|id| self.doc.curves.get(&id)) {
            Some(curve) => format!("along {}", curve.name),
            None => "along a centerline built from the end sections".to_string(),
        };
        self.set_part_graph(graph);
        self.select_body(Some(node));
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        self.say(format!(
            "Lofted {} sections {along} — order, add and remove them in the block",
            sections.len()
        ));
    }

    /// What `node`, a loft, reads as built -- from the cache when the node and the curves are
    /// as they were, built otherwise. `None` for a loft that cannot be built, which the
    /// compile refuses with the reason where the panel shows errors.
    pub(crate) fn loft_readings(&mut self, id: NodeId, node: &Node) -> Option<&LoftReadings> {
        let Node::Loft { sections, spine, smooth, rims: _ } = node else { return None };
        let current = self
            .loft_readings
            .as_ref()
            .is_some_and(|(at, was, rev, _)| *at == id && was == node && *rev == self.curve_revision);
        if !current {
            let readings = fieldform_field::loft_built(&self.doc, sections, *spine, *smooth).map(
                |built| LoftReadings {
                    stretch: built.stretch(),
                    tilts: built.tilts().to_vec(),
                    neck: built.neck(),
                    derived: spine.is_none().then(|| {
                        built
                            .spine
                            .points()
                            .iter()
                            .map(|p| Point3::new(p.x as f32, p.y as f32, p.z as f32))
                            .collect()
                    }),
                },
            );
            self.loft_readings = Some((id, node.clone(), self.curve_revision, readings));
        }
        self.loft_readings.as_ref().and_then(|(_, _, _, readings)| readings.as_ref())
    }

    /// The derived centerline of the selected body, if it is a loft that was given none.
    pub(crate) fn selected_derived_centerline(&mut self) -> Option<Vec<Point3<f32>>> {
        let id = self.selected_body?;
        let node = self.part_graph().node(id)?.clone();
        if !matches!(node, Node::Loft { spine: None, .. }) {
            return None;
        }
        self.loft_readings(id, &node).and_then(|readings| readings.derived.clone())
    }

    /// Adds a straight line, and selects it: a curve of degree one, built as one.
    ///
    /// Its own command rather than something a pipe happens to leave behind. A line is
    /// geometry in its own right — reference to measure against, a path to sweep later, or
    /// nothing at all yet — and making one should not require deciding what it is for.
    ///
    /// From the origin along X, half the size of everything else the Create menu makes: a line
    /// is a span rather than a solid, and one as long as a cube is wide reads as part of the
    /// same drawing.
    ///
    /// **One creation path, not two.** This used to make a two-point polyline while
    /// `add_bezier(1)` made a two-point degree-1 Bézier, and those are the same two points —
    /// `single_span_bezier`'s half sine is zero at both of its ends. Two ways to make one thing
    /// is two things to keep in step, and the drawing-plane lift had already had to be added to
    /// each of them separately.
    ///
    /// Kept as its own entry point because `line` is worth having in the console even though
    /// the Curves menu no longer offers it separately.
    pub(crate) fn add_line(&mut self) -> Option<CurveId> {
        self.add_bezier(1)
    }

    /// Adds a single-span Bézier of the given degree, and selects it.
    ///
    /// One span, which is what makes it a Bézier rather than a general B-spline: every knot
    /// sits at one end or the other, so the curve is one polynomial piece and its control
    /// points are the whole of what shapes it. A degree-`n` one has exactly `n + 1` of them.
    ///
    /// The starting shape is a half-sine raised out of the same span a line occupies — flat at
    /// both ends, highest in the middle. That is a curve that immediately looks like its
    /// degree, where evenly spaced collinear points would draw a straight line and leave the
    /// difference between degree 3 and degree 9 invisible until something had been dragged. It
    /// also degenerates correctly: at degree 1 the sine is zero at both of its two points, so
    /// `Create > Curve > 1` and `Create > Line` agree rather than quietly differing.
    fn add_bezier(&mut self, degree: usize) -> Option<CurveId> {
        let degree = degree.clamp(1, 24);
        let id = CurveId::new();
        match single_span_bezier(id, degree, DEFAULT_BODY_MM * 0.5) {
            Ok(mut curve) => {
                // Named after it is built rather than inside the builder, which is a free
                // function with no document to check against.
                curve.name = self.unique_curve_name(&curve.name);
                // And lifted into the drawing plane for the same reason: `single_span_bezier`
                // has no application to ask where the user is drawing.
                for i in 0..curve.control_points().len() {
                    let local = curve.control_points()[i];
                    curve.set_control_point(i, self.into_drawing_plane(local));
                }
                self.doc.curves.insert(id, curve);
                // Placed now, as everything written past the history is
                // (`block-ownership-spec.md` §4.1).
                self.doc.settle_ownership();
                self.select_curve(id);
                self.rebuild_curve_display();
                self.mark_changed();
                if let Some((centre, radius)) = self.scene_bounds() {
                    self.camera.frame(centre, radius);
                }
                self.say(if degree == 1 {
                    "Added a line".to_string()
                } else {
                    format!("Added a degree {degree} curve")
                });
                Some(id)
            }
            Err(e) => {
                self.warn(format!("could not make a degree {degree} curve: {e}"));
                None
            }
        }
    }

    /// A curve, or a profile, by the name a person would type: a curve's own name first (any
    /// curve of a loop names the loop), then a profile's row name, as `Profile 2` or the name it
    /// was given.
    fn curve_or_profile_named(&mut self, name: &str) -> Option<CurveId> {
        if let Some(id) = self.curve_named(name) {
            return Some(id);
        }
        self.refresh_profiles();
        let keys: Vec<CurveId> = self.profiles.loops.iter().map(|l| l[0]).collect();
        keys.into_iter().find(|key| self.profile_label(*key).eq_ignore_ascii_case(name.trim()))
    }

    /// `profile=` or `curve=` put in hand, so the arms that build on the curve in hand build on
    /// the one named (A6), and the aliases turned into the words those arms read. `None` when
    /// the line was refused, which has been said.
    ///
    /// **A profile named must close.** The arms fall back to a wall when the curve in hand is
    /// open, which is right for a curve picked by clicking and wrong for one named as a
    /// profile: that was a request for a solid, and a wall would be a different block answering
    /// it. `curve=` is the word for an open one.
    pub(crate) fn curve_keys_in_hand(&mut self, verb: &str, args: &Args, aliases: &[(&str, &str)]) -> Option<Args> {
        let args = match console_keys::aliased(verb, args, aliases) {
            Ok(args) => args,
            Err(why) => {
                self.warn(why);
                return None;
            }
        };
        let text = |k: &str| args.get(k).map(|v| match v {
            Value::Str(s) => s.clone(),
            other => format!("{other:?}"),
        });
        match (text("profile"), text("curve")) {
            (Some(_), Some(_)) => {
                self.warn(format!("`{verb}` builds on a profile or a curve — profile= or curve=, not both"));
                return None;
            }
            (Some(name), None) => {
                let Some(id) = self.curve_or_profile_named(&name) else {
                    self.warn(format!("no curve or profile is called {name:?}"));
                    return None;
                };
                if fieldform_field::profile_loop(&self.doc, id, None).is_err() {
                    self.warn(format!(
                        "{name:?} is not part of a closed loop — `{verb} curve={name} wall=2` gives an open curve a thickness"
                    ));
                    return None;
                }
                self.select_curve(id);
            }
            (None, Some(name)) => {
                let Some(id) = self.curve_named(&name) else {
                    self.warn(format!("no curve is called {name:?}"));
                    return None;
                };
                self.select_curve(id);
            }
            (None, None) => {}
        }
        Some(args)
    }

    /// `line`/`curve` with points: placed now, as one undo step, through the same insert a click
    /// makes -- then the profiles refreshed, so a loop the new curve closes joins exactly as a
    /// drawn one does. Without points, click placement as before.
    pub(crate) fn place_or_begin_curve(&mut self, verb: &str, degree: usize, args: &Args) {
        let asked = match console_keys::curve_asked(verb, args) {
            Ok(asked) => asked,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let Some((degree_asked, points, weights)) = asked else {
            self.begin_curve_placement(degree);
            return;
        };
        let id = CurveId::new();
        let mut curve = match single_span(id, degree_asked, points) {
            Ok(curve) => curve,
            Err(e) => {
                self.warn(format!("could not make a degree {degree_asked} curve: {e}"));
                return;
            }
        };
        for (i, w) in weights.iter().enumerate() {
            curve.set_weight(i, *w);
        }
        curve.name = self.unique_curve_name(&curve.name);
        let name = curve.name.clone();
        // Not a graph edit, so it ends any gesture a value drag had open: the next graph edit
        // would otherwise take this entry for the drag's and replace it.
        self.gesture = None;
        if let Err(e) = self.stack.execute(&mut self.doc, Box::new(SetCurve::insert(curve))) {
            self.warn(e.to_string());
            return;
        }
        self.pending_curve = None;
        self.select_curve(id);
        self.rebuild_curve_display();
        self.mark_changed();
        // Now rather than on the next frame: a script runs every line in one, and the next
        // line may name the profile this curve just closed.
        self.refresh_profiles();
        self.say(if degree_asked == 1 {
            format!("Added a line, {name}")
        } else if weights.iter().any(|w| (*w - 1.0).abs() > 1e-12) {
            format!("Added a weighted degree {degree_asked} curve, {name}")
        } else {
            format!("Added a degree {degree_asked} curve, {name}")
        });
    }

    /// The frame `revolve axis= origin=` asks for, given the one the revolve has: a curve's name
    /// makes the frame follow that curve; `x`, `y`, `z` or three numbers, with `origin=`, make a
    /// fixed one; `None` when neither key was given. Asked once with a stand-in frame before the
    /// block is built, so a wrong axis is refused before anything is added, and again after.
    pub(crate) fn revolve_frame(&self, args: &Args, was: &fieldform_scene::Frame) -> Result<Option<fieldform_scene::Frame>, String> {
        if args.get("axis").is_none() && args.get("origin").is_none() {
            return Ok(None);
        }
        let axis_text = args.get("axis").and_then(|v| v.as_str()).map(str::to_string);
        let letter = axis_text.as_deref().is_some_and(|t| Axis::ALL.iter().any(|a| a.label().eq_ignore_ascii_case(t)));
        let along = axis_text.as_deref().filter(|_| !letter).filter(|t| !t.contains(','));
        if let Some(name) = along {
            let Some(curve) = self.curve_named(name) else {
                return Err(format!("no curve is called {name:?} for `revolve axis=`"));
            };
            if args.get("origin").is_some() {
                return Err(
                    "a revolve about a curve turns about that curve — `origin=` goes with axis=x|y|z or three numbers".to_string(),
                );
            }
            return Ok(Some(fieldform_scene::Frame::Along { curve }));
        }
        let (was_origin, was_axis) = match was {
            fieldform_scene::Frame::At { origin, axis } => (*origin, *axis),
            _ => ([ParamRef::Literal(0.0); 3], [ParamRef::Literal(0.0), ParamRef::Literal(0.0), ParamRef::Literal(1.0)]),
        };
        let lookup = |t: &str| self.lookup(t);
        let origin = args
            .get("origin")
            .map(|v| console_keys::param_triple("revolve", "origin", v, console_keys::Slot::Number, &lookup))
            .transpose()?
            .unwrap_or(was_origin);
        let axis = args
            .get("axis")
            .map(|v| console_keys::axis_triple("revolve", "axis", v, &lookup))
            .transpose()?
            .unwrap_or(was_axis);
        Ok(Some(fieldform_scene::Frame::At { origin, axis }))
    }

    /// `sweep` with its curves named: the profile (or the curve a wall is made on) put in hand,
    /// the block built on the spine it was told by the arm that builds it, then the section.
    pub(crate) fn sweep_with_keys(&mut self, args: &Args) {
        let Some(args) = self.curve_keys_in_hand("sweep", args, &[("thickness", "wall")]) else { return };
        let spine = match args.get("spine").and_then(|v| v.as_str()) {
            None => None,
            Some(name) => match self.curve_named(name) {
                Some(id) => Some(id),
                None => {
                    self.warn(format!("no curve is called {name:?} for `sweep spine=`"));
                    return;
                }
            },
        };
        let section = match args.get("section").map(console_keys::section_named).transpose() {
            Ok(section) => section,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        // A spine that is part of what is swept is refused before anything is built.
        if let (Some(spine), Some(held)) = (spine, self.selected_curve) {
            let in_outline = held == spine
                || (!wall_wanted(&args)
                    && fieldform_field::profile_loop(&self.doc, held, None).is_ok_and(|l| l.curves.contains(&spine)));
            if in_outline {
                self.warn("the spine is a curve of the profile itself — a sweep runs along a different curve");
                return;
            }
        }
        // The spine goes in as the block is built (`add_sweep_body` says why), so the line each
        // arm says already names it.
        if wall_wanted(&args) {
            self.add_wall_solid(WallBlock::Sweep { spine }, &args);
        } else {
            self.add_sweep_body(spine);
        }
        if let Some(section) = section
            && let Some(id) = self.selected_body
            && let Some(mut node) = self.part_graph().node(id).cloned()
            && let Node::Sweep { section: across, .. } = &mut node
        {
            *across = section;
            self.replace_node(id, node);
        }
        // **The law too**, which this arm read none of: `sweep profile=p spine=s twist=90` built a
        // sweep with no twist and said nothing, where the arm without curves named read it.
        self.apply_sweep_options(&args);
    }

    /// `pipe curve= radius=`: the curve named, and a radius that a ramp may drive.
    pub(crate) fn pipe_with_keys(&mut self, args: &Args) {
        if let Some(name) = args.get("curve").and_then(|v| v.as_str()) {
            let Some(id) = self.curve_named(name) else {
                self.warn(format!("no curve is called {name:?}"));
                return;
            };
            self.select_curve(id);
        }
        let radius = match args.get("radius").map(|v| {
            console_keys::param_one("pipe", "radius", v, console_keys::Slot::NumberOrField, &|t| self.lookup(t))
        }) {
            None => None,
            Some(Ok(r)) => Some(r),
            Some(Err(why)) => {
                self.warn(why);
                return;
            }
        };
        // Only a ramp may drive a radius; anything else is refused before a block is made.
        if let Some(ParamRef::Field(driver)) = radius
            && !matches!(self.part_graph().node(driver), Some(Node::Ramp { .. }))
        {
            let label = self.body_label(driver);
            self.warn(format!("`pipe radius=` is driven by a ramp, and {label} is not one"));
            return;
        }
        // The tool remembers a number it was given, as it always has; a ramp belongs to this pipe.
        let remembered = self.pipe_radius;
        if let Some(r) = radius {
            self.pipe_radius = r;
        }
        self.add_pipe_body();
        if !matches!(radius, Some(ParamRef::Literal(_))) {
            self.pipe_radius = remembered;
        }
    }

    /// Adds a pipe as a body of its own.
    ///
    /// It joins nothing. A pipe is a solid the moment it exists, and what happens to it — a
    /// cut, a union, a fillet — is the boolean tool's business, not the tool that swept it.
    fn add_pipe_body(&mut self) {
        // A pipe is a curve swept, so it needs one — but needing one is not a reason to refuse
        // to make it. Create > Cage already works this way: it puts back the cage the document
        // still holds, or makes one. Asking the user to go and find a curve first is the tool
        // interrogating them before it will do anything, which is what the block list replaced.
        let curve = match self.selected_curve.filter(|id| self.doc.curves.contains_key(id)) {
            Some(id) => id,
            None => match self.doc.curves.keys().next().copied() {
                Some(id) => id,
                // Nothing to sweep, so make something to sweep. The alternative is a node
                // naming a curve that is not there, and a graph holding one of those does not
                // evaluate at all — adding a pipe would blank the viewport instead of adding
                // a pipe.
                None => match self.add_line() {
                    Some(id) => id,
                    None => return,
                },
            },
        };
        let mut graph = self.part_graph().clone();
        let id = graph.add(Node::Pipe { curve, radius: self.pipe_radius });
        self.set_part_graph(graph);
        self.select_body(Some(id));
        // Point the boolean tool at what was just made, against whatever was already there.
        // Nearly every pipe is swept in order to cut or join with something, and choosing both
        // operands from scratch afterwards is a step with one sensible answer.
        self.boolean_b = Some(id);
        if self.boolean_a.is_none() || self.boolean_a == Some(id) {
            self.boolean_a =
                self.doc.listed_bodies(self.part_graph()).into_iter().find(|b| *b != id);
        }
        self.mark_changed();
        // The pipe is a field, and field mode is what draws one.
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        self.say("Added a pipe — pick its curve in the block");
    }

    /// The blocks a pair-taking tool could start from, for the menu that has to know whether
    /// there are two.
    pub(crate) fn pair_candidates(&self) -> Vec<NodeId> {
        pair_candidates_in(self.part_graph(), &|id| self.doc.made_at(id))
    }

    /// The two operands a block that takes a pair should start from.
    ///
    /// **One spelling, because a boolean and a morph are asked the same question (D9).** Both
    /// mean "these two", so both take it the same way: whatever was last worked with — a body
    /// added or selected becomes one — falling back to the first two in the list. A morph used
    /// to insist that *both* slots be set by hand and warn otherwise, so the same gesture
    /// started one block and refused the other, with nothing on screen saying which slot was
    /// empty or how to fill it.
    ///
    /// It is a *starting* pair rather than a question asked up front, because every block that
    /// takes a pair has both operands in its own panel and changing them there is one click. A
    /// tool that interrogates you before it will do anything is the thing the block list
    /// replaced.
    ///
    /// A block something else already consumes counts as something to start from, named or not —
    /// `pair_candidates_in` says why the name stopped being the gate. **Bodies still sort first**,
    /// so with two bodies in the part the pair a tool opens with is exactly the pair it opened
    /// with before; the tail is what answers a part whose operands have all been consumed, where
    /// the alternative is a tool that refuses with both of them on screen. The two are never the
    /// same node, since the second pick refuses the first.
    fn starting_pair(&self) -> Option<(NodeId, NodeId)> {
        let graph = self.part_graph();
        let blending = self.blend_mode.is_some();
        // **Every remembered operand stands for the block it belongs to** — see `outermost_in`.
        // Asked of the nesting the list is drawn from, so a tool starts from the rows the person
        // can see rather than from something a previous tool pointed at before it was wrapped.
        // One walk for all three, since the list is redrawn from this same answer.
        let nesting = self.doc.nesting_in(graph);
        let out = |id: Option<NodeId>| id.map(|id| outermost_in(&nesting, id));
        starting_pair_among(
            graph,
            |id| self.doc.made_at(id),
            out(self.boolean_a).or_else(|| out(self.selected_body)),
            out(self.boolean_b),
            // A boolean takes no point; a blend may reach to one.
            |id| blending || !matches!(graph.node(id), Some(Node::Point { .. })),
        )
    }

    /// Combines two bodies into one, with a hard edge or a fillet.
    pub(crate) fn apply_boolean(&mut self, args: &Args) {
        // **Which of the three, from a flag.** The tool remembers the last one used, which is
        // what the menu and the panel set; a flag says which *this* one is and leaves that
        // memory pointing at it, so typing `boolean -s` twice does what it says both times.
        //
        // Long names as well as letters, since `-subtract` is what somebody writes when the
        // letter does not come to mind, and `op=subtract` for anyone who prefers the key form.
        //
        // **And the operands by name, and a fillet by its radius** (`benchmark-suite-instructions.md`
        // §2 A2, A3): `boolean -u in=box,ball r=2` joins the two named with a 2 mm fillet; with
        // no `in=` the pair is the one the tool would have picked, as it always was. A fillet
        // joins exactly two, so `r=` with three names is refused rather than chained.
        let asked = match console_keys::boolean_asked(args, &|t| self.lookup(t)) {
            Ok(asked) => asked,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        if let Some(op) = asked.op {
            self.boolean_op = op;
        }

        // Slots rather than blocks, because `in=none` is two of them waiting: a boolean a
        // script made before it had the bodies to join (`notebook-spec.md` §7.2). The pair the
        // tool would have picked is still what nothing typed means, and it is always two blocks
        // that exist.
        let inputs: Vec<fieldform_scene::Operand> = match asked.inputs {
            Some(named) => named,
            None => {
                let Some((a, b)) = self.starting_pair() else {
                    self.warn("Needs two bodies to combine");
                    return;
                };
                vec![fieldform_scene::Operand::filled(a), fieldform_scene::Operand::filled(b)]
            }
        };
        // A radius typed wins over the tool's remembered blend mode; nothing typed, and the
        // tool does what the panel last set it to.
        let fillet = asked.fillet.or_else(|| {
            self.blend_mode.map(|continuity| fieldform_scene::Fillet { radius: self.blend_radius, continuity, second: None })
        });
        let mut graph = self.part_graph().clone();
        let node = match fillet {
            None => graph.add(Node::Boolean { op: self.boolean_op, inputs: inputs.clone() }),
            Some(fillet) => {
                if inputs.len() != 2 {
                    self.warn("a fillet joins two bodies; union the rest first");
                    return;
                }
                graph.add(Node::Blend { op: self.boolean_op, a: inputs[0], b: inputs[1], fillet })
            }
        };
        // Checked before it is committed, not after. This is the click that makes a filleted
        // union, so it is the click D8 refuses, and refusing it here leaves the two bodies
        // exactly where they were rather than folding them into a node that draws nothing.
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        self.set_part_graph(graph);
        // Both operands stopped being bodies the moment something consumed them, so a
        // selection pointing at one would be pointing at an interior node — and, for the same
        // reason, so would an eye. Their hidden flags go with their rows: leaving one behind
        // hides an interior node through a switch that is no longer drawn, and nothing the
        // user can reach would ever turn it back on. `delete_body` already does this for the
        // same reason.
        for input in inputs.iter().filter_map(|slot| slot.source()) {
            self.clear_hidden(input);
        }
        self.select_body(Some(node));
        self.boolean_a = Some(node);
        self.boolean_b = None;
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        self.say(format!("{} applied", boolean_label(self.boolean_op)));
        // **An approximate blend announces itself where its radius was typed** (the project notes on
        // `LowerBound` operands): the panel marks it ≈ beside the number, and the console line
        // is the console's number.
        if fillet.is_some()
            && let Some(why) = self.approximate_join(inputs.iter().filter_map(|slot| slot.source()))
        {
            self.say(format!("\u{2248} {why}"));
        }
    }

    /// Grows or shrinks the selected body by a distance.
    ///
    /// Wraps rather than replaces, so the body underneath keeps its own block and its own
    /// numbers: an offset is a thing done *to* a shape, and undoing it should leave the shape.
    /// That is the same rule the boolean follows, and the reason `delete_body` on an offset
    /// gives the original back.
    pub(crate) fn apply_offset(&mut self, args: &Args) {
        let Ok(named) = self.named_target("offset", args) else { return };
        let bodies = self.doc.listed_bodies(self.part_graph());
        let target = Self::target_operand(named, self.selected_body, &bodies);
        let node = Node::Offset { input: target, distance: ParamRef::Literal(self.offset_distance) };
        let node = match console_keys::offset_with_keys(node, args, &|t| self.lookup(t)) {
            Ok(node) => node,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        // The distance is remembered, so `offset` alone repeats the last one -- the number, when
        // it was one; a ramp driving it is this block's and not the tool's.
        if let Node::Offset { distance: ParamRef::Literal(d), .. } = &node {
            self.offset_distance = *d;
        }
        let mut graph = self.part_graph().clone();
        let node = graph.add(node);
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        self.set_part_graph(graph);
        // The body stopped being a body when the offset consumed it, so its eye goes with its
        // row for the reason `apply_boolean` gives. Nothing to do for an empty slot: there is no
        // row whose eye this stands for until something is put in it.
        if let Some(target) = target.source() {
            self.clear_hidden(target);
        }
        self.select_body(Some(node));
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        // **A success is said, never warned** (`--run` stops at a warning), so a block made with
        // its slot waiting says so and the script carries on to the line that fills it.
        self.say(if target.is_empty() {
            "An offset with an empty slot \u{2014} put a block in it".to_string()
        } else {
            format!("Offset by {:.3} mm", self.offset_distance)
        });
    }

    /// Winds the selected body about the Z axis.
    ///
    /// One node, unlike Shell: a twist replaces the body with the twisted body
    /// rather than producing something to combine with it, so there is nothing to boolean.
    pub(crate) fn apply_twist(&mut self, args: &Args) {
        let Ok(named) = self.named_target("twist", args) else { return };
        // The remembered axis, from a letter, remembered once the block is accepted; a direction
        // typed as three numbers goes on the block's frame below and leaves the memory alone,
        // since a memory is one of three.
        let axis_asked = args
            .get("axis")
            .and_then(|v| v.as_str())
            .and_then(|name| Axis::ALL.into_iter().find(|a| a.label().eq_ignore_ascii_case(name)))
            .unwrap_or(self.twist_axis);
        // **A body this block could actually wind**, which is not the same as any body. The part
        // may hold a plane and a point before it holds a solid, and both are bodies by
        // `Node::is_body` -- so a twist made in that part used to arrive wrapped around the plane,
        // which `operand_refusal` then greys in its own panel. A tool handing itself an operand
        // its panel refuses is the tool disagreeing with itself, and it was reported as exactly
        // that: "I make a point, a plane, and then I twist it, the twist tool automatically
        // selects I think the plane. I don't think that really makes sense."
        //
        // Asked of the same rule the picker asks, so the two cannot drift. There is no node yet
        // to ask *for*, so the question is put to a twist that does not exist -- which is why
        // `operand_refusal` takes the consumer's node rather than its id alone would allow.
        let bodies = self.doc.listed_bodies(self.part_graph());
        let graph = self.part_graph().clone();
        let windable = |id: &NodeId| twist_would_take(&self.doc, &graph, *id);
        let named_cannot = named
            .and_then(|slot| slot.source())
            .is_some_and(|id| !windable(&id));
        if named_cannot {
            self.warn("That block cannot be wound — a twist takes a bounded solid, not a plane or a point");
            return;
        }
        // **`target=none` makes the twist with its slot waiting; nothing named still explains
        // itself.** The refusal below is about what *can* be wound rather than about an unfilled
        // slot -- a part holding only a plane has a body and cannot be twisted -- so it is the
        // one message a guess would have to replace, and an empty slot is not an answer to it.
        let target = match named {
            Some(asked) => asked,
            None => {
                let found = self
                    .selected_body
                    .filter(|id| bodies.contains(id) && windable(id))
                    .or_else(|| bodies.iter().copied().find(|id| windable(id)));
                let Some(found) = found else {
                    // Said in the terms of what is missing rather than "needs a body", since a
                    // part with a plane in it has a body and still cannot be twisted.
                    self.warn("Needs a bounded solid to twist — a plane or a point cannot be wound");
                    return;
                };
                fieldform_scene::Operand::filled(found)
            }
        };
        // A rate that makes a visible quarter turn over whatever the body measures along the
        // chosen axis, so the first result shows what the tool does rather than looking like
        // nothing happened.
        let axis = axis_asked;
        // An empty slot has no extent to read, so the rate falls back to the same degree a
        // millimetre an unmeasurable body gets -- a number the panel can then be dragged from.
        let rate = match target.source().and_then(|id| self.body_extent(id).as_box()) {
            Some((lo, hi)) => {
                let along = match axis {
                    Axis::X => hi.x - lo.x,
                    Axis::Y => hi.y - lo.y,
                    Axis::Z => hi.z - lo.z,
                };
                if along > 1e-6 { 90.0 / along } else { 1.0 }
            }
            _ => 1.0,
        };
        let node = Node::Twist {
            input: target,
            degrees_per_mm: ParamRef::Literal(rate),
            frame: fieldform_scene::Frame::along(axis),
        };
        let node = match console_keys::twist_with_keys(node, args, &|t| self.lookup(t)) {
            Ok(node) => node,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        self.twist_axis = axis_asked;
        // What was done, from the block: a typed rate is said as a rate, and a direction typed
        // as three numbers is not one of the three letters.
        let rate_said = match &node {
            Node::Twist { degrees_per_mm: ParamRef::Literal(r), .. } if args.get("rate").is_some() || args.get("degrees_per_mm").is_some() => {
                format!("at {r}°/mm")
            }
            Node::Twist { degrees_per_mm: ParamRef::Literal(_), .. } => "a quarter turn".to_string(),
            _ => "at a parameter's rate".to_string(),
        };
        let axis_said = match args.get("axis").and_then(|v| v.as_str()) {
            Some(text) if !Axis::ALL.iter().any(|a| a.label().eq_ignore_ascii_case(text)) => "the axis given".to_string(),
            _ => self.twist_axis.label().to_string(),
        };
        let mut graph = self.part_graph().clone();
        let node = graph.add(node);
        self.set_part_graph(graph);
        if let Some(target) = target.source() {
            self.clear_hidden(target);
        }
        self.select_body(Some(node));
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        self.say(if target.is_empty() {
            "A twist with an empty slot — put a block in it to wind".to_string()
        } else {
            format!("Twisted {rate_said} about {axis_said}")
        });
    }

    /// Draws the selected body toward a curve.
    ///
    /// One node, like a twist and for the same reason: an attractor replaces the body with the
    /// attracted body rather than producing a second thing to combine with it.
    ///
    /// The curve is chosen the way a pipe chooses one -- the selected curve, else any curve,
    /// else a fresh line -- because needing a curve is not a reason to refuse to make the
    /// block. A node naming a curve that is not there does not evaluate at all, so the
    /// alternative to making one is blanking the viewport.
    pub(crate) fn apply_attract(&mut self, args: &Args) {
        let Ok(named) = self.named_target("attract", args) else { return };
        // `toward=` by name: a curve first, then a point, since only the app holds both lists.
        let toward_named = match console_keys::toward_named(args, &|n| self.curve_named(n), &|t| self.lookup(t)) {
            Ok(toward) => toward,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        // A block named here must be a point: the compiler refuses anything else, and refusing
        // it by name now is better than committing a graph that cannot be drawn.
        if let Some(fieldform_scene::Attractor::Point(n)) = toward_named
            && !matches!(self.part_graph().node(n), Some(Node::Point { .. }))
        {
            let label = self.body_label(n);
            self.warn(format!("`attract toward=` wants a curve or a point, and {label} is neither"));
            return;
        }
        // Passed through rather than stored on the App the way a twist's axis is. One block has
        // a panel, so the setting belongs there -- and a field here would be a second place the
        // same number lives, remembering a value from a previous attractor that this one never
        // showed anybody.
        let strength = DEFAULT_ATTRACT_STRENGTH;
        let bodies = self.doc.listed_bodies(self.part_graph());
        // `target=none` is taken, as every modifier takes it; nothing named still refuses, since
        // the fall-back is a body and there is none to fall back to.
        let target = match named {
            Some(asked) => asked,
            None => {
                let Some(found) =
                    self.selected_body.filter(|id| bodies.contains(id)).or(bodies.first().copied())
                else {
                    self.warn("Needs a body to attract");
                    return;
                };
                fieldform_scene::Operand::filled(found)
            }
        };
        // **Found before anything is built, and a point counts.** An attractor is drawn toward
        // one thing, and since that thing may now be a point there are two kinds to look for --
        // so drawing a line when the part already holds either is building a block nobody asked
        // for. Reported as exactly that: a line appeared although a curve was already there.
        //
        // The selection first, because reaching for a curve while a point is selected would
        // ignore what was pointed at. Then anything in the part, newest point before oldest
        // curve is not worth deciding -- a curve is the older habit, so it wins the tie. A line
        // is built only when there is genuinely nothing.
        let toward = self
            .selected_curve
            .filter(|id| self.doc.curves.contains_key(id))
            .map(fieldform_scene::Attractor::Curve)
            .or_else(|| {
                self.selected_body
                    .filter(|id| {
                        matches!(self.part_graph().node(*id), Some(Node::Point { .. }))
                    })
                    .map(fieldform_scene::Attractor::Point)
            })
            .or_else(|| {
                self.doc.curves.keys().next().copied().map(fieldform_scene::Attractor::Curve)
            })
            .or_else(|| {
                let mut points: Vec<NodeId> = self
                    .part_graph()
                    .nodes()
                    .filter(|(_, n)| matches!(n, Node::Point { .. }))
                    .map(|(n, _)| n)
                    .collect();
                points.sort_by_key(|n| n.raw());
                points.first().copied().map(fieldform_scene::Attractor::Point)
            });
        // A reach that covers a third of the body, so the first result shows what the tool
        // does rather than looking like nothing happened -- the same reasoning the twist's
        // quarter turn follows.
        let falloff = match target.source().and_then(|id| self.body_extent(id).as_box()) {
            Some((lo, hi)) => {
                let e = hi - lo;
                (e.x.max(e.y).max(e.z) / 3.0).max(1.0)
            }
            _ => 10.0,
        };
        // **Everything a key can refuse is read before a line is drawn**, so a refused
        // `strength=` leaves nothing behind. The target stands in for the attractor until the
        // keys have passed; a line is built only then, and only if nothing was found.
        let found = toward_named.or(toward);
        // With an empty slot there is nothing to stand in, so `attract target=none` says what it
        // needs rather than drawing a curve for a block that was asked for empty.
        let Some(placeholder) =
            found.or_else(|| target.source().map(fieldform_scene::Attractor::Point))
        else {
            self.warn("`attract target=none` needs something to pull toward — give it `toward=`");
            return;
        };
        let node = Node::Attract {
            input: target,
            toward: placeholder,
            strength: ParamRef::Literal(strength),
            falloff: ParamRef::Literal(falloff),
        };
        let mut node = match console_keys::attract_with_keys(node, args, &|t| self.lookup(t)) {
            Ok(node) => node,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let toward = match found {
            Some(t) => t,
            None => match self.add_line() {
                Some(id) => fieldform_scene::Attractor::Curve(id),
                None => return,
            },
        };
        if let Node::Attract { toward: slot, .. } = &mut node {
            *slot = toward;
        }
        let mut graph = self.part_graph().clone();
        let node = graph.add(node);
        // Only the compiler knows the rest of what an attractor refuses, so it is asked before
        // the commit, as every other modifier asks.
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        self.set_part_graph(graph);
        if let Some(target) = target.source() {
            self.clear_hidden(target);
        }
        self.select_body(Some(node));
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        self.say(format!(
            "Drawn toward {} — change the target or the pull in the block",
            match toward {
                fieldform_scene::Attractor::Curve(c) => self
                    .doc
                    .curves
                    .get(&c)
                    .map_or("a curve".to_string(), |c| c.name.clone()),
                fieldform_scene::Attractor::Point(n) => self.body_label(n),
            }
        ));
    }

    /// Why a fillet over these operands will not be the radius that was typed, if it will not.
    ///
    /// **One sentence, said in both places a fillet's radius is entered (D9).** The shared
    /// `Fillet` type carries an obligation with it — an approximate blend must announce itself
    /// where its radius is typed — and an obligation met by two separately worded notes is one
    /// that drifts. A blend asks about its pair; a pattern asks about the body it copies.
    pub(crate) fn approximate_join(
        &self,
        operands: impl IntoIterator<Item = fieldform_scene::NodeId>,
    ) -> Option<String> {
        let blocker = operands.into_iter().find(|id| {
            !fieldform_field::carries_true_distance(&self.doc, self.part_graph(), *id)
        })?;
        Some(format!(
            "Approximate: \u{201c}{}\u{201d} only bounds distance from below, so this fillet \
             comes out at least the radius asked for and up to about 1.7\u{00d7} it. G0 joins \
             them at a crease with no such error.",
            self.body_name(blocker)
                .map(str::to_string)
                .unwrap_or_else(|| self.derived_label(blocker)),
        ))
    }

    /// A second direction to start a grid from: the first step turned a quarter in the ground
    /// plane, the same length, and three rows. A grid whose second step ran along the first would be
    /// refused by name the moment it was switched on, which is a poor welcome.
    pub(crate) fn default_second_direction(&self, first: [ParamRef; 3]) -> fieldform_scene::PatternSecond {
        let (x, y, z) = (self.resolve(first[0]), self.resolve(first[1]), self.resolve(first[2]));
        let length = (x * x + y * y + z * z).sqrt();
        let length = if length > 1e-9 { length } else { DEFAULT_BODY_MM * 0.5 };
        let ground = x.hypot(y);
        let (sx, sy) = if ground > 1e-9 { (-y / ground * length, x / ground * length) } else { (0.0, length) };
        fieldform_scene::PatternSecond {
            step: [ParamRef::Literal(sx), ParamRef::Literal(sy), ParamRef::Literal(0.0)],
            count: ParamRef::Literal(3.0),
        }
    }

    /// Repeats the selected body, round the Z axis or along a line, and unions the copies.
    ///
    /// **Placements of one body, not copies of it.** Every instance is a placement of the
    /// *same* node, so editing the original changes all of them — which is what a pattern
    /// should mean, and what makes this different from pressing paste six times.
    ///
    /// That was not expressible until shader generation learned point contexts (backlog item
    /// 21): a body reached through several placements needed evaluating at several points, and
    /// each node used to get exactly one. This is the first operation built on top of that.
    pub(crate) fn apply_pattern(&mut self, args: &Args) {
        let Ok(named) = self.named_target("pattern", args) else { return };
        // What the tool remembers between uses, from what it was given: a count, a kind and a
        // single-number step -- read into locals here and remembered only once the block has
        // been accepted, so a refused line leaves the tool as it was. The block itself is then
        // read in full by `console_keys::pattern_with_keys`, which is where a three-number step,
        // a ring's frame and a filleted join land.
        let count_asked = args
            .get("count")
            .and_then(|v| v.as_f64())
            .map(|n| n.round().clamp(2.0, 512.0) as u32)
            .unwrap_or(self.pattern_count);
        let radial = match args.get("kind").and_then(|v| v.as_str()) {
            Some(kind) => !kind.eq_ignore_ascii_case("linear"),
            None => self.pattern_radial,
        };
        let step_mm = args.get("step").and_then(|v| v.as_f64()).unwrap_or(self.pattern_step_mm);
        let options = match pattern_options_from(args) {
            Ok(options) => options,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let bodies = self.doc.listed_bodies(self.part_graph());
        // Nothing to repeat is a pattern with its slot waiting rather than a refusal
        // ([`Self::target_operand`]), so a script can say what the part is made of in the order
        // it thinks of it.
        let target = Self::target_operand(named, self.selected_body, &bodies);
        let count = count_asked.clamp(1, 512);
        let kind = if radial {
            PatternKind::Radial { frame: fieldform_scene::Frame::at_along(Axis::Z, [ParamRef::Literal(0.0); 3]) }
        } else {
            PatternKind::Linear {
                step: [ParamRef::Literal(step_mm), ParamRef::Literal(0.0), ParamRef::Literal(0.0)],
            }
        };

        let node = Node::Pattern {
            input: target,
            // The tool's remembered count stays a `u32` — it is transient interface state, and a
            // number is honest for it. A pattern is born with a literal and named afterwards.
            count: ParamRef::Literal(f64::from(count)),
            kind,
            // **The copies, and only the copies** -- the same rule Mirror follows, and asked
            // for by the same reasoning. A block holding instance zero swallows the body it
            // patterns: one row, one eye, and no way to look at the body on its own or hand it
            // to a boolean. The picture is unchanged, because instance zero is still on screen
            // -- as its own body, beside the block.
            join: false,
            fillet: Fillet::hard(),
            options,
        };
        let node = match console_keys::pattern_with_keys(node, args, &|t| self.lookup(t)) {
            Ok(node) => node,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let filleted = matches!(&node, Node::Pattern { join: true, fillet, .. } if !matches!(fillet.radius, ParamRef::Literal(r) if r <= 0.0));
        // What the status line says about the step: the block's own, which a typed `step=0,30,0`
        // sets, and not the tool's single-number memory, which it does not.
        let step_said = match &node {
            Node::Pattern { kind: PatternKind::Linear { step }, .. } => {
                let numbers: Option<Vec<f64>> =
                    step.iter().map(|s| if let ParamRef::Literal(v) = s { Some(*v) } else { None }).collect();
                match numbers {
                    Some(n) => format!("{:.1} mm apart", (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()),
                    None => "a parameter's step apart".to_string(),
                }
            }
            _ => String::new(),
        };
        let mut graph = self.part_graph().clone();
        let node = graph.add(node);
        // A filleted join is the one thing here D8 can refuse, so it is asked before the commit.
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        self.pattern_count = count_asked;
        self.pattern_radial = radial;
        self.pattern_step_mm = step_mm;
        // Consumed by the pattern, so no longer a root -- and a block that is not a root draws
        // only when it is asked for. Asked for here, because it was on screen a moment ago and
        // nothing this operation did should take it off.
        //
        // **It is not named, and that is the change.** This used to give it one, because a name
        // was what let a consumed block into the operand lists the owner reaches for next. Two
        // things undid that: `operand_choices_in` and `pair_candidates_in` no longer ask, so the
        // name bought nothing; and naming was then what lifted a block out of the block it is
        // drawn inside, so writing one on the user's behalf was the app deciding a question that
        // is theirs. Neither a name nor the eye moves a row now (`block-ownership-spec.md` §2.2):
        // the pattern adopts its target, which draws from inside the pattern's row.
        //
        // Through the eye, asked of the graph the edit writes, so the flag it reaches is the one a
        // consumed block's eye means; and in the graph edit's own undo entry, after it, so a redo
        // of the pattern draws the target again rather than leaving that to an entry of its own
        // (§7.3).
        // Nothing to keep drawn where the slot is empty: there is no target whose row the eye
        // stands for until something is put in it.
        match target.source() {
            Some(target) => {
                let keep_drawn = eye_command(&graph, target, true);
                self.set_part_graph_with(graph, vec![Box::new(keep_drawn)]);
                self.clear_hidden(target);
            }
            None => self.set_part_graph(graph),
        }
        self.select_body(Some(node));
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        // "Still beside it" is only true of a block that was there, so an empty slot says what
        // it is waiting for instead -- and says it, never warns it, so a script runs on.
        match target.source() {
            None => self.say("A pattern with an empty slot — put a block in it to repeat"),
            Some(target) => {
                let original = self.body_label(target);
                self.say(if radial {
                    format!("{count} around the axis — {original} is still beside it")
                } else {
                    format!("{count}, {step_said} — {original} is still beside it")
                });
            }
        }
        // Nothing to announce about an empty slot: there is no operand yet whose distance could
        // be a lower bound. `Option` is the iterator of nought or one, so the question is asked
        // of exactly the operands there are.
        if filleted && let Some(why) = self.approximate_join(target.source()) {
            self.say(format!("\u{2248} {why}"));
        }
    }

    /// Mirrors the selected body, keeping the original, so the result is symmetric.
    ///
    /// **One block.** It used to add a reflection and a boolean, which is one operation
    /// described twice: the list showed a Mirror followed by a Union, and a Union that nobody
    /// asked for reads as though something else had happened. The join lives inside the node
    /// now, and the shader expresses it by evaluating the body at two points — which it can do
    /// since point contexts landed.
    ///
    /// The plane goes through the world origin. Not through the body's own centre, which would
    /// mirror it onto itself and give the body back: a mirror needs a plane the geometry is to
    /// one side of, and only the user knows which.
    pub(crate) fn apply_mirror(&mut self, args: &Args) {
        let Ok(named) = self.named_target("mirror", args) else { return };
        let axis_asked = args
            .get("axis")
            .and_then(|v| v.as_str())
            .and_then(|name| Axis::ALL.into_iter().find(|a| a.label().eq_ignore_ascii_case(name)))
            .unwrap_or(self.mirror_axis);
        let bodies = self.doc.listed_bodies(self.part_graph());
        // Nothing to mirror leaves the slot waiting rather than refusing, as a shell's does.
        let target = Self::target_operand(named, self.selected_body, &bodies);
        let node = Node::Mirror {
            input: target,
            frame: fieldform_scene::Frame::at_along(axis_asked, [ParamRef::Literal(0.0); 3]),
            // **The reflection, and only the reflection.** It used to hold both halves, so
            // mirroring a body made that body vanish into a block that drew two of it -- one
            // row, one eye, and no way to look at either half alone. Asked for directly: the
            // block holds the mirrored body, the original stays a body of its own, and a
            // boolean is what puts them together if that is what was wanted. `-join` asks for
            // both halves in one block, for the script that wants the symmetric part whole.
            join: false,
        };
        let node = match console_keys::mirror_with_keys(node, args, &|t| self.lookup(t)) {
            Ok(node) => node,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        self.mirror_axis = axis_asked;
        let joined = matches!(&node, Node::Mirror { join: true, .. });
        let mut graph = self.part_graph().clone();
        let node = graph.add(node);
        // The original is consumed by the mirror, so it is no longer a root -- and a block that
        // is not a root draws only when it is asked for. Asked for here, because it was on
        // screen a moment ago. Not *named*, for the reasons `apply_pattern` states: the operand
        // lists no longer gate on a name, and taking a block out of its parent in the list is
        // the user's to ask for. Through the eye, in the graph edit's own entry and after it, as
        // `apply_pattern` does.
        // For the reason `apply_pattern` gives: with an empty slot there is no row whose eye
        // this stands for.
        match target.source() {
            Some(target) => {
                let keep_drawn = eye_command(&graph, target, true);
                self.set_part_graph_with(graph, vec![Box::new(keep_drawn)]);
                self.clear_hidden(target);
            }
            None => self.set_part_graph(graph),
        }
        self.select_body(Some(node));
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        match target.source() {
            None => self.say("A mirror with an empty slot — put a block in it to reflect"),
            Some(target) => {
                let original = self.body_label(target);
                self.say(if joined {
                    format!("Mirrored in {} — both halves in one block", self.mirror_axis.label())
                } else {
                    format!(
                        "Mirrored in {} — the reflection is its own block, and {original} is still beside it",
                        self.mirror_axis.label(),
                    )
                });
            }
        }
    }

    /// Morphs two bodies into one, so that one becomes the other as the amount is dragged.
    ///
    /// **Two operands, so it takes them the way a boolean does** — `starting_pair`, not a
    /// second way to say "these two". It used to require both slots to have been set by hand,
    /// which meant a part full of bodies could still refuse to morph and say only that it
    /// needed two of them. From and To are in the block, so a pair it guessed wrongly is a
    /// dropdown away from right.
    pub(crate) fn apply_morph(&mut self, args: &Args) {
        // `morph a=x b=y t=0.5`: both by name, or neither -- one named and one guessed would be
        // a pair nobody asked for.
        let lookup = |t: &str| self.lookup(t);
        let named = (
            console_keys::block_named("morph", "a", args, &lookup),
            console_keys::block_named("morph", "b", args, &lookup),
        );
        let pair = match named {
            (Err(why), _) | (_, Err(why)) => {
                self.warn(why);
                return;
            }
            (Ok(Some(a)), Ok(Some(b))) => Some((a, b)),
            (Ok(None), Ok(None)) => None,
            _ => {
                self.warn("`morph` names both operands or neither — `morph a=box b=ball`");
                return;
            }
        };
        let Some((a, b)) = pair.or_else(|| self.starting_pair()) else {
            self.warn("Needs two bodies to morph between");
            return;
        };
        // Half way, which is the only starting value that shows what the block does. Nought or
        // one would look like nothing happened.
        let node = Node::Morph {
            a: fieldform_scene::Operand::filled(a),
            b: fieldform_scene::Operand::filled(b),
            t: ParamRef::Literal(0.5),
        };
        let node = match console_keys::morph_with_keys(node, args, &|t| self.lookup(t)) {
            Ok(node) => node,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        let mut graph = self.part_graph().clone();
        let node = graph.add(node);
        if let Some(why) = self.refuse_reason(&graph) {
            self.warn(why);
            return;
        }
        self.set_part_graph(graph);
        self.clear_hidden(a);
        self.clear_hidden(b);
        self.select_body(Some(node));
        // The result becomes the next pair's first operand, the way a boolean's does: what you
        // just made is what you are most likely to reach for next.
        self.boolean_a = Some(node);
        self.boolean_b = None;
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        // Which two it picked, by name, because it picks rather than asks: a message that did
        // not say would leave the guess to be discovered by looking at the shape.
        self.say(format!(
            "Morphing {} to {} — change either, or the amount, in the block",
            self.body_label(a),
            self.body_label(b)
        ));
    }

    /// Shells the selected body: hollows it out, leaving a wall of the stated thickness.
    ///
    /// Not `Thicken`, which builds a solid around a surface that has none — another CAD package
    /// keeps a support article about people conflating those two. And not `Offset` either, which
    /// is the other thing it gets confused with: a negative offset is a *smaller solid*, where
    /// this is that solid subtracted from the original.
    pub(crate) fn apply_shell(&mut self, args: &Args) {
        let Ok(named) = self.named_target("shell", args) else { return };
        let bodies = self.doc.listed_bodies(self.part_graph());
        // **No body is no refusal**, which is the whole of §7.2 reaching the console: the block
        // is made with its slot waiting. [`Self::target_operand`] carries the reasoning.
        let target = Self::target_operand(named, self.selected_body, &bodies);
        // One node. It used to be an `Offset` and a `Boolean` side by side, which is what a
        // shell *is* and was still the wrong thing to show: the block named for the operation
        // had no thickness on it — that lived on the other one — and it offered a boolean's
        // union and intersect and a blend radius to something that is neither.
        let node = Node::Shell { input: target, thickness: ParamRef::Literal(self.shell_thickness) };
        let node = match console_keys::shell_with_keys(node, args, &|t| self.lookup(t)) {
            Ok(node) => node,
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        if let Node::Shell { thickness: ParamRef::Literal(t), .. } = &node {
            self.shell_thickness = *t;
        }
        let mut graph = self.part_graph().clone();
        let node = graph.add(node);
        self.set_part_graph(graph);
        if let Some(target) = target.source() {
            self.clear_hidden(target);
        }
        self.select_body(Some(node));
        self.mark_changed();
        self.view_mode = ViewMode::Field;
        self.rebuild_render_resources();
        self.say(if target.is_empty() {
            "A shell with an empty slot \u{2014} put a block in it".to_string()
        } else {
            format!("Shelled to a {:.3} mm wall", self.shell_thickness)
        });
    }

    /// Deletes one block, leaving whatever it was made from.
    ///
    /// Deleting a boolean leaves the two solids it joined — you are undoing the operation, not
    /// discarding the shapes. Deleting a sphere removes the sphere. One rule, and it is the
    /// same one the graph enforces.
    ///
    /// Answers whether it deleted anything, so a caller says "Deleted" only when it did.
    pub(crate) fn delete_body(&mut self, id: NodeId) -> bool {
        let freed = self.part_graph().node(id).map(|n| n.inputs()).unwrap_or_default();
        if self.part_graph().node(id).is_none() {
            return false;
        }
        let graph = match Self::after_deleting(&self.doc, self.part_graph(), id, &|e| {
            self.explain_compile(e)
        }) {
            Ok(graph) => graph,
            Err(why) => {
                self.warn(Self::delete_refused(&self.body_label(id), &why));
                return false;
            }
        };
        self.let_curves_go(id);
        self.set_part_graph(graph);
        self.clear_hidden(id);
        // What it was made from is on its own again, so the selection and the boolean tool
        // follow it rather than pointing at something that no longer exists.
        self.select_body(freed.first().copied());
        self.boolean_a = freed.first().copied().or(self.boolean_a.filter(|b| *b != id));
        self.boolean_b = freed.get(1).copied().or(self.boolean_b.filter(|b| *b != id));
        self.mark_changed();
        self.rebuild_render_resources();
        true
    }

    /// The graph deleting `id` leaves, or why that graph may not be committed: which is only ever
    /// that the part builds and the graph left would not.
    ///
    /// Free of the `App` for the reason `refusal_of` is, and the whole of the delete that a test
    /// has to see: the letting go, the cascade, and the check.
    pub(crate) fn after_deleting(
        doc: &Document,
        graph: &Graph,
        id: NodeId,
        explain: &dyn Fn(&fieldform_field::CompileError) -> String,
    ) -> Result<Graph, String> {
        let before = graph;
        let mut graph = graph.clone();
        // **Anything merely sitting on this keeps its position and lets go**, before the
        // cascade runs.
        //
        // A place or a frame taken from a block is an input — that is what makes it evaluate
        // first and what keeps it out of the body list — but it is not the same *kind* of input
        // as a boolean's operand, and `remove_cascading` treats every input alike. A boolean
        // without its operand has nothing left to be, so collapsing it is right. A sphere
        // sitting on a point is still a sphere: the point told it where, and deleting the point
        // should leave it where it was told.
        //
        // Reported as a plane vanishing when the point beneath it was deleted. It was not the
        // plane that vanished — it was every body that had been placed on that point, cascaded
        // through and taken with it, silently. Impossible before a place could name a block,
        // which is why the cascade never had to tell the two apart until now.
        Self::detach_from(doc, &mut graph, id);
        graph.remove_cascading(id);
        // **Asked before anything is written, as every other graph edit asks it.** The cascade
        // hands a consumer its surviving operand, and that is not always one it can be built
        // on: delete the point an attract pulls toward and the attract is handed its own body to
        // pull toward, which is not a point. Committed, the part went dark on a click that then
        // said "Deleted".
        match Self::delete_breaks(doc, before, &graph, explain) {
            Some(why) => Err(why),
            None => Ok(graph),
        }
    }

    /// Why a delete that leaves `after` of the part `before` may not be committed, if it may not:
    /// only ever that `before` builds and `after` would not, in [`Self::refusal_of`]'s words.
    ///
    /// **Refused only for breaking a part that built.** Asked of the graph left and of nothing
    /// else, the check refused every delete in a part that already could not be built, unless that
    /// one delete mended it -- and a part with two faults has no such delete, so it could not be
    /// repaired at all: two twists the old stand-in left reading the planes they were aimed at,
    /// each delete leaving the other. A delete makes nothing worse in a part already broken, and
    /// it is the way out of one. The part as it stands is asked only once the graph left has
    /// failed, so a delete that leaves a part that builds compiles once, as it did.
    ///
    /// **One rule for every delete**, a block's ([`Self::after_deleting`]) and a curve's out of a
    /// held loop ([`Self::loop_member_deletion`]) alike. The second had a copy of the check that
    /// asked the graph left alone, so it went on refusing in a part already broken after the
    /// first stopped.
    pub(crate) fn delete_breaks(
        doc: &Document,
        before: &Graph,
        after: &Graph,
        explain: &dyn Fn(&fieldform_field::CompileError) -> String,
    ) -> Option<String> {
        let why = Self::fault_in(doc, after, explain)?;
        match Self::fault_in(doc, before, explain) {
            None => Some(why),
            Some(_) => None,
        }
    }

    /// What stops `graph` being built, in [`Self::refusal_of`]'s words, except that a part with
    /// no body at all is not at fault.
    ///
    /// The compilers refuse that part because there is nothing to draw -- not because anything
    /// is wrong. A delete is the edit that gets there on purpose, and asking the check of it would
    /// make the last body of a part undeletable. Still in order, though: no body is not the same
    /// as no way to evaluate.
    fn fault_in(
        doc: &Document,
        graph: &Graph,
        explain: &dyn Fn(&fieldform_field::CompileError) -> String,
    ) -> Option<String> {
        if graph.roots().is_empty() && graph.evaluation_order().is_some() {
            return None;
        }
        Self::refusal_of(doc, graph, explain)
    }

    /// What a refused delete says: which block or curve, that the part builds now, and the check's
    /// reason.
    ///
    /// The reason is quoted after a colon, and the check's own sentences open with a capital,
    /// being written to stand alone as any other edit's warning; so its first letter is lowered.
    /// "Builds now" because a delete is refused only for breaking a part that built
    /// ([`Self::delete_breaks`]). It used to say the delete "would leave" a block that could not
    /// be built, which in a part already broken was a fault the delete had not made.
    pub(crate) fn delete_refused(label: &str, why: &str) -> String {
        let mut letters = why.chars();
        let why: String = match letters.next() {
            Some(first) => first.to_lowercase().chain(letters).collect(),
            None => String::new(),
        };
        format!("Deleting {label} would break a part that builds now: {why}")
    }

    /// Turns every place and frame taken from `id` into the typed equivalent it currently
    /// resolves to, so deleting `id` leaves those blocks where they are.
    ///
    /// Resolved first and written after, because reading a position needs the graph as it
    /// stands and writing one needs it mutable — and a body that moved because the thing it sat
    /// on was deleted would be a worse bug than the one this fixes.
    fn detach_from(doc: &Document, graph: &mut Graph, id: NodeId) {
        let number = |r: &ParamRef| doc.resolve(r).unwrap_or(0.0);
        let numbers = |r: &ParamRef| Some(number(r));

        // Places: the position each one is at right now.
        let mut settled: Vec<(NodeId, usize, fieldform_scene::Place)> = Vec::new();
        for (holder, node) in graph.nodes() {
            for (i, place) in node.places().iter().enumerate() {
                if place.source() == Some(id) {
                    let at = fieldform_field::resolve_place(doc, graph, place, &numbers)
                        .unwrap_or_else(|_| Point3::origin());
                    settled.push((
                        holder,
                        i,
                        fieldform_scene::Place::At([
                            ParamRef::Literal(at.x),
                            ParamRef::Literal(at.y),
                            ParamRef::Literal(at.z),
                        ]),
                    ));
                }
            }
        }

        // Frames: the place *and* the facing, since a frame carries both and half of one is a
        // block pointing somewhere it never pointed.
        //
        // Every block a frame reads, not only the one a `From` names: a frame between two points
        // with either deleted has no other way to stand, and left holding the deleted one it was
        // handed the block's surviving input by the cascade -- a mirror between two points became
        // a mirror aimed from the body it reflects.
        let mut aimed: Vec<(NodeId, usize, fieldform_scene::Frame)> = Vec::new();
        for (holder, node) in graph.nodes() {
            for (i, frame) in node.frames().iter().enumerate() {
                if frame.nodes().any(|read| read == id)
                    && let Ok((origin, rotation)) =
                        fieldform_field::resolve_frame(doc, graph, frame, &numbers)
                {
                    let axis = rotation * Vector3::z();
                    aimed.push((
                        holder,
                        i,
                        fieldform_scene::Frame::At {
                            origin: [
                                ParamRef::Literal(origin.x),
                                ParamRef::Literal(origin.y),
                                ParamRef::Literal(origin.z),
                            ],
                            axis: [
                                ParamRef::Literal(axis.x),
                                ParamRef::Literal(axis.y),
                                ParamRef::Literal(axis.z),
                            ],
                        },
                    ));
                }
            }
        }

        // **Turns: the angles the block is at right now.** A facing taken from a plane is an
        // input like a place is, and `remove_cascading` treats every input alike -- so deleting
        // a plane a cuboid was aimed at found that the cuboid had no *other* input to fall back
        // to and deleted the cuboid. A block aimed at a plane is still that block: the plane told
        // it which way, and deleting the plane should leave it facing where it was told. Either
        // point of a facing between two goes the same way, for the frame's reason.
        let mut facing: Vec<(NodeId, Turn)> = Vec::new();
        for (holder, node) in graph.nodes() {
            if let Some(turn) = node.turn()
                && turn.nodes().any(|read| read == id)
            {
                let (x, y, z) =
                    fieldform_field::turn_rotation(doc, graph, turn, &numbers).euler_angles();
                facing.push((
                    holder,
                    Turn::By([
                        ParamRef::Literal(x.to_degrees()),
                        ParamRef::Literal(y.to_degrees()),
                        ParamRef::Literal(z.to_degrees()),
                    ]),
                ));
            }
        }

        // **And a pair of ends standing on it**, which keeps both points rather than the
        // direction alone: the block runs from where it runs to where it stops, and that is two
        // places once it is no longer standing on anything.
        let mut standing_on: Vec<(NodeId, usize, Ends)> = Vec::new();
        for (holder, node) in graph.nodes() {
            for (i, ends) in node.ends().iter().enumerate() {
                if ends.node() == Some(id) {
                    let (a, b) = fieldform_field::ends_at(ends, doc, graph, &number);
                    standing_on.push((
                        holder,
                        i,
                        Ends::between(
                            fieldform_scene::Place::At(Self::as_literals(a)),
                            fieldform_scene::Place::At(Self::as_literals(b)),
                        ),
                    ));
                }
            }
        }

        // **And a parameter this was driving, which settles to the number it currently reads.**
        //
        // A ramp is a block like any other now (`ramp-decision.md` §7.2), so it can be selected
        // and deleted while a morph or an offset is reading it. `remove_cascading` below moves a
        // driver's edge as it moves any other and would hand that socket one of the block's
        // operands — a body's field driving an amount, which is refused at compile as
        // `DrivenParameterNeedsARamp`. The part would go dark on a delete, with the error naming
        // a node rather than the click.
        //
        // The midpoint of the ramp's output rather than either end, so letting go lands between
        // what the varying value was doing at its extremes instead of at whichever end the code
        // happened to pick — the same number `drop_driver` hands back when the switch is turned
        // off, so deleting the block and switching it off agree.
        let settle_to = match graph.node(id) {
            Some(Node::Ramp { out_min, out_max, .. }) => {
                (number(out_min) + number(out_max)) * 0.5
            }
            // Not a ramp: nothing else can legally drive a parameter, so nothing routes here.
            _ => 0.0,
        };
        graph.release_drivers(id, settle_to);

        for (holder, i, place) in settled {
            if let Some(node) = graph.node_mut(holder)
                && let Some(slot) = node.places_mut().get_mut(i)
            {
                **slot = place;
            }
        }
        for (holder, i, frame) in aimed {
            if let Some(node) = graph.node_mut(holder)
                && let Some(slot) = node.frames_mut().get_mut(i)
            {
                **slot = frame;
            }
        }
        for (holder, turn) in facing {
            if let Some(node) = graph.node_mut(holder)
                && let Some(slot) = node.turn_mut()
            {
                *slot = turn;
            }
        }
        for (holder, i, ends) in standing_on {
            if let Some(node) = graph.node_mut(holder)
                && let Some(slot) = node.ends_mut().get_mut(i)
            {
                **slot = ends;
            }
        }
    }

    /// **And the curves, which the graph walk in `detach_from` cannot see.** A curve is not a
    /// node, so nothing about it appears in `inputs` and the delete cascade never touches it —
    /// which means a curve sitting on a deleted point would not be taken with the point, it would
    /// simply stop following and keep a position whose source no longer exists. Letting go here
    /// makes that explicit rather than accidental, and keeps the curve where it is.
    ///
    /// Apart from the graph's half because it writes the document, so it runs only once the
    /// delete is past its check: a refused delete leaves every curve still standing on the block.
    fn let_curves_go(&mut self, id: NodeId) {
        let standing: Vec<CurveId> = self
            .doc
            .curves
            .iter()
            .filter(|(_, c)| c.sits_on().any(|n| n == id))
            .map(|(cid, _)| *cid)
            .collect();
        for cid in standing {
            let Some(curve) = self.doc.curves.get(&cid) else { continue };
            let placed: Vec<usize> = (0..curve.control_points().len())
                .filter(|i| curve.placed_on(*i) == Some(id))
                .collect();
            if let Some(curve) = self.doc.curve_mut(cid) {
                for i in placed {
                    curve.place_on(i, None);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Spelled out rather than taken from the module's own glob: these tests hold a rule about
    // which graphs may be committed, and the names the rule is written in are worth reading.
    use crate::App;
    use fieldform_ops::Document;
    use fieldform_scene::{
        Axis, BooleanOp, CellMap, Continuity, Fillet, Frame, Graph, LatticeCell, LatticePhase,
        Node, NodeId, Operand, ParamRef, Place,
    };

    /// The kernel's refusal in its own words, as every caller of [`App::refusal_of`] that has no
    /// window hands it (`a_delete_that_would_break_a_part_that_builds_is_refused_with_the_check_s_reason`).
    fn explain(e: &fieldform_field::CompileError) -> String {
        e.to_string()
    }

    fn lit(v: f64) -> ParamRef {
        ParamRef::Literal(v)
    }

    fn ball_in(graph: &mut Graph, x: f64) -> NodeId {
        graph.add(Node::Sphere { centre: Place::At([lit(x), lit(0.0), lit(0.0)]), radius: lit(10.0) })
    }

    /// **The edit gate stands down for incompleteness and for nothing else.**
    ///
    /// An empty slot is a legal, savable state, and it arises from a gesture — dragging a block
    /// out of a slot (`notebook-spec.md` §7.2) — so a gate that refused the graph it leaves would
    /// make that gesture impossible to perform: the drag would be taken back the instant it was
    /// finished, with a message about a block that cannot be drawn. The viewport prunes the holed
    /// body and draws its siblings, and this is the same answer at the point of commit.
    ///
    /// Three cases, because the hole can be anywhere: under one body of several, under the only
    /// body there is, and under a block that something else reads. And then the control that
    /// matters: a cycle is still refused, so the gate was narrowed and not opened.
    #[test]
    fn emptying_a_slot_is_never_refused_and_a_cycle_still_is() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let ball = ball_in(&mut graph, 0.0);
        let beside = ball_in(&mut graph, 40.0);
        let shell = graph.add(Node::Shell { input: Operand::filled(ball), thickness: lit(1.0) });
        assert_eq!(App::refusal_of(&doc, &graph, &explain), None, "the part builds to begin with");

        // One body of two, dragged out of its slot. The sphere beside it is untouched, so there
        // is a part left to compile -- and the holed body is not what the gate asks about.
        if let Some(Node::Shell { input, .. }) = graph.node_mut(shell) {
            input.clear();
        }
        assert_eq!(
            App::refusal_of(&doc, &graph, &explain),
            None,
            "a body with an empty slot is pruned, not refused"
        );

        // And with something standing on the hole, which is the case `Graph::incomplete` closes
        // downstream: the twist above the holed shell has no value either, and neither is at
        // fault for the edit that emptied the slot below them.
        let twist = graph.add(Node::Twist {
            input: Operand::filled(shell),
            degrees_per_mm: lit(1.0),
            frame: Frame::along(Axis::Z),
        });
        assert!(graph.incomplete().contains(&twist), "the block reading the hole is incomplete too");
        assert_eq!(App::refusal_of(&doc, &graph, &explain), None, "and still not refused");

        // The whole part waiting to be told what to read: nothing is left to ask either compiler
        // about, which is a part in the making rather than a part at fault.
        let mut all_holed = Graph::empty();
        all_holed.add(Node::Shell { input: Operand::empty(), thickness: lit(1.0) });
        assert_eq!(
            App::refusal_of(&doc, &all_holed, &explain),
            None,
            "a part whose only body is holed is not a part at fault"
        );
        assert!(!graph.incomplete().contains(&beside), "and a whole body is never swept up by one");

        // The control. A loop is a walk that does not come back rather than a block short of an
        // input, so the cycle check is still asked of the whole graph, holes and all.
        let mut looped = Graph::empty();
        let inner = ball_in(&mut looped, 0.0);
        let outer = looped.add(Node::Shell { input: Operand::filled(inner), thickness: lit(1.0) });
        let above = looped.add(Node::Shell { input: Operand::filled(outer), thickness: lit(1.0) });
        if let Some(Node::Shell { input, .. }) = looped.node_mut(outer) {
            input.fill(above);
        }
        assert!(
            App::refusal_of(&doc, &looped, &explain).is_some(),
            "a graph that depends on itself is refused as it always was"
        );
        // A hole somewhere else in that graph does not excuse the loop.
        looped.add(Node::Offset { input: Operand::empty(), distance: lit(1.0) });
        assert!(
            App::refusal_of(&doc, &looped, &explain).is_some(),
            "and a hole elsewhere is not a way past the cycle check"
        );
    }

    /// **An approximate blend still announces itself**, which is the thing the gate standing down
    /// must not quietly stop happening.
    ///
    /// The name this test was asked for was `a_blend_against_a_lower_bound_operand_is_still_refused`,
    /// and it is not refused: the project notes records the owner reversing that on 2026-09-02, deliberately
    /// and with reasons — the error is bounded and one-directional, so a fillet against a lattice
    /// comes out at least the radius asked for and never smaller. What replaced the refusal is the
    /// announcement, and the rule that remains is that an approximate blend must say so where its
    /// radius is typed. So that is what is asserted here, under the property's own name: the gate
    /// lets the blend through, and `approximate_blends` names it, which is what the panel's `≈`
    /// and `apply_boolean`'s console line both read.
    #[test]
    fn a_blend_against_a_lower_bound_operand_is_accepted_and_still_announces_itself() {
        let doc = Document::new();
        let mut graph = Graph::empty();
        let ball = ball_in(&mut graph, 0.0);
        // A gyroid sheet bounds its distance from below, which is the operand the old refusal was
        // written for: its measure is in millimetres, divided by its own gradient bound.
        let lattice = graph.add(Node::Lattice {
            cell: LatticeCell::Gyroid,
            frame: Frame::along(Axis::Z),
            map: CellMap::Rectangular { cell: lit(8.0) },
            phase: LatticePhase::Sheet { thickness: lit(1.0), offset: lit(0.0) },
        });
        assert!(
            !fieldform_field::carries_true_distance(&doc, &graph, lattice),
            "the operand this is about only bounds distance from below"
        );
        let blend = graph.add(Node::Blend {
            op: BooleanOp::Union,
            a: Operand::filled(ball),
            b: Operand::filled(lattice),
            fillet: Fillet { radius: lit(2.0), continuity: Continuity::G1, second: None },
        });
        assert_eq!(
            App::refusal_of(&doc, &graph, &explain),
            None,
            "accepted, since the error is bounded and one-directional"
        );
        assert!(
            fieldform_field::approximate_blends(&doc, &graph).contains(&blend),
            "and reported, which is what the panel marks beside the radius"
        );
    }

    /// **A verb with nothing to work on makes the block and leaves its slot waiting**, which is
    /// what lets a script build a part top-down: `shell` before there is anything to hollow, and
    /// the block it made is there to be filled by the next line.
    ///
    /// **A red cage is pruned by the gate, not refused** (R12, R4): a cage opened by a tool is a
    /// state, like an empty slot, and an edit to anything else in the part while it is open has
    /// to go through. Both when the cage is one body of two and when it is the only one.
    #[test]
    fn a_red_cage_is_never_what_the_gate_refuses() {
        let cube = fieldform_cage::Cage::cube_of(10.0);
        let positions = cube.vertex_ids().map(|v| cube.vertex_position(v)).collect();
        let faces: Vec<Vec<u32>> = cube
            .face_ids()
            .skip(1)
            .map(|f| cube.face_vertices(f).iter().map(|v| v.index() as u32).collect())
            .collect();
        let mut doc = Document::new();
        let red = doc.add_cage_body(fieldform_cage::CageBody::new(
            fieldform_cage::QuadMesh::from_faces(positions, &faces).unwrap(),
        ));

        let mut graph = Graph::from_cage(red);
        assert_eq!(App::refusal_of(&doc, &graph, &explain), None, "a part that is only a red cage is a part in the making");
        ball_in(&mut graph, 40.0);
        assert_eq!(App::refusal_of(&doc, &graph, &explain), None, "and the sphere beside it is what the gate asks about");
    }

    /// Asked of [`App::target_operand`], which is the whole of that decision and is free of the
    /// `App` so it can be asked without a window. The block built from the answer is then held to
    /// the two things that make the state legal: `Graph::incomplete` names it, and the edit gate
    /// does not refuse it.
    #[test]
    fn shell_with_no_target_makes_a_block_with_an_empty_slot() {
        let doc = Document::new();
        let nothing: Vec<NodeId> = Vec::new();
        let slot = App::target_operand(None, None, &nothing);
        assert!(slot.is_empty(), "nothing to shell leaves the slot waiting rather than refusing");

        let mut graph = Graph::empty();
        let shell = graph.add(Node::Shell { input: slot, thickness: lit(1.0) });
        assert!(graph.incomplete().contains(&shell), "the block is incomplete, which is a state");
        assert_eq!(
            App::refusal_of(&doc, &graph, &explain),
            None,
            "and committable, or the verb could not have made it"
        );

        // `target=none` is a choice and not a failure to find anything, so it is not guessed past
        // where there *is* something to guess -- and nothing typed still takes what it always took.
        let ball = ball_in(&mut graph, 0.0);
        let bodies = vec![ball];
        assert!(
            App::target_operand(Some(Operand::empty()), Some(ball), &bodies).is_empty(),
            "`target=none` empties the slot although a body was selected"
        );
        assert_eq!(
            App::target_operand(None, Some(ball), &bodies).source(),
            Some(ball),
            "and with no key the selection is the target, as it always was"
        );
        assert_eq!(
            App::target_operand(None, None, &bodies).source(),
            Some(ball),
            "with nothing selected either, the first body"
        );
    }
}
