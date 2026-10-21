// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Curves: profiles, relations and joints, the gestures that edit them, and placing a new one.

use crate::*;

impl App {
    /// Brings the profile rows up to date with the curves, and says what changed.
    ///
    /// A no-op while no curve has changed since the last call, which is the common case and
    /// the whole reason for the counter: this is called before every draw of the block list.
    ///
    /// **Closing a loop is reported once, and so is opening one.** The comparison is between
    /// the previous partition and this one — which rows appeared and which vanished — so a
    /// drag that carries an end away says "opened" on its first frame and nothing on the
    /// hundreds after it, rather than re-measuring the gap into the console sixty times a
    /// second. A vanished row is looked up again from its key to say *where* the gap is, in
    /// `LoopError::NotClosed`'s own words, since that is the sentence that tells someone what to
    /// do about it. When the key still closes some other loop the row it now heads is reported
    /// as the new one, and nothing is said twice.
    pub(crate) fn refresh_profiles(&mut self) {
        self.refresh_profiles_with(true);
    }

    /// The same, saying whether a loop this refresh finds newly closed should be held with
    /// joints. A pasted curve re-forming a loop is the one case it should not
    /// (`profile-group-spec.md` §4.4): nothing rejoins behind your back.
    ///
    /// **Joints wait for the hand to come off.** A loop that closes mid-drag under Ctrl would
    /// otherwise be joined on that frame and then dragged off its own joint, which is exactly
    /// the snap-back §4.3 forbids. So a closure found while a drag or a placement is in
    /// flight is reported at once and queued in `pending_joins`, and held on the first call
    /// after the gesture ends — if it is still a loop by then.
    pub(crate) fn refresh_profiles_with(&mut self, join: bool) {
        if self.profiles.at != Some(self.curve_revision) {
            let loops = fieldform_field::profile_partition(&self.doc);
            // First sight of this document: found, not announced, not joined.
            let previous = self.profiles.at.map(|_| std::mem::take(&mut self.profiles.loops));
            let mut said = Vec::new();
            if let Some(before) = &previous {
                for i in rows_not_in(before, &loops) {
                    let old = &before[i];
                    // A row whose every curve is gone was deleted as a row, and the deletion
                    // said so.
                    if old.iter().all(|c| !self.doc.curves.contains_key(c)) {
                        continue;
                    }
                    match fieldform_field::loop_from(&self.doc, old[0], None) {
                        Err(fieldform_field::LoopError::MissingCurve(_)) => said.push(format!(
                            "profile: Profile {} opened — one of its curves was deleted",
                            i + 1
                        )),
                        Err(e) => {
                            said.push(format!("profile: Profile {} opened — {e}", i + 1))
                        }
                        // Still a loop, just a different one; the new row says so below.
                        Ok(_) => {}
                    }
                }
            }
            let closed = newly_closed_rows(previous.as_deref(), &loops);
            self.profiles = ProfileRows { at: Some(self.curve_revision), loops };
            let can_join = join && self.point_drag.is_none() && self.pending_curve.is_none();
            for i in closed {
                let key = self.profiles.loops[i][0];
                let n = self.profiles.loops[i].len();
                let mut line =
                    format!("profile: {n} curves closed a loop ({})", self.profile_label(key));
                if can_join {
                    let made = self.join_profile(key);
                    if made > 0 {
                        line.push_str(&format!(", {made} joints"));
                    }
                } else if join {
                    self.pending_joins.push(key);
                }
                said.push(line);
            }
            // Holding a loop rebuilt the curve display, which moved the counter; the loops are
            // the same, since a joint moves nothing.
            self.profiles.at = Some(self.curve_revision);
            for line in said {
                self.say(line);
            }
        }
        if self.point_drag.is_some() || self.pending_curve.is_some() || self.pending_joins.is_empty() {
            return;
        }
        let keys = std::mem::take(&mut self.pending_joins);
        for key in keys {
            // Opened again before the hand came off: nothing to hold.
            if self.profiles.members(key).is_none() {
                continue;
            }
            let label = self.profile_label(key);
            let made = self.join_profile(key);
            if made > 0 {
                self.say(format!("profile: {label} held with {made} joints"));
            }
        }
        self.profiles.at = Some(self.curve_revision);
    }

    /// Holds a loop together with joints, as one entry. How many meetings gained one.
    fn join_profile(&mut self, key: CurveId) -> usize {
        let Some(members) = self.profiles.members(key).map(|m| m.to_vec()) else { return 0 };
        let (commands, joints, loose) =
            join_loop_commands(&self.doc, self.active_part, &members);
        // **Said, because a corner left loose is the one thing a silent join cannot report.** A
        // relation owns the points it derives (§6), so the join leaves those corners alone; without
        // this the loop simply stayed open and the next refresh said "opened" with no hint why.
        if !loose.is_empty() {
            self.warn(format!(
                "{} left loose: {} — break that to join the corner",
                if loose.len() == 1 { "one corner" } else { "corners" },
                loose.join(", "),
            ));
        }
        if commands.is_empty() {
            return 0;
        }
        // A graph gesture in flight must not fold this into itself, or undoing a slider drag
        // would take the joints with it.
        self.gesture = None;
        let batch = Box::new(fieldform_ops::Batch::new("profile.join", commands));
        if let Err(e) = self.stack.execute(&mut self.doc, batch) {
            self.warn(e.to_string());
            return 0;
        }
        self.after_document_change();
        joints
    }

    /// Takes a loop's joints out, as one entry, moving nothing. How many points were removed.
    fn ungroup_profile(&mut self, key: CurveId) -> usize {
        let Some(members) = self.profiles.members(key).map(|m| m.to_vec()) else { return 0 };
        let (commands, removed) = ungroup_commands(&self.doc, self.active_part, &members);
        if commands.is_empty() {
            return 0;
        }
        self.gesture = None;
        let batch = Box::new(fieldform_ops::Batch::new("profile.ungroup", commands));
        if let Err(e) = self.stack.execute(&mut self.doc, batch) {
            self.warn(e.to_string());
            return 0;
        }
        self.after_document_change();
        removed
    }

    /// `join`, and the row's Join: holds this row's loop, and says so.
    pub(crate) fn join_row(&mut self, key: CurveId) {
        let label = self.profile_label(key);
        match self.join_profile(key) {
            0 => self.say(format!("{label} is already held")),
            made => self.say(format!("profile: {label} held with {made} joints")),
        }
    }

    /// `ungroup`, and the row's Ungroup: takes this row's joints out, and says so.
    pub(crate) fn ungroup_row(&mut self, key: CurveId) {
        let label = self.profile_label(key);
        match self.ungroup_profile(key) {
            0 => self.say(format!("{label} has no joints to take out")),
            n => self.say(format!(
                "profile: {label} ungrouped — {n} joints removed, nothing moved"
            )),
        }
    }

    /// The row the selection points at: the selected profile row, or the row of the selected
    /// curve. Refreshed first, for the same reason `curve_in_hand` is.
    fn selected_profile_key(&mut self) -> Option<CurveId> {
        let curve = self.curve_in_hand()?;
        self.profiles.key_of(curve)
    }

    pub(crate) fn join_selected(&mut self) {
        match self.selected_profile_key() {
            Some(key) => self.join_row(key),
            None => self.warn(
                "Select a curve of a closed loop, or its Profile row, to hold it together",
            ),
        }
    }

    pub(crate) fn ungroup_selected(&mut self) {
        match self.selected_profile_key() {
            Some(key) => self.ungroup_row(key),
            None => self.warn("Select a curve of a closed loop, or its Profile row, to ungroup"),
        }
    }

    /// What a profile row is called: the name given to it, or "Profile N" by its place in
    /// the list.
    pub(crate) fn profile_label(&self, key: CurveId) -> String {
        self.doc
            .profile_names
            .get(&key)
            .cloned()
            .unwrap_or_else(|| format!("Profile {}", self.profiles.number(key).unwrap_or(0)))
    }

    /// Names a profile row, or clears its name with an empty one so it reads "Profile N"
    /// again. Like a body's name, one undo step through [`fieldform_ops::SetName`], which says
    /// why a name is on the history although it is not an edit to the geometry.
    pub(crate) fn rename_profile(&mut self, key: CurveId, name: &str) {
        self.set_name(fieldform_ops::Named::Profile(key), name);
    }

    /// Drops the names of rows that no longer exist, before a save.
    ///
    /// A name is keyed by a row's first curve, and when that curve leaves its loop the name
    /// names nothing. Removed on save rather than the moment the loop opens, so a loop that
    /// is opened and closed again in one sitting keeps what it was called
    /// (`profile-group-spec.md` §3.3).
    pub(crate) fn prune_profile_names(&mut self) {
        self.refresh_profiles();
        let keys: std::collections::HashSet<CurveId> =
            self.profiles.loops.iter().map(|l| l[0]).collect();
        self.doc.profile_names.retain(|key, _| keys.contains(key));
    }

    /// The curve a console argument names, by its document name, case aside.
    pub(crate) fn curve_named(&self, name: &str) -> Option<CurveId> {
        let wanted = name.trim();
        let mut ids: Vec<&fieldform_scene::Curve> = self.doc.curves.values().collect();
        ids.sort_by_key(|c| (self.doc.curves_created.get(&c.id).copied().unwrap_or(0), c.id.raw()));
        ids.iter().find(|c| c.name.eq_ignore_ascii_case(wanted)).map(|c| c.id)
    }

    /// A curve argument with an optional `.start` / `.end` suffix, the way `align` takes one.
    fn curve_end_argument(
        &self,
        args: &Args,
        key: &str,
        end_key: &str,
        default: fieldform_scene::End,
    ) -> Result<fieldform_scene::CurveEnd, String> {
        let text = args
            .get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("`{key}=<curve>` is needed"))?
            .to_string();
        let (name, suffix) = match text.rsplit_once('.') {
            Some((n, "start")) => (n.to_string(), Some(fieldform_scene::End::Start)),
            Some((n, "end")) => (n.to_string(), Some(fieldform_scene::End::End)),
            _ => (text.clone(), None),
        };
        let curve = self
            .curve_named(&name)
            .ok_or_else(|| format!("no curve is called `{name}`"))?;
        let end = match args.get(end_key).and_then(|v| v.as_str()) {
            Some(s) if s.eq_ignore_ascii_case("start") => fieldform_scene::End::Start,
            Some(s) if s.eq_ignore_ascii_case("end") => fieldform_scene::End::End,
            Some(other) => return Err(format!("`{end_key}={other}` — say start or end")),
            None => suffix.unwrap_or(default),
        };
        Ok(fieldform_scene::CurveEnd { curve, end })
    }

    /// Makes a relation: refused with the reason, or one entry that both records it and
    /// derives what it claims.
    fn make_relation(&mut self, relation: fieldform_scene::Relation) {
        self.set_relation(None, relation);
    }

    /// Makes a relation in place of `replace`, as one entry: the old record out, the new one
    /// in, and the points the new one claims moved to where it puts them. What the panel does
    /// when a corner's G1 becomes a G2, or a line is pointed at a different master; with no
    /// `replace` it is `make_relation`.
    fn set_relation(
        &mut self,
        replace: Option<fieldform_scene::RelationId>,
        relation: fieldform_scene::Relation,
    ) {
        if let Err(why) =
            fieldform_scene::relation::refuse_reason_replacing(&self.doc, &relation, replace)
        {
            self.warn(why.to_string());
            return;
        }
        let id = fieldform_scene::RelationId::new();
        let record = fieldform_scene::RelationRecord { relation: relation.clone(), broken: None };
        // Derived on a copy, so the command records the move from where the points were.
        let mut after = self.doc.clone();
        let mut commands: Vec<Box<dyn fieldform_ops::Command>> = Vec::new();
        if let Some(old) = replace
            && let Some(was) = self.doc.relations.get(&old).cloned()
        {
            after.relations.remove(&old);
            commands.push(Box::new(SetRelation::remove(old, was)));
        }
        after.relations.insert(id, record.clone());
        let moved = fieldform_scene::relation::rederive(&mut after);
        commands.push(Box::new(SetRelation::insert(id, record)));
        commands.extend(rederived_commands(&self.doc, &moved));
        self.gesture = None;
        let batch = Box::new(fieldform_ops::Batch::new("relation.make", commands));
        if let Err(e) = self.stack.execute(&mut self.doc, batch) {
            self.warn(e.to_string());
            return;
        }
        self.after_document_change();
        let what = relation.describe(&self.doc);
        self.say(match moved.len() {
            0 => format!("{what} — it already held"),
            n => format!("{what} — {n} point{} moved to make it so", if n == 1 { "" } else { "s" }),
        });
    }

    /// `align master=<curve> end=start|end slave=<curve> at=start|end g=1|2`, or the curve
    /// names carrying `.start` / `.end`. A G1 or G2 joint, the slave following the master.
    pub(crate) fn align_command(&mut self, args: &Args) {
        let master = self.curve_end_argument(args, "master", "end", fieldform_scene::End::End);
        let slave = self.curve_end_argument(args, "slave", "at", fieldform_scene::End::Start);
        let (master, slave) = match (master, slave) {
            (Ok(m), Ok(s)) => (m, s),
            (Err(why), _) | (_, Err(why)) => {
                self.warn(format!(
                    "{why} — `align master=Line end=end slave=Curve at=start g=1`"
                ));
                return;
            }
        };
        let continuity = match args.get("g").and_then(|v| v.as_f64()).map(|g| g as i64) {
            None | Some(1) => fieldform_scene::JointContinuity::G1,
            Some(2) => fieldform_scene::JointContinuity::G2,
            Some(other) => {
                self.warn(format!("g={other} — G1 or G2 are what a joint can be"));
                return;
            }
        };
        self.make_relation(fieldform_scene::Relation::Joint { master, slave, continuity });
    }

    /// `set <curve>.p<i>=x,y,z` -- spec §6 and §9.3, the word an outside driver or an AI edit
    /// needs to move a control point by name. Several in one line are one entry. A point that
    /// sits on a block moves the block, exactly as a drag on it does, so its neighbours follow
    /// and its relations re-derive; a point a relation owns is refused, naming the relation
    /// (§5.2 rule 1). A curve name with a space is quoted with its point: `set "Curve 3.p1"=0,0,0`.
    pub(crate) fn set_command(&mut self, args: &Args) {
        // **A block's keys first** (`interaction-spec.md` §6.2): `set Sphere_0.radius=12` sets the
        // field its create verb reads by that word. A curve's point -- `Line.p1`, `.start`, `.end`
        // -- stays the curve's, as it always was, even where a block shares the name.
        let mut blocks: Vec<(NodeId, String, Args)> = Vec::new();
        let mut points = Args::new();
        for (key, value) in args.iter() {
            let block = key.rsplit_once('.').and_then(|(name, field)| {
                let a_point = field == "start" || field == "end" || field.strip_prefix('p').is_some_and(|i| i.parse::<usize>().is_ok());
                if a_point && self.curve_named(name).is_some() {
                    return None;
                }
                self.node_named(name).map(|id| (id, name.to_string(), field.to_string()))
            });
            match block {
                Some((id, name, field)) => match blocks.iter_mut().find(|(b, _, _)| *b == id) {
                    Some((_, _, a)) => *a = a.clone().with(&field, value.clone()),
                    None => blocks.push((id, name, Args::new().with(&field, value.clone()))),
                },
                None => points = points.with(key, value.clone()),
            }
        }
        for (id, name, keys) in &blocks {
            if let Err(why) = self.set_block_keys(*id, keys) {
                self.warn(format!("`set {name}`: {why}"));
                return;
            }
        }
        if !blocks.is_empty() {
            let said: Vec<String> = blocks.iter().map(|(_, name, _)| name.clone()).collect();
            self.say(format!("Set {}", said.join(", ")));
            if points.iter().next().is_none() {
                return;
            }
        }
        let args = &points;
        let pairs: Vec<(String, String)> = args
            .iter()
            .map(|(k, v)| {
                let text = match v {
                    Value::Str(s) => s.clone(),
                    Value::Int(i) => i.to_string(),
                    Value::Float(f) => f.to_string(),
                    _ => String::new(),
                };
                (k.to_string(), text)
            })
            .collect();
        let wanted = match set_targets(&self.doc, &pairs, |n| self.curve_named(n)) {
            Ok(w) if !w.is_empty() => w,
            Ok(_) => {
                self.warn("Name a point and where it goes: `set Line.p1=10,20,0`, or `Line.end=...`");
                return;
            }
            Err(why) => {
                self.warn(why);
                return;
            }
        };
        for (curve, index, _) in &wanted {
            if let Some(owner) = fieldform_scene::relation::owners_of(&self.doc, *curve, *index, None)
                .first()
                .and_then(|id| self.doc.relations.get(id))
            {
                let name = self.doc.curve(*curve).map(|c| c.name.clone()).unwrap_or_default();
                self.warn(format!(
                    "point {index} of {name} is owned by {} — break it to set the point",
                    owner.relation.describe(&self.doc)
                ));
                return;
            }
        }
        // One gesture per curve, committed together: the path a drag takes, so held ends move
        // their point and the whole thing is one entry.
        let mut by_curve: Vec<(CurveId, Vec<(usize, Point3<f64>)>)> = Vec::new();
        for (curve, index, at) in &wanted {
            match by_curve.iter_mut().find(|(c, _)| c == curve) {
                Some((_, to)) => to.push((*index, *at)),
                None => by_curve.push((*curve, vec![(*index, *at)])),
            }
        }
        let mut moves: Vec<CurveMove> = Vec::new();
        for (curve, to) in &by_curve {
            let Some(points) = self.doc.curve(*curve).map(|c| c.control_points().to_vec()) else {
                continue;
            };
            let indices: Vec<usize> = to.iter().map(|(i, _)| *i).collect();
            let before: Vec<Point3<f64>> = indices.iter().map(|&i| points[i]).collect();
            let Some(gesture) = self.begin_curve_gesture(*curve, &indices) else { return };
            moves.push(CurveMove { curve: *curve, indices, before, gesture });
        }
        for (m, (_, to)) in moves.clone().iter().zip(&by_curve) {
            self.move_curve_points_live(m.curve, to, &m.gesture);
        }
        let graph_before = moves.iter().find_map(|m| m.gesture.graph_before.clone());
        let shapes_field = moves.iter().any(|m| m.gesture.shapes_field);
        if !self.commit_curves_gesture(moves, graph_before) {
            return;
        }
        self.after_curve_gesture(shapes_field);
        self.say(format!(
            "Set {} point{}",
            wanted.len(),
            if wanted.len() == 1 { "" } else { "s" }
        ));
    }

    /// A mirrored copy of `master` across the line `axis`, following both, as one entry: the
    /// new curve already where the relation puts it, and the relation. What `relate -mirror`
    /// does without a slave, and the panel's Mirror across.
    fn mirror_curve(&mut self, master: CurveId, axis: CurveId) {
        let Some(original) = self.doc.curve(master).cloned() else { return };
        let id = CurveId::new();
        let mut copy = original.clone();
        copy.id = id;
        copy.name = self.unique_curve_name(&format!("{} mirrored", original.name));
        // The copy is a curve of its own: the original's ends keep their points.
        for i in 0..copy.control_points().len() {
            copy.place_on(i, None);
        }
        let relation = fieldform_scene::Relation::Mirror { master, slave: id, axis };
        let mut with_copy = self.doc.clone();
        with_copy.curves.insert(id, copy.clone());
        if let Err(why) = fieldform_scene::relation::refuse_reason(&with_copy, &relation) {
            self.warn(why.to_string());
            return;
        }
        if let Some(targets) = fieldform_scene::relation::derive(&with_copy, &relation) {
            for (i, p) in targets {
                copy.set_control_point(i, p);
            }
        }
        let rid = fieldform_scene::RelationId::new();
        let record = fieldform_scene::RelationRecord { relation: relation.clone(), broken: None };
        let commands: Vec<Box<dyn fieldform_ops::Command>> =
            vec![Box::new(SetCurve::insert(copy)), Box::new(SetRelation::insert(rid, record))];
        self.gesture = None;
        let batch = Box::new(fieldform_ops::Batch::new("relation.mirror", commands));
        if let Err(e) = self.stack.execute(&mut self.doc, batch) {
            self.warn(e.to_string());
            return;
        }
        self.select_curve(id);
        self.rebuild_curve_display();
        self.after_document_change();
        let what = relation.describe(&self.doc);
        self.say(format!("{what} — a new curve"));
    }

    /// `relate -parallel|-perpendicular|-equal master=<line> slave=<line>`, or `kind=`.
    pub(crate) fn relate_command(&mut self, args: &Args) {
        let flag = |name: &str| args.get(name).is_some_and(|v| !matches!(v, Value::Bool(false)));
        let kind = args
            .get("kind")
            .and_then(|v| v.as_str())
            .map(|s| s.to_ascii_lowercase())
            .or_else(|| {
                ["parallel", "perpendicular", "equal", "concentric", "mirror"]
                    .into_iter()
                    .find(|k| flag(k))
                    .map(String::from)
            });
        let names = |key: &str| -> Result<CurveId, String> {
            let name = args
                .get(key)
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("`{key}=<curve>` is needed"))?;
            self.curve_named(name).ok_or_else(|| format!("no curve is called `{name}`"))
        };
        // A mirror names its axis, and makes its own slave unless it is told one.
        if kind.as_deref() == Some("mirror") {
            let (master, axis) = match (names("master"), names("axis")) {
                (Ok(m), Ok(a)) => (m, a),
                (Err(why), _) | (_, Err(why)) => {
                    self.warn(format!(
                        "{why} — `relate -mirror master=Curve axis=Line`, with `slave=` \
                         to mirror onto a curve that is already there"
                    ));
                    return;
                }
            };
            match args.get("slave") {
                None => self.mirror_curve(master, axis),
                Some(_) => match names("slave") {
                    Ok(slave) => self.make_relation(fieldform_scene::Relation::Mirror {
                        master,
                        slave,
                        axis,
                    }),
                    Err(why) => self.warn(why),
                },
            }
            return;
        }
        let (master, slave) = match (names("master"), names("slave")) {
            (Ok(m), Ok(s)) => (m, s),
            (Err(why), _) | (_, Err(why)) => {
                self.warn(format!("{why} — `relate -parallel master=Line slave=\"Line 2\"`"));
                return;
            }
        };
        let relation = match kind.as_deref() {
            Some("parallel") => fieldform_scene::Relation::Parallel { master, slave },
            Some("perpendicular") => fieldform_scene::Relation::Perpendicular { master, slave },
            Some("equal") | Some("equal length") => {
                fieldform_scene::Relation::EqualLength { master, slave }
            }
            Some("concentric") => fieldform_scene::Relation::Concentric { master, slave },
            _ => {
                self.warn(
                    "Say which: `relate -parallel`, `-perpendicular`, `-equal`, `-concentric` \
                     or `-mirror`",
                );
                return;
            }
        };
        self.make_relation(relation);
    }

    /// The relations driving the curve a command names, or the selected curve.
    fn relations_driving(&mut self, args: &Args) -> Option<(CurveId, Vec<fieldform_scene::RelationId>)> {
        let curve = match args.get("slave").or_else(|| args.get("curve")).and_then(|v| v.as_str()) {
            Some(name) => match self.curve_named(name) {
                Some(id) => id,
                None => {
                    self.warn(format!("no curve is called `{name}`"));
                    return None;
                }
            },
            None => match self.curve_in_hand() {
                Some(id) => id,
                None => {
                    self.warn("Select a curve, or say `slave=<curve>`");
                    return None;
                }
            },
        };
        let kind = args.get("kind").and_then(|v| v.as_str()).map(|s| s.to_ascii_lowercase());
        let ids: Vec<fieldform_scene::RelationId> = self
            .doc
            .relations
            .iter()
            .filter(|(_, r)| r.relation.slave() == curve)
            .filter(|(_, r)| kind.as_deref().is_none_or(|k| r.relation.kind().eq_ignore_ascii_case(k)))
            .map(|(id, _)| *id)
            .collect();
        Some((curve, ids))
    }

    /// `break slave=<curve>`, or `break` on the selected curve: deletes what drives it, and
    /// moves nothing (`profile-group-spec.md` §5.4).
    pub(crate) fn break_command(&mut self, args: &Args) {
        let Some((curve, ids)) = self.relations_driving(args) else { return };
        let name = self.doc.curve(curve).map(|c| c.name.clone()).unwrap_or_default();
        if ids.is_empty() {
            self.say(format!("Nothing drives {name}"));
            return;
        }
        self.break_relations(&ids);
    }

    /// Deletes relations as one entry, moving nothing: the word's and the panel's Break.
    pub(crate) fn break_relations(&mut self, ids: &[fieldform_scene::RelationId]) {
        let mut commands: Vec<Box<dyn fieldform_ops::Command>> = Vec::new();
        let mut said = Vec::new();
        for id in ids {
            if let Some(record) = self.doc.relations.get(id).cloned() {
                said.push(record.relation.describe(&self.doc));
                commands.push(Box::new(SetRelation::remove(*id, record)));
            }
        }
        if commands.is_empty() {
            return;
        }
        self.gesture = None;
        let batch = Box::new(fieldform_ops::Batch::new("relation.break", commands));
        if let Err(e) = self.stack.execute(&mut self.doc, batch) {
            self.warn(e.to_string());
            return;
        }
        self.mark_changed();
        self.say(format!("Broke {} — nothing moved", said.join("; ")));
    }

    /// `realign slave=<curve>`, or `realign` on the selected curve: re-derives a relation found
    /// broken on open, from the master's current state, as one undoable edit.
    pub(crate) fn realign_command(&mut self, args: &Args) {
        let Some((curve, ids)) = self.relations_driving(args) else { return };
        let name = self.doc.curve(curve).map(|c| c.name.clone()).unwrap_or_default();
        let broken: Vec<fieldform_scene::RelationId> = ids
            .into_iter()
            .filter(|id| self.doc.relations.get(id).is_some_and(|r| r.broken.is_some()))
            .collect();
        if broken.is_empty() {
            self.say(format!("Nothing driving {name} is broken"));
            return;
        }
        self.realign_relations(&broken);
    }

    /// Re-derives relations found broken on open, from their masters' current state, as one
    /// undoable edit: the word's and the panel's Realign.
    pub(crate) fn realign_relations(&mut self, broken: &[fieldform_scene::RelationId]) {
        let Some((commands, moved, said)) = realign_commands(&self.doc, broken) else { return };
        self.gesture = None;
        let batch = Box::new(fieldform_ops::Batch::new("relation.realign", commands));
        if let Err(e) = self.stack.execute(&mut self.doc, batch) {
            self.warn(e.to_string());
            return;
        }
        self.after_document_change();
        self.say(format!(
            "Re-aligned {} — {} point{} moved",
            said.join("; "),
            moved.len(),
            if moved.len() == 1 { "" } else { "s" }
        ));
    }

    /// The point blocks a loop's ends sit on, in the order the curves are walked.
    pub(crate) fn joint_points_of(&self, members: &[CurveId]) -> Vec<NodeId> {
        joint_points_in(&self.doc, self.active_part, members)
    }

    /// A corner's choices as a list -- none, G1 or G2, either curve leading -- with a choice
    /// that would be refused greyed and the reason on hover, which is `refuse_reason` said
    /// where the choice is made rather than after it. What was picked, if anything: the
    /// corner's dropdown under its profile and the viewport's right-click both draw this.
    fn joint_choice_list(
        &self,
        ui: &mut egui::Ui,
        a: fieldform_scene::CurveEnd,
        b: fieldform_scene::CurveEnd,
        existing: Option<&(fieldform_scene::RelationId, fieldform_scene::Relation)>,
    ) -> Option<Option<fieldform_scene::Relation>> {
        let mut picked = None;
        for (label, relation) in joint_choices(&self.doc, a, b) {
            let is_current = match (&relation, existing) {
                (None, None) => true,
                (Some(r), Some((_, e))) => r == e,
                _ => false,
            };
            let refused = relation.as_ref().and_then(|r| {
                fieldform_scene::relation::refuse_reason_replacing(
                    &self.doc,
                    r,
                    existing.map(|(id, _)| *id),
                )
                .err()
            });
            let response = ui.add_enabled(
                refused.is_none(),
                egui::Button::selectable(is_current, label.as_str()),
            );
            let response = match &refused {
                Some(why) => response.on_disabled_hover_text(why.to_string()),
                None => response,
            };
            if response.clicked() && !is_current {
                picked = Some(relation);
                ui.close();
            }
        }
        picked
    }

    /// Applies a corner's pick: a relation in place of the one there, or none.
    fn apply_joint_choice(
        &mut self,
        existing: Option<fieldform_scene::RelationId>,
        choice: Option<fieldform_scene::Relation>,
    ) {
        match (choice, existing) {
            (Some(relation), replace) => self.set_relation(replace, relation),
            (None, Some(id)) => self.break_relations(&[id]),
            (None, None) => {}
        }
    }

    /// A corner's continuity, chosen where the corner is listed under its profile: what
    /// `align` and `break` do from the console, as one dropdown. Drawn only for a point two
    /// ends sit on, which is what a joint is.
    pub(crate) fn joint_controls(&mut self, ui: &mut egui::Ui, point: NodeId) {
        let ends = ends_on_point(&self.doc, point);
        let &[a, b] = ends.as_slice() else { return };
        let existing = joint_relation_between(&self.doc, a, b);
        let current = existing
            .as_ref()
            .map(|(_, r)| r.describe(&self.doc))
            .unwrap_or_else(|| "none".to_string());
        let mut picked = None;
        ui.horizontal(|ui| {
            row_label(ui, "Continuity:");
            egui::ComboBox::from_id_salt((point.raw(), "joint_continuity"))
                .selected_text(current)
                .show_ui(ui, |ui| {
                    picked = self.joint_choice_list(ui, a, b, existing.as_ref());
                });
        });
        if let Some(choice) = picked {
            self.apply_joint_choice(existing.map(|(id, _)| id), choice);
        }
    }

    /// The same choices on a right-click of a corner in the viewport.
    pub(crate) fn joint_menu(&mut self, ui: &mut egui::Ui, point: NodeId) {
        let ends = ends_on_point(&self.doc, point);
        let &[a, b] = ends.as_slice() else { return };
        let existing = joint_relation_between(&self.doc, a, b);
        ui.label(egui::RichText::new("Corner continuity").weak());
        if let Some(choice) = self.joint_choice_list(ui, a, b, existing.as_ref()) {
            self.apply_joint_choice(existing.map(|(id, _)| id), choice);
        }
    }

    /// The joint under a viewport position, within a few pixels: the point two ends sit on
    /// whose dot is nearest the pointer.
    pub(crate) fn joint_at(&self, aspect: f32, rect: egui::Rect, pos: egui::Pos2) -> Option<NodeId> {
        const REACH: f32 = 10.0;
        self.point_positions()
            .into_iter()
            .filter_map(|(id, at)| {
                self.camera
                    .project_to_screen(aspect, rect, at.cast::<f32>())
                    .map(|screen| (id, screen.distance(pos)))
            })
            .filter(|(id, d)| *d <= REACH && ends_on_point(&self.doc, *id).len() == 2)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(id, _)| id)
    }

    /// What this curve follows, chosen on its own panel: parallel, perpendicular and equal
    /// length for a line, concentric for a curve with a centre -- each a dropdown of the
    /// curves it could follow, with "nothing" to break the one there and a refused choice
    /// greyed with the reason -- and, for any curve, a mirrored copy across a line. What
    /// `relate` and `break` do from the console. One way: this curve follows the one chosen.
    pub(crate) fn relation_controls(&mut self, ui: &mut egui::Ui, id: CurveId) {
        let Some(this) = self.doc.curve(id) else { return };
        let is_line =
            |c: &fieldform_scene::Curve| c.degree() == 1 && c.control_points().len() == 2;
        let this_is_line = is_line(this);
        let has_centre = fieldform_scene::relation::centre_of(this).is_some();
        let by_creation = |mut v: Vec<(CurveId, String)>| {
            v.sort_by_key(|(other, _)| {
                (self.doc.curves_created.get(other).copied().unwrap_or(0), other.raw())
            });
            v
        };
        let others = |keep: &dyn Fn(&fieldform_scene::Curve) -> bool| {
            by_creation(
                self.doc
                    .curves
                    .values()
                    .filter(|c| c.id != id && keep(c))
                    .map(|c| (c.id, c.name.clone()))
                    .collect(),
            )
        };
        let lines = others(&|c| is_line(c));
        let round = others(&|c| fieldform_scene::relation::centre_of(c).is_some());
        // What this curve mirrors, if it is a mirror: shown with its Break, since a mirror is
        // made by the curve it copies rather than chosen here.
        let mirrors: Option<(fieldform_scene::RelationId, String)> = self
            .doc
            .relations
            .iter()
            .find(|(_, r)| {
                r.relation.slave() == id
                    && matches!(r.relation, fieldform_scene::Relation::Mirror { .. })
            })
            .map(|(rid, r)| (*rid, r.relation.describe(&self.doc)));

        type Make = fn(CurveId, CurveId) -> fieldform_scene::Relation;
        let mut kinds: Vec<(&str, &str, Make, &Vec<(CurveId, String)>)> = Vec::new();
        if this_is_line && !lines.is_empty() {
            kinds.push(("Parallel to", "parallel", |master, slave| {
                fieldform_scene::Relation::Parallel { master, slave }
            }, &lines));
            kinds.push(("Perpendicular to", "perpendicular", |master, slave| {
                fieldform_scene::Relation::Perpendicular { master, slave }
            }, &lines));
            kinds.push(("Equal length to", "equal length", |master, slave| {
                fieldform_scene::Relation::EqualLength { master, slave }
            }, &lines));
        }
        if has_centre && !round.is_empty() {
            kinds.push(("Concentric with", "concentric", |master, slave| {
                fieldform_scene::Relation::Concentric { master, slave }
            }, &round));
        }
        if kinds.is_empty() && lines.is_empty() && mirrors.is_none() {
            return;
        }
        ui.separator();
        ui.label(egui::RichText::new("Follows").weak()).on_hover_text(
            "One way: this curve follows the one chosen, and moves when it does. A choice that \
             would be refused is greyed, with the reason on hover.",
        );
        // Picks are applied after the grid, since one edits the document the grid is reading.
        let mut picks: Vec<(Option<CurveId>, Make, Option<fieldform_scene::RelationId>)> =
            Vec::new();
        let mut mirror_across: Option<CurveId> = None;
        let mut break_mirror: Option<fieldform_scene::RelationId> = None;
        egui::Grid::new((id.raw(), "curve_relations")).num_columns(2).show(ui, |ui| {
            for (label, kind, make, candidates) in &kinds {
                let (make, kind) = (*make, *kind);
                let existing = self
                    .doc
                    .relations
                    .iter()
                    .find(|(_, r)| r.relation.slave() == id && r.relation.kind() == kind)
                    .map(|(rid, r)| (*rid, r.relation.master()));
                let current = existing
                    .and_then(|(_, master)| self.doc.curve(master))
                    .map(|c| c.name.clone())
                    .unwrap_or_else(|| "nothing".to_string());
                let mut picked: Option<Option<CurveId>> = None;
                // Extending, because a label in a grid cell truncates to the cell's width
                // from the frame before and so never asks for the room it needs.
                ui.add(egui::Label::new(*label).extend());
                egui::ComboBox::from_id_salt((id.raw(), "curve_relation", kind))
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        if ui
                            .add(egui::Button::selectable(existing.is_none(), "nothing"))
                            .clicked()
                            && existing.is_some()
                        {
                            picked = Some(None);
                            ui.close();
                        }
                        for (other, name) in candidates.iter() {
                            let is_current = existing.is_some_and(|(_, m)| m == *other);
                            let refused = fieldform_scene::relation::refuse_reason_replacing(
                                &self.doc,
                                &make(*other, id),
                                existing.map(|(rid, _)| rid),
                            )
                            .err();
                            let response = ui.add_enabled(
                                refused.is_none(),
                                egui::Button::selectable(is_current, name.as_str()),
                            );
                            let response = match &refused {
                                Some(why) => response.on_disabled_hover_text(why.to_string()),
                                None => response,
                            };
                            if response.clicked() && !is_current {
                                picked = Some(Some(*other));
                                ui.close();
                            }
                        }
                    });
                ui.end_row();
                if let Some(pick) = picked {
                    picks.push((pick, make, existing.map(|(rid, _)| rid)));
                }
            }
            if let Some((rid, what)) = &mirrors {
                ui.add(egui::Label::new("Mirror").extend());
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(what).small());
                    if ui
                        .small_button("Break")
                        .on_hover_text("Deletes the relation. Nothing moves.")
                        .clicked()
                    {
                        break_mirror = Some(*rid);
                    }
                });
                ui.end_row();
            }
            if !lines.is_empty() {
                ui.add(egui::Label::new("Mirror across").extend());
                ui.menu_button("a line\u{2026}", |ui| {
                    for (axis, name) in &lines {
                        if ui.button(name).clicked() {
                            mirror_across = Some(*axis);
                            ui.close();
                        }
                    }
                })
                .response
                .on_hover_text(
                    "Makes a mirrored copy of this curve across the line chosen. The copy \
                     follows this curve and the line, point for point.",
                );
                ui.end_row();
            }
        });
        for (pick, make, existing) in picks {
            match (pick, existing) {
                (Some(master), replace) => self.set_relation(replace, make(master, id)),
                (None, Some(rid)) => self.break_relations(&[rid]),
                (None, None) => {}
            }
        }
        if let Some(rid) = break_mirror {
            self.break_relations(&[rid]);
        }
        if let Some(axis) = mirror_across {
            self.mirror_curve(id, axis);
        }
    }

    /// How many of a loop's ends sit on points, out of how many ends there are.
    pub(crate) fn profile_hold(&self, members: &[CurveId]) -> (usize, usize) {
        let mut held = 0;
        let mut total = 0;
        for id in members {
            let Some(c) = self.doc.curve(*id) else { continue };
            let n = c.control_points().len();
            if n < 2 {
                continue;
            }
            total += 2;
            held += [0, n - 1].iter().filter(|i| c.placed_on(**i).is_some()).count();
        }
        (held, total)
    }

    /// What a curve's row becomes once loops are folded: its own row, the profile row it
    /// heads, or nothing, because it is inside a row headed by another curve.
    ///
    /// One rule, used by the block list and by an import's own list of curves, so a loop is
    /// folded the same way wherever its curves happen to be shown. `nesting` is the list's, for
    /// the reader counts the rule reads.
    pub(crate) fn folded_curve_row(
        &self,
        id: CurveId,
        nesting: &fieldform_scene::Nesting,
    ) -> Option<Block> {
        folded_curve_row_in(&self.profiles, &self.doc, &|c| nesting.readers_of_curve(c), id)
    }

    /// The curve a tool that wants one should start from: the selected curve, or the first
    /// curve of the selected profile row.
    ///
    /// Selecting a row selects every curve in it, and Extrude, Revolve and Sweep name *one*
    /// curve and walk outward from it — so the row's first is what they are handed, which is
    /// exactly what they already accept. Refreshed first, because a command typed in the same
    /// frame as the edit that closed the loop should see the loop.
    pub(crate) fn curve_in_hand(&mut self) -> Option<CurveId> {
        if let Some(id) = self.selected_curve.filter(|id| self.doc.curves.contains_key(id)) {
            return Some(id);
        }
        self.refresh_profiles();
        match self.selected_block {
            Some(outputs::BlockRef::Profile(key)) => self.profiles.members(key).map(|m| m[0]),
            _ => None,
        }
    }

    /// The curves the viewport draws as selected: the selected curve, or every curve of the
    /// selected profile row.
    pub(crate) fn highlighted_curves(&self) -> Vec<CurveId> {
        if let Some(id) = self.selected_curve {
            return vec![id];
        }
        match self.selected_block {
            Some(outputs::BlockRef::Profile(key)) => {
                self.profiles.members(key).map(|m| m.to_vec()).unwrap_or_default()
            }
            _ => Vec::new(),
        }
    }

    /// Takes a curve and everything in its loop back out of `ungrouped_curves`, so the row re-forms.
    ///
    /// **Its loop-mates too, because the dismissal is a property of the set.** `profile_partition`
    /// skips a loop when *any* member is dismissed, so clearing one curve alone would leave the row
    /// still missing and the gesture would look broken. The loop is asked for rather than remembered —
    /// it is the same walk the grouping itself uses, so the two cannot disagree about who is in it.
    pub(crate) fn regroup_curve(&mut self, id: CurveId) {
        let members = fieldform_field::loop_from(&self.doc, id, None).unwrap_or_else(|_| vec![id]);
        for member in &members {
            self.doc.ungrouped_curves.remove(member);
        }
        self.mark_changed();
        self.refresh_profiles();
        self.say(format!("{} curves gathered back into one row", members.len()));
    }

    /// Dismisses a profile row, leaving its curves as separate rows and its geometry alone.
    ///
    /// **The escape a derived row needs, because it cannot be deleted.** The grouping is
    /// `profile_partition`'s reading of which chains close, so removing the row without storing
    /// something re-forms it on the next refresh — and Ungroup, which takes the joint points out,
    /// leaves it standing because the loop still closes without them. `Document::ungrouped_curves` is
    /// what makes the dismissal stick, and it is read when rows are gathered and nowhere else.
    ///
    /// **The joints stay, deliberately.** They hold the corners at one place and holding them there
    /// is not the grouping — taking them out would let the corners drift apart on the next drag, which
    /// is a change to the geometry on a gesture that is supposed to make none. Separating a row is
    /// about where its curves are *listed*.
    ///
    /// Not on the undo stack, as a curve's declaration was not before the curve row's V became a
    /// command (`fieldform_ops::SetOwnership`): this writes no geometry, and the inverse is the
    /// same gesture again. Ctrl+Z after it would otherwise undo whatever real edit came before,
    /// which is worse than the gesture not being on the stack at all.
    fn separate_profile(&mut self, members: &[CurveId], label: &str, users: &[String]) {
        self.doc.ungrouped_curves.extend(members.iter().copied());
        self.mark_changed();
        self.refresh_profiles();
        self.selected_curve = None;
        self.say(format!(
            "{label} is {} separate curves now — {} still reads the loop, which has not moved",
            members.len(),
            users.join(", "),
        ));
    }

    /// Deletes every curve of a profile row, as one entry on the history.
    ///
    /// **Not while anything still reads one of them**: an extrude whose profile vanished stops
    /// evaluating, and deleting that as well would take a block nobody selected. A row something
    /// in this part reads is separated instead ([`Self::separate_profile`]), for the reason given
    /// inside; one only another part reads is refused, naming the block and its part.
    pub(crate) fn delete_profile(&mut self, key: CurveId) {
        self.refresh_profiles();
        let Some(members) = self.profiles.members(key).map(|m| m.to_vec()) else { return };
        let label = self.profile_label(key);
        let mut users: Vec<String> = self
            .part_graph()
            .nodes()
            .filter(|(_, n)| n.curves().iter().any(|c| members.contains(c)))
            .map(|(node, _)| self.body_label(node))
            .collect();
        // **Used by something, so the row is separated rather than refused.** It used to say "delete
        // that first" and stop, which left no way forward at all: the curves could not go because the
        // extrude reads them, and the *row* could not go either, because a row is a reading of the
        // geometry and there was nothing to delete. Reported as "we are stuck!! cant delete the
        // profile or the curves inside it!", with the remedy the owner wanted — "the components needs
        // to stay/seperate on profile deletion."
        //
        // So this is what Del on a consumed row means now: the grouping is dismissed, the curves stay
        // as separate rows, and whatever was built on the loop goes on being built, because
        // `profile_loop` does not consult the dismissal. Nothing is destroyed, which is what makes it
        // the safe thing to do to a row somebody else depends on.
        if !users.is_empty() {
            users.sort();
            users.dedup();
            self.separate_profile(&members, &label, &users);
            return;
        }
        // **A block in another part refuses the delete** (`readers_in_other_parts` says why). The
        // readers above are the part on screen's, so with none here the delete below would go
        // ahead and take curves another part is built on.
        let elsewhere = readers_in_other_parts(&self.doc, self.active_part, &members, |g, n| {
            self.body_label_in(g, n)
        });
        if !elsewhere.is_empty() {
            self.say(format!(
                "{label} is used by {} — {}",
                elsewhere.readers.join(", "),
                elsewhere.built_on(),
            ));
            return;
        }
        // The row's joints go with it; `delete_row_commands` says why that is not
        // `delete_curves_commands`.
        let part = self.active_part;
        let commands = delete_row_commands(&self.doc, part, &members);
        // Ended for the reason `join_profile` gives: a graph gesture in flight must not fold this
        // into itself, or undoing a slider drag would take the curves' deletion with it.
        self.gesture = None;
        let batch = Box::new(fieldform_ops::Batch::new("profile.delete", commands));
        if let Err(e) = self.stack.execute(&mut self.doc, batch) {
            self.warn(e.to_string());
            return;
        }
        self.selected_curve = None;
        self.clear_point_selection();
        self.after_document_change();
        self.say(format!("Deleted {label} and its {} curves", members.len()));
    }

    /// Moves every curve control point that sits on a block to where that block now is.
    ///
    /// **One call site's worth of work, in one place.** `Curve::apply_places` is the single
    /// derivation for one curve; this is the single derivation for the document, so "when does a
    /// curve catch up with the points it sits on" has one answer: whenever the part changes,
    /// because `rebuild_render_resources` runs it.
    pub(crate) fn settle_curves_on_points(&mut self) -> bool {
        // Where each point block currently is, read once. Resolved through the same
        // `resolve_place` the field and the shader use, so a curve end lands exactly where the
        // body sitting on that point lands. The rule itself is the free function, so a test
        // can run it on a document without a window.
        let settled = settle_curves_on_points(&mut self.doc, self.active_part);
        // **And the relations, after the places.** The re-derivation is the second half of the
        // single derivation: an end that followed its point may be a master, and what follows
        // it follows now. Run here rather than folded into each command, because this is what
        // runs after *every* change -- an edit, an undo, a redo, an open -- so a slave is
        // always where its master puts it, whichever path moved the master.
        let followed = !fieldform_scene::relation::rederive(&mut self.doc).is_empty();
        settled || followed
    }

    /// Whether moving `curve` changes a field: a block reads it, or a block reads a point one
    /// of its ends sits on.
    pub(crate) fn curve_shapes_a_body(&self, curve: CurveId) -> bool {
        curve_shapes_a_body_in(&self.doc, self.part_graph(), curve)
    }

    /// What a gesture on `indices` of `curve` needs to know, or `None` -- said -- when one of
    /// them sits on a block a parameter drives: the variable is where that number lives, and
    /// overwriting it from a drag would be the silent edit an expression exists to prevent.
    pub(crate) fn begin_curve_gesture(&mut self, curve: CurveId, indices: &[usize]) -> Option<CurveGesture> {
        // An axis on a surface is derived from its body, so a hand on its points would be
        // overruled by the next derivation. Refused here, once, for every path that gets this far.
        if let Some(why) = self.surface_axis_refusal(curve) {
            self.warn(why);
            return None;
        }
        let c = self.doc.curve(curve)?;
        let (bound, _) = drag_targets(c, indices);
        if let Some((_, node)) =
            bound.iter().find(|(_, node)| point_literals(self.part_graph(), *node).is_none())
        {
            let name = self.body_label(*node);
            self.warn(format!(
                "\u{201c}{name}\u{201d} is driven by a variable — change the variable to \
                 move it"
            ));
            return None;
        }
        let graph = self.part_graph();
        let read = |p: &NodeId| graph.is_consumed(*p);
        let bodies_follow = bound.iter().any(|(_, p)| read(p));
        // Another curve on the same block moves too, and it may be what a body is made of.
        let neighbours_shape = bound.iter().any(|(_, p)| {
            self.doc
                .curves
                .values()
                .any(|other| other.sits_on().any(|q| q == *p) && self.curve_shapes_a_body(other.id))
        });
        let shapes_field = bodies_follow || neighbours_shape || self.curve_shapes_a_body(curve);
        Some(CurveGesture {
            graph_before: (!bound.is_empty()).then(|| graph.clone()),
            bound,
            bodies_follow,
            shapes_field,
        })
    }

    /// Moves a curve's picked points live: a free point moves itself, a point on a block moves
    /// the block and every end on that block follows (`profile-group-spec.md` §4.3).
    pub(crate) fn move_curve_points_live(
        &mut self,
        curve: CurveId,
        to: &[(usize, Point3<f64>)],
        gesture: &CurveGesture,
    ) {
        let mut moved_point = false;
        for (i, p) in to {
            if let Some((_, node)) = gesture.bound.iter().find(|(b, _)| b == i) {
                moved_point |= move_point_node(self.part_graph_mut(), *node, *p);
            } else if let Some(c) = self.doc.curve_mut(curve) {
                c.set_control_point(*i, *p);
            }
        }
        if moved_point {
            self.settle_curves_on_points();
            // A body sitting on the block follows too, which is what the block panel's own
            // point drag rebuilds for.
            if gesture.bodies_follow {
                self.rebuild_render_resources();
            }
        } else {
            // What follows the moved curve follows it live, on every frame of the drag.
            fieldform_scene::relation::rederive(&mut self.doc);
        }
        // Drawn live, and the traced view redrawn at draft quality only when a field moved:
        // the shape under the cursor is what a drag is for, and the settled bake follows when
        // it stops.
        self.rebuild_curve_display();
        if gesture.shapes_field {
            self.refresh_scene_bounds();
            self.traced.invalidate(traced::Quality::Draft);
        }
    }

    /// Commits a gesture that moved a curve's points, as one entry whatever it moved: the free
    /// points as a `SetControlPoints`, the blocks as one graph edit from before the gesture to
    /// where they settled, batched when both moved so the two halves of one drag undo together.
    /// Returns whether anything moved at all -- a click that never moved anything is not an
    /// edit, and should not fill the history.
    pub(crate) fn commit_curve_gesture(
        &mut self,
        curve: CurveId,
        indices: &[usize],
        before: &[Point3<f64>],
        gesture: CurveGesture,
    ) -> bool {
        let graph_before = gesture.graph_before.clone();
        let one = CurveMove { curve, indices: indices.to_vec(), before: before.to_vec(), gesture };
        self.commit_curves_gesture(vec![one], graph_before)
    }

    /// The same for several curves moved together -- a profile row under the manipulator --
    /// as one entry: one graph edit for every block that moved, and a `SetControlPoints` per
    /// curve whose free points did.
    pub(crate) fn commit_curves_gesture(&mut self, moves: Vec<CurveMove>, graph_before: Option<Graph>) -> bool {
        // Each curve's free points: what its command records. The blocks are the graph's.
        struct Free {
            curve: CurveId,
            indices: Vec<usize>,
            before: Vec<Point3<f64>>,
            after: Vec<Point3<f64>>,
        }
        let mut frees: Vec<Free> = Vec::new();
        for m in &moves {
            let Some(c) = self.doc.curve(m.curve) else { return false };
            let free: Vec<(usize, Point3<f64>)> = m
                .indices
                .iter()
                .zip(&m.before)
                .filter(|(i, _)| !m.gesture.bound.iter().any(|(b, _)| b == *i))
                .map(|(i, p)| (*i, *p))
                .collect();
            let after: Vec<Point3<f64>> = free
                .iter()
                .filter_map(|(i, _)| c.control_points().get(*i).copied())
                .collect();
            if after.len() != free.len() {
                return false;
            }
            let before: Vec<Point3<f64>> = free.iter().map(|(_, p)| *p).collect();
            if before.iter().zip(&after).any(|(b, a)| (a - b).norm_squared() > 1e-12) {
                frees.push(Free {
                    curve: m.curve,
                    indices: free.iter().map(|(i, _)| *i).collect(),
                    before,
                    after,
                });
            }
        }
        let graph_after = self.part_graph().clone();
        let points_moved = graph_before.as_ref().is_some_and(|was| {
            moves.iter().flat_map(|m| m.gesture.bound.iter()).any(|(_, node)| {
                point_literals(was, *node) != point_literals(&graph_after, *node)
            })
        });
        if frees.is_empty() && !points_moved {
            return false;
        }
        // Back to where the gesture started, so the command's execute is what puts everything
        // where it ended.
        let mut commands: Vec<Box<dyn fieldform_ops::Command>> = Vec::new();
        if points_moved && let Some(was) = graph_before {
            self.restore_part_graph(was.clone());
            commands.push(Box::new(SetPartGraph::new(self.active_part, was, graph_after)));
        }
        for f in frees {
            if let Some(c) = self.doc.curve_mut(f.curve) {
                for (i, p) in f.indices.iter().zip(&f.before) {
                    c.set_control_point(*i, *p);
                }
            }
            commands.push(Box::new(SetControlPoints::new(f.curve, f.indices, f.before, f.after)));
        }
        self.gesture = None;
        let cmd: Box<dyn fieldform_ops::Command> = match commands.len() {
            1 => commands.pop().expect("one command"),
            _ => Box::new(fieldform_ops::Batch::new("curve.drag", commands)),
        };
        if let Err(e) = self.stack.execute(&mut self.doc, cmd) {
            self.warn(e.to_string());
        }
        true
    }

    /// What follows a curve gesture. The traced view is rebuilt only when a field changed;
    /// the curve overlay is redrawn either way, and the ends settle onto their blocks.
    pub(crate) fn after_curve_gesture(&mut self, shapes_field: bool) {
        self.mark_changed();
        if shapes_field {
            self.rebuild_render_resources();
        } else {
            self.settle_curves_on_points();
            self.rebuild_curve_display();
        }
    }

    /// Begins placing a curve: the next two clicks in the viewport are its start and its end.
    ///
    /// Replaces any placement already waiting rather than refusing, since asking for a curve
    /// twice means one curve, not an error about the first.
    pub(crate) fn begin_curve_placement(&mut self, degree: usize) {
        let degree = degree.clamp(1, 24);
        self.pending_curve =
            Some(PendingCurve {
                degree,
                first: None,
                preview: None,
                axis: None,
                restanding: None,
                placed: 0,
            });
        let what = if degree == 1 { "Line".to_string() } else { format!("Degree {degree} curve") };
        self.say(format!(
            "{what} — click the start point, then the end point. Esc cancels"
        ));
    }

    /// Forgets a placement in progress. Nothing was drawn, so there is nothing to undo.
    pub(crate) fn cancel_curve_placement(&mut self) {
        if let Some(pending) = self.pending_curve.take() {
            self.say(crate::curve_tool_ended(&pending, "Curve tool off"));
        }
    }

    /// Where a click at `pos` lands in space, or why it cannot.
    ///
    /// The three-way rule is [`landing_point`]'s; this only gathers what it needs from the
    /// application. The plane is read through the same `resolve_frame` the geometry is built
    /// with, so a point placed on it and a point lifted into it agree exactly; the surface is
    /// `pick_pivot`, which already answers "what is under the cursor" by what the mode draws.
    pub(crate) fn landing_under(
        &self,
        aspect: f32,
        rect: egui::Rect,
        pos: egui::Pos2,
    ) -> Result<Point3<f64>, String> {
        let (origin, dir) = self.camera.pick_ray(aspect, pos, rect);
        let plane = self.drawing_frame().and_then(|frame| {
            let numbers = |r: &ParamRef| Some(self.resolve(*r));
            let (translation, rotation) =
                fieldform_field::resolve_frame(&self.doc, self.part_graph(), &frame, &numbers)
                    .ok()?;
            Some((
                Point3::from(translation).cast::<f32>(),
                (rotation * Vector3::z()).cast::<f32>(),
            ))
        });
        let surface = || self.pick_pivot(origin, dir, aspect, rect, pos);
        match landing_point(
            origin,
            dir,
            plane,
            surface,
            self.camera.last_poi,
            self.camera.forward(),
        ) {
            Some(at) => Ok(at.cast::<f64>()),
            None if plane.is_some() => Err(format!(
                "Looking at {} edge-on — turn the view a little, or say draw on=none",
                self.drawing_plane.label(self)
            )),
            None => Err("Nothing under the cursor to place a point on".to_string()),
        }
    }

    /// Takes a click for the curve being placed: the start if none is down yet, else the end.
    ///
    /// **Ctrl snaps to the nearest curve end in reach, before anything else is asked** — in 3D
    /// and off the drawing plane if that is where the end is, which is what the spec means by
    /// snapping being deliberate. What is stored is a position, not a reference to the other
    /// curve: move it later and this point stays.
    pub(crate) fn place_curve_point(
        &mut self,
        aspect: f32,
        rect: egui::Rect,
        pos: egui::Pos2,
        snapping: bool,
        grid: bool,
    ) {
        let Some(pending) = self.pending_curve.as_ref() else { return };
        let (degree, first, axis, placed) =
            (pending.degree, pending.first, pending.axis, pending.placed);

        // One click on the surface, and the field says which way the axis goes — nothing here
        // snaps or lands on a drawing plane, since the point has to be on the surface itself.
        if axis == Some(AxisKind::Normal) {
            self.place_normal_axis(aspect, rect, pos);
            return;
        }

        let snapped = if snapping { self.placement_snap(aspect, rect, pos) } else { None };
        let (at, onto) = match snapped {
            Some((at, what)) => (at, Some(what)),
            None => match self.landing_under(aspect, rect, pos) {
                // Shift rounds the landing to the grid; a curve end in reach has already won,
                // since an exact meeting beats a round number.
                Ok(at) if grid => (self.grid_snap(aspect, rect, at).unwrap_or(at), None),
                Ok(at) => (at, None),
                Err(why) => {
                    self.warn(why);
                    return;
                }
            },
        };

        let Some(first) = first else {
            if let Some(pending) = self.pending_curve.as_mut() {
                pending.first = Some(at);
                pending.preview = None;
            }
            self.say(match onto {
                Some(what) => {
                    format!("Start snapped to {what} — click the end point")
                }
                None => "Start placed — click the end point".to_string(),
            });
            return;
        };

        // Two clicks on one spot make no curve, and a zero-length line is a division waiting
        // in every tangent that would ever be asked of it. Kept waiting, not cancelled.
        if (at - first).norm() < 1e-6 {
            self.warn("That is where it starts — click somewhere else for the end");
            return;
        }

        let id = CurveId::new();
        let mut curve = match bezier_between(id, degree, first, at) {
            Ok(curve) => curve,
            Err(e) => {
                self.pending_curve = None;
                self.warn(format!("could not make a degree {degree} curve: {e}"));
                return;
            }
        };
        curve.name = if axis.is_some() {
            self.axis_name()
        } else {
            self.unique_curve_name(&curve.name)
        };
        // Through the stack, so the line just clicked is one Ctrl+Z from gone — the first
        // thing anyone tries after a click lands somewhere they did not mean. And not a graph
        // edit, so it ends any gesture a value drag had open, or the next graph edit would take
        // this entry for the drag's and replace it.
        self.gesture = None;
        if let Err(e) = self.stack.execute(&mut self.doc, Box::new(SetCurve::insert(curve))) {
            self.pending_curve = None;
            self.warn(e.to_string());
            return;
        }
        // **The tool stays on until Escape** (item 111, the owner's request 2026-10-15): drawing
        // the four sides of a profile used to mean choosing the tool four times. Ready for the
        // next start, same degree; the next curve is not chained to this one's end, by the owner's
        // ruling the same day — Ctrl's snap already chains a profile, and an automatic start would
        // put an extra step in front of every curve that is not meant to touch.
        //
        // An axis stays one-shot: it is one object at a time, and re-pointing one is a single
        // gesture by nature. A console line never reaches here (this has one caller, the viewport's
        // click), so a script cannot end with the tool armed.
        self.pending_curve = axis.is_none().then_some(PendingCurve {
            degree,
            first: None,
            preview: None,
            axis: None,
            restanding: None,
            placed: placed + 1,
        });
        self.select_curve(id);
        self.rebuild_curve_display();
        self.mark_changed();
        // No reframing: the curve was placed where the person is looking, and moving the view
        // to show them what they just drew would take it away from them.
        let what = if axis.is_some() {
            "an axis".to_string()
        } else if degree == 1 {
            "a line".to_string()
        } else {
            format!("a degree {degree} curve")
        };
        // **Said every time, because the tool is still live.** A mode you are in without knowing
        // is how a click meant to select something draws a line instead.
        let next = if axis.is_none() { " — click the next start, Esc to finish" } else { "" };
        self.say(match onto {
            Some(snap) => format!("Added {what}, its end snapped to {snap}{next}"),
            None => format!("Added {what}{next}"),
        });
    }

    /// Brings the placement preview up to date with the cursor, recomputing only when it must.
    ///
    /// The landing can be a field trace, so it is keyed on the cursor, the eye and the modifier:
    /// a frame in which none of them moved reuses the last answer. Off the viewport there is
    /// no preview, since there is nowhere for the point to be.
    pub(crate) fn refresh_placement_preview(
        &mut self,
        aspect: f32,
        rect: egui::Rect,
        hover: Option<egui::Pos2>,
        snapping: bool,
        grid: bool,
    ) {
        let Some(cursor) = hover.filter(|p| rect.contains(*p)) else {
            if let Some(pending) = self.pending_curve.as_mut() {
                pending.preview = None;
            }
            return;
        };
        let eye = self.camera.eye;
        let fresh = self.pending_curve.as_ref().is_some_and(|p| {
            p.preview.as_ref().is_some_and(|l| {
                l.cursor == cursor && l.eye == eye && l.snapping == snapping && l.grid == grid
            })
        });
        if fresh {
            return;
        }
        // An axis that stands on a surface is shown where the *surface* is, not where a curve
        // point would land (the drawing plane, a curve end): the marker is the click's answer.
        let normal_axis = self
            .pending_curve
            .as_ref()
            .is_some_and(|p| p.axis == Some(AxisKind::Normal));
        let at = if normal_axis {
            self.normal_axis_landing(aspect, rect, cursor)
        } else {
            (if snapping {
                self.placement_snap(aspect, rect, cursor).map(|(at, _)| at)
            } else {
                None
            })
            .or_else(|| {
                let at = self.landing_under(aspect, rect, cursor).ok()?;
                Some(if grid { self.grid_snap(aspect, rect, at).unwrap_or(at) } else { at })
            })
        };
        if let Some(pending) = self.pending_curve.as_mut() {
            pending.preview = at.map(|at| Landing { cursor, eye, snapping, grid, at });
        }
    }

    /// Draws the curve being placed: where the next click would land, the start once it is
    /// down, and a dashed run between them.
    ///
    /// Drawn to the *landing* point rather than to the cursor. In perspective the depth is the
    /// whole question, and where the point will actually sit — on the model, on the plane — is
    /// only legible when the marker is drawn there. A hollow square for what is not yet a point,
    /// a filled one for the start, which is a control point in all but name.
    pub(crate) fn draw_placement(&self, painter: &egui::Painter, aspect: f32, rect: egui::Rect) {
        let Some(pending) = &self.pending_curve else { return };
        let stroke = egui::Stroke::new(1.5_f32, CURVE_SELECTED_COLOR);
        let start = pending
            .first
            .and_then(|p| self.camera.project_to_screen(aspect, rect, p.cast::<f32>()));
        let next = pending
            .preview
            .as_ref()
            .and_then(|l| self.camera.project_to_screen(aspect, rect, l.at.cast::<f32>()));
        if let (Some(a), Some(b)) = (start, next) {
            for shape in egui::Shape::dashed_line(&[a, b], stroke, 6.0, 4.0) {
                painter.add(shape);
            }
        }
        if let Some(b) = next {
            painter.rect_stroke(
                egui::Rect::from_center_size(b, egui::vec2(9.0, 9.0)),
                1.0,
                stroke,
                egui::StrokeKind::Middle,
            );
        }
        if let Some(a) = start {
            painter.rect_filled(
                egui::Rect::from_center_size(a, egui::vec2(9.0, 9.0)),
                1.0,
                SELECTION_COLOR,
            );
        }
    }

    /// Starts dragging the selected control points, in the plane they should slide in.
    ///
    /// **Which plane matters more here than it does for a cage vertex.** A vertex slides in the
    /// plane facing the camera, which is the only sensible answer when there is no other. A
    /// control point often has one: if a drawing plane is set, the point stays on it, so a curve
    /// drawn flat stays flat through an edit rather than being pulled out of its plane by a drag
    /// taken at some angle. With no drawing plane it falls back to facing the camera, which in a
    /// dead-on view is the same plane anyway.
    pub(crate) fn begin_point_drag(&mut self, aspect: f32, rect: egui::Rect, click_pos: egui::Pos2) {
        // Every picked point grouped by curve, the selected curve's first: the first group is
        // what the drag anchors on and snaps, the rest move by the same offset.
        let groups = group_by_curve(&self.picked_points());
        let Some((curve, indices)) = groups.first().cloned() else { return };
        let mut others = Vec::new();
        for (other, at) in groups.into_iter().skip(1) {
            let Some(points) = self.doc.curve(other).map(|c| c.control_points().to_vec()) else {
                continue;
            };
            let before: Vec<Point3<f64>> = at.iter().filter_map(|i| points.get(*i).copied()).collect();
            if before.len() != at.len() {
                return;
            }
            let Some(gesture) = self.begin_curve_gesture(other, &at) else { return };
            others.push(CurveMove { curve: other, indices: at, before, gesture });
        }
        let Some(c) = self.doc.curve(curve) else { return };
        let start_positions: Vec<Point3<f64>> = indices
            .iter()
            .filter_map(|i| c.control_points().get(*i).copied())
            .collect();
        if start_positions.len() != indices.len() {
            return;
        }

        // An end sitting on a point block moves the block (§4.3); the gesture says which.
        let Some(gesture) = self.begin_curve_gesture(curve, &indices) else { return };

        // The anchor is the first point moving, and everything else keeps its offset from it.
        let anchor = start_positions[0].cast::<f32>();
        let plane_normal = match self.drawing_normal() {
            Some(n) => n.cast::<f32>(),
            None => self.camera.forward(),
        };
        let (origin, dir) = self.camera.pick_ray(aspect, click_pos, rect);
        let click_offset = ray_plane_intersect(origin, dir, anchor, plane_normal)
            .map(|p| p - anchor)
            .unwrap_or_else(Vector3::zeros);

        self.point_drag = Some(PointDrag {
            curve,
            indices,
            start_positions,
            plane_point: anchor,
            plane_normal,
            click_offset,
            snapped_to: None,
            gesture,
            others,
        });
    }

    /// Continues a control-point drag: every point moves by the same offset on the drag plane.
    ///
    /// **The snap is applied here, while the drag is running**, not when it ends. A snap you
    /// cannot see until you let go is one you cannot aim: the point jumps onto the end it has
    /// found, so holding the modifier and moving near an end shows you it has taken.
    pub(crate) fn continue_point_drag(
        &mut self,
        aspect: f32,
        rect: egui::Rect,
        pos: egui::Pos2,
        snapping: bool,
        grid: bool,
    ) {
        let mut state = match self.point_drag.take() {
            Some(state) => state,
            None => return,
        };
        let (origin, dir) = self.camera.pick_ray(aspect, pos, rect);
        if let Some(hit) = ray_plane_intersect(origin, dir, state.plane_point, state.plane_normal)
        {
            let delta = ((hit - state.click_offset) - state.plane_point).cast::<f64>();
            // Shift snaps the anchor to the grid and carries the rest by the same offset, so a
            // shape dragged as a group lands on the grid without being reshaped by it. With a
            // drawing plane set the drag is already in that plane, so the landing is exact.
            let delta = if grid {
                let anchor = state.start_positions[0];
                self.grid_snap(aspect, rect, anchor + delta)
                    .map(|snapped| snapped - anchor)
                    .unwrap_or(delta)
            } else {
                delta
            };

            // Only a single end can snap: dragging several points is a move of a shape, and an
            // interior control point is not somewhere two curves meet.
            let count = self.doc.curve(state.curve).map(|c| c.control_points().len());
            let lone_end = match (state.indices.as_slice(), count) {
                ([i], Some(n)) if state.others.is_empty() && (*i == 0 || *i + 1 == n) => Some(*i),
                _ => None,
            };
            // **Measured from where the point would be, not from the cursor.** They are not
            // the same place: grabbing a point ten pixels off centre leaves the cursor ten
            // pixels from what is being aimed, so hovering the *point* over a target was
            // measuring from somewhere else entirely. Where the point is, is what the person is
            // looking at.
            let target = if snapping && lone_end.is_some() {
                let would_be = state.start_positions[0] + delta;
                self.camera
                    .project_to_screen(aspect, rect, would_be.cast::<f32>())
                    .and_then(|at| self.endpoint_near(aspect, rect, at, Some(state.curve)))
            } else {
                None
            };
            state.snapped_to = target.map(|(id, _)| id);

            // Where each point goes: a snapped end onto its target, everything else by the
            // delta. Moving them is the gesture's shared path, so an end sitting on a block
            // moves the block here exactly as it does under the manipulator.
            let snapped_at = target.map(|(_, at)| at);
            let to: Vec<(usize, Point3<f64>)> = state
                .indices
                .iter()
                .zip(&state.start_positions)
                .map(|(i, start)| match snapped_at {
                    Some(at) if lone_end == Some(*i) => (*i, at),
                    _ => (*i, start + delta),
                })
                .collect();
            self.move_curve_points_live(state.curve, &to, &state.gesture);
            for m in &state.others {
                let to: Vec<(usize, Point3<f64>)> =
                    m.indices.iter().zip(&m.before).map(|(i, start)| (*i, start + delta)).collect();
                self.move_curve_points_live(m.curve, &to, &m.gesture);
            }
        }
        self.point_drag = Some(state);
    }

    /// Finishes a control-point drag, snapping onto another curve's end when `snapping` is set.
    ///
    /// Committed the same way every other drag is: the document was moved live for feedback, and
    /// the gesture goes on the history once, as the value it settled at, rather than sixty times
    /// a second.
    pub(crate) fn end_point_drag(&mut self) {
        let Some(state) = self.point_drag.take() else { return };
        let PointDrag { curve, indices, start_positions, snapped_to: snapped, gesture, others, .. } =
            state;
        let shapes_field = gesture.shapes_field || others.iter().any(|m| m.gesture.shapes_field);

        // Nothing to decide here: the drag already put the point where it goes, snapped or
        // not. This only reports it, and reports it by name because an exact landing and a near
        // miss look identical on screen and only one of them closes a profile.
        let landed = snapped.and_then(|id| self.doc.curve(id).map(|c| c.name.clone()));

        let committed = if others.is_empty() {
            self.commit_curve_gesture(curve, &indices, &start_positions, gesture)
        } else {
            let graph_before = gesture
                .graph_before
                .clone()
                .or_else(|| others.iter().find_map(|m| m.gesture.graph_before.clone()));
            let mut moves = vec![CurveMove { curve, indices, before: start_positions, gesture }];
            moves.extend(others);
            self.commit_curves_gesture(moves, graph_before)
        };
        if !committed {
            return;
        }
        if let Some(onto) = landed {
            self.say(format!("Snapped to the end of {onto}"));
        }
        self.after_curve_gesture(shapes_field);
    }
}
