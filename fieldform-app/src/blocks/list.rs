// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The block list's rows: what a row is, how rows nest, and the profile rows loops of curves make.

use crate::*;

/// Which row in the block list is being renamed.
///
/// A node and a curve are both rows with a name on them, so both rename the same way (D9). They
/// cannot share `Row`, which carries a node's body flag and is about the context menu.
///
/// `Hash` so the name field can be salted with the row it belongs to rather than placed by
/// position — see [`App::list_row`]. A row moves while it is being renamed: pulling a block out of
/// the block it was drawn inside names it, and the row it is on rises in the same frame.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum RowKey {
    Node(NodeId),
    Curve(CurveId),
    /// A surface's row. It has a name like the other two, though nothing renames one yet — the
    /// name comes from the file and there has been no reason to change it.
    Surface(fieldform_scene::QuadMeshId),
    Tessellation(fieldform_scene::TessellationId),
    Export(fieldform_scene::ExportId),
    Import(fieldform_scene::id::SourceId),
    /// A profile row, keyed by its first curve. Its name is `Document::profile_names`
    /// (`profile-group-spec.md` §3.3), and an empty one puts "Profile N" back.
    Profile(CurveId),
}

/// The contents of a dropdown you can type into: a filter box, then whatever matches.
///
/// **Because a list of a hundred bodies is not a list anybody can use.** Asked for directly: a
/// part with that many blocks makes every operand picker a scroll, and the name being looked for
/// is already known. Typing narrows it; Enter takes the only match when there is one.
///
/// The box appears only past [`FILTER_FROM`] entries. Below that the list is quicker to read
/// than to type into, and a search field over four items is furniture.
///
/// The query lives in egui's own temporary memory keyed by this menu's id, not on the `App`: it
/// is per-menu, it should not survive a reopen, and egui already cleans up what it holds. A
/// field on the app would be a map of every dropdown ever opened, cleared by nobody.
///
/// Call inside `ComboBox::show_ui`. Returns the item that was clicked, if one was.
pub(crate) fn filtered_choices<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    items: &[(T, String)],
    current: Option<T>,
) -> Option<T> {
    filtered_choices_gated(ui, items, current, |_| None)
}

/// The same, with a reason some entries cannot be chosen.
///
/// **Greyed and left in rather than taken out**, which is the owner's ruling and the better one:
/// a row that is missing answers nothing, and the question somebody has while looking down this
/// list is *why* their body is not an option. So a refused entry keeps its place, draws weak, does
/// not respond to a click, and says on hover what the block would do with it if it could.
fn filtered_choices_gated<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    items: &[(T, String)],
    current: Option<T>,
    refused: impl Fn(T) -> Option<&'static str>,
) -> Option<T> {
    let key = ui.id().with("filter");
    let mut query: String = ui.memory(|m| m.data.get_temp(key)).unwrap_or_default();
    let filtering = items.len() > FILTER_FROM;
    if filtering {
        let box_ = ui.add(
            egui::TextEdit::singleline(&mut query)
                .hint_text("type to narrow")
                .desired_width(f32::INFINITY),
        );
        // Focused when the menu opens, so typing starts working without a click. Asking every
        // frame would take the keyboard away from the list's own arrow keys.
        if box_.has_focus() {
            // Already ours.
        } else if !box_.lost_focus() && query.is_empty() {
            box_.request_focus();
        }
        ui.memory_mut(|m| m.data.insert_temp(key, query.clone()));
        ui.separator();
    }

    let needle = query.trim().to_lowercase();
    let matching: Vec<&(T, String)> = items
        .iter()
        .filter(|(_, name)| needle.is_empty() || name.to_lowercase().contains(&needle))
        .collect();

    // **Enter takes the one match**, which is what makes this typing rather than scrolling: the
    // name is known, so narrowing to it and pressing Enter is the whole gesture. Ambiguous or
    // empty does nothing, because guessing between two is worse than another keystroke -- and a
    // refused match is not one, or the keyboard would take what the mouse will not.
    if filtering
        && let [(only, _)] = matching.as_slice()
        && refused(*only).is_none()
        && ui.input(|i| i.key_pressed(egui::Key::Enter))
    {
        ui.memory_mut(|m| m.data.remove::<String>(key));
        ui.close();
        return Some(*only);
    }

    let mut picked = None;
    if matching.is_empty() {
        ui.label(egui::RichText::new("nothing matches").weak());
    }
    for (item, name) in matching {
        match refused(*item) {
            Some(why) => {
                ui.add_enabled(
                    false,
                    egui::Button::selectable(false, egui::RichText::new(name).weak()),
                )
                .on_disabled_hover_text(why);
            }
            None => {
                if ui.selectable_label(current == Some(*item), name).clicked() {
                    picked = Some(*item);
                }
            }
        }
    }
    if picked.is_some() {
        // The next open starts clean rather than showing the last search, which would look like
        // a list that had lost most of its entries.
        ui.memory_mut(|m| m.data.remove::<String>(key));
    }
    picked
}

/// How many entries a dropdown needs before it offers to be typed into. See
/// [`filtered_choices`].
const FILTER_FROM: usize = 8;

/// Which row a context menu was opened on.
#[derive(Clone, Copy)]
pub(crate) enum Row {
    /// A node in the part's graph. `is_body` is whether it is a *root* — only those have an eye,
    /// so only those have anything to show or hide. `readers` is the count the row's V was drawn
    /// from, so the menu's Variable item asks the V's question of the same number.
    Node { id: NodeId, is_body: bool },
    /// A curve's row, with the count its V was drawn from for the same reason, and because the
    /// count is also what says whether that V is the curve's own or its loop's.
    Curve { id: CurveId, readers: usize },
}

/// What a copy put aside, ready to be pasted.
///
/// Held as *geometry*, not as ids. A pasted body must be independent of the one it came from —
/// dragging the copy must not move the original — so the cages and curves its nodes name travel
/// with it and are minted afresh on paste. Referencing the originals would give two bodies one
/// cage, which is a different feature and not the one anybody means by paste.
///
/// Lives in the app rather than the document: a clipboard is not part of what a file contains.
#[derive(Clone)]
pub(crate) enum Clipboard {
    /// A body and everything beneath it.
    Body {
        /// The subtree, root last, so inputs are always already inserted when a node arrives.
        nodes: Vec<(NodeId, Node)>,
        root: NodeId,
        cages: Vec<(CageId, Cage)>,
        curves: Vec<(CurveId, fieldform_scene::Curve)>,
    },
    Curve(fieldform_scene::Curve),
}

/// One row of the block list.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Block {
    /// A named value, which is a block in its own right.
    ///
    /// **It leaves its parent when it is named.** A number typed into a radius belongs to that
    /// sphere; a number that fifty spheres read belongs to none of them, and leaving it shown
    /// inside the first one that happened to name it says the opposite of what it now is. So a
    /// variable rises out of the block it was declared in and takes a row of its own, at the top
    /// where inputs belong.
    Parameter(ParamId),
    Curve(CurveId),
    /// An open surface a conformal lattice can follow.
    ///
    /// **A block, though it is not a body**, for the same reason a curve is one: it is
    /// geometry this document holds and other blocks read. The first cut of the surface import
    /// put one in the document and showed nothing at all, which is indistinguishable from an
    /// import that failed -- reported immediately, and rightly.
    Surface(fieldform_scene::QuadMeshId),
    Node(NodeId),
    /// A body turned into triangles — the first of the two export blocks (item 53).
    Tessellation(fieldform_scene::TessellationId),
    /// A file written from a tessellation — the second.
    Export(fieldform_scene::ExportId),
    /// Everything one file brought in, as one row that opens.
    ///
    /// A STEP assembly is one solid per body — forty of them for a real machine — and forty
    /// rows saying "imported (17)" is a list nobody reads. The bodies are still bodies: this
    /// groups the *rows*, and nothing in the document describes the grouping, which is the
    /// same rule the bodies themselves follow. A file that produced one solid stays one
    /// ordinary row, because a group of one is a disclosure triangle over nothing.
    Import(fieldform_scene::id::SourceId),
    /// A closed loop of two or more open curves, as one row that opens.
    ///
    /// **Derived on every read, and nothing in the document describes it** — the same rule an
    /// import's row follows, and the one `profile_loop` already made for the loop itself: a
    /// stored list of members is a second description of what the geometry says, and the two
    /// part company the moment a curve moves. The key is the loop's first curve in block-list
    /// order, which is stable while that curve stays in the loop. A single closed curve is not
    /// one of these; it is its own row, since a group of one is a disclosure triangle over
    /// nothing. `profile-group-spec.md` §3.
    Profile(CurveId),
}

/// One row of the block list, together with the rows drawn inside it.
///
/// **Nesting is a reading of the list, not a second list.** Which block a block is drawn inside is
/// the ownership the document stores, read by [`fieldform_scene::Document::nesting_in`]; this is
/// only that reading arranged into the shape the panel draws. Nothing about the tree is stored,
/// the graph is untouched, and the part's bodies are still the roots of its graph.
///
/// A row is drawn inside its owner's *body*, the way an import's solids and a profile's curves
/// already are — "one row that opens" — so opening a block shows what it is made of and closing it
/// puts it away again.
#[derive(Clone)]
pub(crate) struct BlockRow {
    pub(crate) block: Block,
    /// The blocks stored inside this one, in the order the flat list had them — which is
    /// dependency order, since `listed_blocks` is a post-order walk.
    pub(crate) inside: Vec<BlockRow>,
    /// Whether this row is itself drawn inside another. It decides one thing: whether the row
    /// offers a grip, which for a nested row is the drag that takes it out to the top level
    /// (`notebook-spec.md` §7.2) — a plain block, not a variable.
    pub(crate) nested: bool,
    /// How many blocks read this one, taken from the same walk that decided the nesting.
    ///
    /// Carried on the row rather than asked for while the row is drawn, because asking is a pass
    /// over every node and drawing happens once per row every frame — the shape
    /// `Document::nesting_in` warns about in its own first comment. Derived here, in the pass that
    /// builds the tree, so it cannot fall out of step with the nesting the way a cache on the
    /// application would.
    pub(crate) readers: usize,
    /// The inputs this block reads that live outside it as variables, in the order it names them
    /// (backlog item 108) — the third kind of input beside a value and an owned block, which the
    /// row now draws rather than leaving out. From the same `Nesting` the tree is built from.
    pub(crate) references: Vec<BlockKey>,
}

/// How tall the strip between two rows is, and therefore how much room the rows are given when
/// there is no strip — the list should not breathe differently where a block happens to be able to
/// land.
///
/// **Eight rather than the six it was first written with.** Six was aimed at by hand and missed by
/// two pixels, which is the sort of thing only running it says. A pointer target rather than a
/// hairline.
const GUTTER_HEIGHT: f32 = 8.0;

/// Where each row's branch begins: the body a drop above this row would put a block in front of, or
/// `None` if a drop there means nothing.
///
/// **A drag moves a body together with what only it reads**, which `block-order-spec.md` §2 settled
/// and `listed_blocks` arranges: a block pinned above the body that consumes it travels with it. So
/// the line has to be drawn above the first row that moves, not above the body's own row — drawn on
/// the body, a drop lands one row higher than the line promised, which is exactly what running it
/// showed.
///
/// Read backwards, because a branch ends at its body and the rows above it are what it took with
/// it. A row belongs to the run only if the body can actually reach it; a curve nothing in the
/// branch reads happens to sit there and stays where it is when the body moves.
pub(crate) fn branch_starts(rows: &[BlockRow], graph: &Graph) -> Vec<Option<NodeId>> {
    let roots = graph.roots();
    let mut starts: Vec<Option<NodeId>> = vec![None; rows.len()];
    // The body whose branch the walk is inside, and everything that body reads.
    let mut branch: Option<(NodeId, std::collections::HashSet<NodeId>)> = None;
    for (i, row) in rows.iter().enumerate().rev() {
        if let Block::Node(id) = row.block
            && roots.contains(&id)
        {
            branch = Some((id, graph.subtree_of(id)));
        }
        let belongs = match (&branch, row.block) {
            (Some((_, own)), Block::Node(id)) => own.contains(&id),
            (Some((_, own)), Block::Curve(curve) | Block::Profile(curve)) => own
                .iter()
                .any(|n| graph.node(*n).is_some_and(|node| node.curves().contains(&curve))),
            _ => false,
        };
        if !belongs {
            branch = None;
        }
        starts[i] = branch.as_ref().map(|(body, _)| *body);
    }
    starts
}

/// Several names as a line reads them: "A", "A and B", "A, B and C".
///
/// **What is left of a refusal that is gone.** This was `put_back_refusal`, which said why a
/// variable two blocks read could not go back inside one of them: a nested row had one owner, so
/// two readers left nowhere for the row to go. `notebook-spec.md` §6.6 puts the block back inside
/// **every** reader instead — a shared instance (§7.3) — so there is nothing to refuse, and the
/// names are wanted for the opposite line: what the row went into.
///
/// **It names the blocks rather than counting them**, which is why the joining is worth a function
/// of its own. "is part of Union and Shell again" is something a person can go and look at; "is
/// part of 2 blocks again" is not. Written once because the bright V, the row menu's untick and
/// the console all say it, and three spellings of one line drift.
///
/// No names is allowed, and reads "nothing": a block that goes back inside nothing stands on its
/// own, which is a destination too.
pub(crate) fn names_in_a_row(names: &[String]) -> String {
    match names {
        [] => "nothing".to_string(),
        [only] => only.clone(),
        [first, second] => format!("{first} and {second}"),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// The refusal for clearing a variable's name (`block-ownership-spec.md` §5.3, I4).
const A_VARIABLE_NEEDS_A_NAME: &str =
    "A variable needs a name \u{2014} put it back inside first to clear it";

/// *Make variable* (`block-ownership-spec.md` §5.1): the command, or `None` for a block that
/// already is one.
///
/// **The name it keeps, or one `invented` for it**, since a variable is what other blocks refer
/// to by name and an unnamed one could be reused and never typed (I4). Asked only when wanted,
/// because inventing one walks the document for the names in use. A curve is born named, so one
/// is never invented for a curve's row or a profile row, whose V acts on a curve of its loop.
///
/// Free of the `App`, with the V, the row menu and a drop on the list all reaching it, so a test
/// can hold what all of them do: one command, so one undo takes back the row and the invented
/// name together. One function for a block and a curve, because they are one act (§10).
pub(crate) fn make_variable(
    doc: &Document,
    block: BlockKey,
    invented: impl FnOnce() -> String,
) -> Option<fieldform_ops::SetOwnership> {
    if doc.ownership(block) == Ownership::Variable {
        return None;
    }
    let name = name_of(doc, block).unwrap_or_else(invented);
    Some(fieldform_ops::SetOwnership::new(block, Ownership::Variable, Some(&name)))
}

/// *Put back inside* (`block-ownership-spec.md` §5.2): the command, or the readers that refuse it.
///
/// Inside the one block that reads it, or free when nothing does; with two or more there is no
/// one block to go into, and the answer is who they are, ascending, for the refusal to name.
/// [`Document::put_back_target`] answers where, counting readers as I2 does — a curve's in every
/// part.
///
/// **The name stays.** Clearing it used to *be* the move back inside, when the list was laid out
/// by names, so a name somebody typed was lost to the gesture; now it is a label and the row is
/// the ownership's, so the two come apart. A profile row's was the same, and is a label too.
pub(crate) fn put_back_inside(doc: &Document, block: BlockKey) -> fieldform_ops::SetOwnership {
    let target = doc.put_back_target(block);
    fieldform_ops::SetOwnership::new(block, target, name_of(doc, block).as_deref())
}

/// *Put back inside* for a parameter (`block-ownership-spec.md` §11): one entry that writes its
/// number into the one socket holding it and takes the parameter out, or why it has to stay a
/// variable.
///
/// **A parameter is already this model's variable.** A socket holds a number or a name
/// (`ParamRef::Literal`, `ParamRef::Named`), and *Name this value* is make variable; this is the
/// way back, from the bright V on the parameter's row. It goes back only where there is one place
/// for it to go:
///
/// - a number, not an expression, since `ParamRef` has no arm for a formula in a field, so a
///   computed number is always a variable;
/// - held by exactly one socket in the document — every part, since a parameter is the
///   document's, and counted per socket, so a box whose width and depth both read it has two
///   places it could go and no one place;
/// - and read by no other parameter's expression, which can read only a variable.
///
/// **The graph edit first and the parameter after it, in one entry**, as the × beside a named
/// value has them: undo runs the two backwards, so the parameter is back before the graph reads
/// it again, and one Ctrl+Z restores both. Turning a name into a number moves no row, so the
/// settle the graph edit runs finds every block where it was.
///
/// `label` is what the application calls a block of a part, for a refusal to name it by. Free of
/// the `App` otherwise, so a test can hold what the V does.
pub(crate) fn put_number_back(
    doc: &Document,
    id: ParamId,
    label: impl Fn(PartId, NodeId) -> String,
) -> Result<fieldform_ops::Batch, String> {
    let Some(parameter) = doc.parameters.get(&id) else {
        return Err("It is not in the document any more.".to_string());
    };
    let Some(value) = parameter.value.as_literal() else {
        return Err("It computes its value, and an expression has to stay a variable: a field \
                    holds a number or a name, never a formula."
            .to_string());
    };
    let wanted = ParamRef::Named(id);
    // Each block holding it, with how many of its sockets do.
    let mut holders: Vec<(PartId, NodeId, usize)> = Vec::new();
    for (&part, held) in &doc.parts {
        for (node, block) in held.graph.nodes() {
            let sockets = block.param_refs().into_iter().filter(|r| **r == wanted).count();
            if sockets > 0 {
                holders.push((part, node, sockets));
            }
        }
    }
    let expressions = doc.parameters_reading(id);
    if let ([(part, node, 1)], []) = (holders.as_slice(), expressions.as_slice()) {
        let before = doc.parts[part].graph.clone();
        let mut after = before.clone();
        if let Some(block) = after.node_mut(*node) {
            for socket in block.param_refs_mut() {
                if *socket == wanted {
                    *socket = ParamRef::Literal(value);
                }
            }
        }
        return Ok(fieldform_ops::Batch::new(
            "parameter.put_back_inside",
            vec![
                Box::new(SetPartGraph::new(*part, before, after)),
                Box::new(SetParameter::remove(parameter.clone())),
            ],
        ));
    }
    if holders.is_empty() && expressions.is_empty() {
        return Err("There is no field for it to go back into: nothing uses it. The \u{d7} \
                    beside it deletes it."
            .to_string());
    }
    // The blocks in the order they were made, as a block's refusal names its readers, then the
    // parameters by name.
    holders.sort_by_key(|(_, node, _)| (doc.made_at(*node), node.raw()));
    let mut names: Vec<String> = holders
        .into_iter()
        .map(|(part, node, sockets)| match sockets {
            1 => label(part, node),
            2 => format!("{} twice", label(part, node)),
            n => format!("{} {n} times", label(part, node)),
        })
        .collect();
    let mut reading: Vec<String> =
        expressions.iter().map(|p| doc.parameters[p].name.clone()).collect();
    reading.sort();
    names.extend(reading);
    let names = match names.split_last() {
        Some((only, [])) => only.clone(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
        // Refused for its readers, so there are some.
        None => String::new(),
    };
    // An expression is the stronger reason: with one socket beside it, the socket alone would
    // have had somewhere to go.
    let so = if expressions.is_empty() {
        "so there is no single field for it to go back into"
    } else {
        "and an expression can read only a variable"
    };
    Err(format!("A variable, and it has to stay one: it is used by {names}, {so}."))
}

/// The name `block` carries where `SetOwnership` writes it: a block's, which it may not have, or a
/// curve's, which it always has.
fn name_of(doc: &Document, block: BlockKey) -> Option<String> {
    match block {
        BlockKey::Node(id) => doc.node_meta.get(&id).and_then(|m| m.name.clone()),
        BlockKey::Curve(id) => doc.curve(id).map(|c| c.name.clone()),
    }
}

/// A rename (`block-ownership-spec.md` §5.3): the command, carrying the name and the ownership
/// the block already has, or why it may not.
///
/// **It never moves a block**, because a name is a label now: the ownership it carries is the
/// one stored, so the settle the command runs finds everything where it was. The one refusal is
/// clearing a variable's name, since a variable is typed by its name (I4); an `Inside` or free
/// block's may always go. Whether another block holds the name is the caller's to ask first, as
/// `SetName`'s clash check always was: the answer has to name that block, which is the
/// application's to call.
pub(crate) fn rename_block(
    doc: &Document,
    id: NodeId,
    name: &str,
) -> Result<fieldform_ops::SetOwnership, &'static str> {
    let block = BlockKey::Node(id);
    let ownership = doc.ownership(block);
    if name.trim().is_empty() && ownership == Ownership::Variable {
        return Err(A_VARIABLE_NEEDS_A_NAME);
    }
    Ok(fieldform_ops::SetOwnership::new(block, ownership, Some(name)))
}

/// The row menu's Variable item for a block, and what a click on it asks for: `Some(true)` to
/// make the block a variable, `Some(false)` to put it back inside, `None` when nothing was asked.
///
/// Free of the app so a headless click can show the refusal holds, which a window and a device
/// would otherwise stand between.
///
/// **Nothing greys the untick, since 112.** It used to go dead, with the reason on its hover,
/// where two or more blocks read the variable: there was no single block for it to go back
/// inside. A block goes back inside every reader now (`notebook-spec.md` §6.6), so the untick
/// always has an answer and the guard — and the reader names it was given to say — are gone.
///
/// **Greyed both ways where the row's V is another row's**, `elsewhere` saying why: a curve drawn
/// in its loop's row ([`curve_v_elsewhere`]). The menu is a second way to the V's act, not a way
/// round a V that has none.
pub(crate) fn variable_item(
    ui: &mut egui::Ui,
    is_variable: bool,
    elsewhere: Option<&str>,
) -> Option<bool> {
    let mut ticked = is_variable;
    if let Some(why) = elsewhere {
        ui.add_enabled(false, egui::Checkbox::new(&mut ticked, "Variable"))
            .on_disabled_hover_text(why);
        return None;
    }
    ui.checkbox(&mut ticked, "Variable").clicked().then_some(ticked)
}

/// The three bars a row is dragged by, painted rather than typed.
///
/// **A glyph was simpler and was wrong.** This was `\u{2261}`, chosen on the reasoning that egui's
/// bundled faces -- Ubuntu-Light, Hack and the two emoji faces -- cover a basic mathematical
/// operator. They do not: it rendered as the empty box that the comment defending the choice was
/// itself warning about, which reads as a broken feature rather than a missing character. Found by
/// running the list under a display for the first time, which is what `block-order-spec.md` §8.4
/// said to do before building anything else on this drag. Three filled rectangles cannot be
/// missing from any font, and this is the mark every list uses for the job anyway.
pub(crate) fn grip_mark(ui: &mut egui::Ui) {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(14.0, ui.spacing().interact_size.y),
        egui::Sense::hover(),
    );
    if !ui.is_rect_visible(rect) {
        return;
    }
    // Brighter under the pointer, because a grip that does not answer looks like decoration. The
    // cursor becomes a hand as well -- `dnd_drag_source` sets that on the response it returns.
    let colour = if response.contains_pointer() {
        ui.visuals().strong_text_color()
    } else {
        ui.visuals().weak_text_color()
    };
    for offset in [-3.5_f32, 0.0, 3.5] {
        ui.painter().rect_filled(
            egui::Rect::from_center_size(
                rect.center() + egui::vec2(0.0, offset),
                egui::vec2(10.0, 1.5),
            ),
            0.5,
            colour,
        );
    }
}

/// A block being dragged by its grip, and the block whose slot it was drawn in.
///
/// **The holder travels with the drag because the drop needs it** (`notebook-spec.md` §7.2):
/// taking a nested row out empties the slot it came from, and a bare `NodeId` cannot say which
/// block that was. A shared instance is drawn in every reader — `listed_blocks` clones its row per
/// owner — so "the slot it came from" is a fact about the *row* that was grabbed, and the row is
/// the only place that knows it.
///
/// **The holder, not the slot's index.** The index would be read a frame or more after it was
/// taken, and an edit arriving in between (a console line, another panel) would have it empty
/// whatever slot now sits at that number. The holder is an id, so it either still reads the block
/// or no longer does, and [`fieldform_scene::Node::stop_reading`] asks the graph at the moment of
/// the release. Nothing is carried that could go stale into a wrong edit.
///
/// `from` is `None` for a row at the top level, which came out of no slot: that drag reorders the
/// bodies, and it is the same payload because to the hand it is the same gesture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DraggedBlock {
    pub(crate) id: NodeId,
    pub(crate) from: Option<NodeId>,
}

/// The thin strip between two rows, which answers a drop as "put it here".
///
/// **Because the block's own frame cannot mean both things.** A drop means "before this row",
/// and once a block's frame contains the blocks it is made of, highlighting that frame says the
/// opposite -- goes inside. A line between two rows can only mean one thing, which is what an
/// insertion point is for. `block-order-spec.md` §9 carries the decision and §8.3 the bug that
/// made a separate end-of-list target necessary in the first place.
///
/// Invisible until there is something to drop, because a permanent line between every row is a
/// list full of separators nobody asked for. While a drag is in flight every gutter shows where
/// a row could land and the one under the pointer shows it brightly.
///
/// **`live` is whether this gutter answers the drag in flight, and it is not a property of the
/// gutter alone.** A body may only take the rank of another body, so its drop lands at a branch
/// start and nowhere else. A nested block has no rank to take — any drop on the list pulls it out
/// (`drop_on_the_list`) — so for *that* drag every gutter is a valid target. Keeping the branch
/// rule for both is what left a single live strip at the top of the list: with one boolean body
/// made of two unnamed blocks there was exactly one place in the whole panel that would accept a
/// drop, and every other gap looked identical and did nothing. Reported by the owner as "if you
/// drag one out you're basically stuck with the other one."
pub(crate) fn drop_gutter(ui: &mut egui::Ui, live: bool) -> Option<DraggedBlock> {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), GUTTER_HEIGHT),
        egui::Sense::hover(),
    );
    // **The space is taken whether or not the drop is offered**, so the rows sit at the same
    // pitch either way. A gutter that appeared only where it was live would move every row below
    // it the moment a drag started.
    if !live {
        return None;
    }
    if egui::DragAndDrop::has_payload_of_type::<DraggedBlock>(ui.ctx()) && ui.is_rect_visible(rect) {
        // `contains_pointer` rather than `hovered`, which is false for everything while
        // something is being dragged -- the same note egui's own drop zone carries.
        let under = response.contains_pointer();
        let line = egui::Rect::from_center_size(
            rect.center(),
            egui::vec2(rect.width(), if under { 3.0 } else { 1.0 }),
        );
        ui.painter().rect_filled(
            line,
            1.0,
            if under {
                ui.visuals().selection.bg_fill
            } else {
                ui.visuals().widgets.noninteractive.bg_stroke.color
            },
        );
    }
    response.dnd_release_payload::<DraggedBlock>().map(|dragged| *dragged)
}

/// Arranges the flat block list into the tree the panel draws.
///
/// **The order is already right; what this adds is depth.** `Document::listed_blocks` gathers each
/// branch contiguously (`block-order-spec.md` §8.2), so a block is always directly above the block
/// that consumes it and the rows arrive in the order this walk needs them.
/// [`fieldform_scene::Document::nesting_in`] says *which* block owns which, read from the stored
/// ownership, and carries the reasoning.
///
/// Built from the bottom up in one pass: a row is finished when the rows it holds have been
/// gathered, and post-order means they already have been.
pub(crate) fn nest_rows(
    flat: Vec<Block>,
    nesting: &fieldform_scene::Nesting,
    loop_of: &dyn Fn(CurveId) -> Vec<CurveId>,
) -> Vec<BlockRow> {
    // Where each block's row is, because a row may only be nested inside one that comes *after*
    // it. Post-order gives that for a graph edge, but a curve is merged into the list by when it
    // was made, and a file can hold a curve stamped later than the block that reads it. Checked
    // rather than assumed: the failure would be a row nobody can reach.
    let at: std::collections::HashMap<NodeId, usize> = flat
        .iter()
        .enumerate()
        .filter_map(|(i, block)| match block {
            Block::Node(id) => Some((*id, i)),
            _ => None,
        })
        .collect();
    // **Every block that draws it, not the one it belongs to** (`notebook-spec.md` R1, §7.3). A
    // block two blocks read is a shared instance: one block, drawn in full inside each of them,
    // so it comes back here as two owners and is cloned into both.
    let owners_of = |i: usize, block: &Block| -> Vec<NodeId> {
        let owners: Vec<NodeId> = match block {
            Block::Node(id) => nesting.owners_of(*id).to_vec(),
            Block::Curve(id) => nesting.curve_owners_of(*id).to_vec(),
            // **A profile row goes inside whatever reads *any* curve of its loop.**
            //
            // It used to ask about its key alone — the loop's first curve in walk order, which is
            // the curve the row is drawn under. An extrude names *one* curve of the loop, and
            // which one is whichever was picked, so when that was not the key the key had no
            // reader, the row had no owner, and a profile sat outside the extrude made from it
            // while one of its own curves sat inside. Reported as "I made two curves into a
            // profile, I then went to extrude the profile, the profile is no longer inside the
            // extrude" — and it was a coin flip on walk order, which is why it looked arbitrary.
            //
            // The loop is one thing, so it has one row — and since 112 that row is drawn in
            // every block that reads any of its curves, which is the same shared instance a block
            // gets. It used to take the *first* member with an owner, to pick one place
            // deterministically; a loop two blocks read now lands in both. `loop_of` is the
            // caller's because the walk lives in `fieldform-field` and `Nesting` is built in
            // `fieldform-scene`, which cannot reach it — the same reason `made_at` arrives as a
            // closure.
            Block::Profile(id) => {
                let mut owners: Vec<NodeId> = Vec::new();
                for member in loop_of(*id) {
                    for owner in nesting.curve_owners_of(member) {
                        if !owners.contains(owner) {
                            owners.push(*owner);
                        }
                    }
                }
                owners.sort();
                owners
            }
            // The rest are pinned to sections of their own — a parameter and a surface are inputs
            // at the top, an import is a row that already opens, an output is below everything it
            // reads — so there is no block for one of them to go inside. `App::ordered_blocks`
            // says why each is where it is.
            Block::Parameter(_)
            | Block::Surface(_)
            | Block::Import(_)
            | Block::Tessellation(_)
            | Block::Export(_) => Vec::new(),
        };
        // An owner further down the list holds this row; one that is not is not a row of this
        // list, and the block keeps a row of its own rather than vanishing into nothing.
        owners.into_iter().filter(|owner| at.get(owner).is_some_and(|at| *at > i)).collect()
    };

    // A Vec rather than a map, so the rows inside a block stay in the order the list had them and
    // two runs cannot disagree about it.
    let mut waiting: Vec<(NodeId, Vec<BlockRow>)> = Vec::new();
    let mut out: Vec<BlockRow> = Vec::new();
    for (i, block) in flat.iter().enumerate() {
        let inside = match block {
            Block::Node(id) => waiting
                .iter()
                .position(|(owner, _)| owner == id)
                .map(|k| waiting.remove(k).1)
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        let owners = owners_of(i, block);
        let readers = match block {
            Block::Node(id) => nesting.readers_of(*id),
            // A curve's row has the V too, so it needs the same count to know whether putting it
            // back has one block to go into.
            Block::Curve(id) => nesting.readers_of_curve(*id),
            // A profile row's V writes whichever curve of its loop puts the row where it is
            // (`profile_v_curve_in`), which takes the stored ownership this walk has no view of,
            // so the row counts that curve's readers from this same `Nesting` when it is drawn.
            _ => 0,
        };
        let references = match block {
            Block::Node(id) => nesting.references_of(*id).to_vec(),
            _ => Vec::new(),
        };
        let row = BlockRow { block: *block, inside, nested: !owners.is_empty(), readers, references };
        if owners.is_empty() {
            out.push(row);
        } else {
            // Cloned per owner, contents and all: §3 says a shared instance is drawn in full in
            // every reader — the same rows in each, not a summary in all but one.
            for owner in owners {
                let here = row.clone();
                match waiting.iter_mut().find(|(id, _)| *id == owner) {
                    Some((_, rows)) => rows.push(here),
                    None => waiting.push((owner, vec![here])),
                }
            }
        }
    }
    // Unreachable, because every owner is a row further down the list and its contents are
    // collected when that row is reached. Put back at the top level rather than dropped anyway: a
    // row in the wrong place is a bug somebody can see, and a row that vanished is a block they
    // cannot get to.
    out.extend(waiting.into_iter().flat_map(|(_, rows)| rows));
    out
}

/// The block list's profile rows: the closed loops the document's open curves make, found by
/// the walk an extrude uses and kept until a curve changes.
///
/// **Cached behind a revision counter rather than walked per frame.** `profile_partition`
/// samples every open curve for every seed, and the block list is drawn sixty times a second;
/// `App::curve_revision` moves whenever the curve display is rebuilt, which is every moment a
/// curve changes, and [`App::refresh_profiles`] re-walks only when it has. A document with no
/// open curves costs nothing either way.
///
/// `at` is `None` until the first refresh and after a document is replaced, which is how the
/// first partition of an opened file is computed without being *reported*: a loop that was in
/// the file when it opened did not close because of an edit, and saying so would be false.
#[derive(Default)]
pub(crate) struct ProfileRows {
    /// The `curve_revision` the loops were found at.
    pub(crate) at: Option<u64>,
    /// Each loop's curves in walk order; the first is its key.
    pub(crate) loops: Vec<Vec<CurveId>>,
}

/// What a curve's row becomes once loops are folded: its own row, the profile row it heads, or
/// nothing, because it is drawn inside a row headed by another curve.
///
/// **Free of the app because it carries an invariant worth asserting**, and because breaking that
/// invariant made a curve disappear: `profile_block` decided for itself which members to skip, read
/// the declarations directly, and so disagreed with this function about the loop's *key* — whose
/// row **is** the profile row, so declaring it cannot give it one of its own. The member list
/// skipped it, this said "the profile row has it", and the curve was drawn in neither place.
///
/// The invariant: for every curve, exactly one of — it has a top-level row, or it is drawn inside the
/// row that heads its loop. `every_curve_is_drawn_in_exactly_one_place` holds it.
///
/// `readers` is how many blocks read a curve, counted in every part as `Nesting::readers_of_curve`
/// counts them.
pub(crate) fn folded_curve_row_in(
    profiles: &ProfileRows,
    doc: &Document,
    readers: &dyn Fn(CurveId) -> usize,
    id: CurveId,
) -> Option<Block> {
    match profiles.key_of(id) {
        None => Some(Block::Curve(id)),
        // The key heads the row, so the row *is* its row. Tested before its ownership, because a
        // key that is a variable is the profile row being one, and still has nowhere else to go.
        Some(key) if key == id => Some(Block::Profile(key)),
        // **A member that is a variable comes out of the row, which is what makes its V mean
        // anything.** Reported: "when you have a profile of curves and you try to declare them as
        // variables they stay inside the profile." They did: every non-key member got `None`
        // regardless, so the declaration was read by `nesting_in` and by nothing that decides where
        // a member's row goes.
        //
        // **But only one nothing reads.** A block reading a member reads the loop through it — an
        // extrude names one curve and walks the rest — so the member goes where the loop's row
        // goes, and the profile row's V is what moves it (`profile_v_curve_in`). Two blocks
        // reading one make it a variable without anybody asking for a row, and the key, which
        // heads the row, could never come out for that reason: lifting a read member out split the
        // loop exactly when the blocks were made with a curve other than the first selected, and
        // opened a file from before ownership was stored with its list changed. The curve's own V
        // is the loop's while a block reads it (`curve_v_elsewhere`), so the only variables this
        // takes out are the ones that V made.
        Some(_)
            if readers(id) == 0
                && doc.ownership(BlockKey::Curve(id)) == Ownership::Variable =>
        {
            Some(Block::Curve(id))
        }
        Some(_) => None,
    }
}

/// The curve of a loop whose ownership the profile row's V reads and writes: the one that puts
/// the row where it is.
///
/// **Not the key**, which is what the row is drawn under and was what the V wrote. An extrude
/// names whichever curve was selected, and [`nest_rows`] draws the row inside whatever holds any
/// curve of its loop — the first, in walk order, that a block of this part owns. So that curve is
/// the first answer, read from the same `nesting`, and making it a variable is what takes the row
/// out. With none owned the row already stands on its own, and the V's state is that of the first
/// variable a block reads, which is what putting it back moves; failing that the key's, so a loop
/// nothing reads is made a variable as a body is, in place. A variable nothing reads that is not
/// the key is a row of its own ([`folded_curve_row_in`]), not this row's.
///
/// One curve, though two blocks may each hold a different curve of one loop (loop members are not
/// adopted, `block-ownership-spec.md` §10): the row moves to the other block, and the V moves it
/// again from there.
pub(crate) fn profile_v_curve_in(
    doc: &Document,
    nesting: &fieldform_scene::Nesting,
    members: &[CurveId],
) -> Option<CurveId> {
    let variable = |id: CurveId| doc.ownership(BlockKey::Curve(id)) == Ownership::Variable;
    members
        .iter()
        .copied()
        .find(|id| nesting.curve_is_nested(*id))
        .or_else(|| {
            let read = |id: &CurveId| nesting.readers_of_curve(*id) > 0;
            members.iter().copied().find(|id| variable(*id) && read(id))
        })
        .or(members.first().copied())
}

/// Why a curve's own V is not the one that moves it, or `None` when it is: the hover its V shows
/// instead, and its row menu's Variable item, greyed, the same.
///
/// **The loop's, for every curve drawn in its row that it could move.** The curve heading a loop
/// has the profile row for its row, so declaring it cannot give it one of its own, and a click
/// there was a click that did nothing — what sent somebody looking for a curve that had vanished
/// (`profile-group-spec.md` §11.2). A member a block reads is the loop to that block and stays in
/// the row ([`folded_curve_row_in`]), so its V could only do what the profile row's does, as a
/// second handle on it. A member nothing reads keeps its own: making it a variable gives it a row
/// of its own, and putting it back returns it to the loop.
///
/// `readers` is how many blocks read `id`, counted in every part.
pub(crate) fn curve_v_elsewhere(
    profiles: &ProfileRows,
    id: CurveId,
    readers: usize,
) -> Option<&'static str> {
    let key = profiles.key_of(id)?;
    (key == id || readers > 0).then_some(CURVE_V_IS_THE_LOOP_S)
}

/// What a curve's V says, and its row menu's Variable item, where the V that moves it is its
/// profile row's ([`curve_v_elsewhere`]).
pub(crate) const CURVE_V_IS_THE_LOOP_S: &str =
    "Drawn in its loop's row, which is where it goes: the V on the profile row above is the one \
     that moves the loop, making it a variable or putting it back.";

/// The point blocks a loop's ends sit on, in the order the curves are walked. A point a
/// curve names but the part no longer has is left out.
pub(crate) fn joint_points_in(doc: &Document, part: PartId, members: &[CurveId]) -> Vec<NodeId> {
    let graph = doc.parts.get(&part).map(|p| &p.graph);
    let mut out = Vec::new();
    for id in members {
        let Some(c) = doc.curve(*id) else { continue };
        let n = c.control_points().len();
        if n < 2 {
            continue;
        }
        for i in [0, n - 1] {
            if let Some(p) = c.placed_on(i)
                && !out.contains(&p)
                && graph.is_some_and(|g| g.node(p).is_some())
            {
                out.push(p);
            }
        }
    }
    out
}

/// Every held loop's joint points in `part`: drawn inside their profile row (§4.3), never as rows
/// of their own. One set for the list and the console, so the two cannot disagree about which
/// points have a row.
pub(crate) fn held_joint_points(
    doc: &Document,
    part: PartId,
    profiles: &ProfileRows,
) -> std::collections::HashSet<NodeId> {
    profiles.loops.iter().flat_map(|l| joint_points_in(doc, part, l)).collect()
}

/// The blocks of `part` a console line may call by name or label, oldest first: those the list
/// gives a row, since `variable` and `inline` stand for the V on one.
///
/// **A held loop's joints are left out**, as the list leaves them out ([`held_joint_points`]).
/// They are unnamed points with no V, so counted here they made `variable 'Point'` ambiguous
/// beside the one point the list shows, or made a joint a variable where it was the only one.
pub(crate) fn blocks_a_line_can_call(doc: &Document, part: PartId, profiles: &ProfileRows) -> Vec<NodeId> {
    let Some(graph) = doc.parts.get(&part).map(|p| &p.graph) else { return Vec::new() };
    let joints = held_joint_points(doc, part, profiles);
    let mut blocks: Vec<NodeId> = graph.nodes().map(|(id, _)| id).filter(|id| !joints.contains(id)).collect();
    blocks.sort_by_key(|id| (doc.made_at(*id), id.raw()));
    blocks
}

/// The curve ends sitting on a point, in curve-creation order. A joint has exactly two.
pub(crate) fn ends_on_point(doc: &Document, point: NodeId) -> Vec<fieldform_scene::CurveEnd> {
    let mut curves: Vec<&fieldform_scene::Curve> = doc.curves.values().collect();
    curves.sort_by_key(|c| (doc.curves_created.get(&c.id).copied().unwrap_or(0), c.id.raw()));
    let mut out = Vec::new();
    for c in curves {
        let n = c.control_points().len();
        if n < 2 {
            continue;
        }
        if c.placed_on(0) == Some(point) {
            out.push(fieldform_scene::CurveEnd { curve: c.id, end: fieldform_scene::End::Start });
        }
        if c.placed_on(n - 1) == Some(point) {
            out.push(fieldform_scene::CurveEnd { curve: c.id, end: fieldform_scene::End::End });
        }
    }
    out
}

/// The joint relation between two ends, whichever leads, if there is one.
pub(crate) fn joint_relation_between(
    doc: &Document,
    a: fieldform_scene::CurveEnd,
    b: fieldform_scene::CurveEnd,
) -> Option<(fieldform_scene::RelationId, fieldform_scene::Relation)> {
    doc.relations.iter().find_map(|(id, r)| match &r.relation {
        fieldform_scene::Relation::Joint { master, slave, .. }
            if (*master == a && *slave == b) || (*master == b && *slave == a) =>
        {
            Some((*id, r.relation.clone()))
        }
        _ => None,
    })
}

/// What a corner offers: none, then G1 and G2 with either curve leading, each as the
/// sentence the console would say for it.
pub(crate) fn joint_choices(
    doc: &Document,
    a: fieldform_scene::CurveEnd,
    b: fieldform_scene::CurveEnd,
) -> Vec<(String, Option<fieldform_scene::Relation>)> {
    let name =
        |e: fieldform_scene::CurveEnd| doc.curve(e.curve).map(|c| c.name.clone()).unwrap_or_default();
    // **"G0 — positional only", not "none", because that is what it is.** A joint point holds two
    // ends at the same place and nothing more, which *is* G0; `JointContinuity` carries only G1 and G2
    // because G0 is the absence of a relation rather than one of its kinds. Labelling it "none" read
    // as "no corner treatment at all" and hid a base feature the owner asked for as if it were
    // missing: "I should be able to click on a join, right click on it and set it as G0, meaning
    // positional. That's kind of as a base feature."
    let mut out = vec![("G0 \u{2014} positional only".to_string(), None)];
    for (g, continuity) in [
        ("G1", fieldform_scene::JointContinuity::G1),
        ("G2", fieldform_scene::JointContinuity::G2),
    ] {
        for (master, slave) in [(a, b), (b, a)] {
            out.push((
                format!("{g}: {} follows {}", name(slave), name(master)),
                Some(fieldform_scene::Relation::Joint { master, slave, continuity }),
            ));
        }
    }
    out
}

/// The joint points that go unseen with their row: those of every row in `rows` whose curves
/// are all in `hidden`.
///
/// Derived from the curves rather than written onto the point. The row's eye is the one
/// switch, and a point marked hidden on its own account would stay hidden after its curves
/// came back -- a reopen keeps a body's flag and not a curve's. Before this rule, hiding a
/// held loop left its corners on screen as three dots: visible, and not pickable either, a
/// point being an overlay rather than a traced surface. A joint holds exactly one row, since
/// none is made where three ends meet, so "its row" is well defined.
pub(crate) fn points_hidden_with_rows(
    doc: &Document,
    part: PartId,
    hidden: &std::collections::HashSet<CurveId>,
    rows: &[Vec<CurveId>],
) -> Vec<NodeId> {
    rows.iter()
        .filter(|members| !members.is_empty() && members.iter().all(|id| hidden.contains(id)))
        .flat_map(|members| joint_points_in(doc, part, members))
        .collect()
}

impl ProfileRows {
    /// The key of the row `curve` is in, if it is in one.
    pub(crate) fn key_of(&self, curve: CurveId) -> Option<CurveId> {
        self.loops.iter().find(|l| l.contains(&curve)).map(|l| l[0])
    }

    /// The row's curves in walk order, or `None` when `key` heads no row.
    pub(crate) fn members(&self, key: CurveId) -> Option<&[CurveId]> {
        self.loops.iter().find(|l| l[0] == key).map(|l| l.as_slice())
    }

    /// The row's number, counted from one in list order — what "Profile 2" means.
    pub(crate) fn number(&self, key: CurveId) -> Option<usize> {
        self.loops.iter().position(|l| l[0] == key).map(|i| i + 1)
    }
}
