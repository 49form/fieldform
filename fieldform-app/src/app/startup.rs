// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! Building the application: its first document, its settings and its GPU resources.

use crate::*;

impl App {
    pub(crate) fn new(cc: &eframe::CreationContext) -> Self {
        let wgpu_render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("fieldform-app requires the wgpu backend (eframe's \"wgpu\" feature)")
            .clone();

        // **A lost GPU names itself in the crash log, before anything else trips over it.**
        //
        // Windows resets a display driver whose work runs longer than about two seconds on one
        // packet (TDR), and a Maxwell-era card cannot preempt a compute dispatch, so a heavy
        // frame on one is killed rather than paused. wgpu then marks the device invalid and
        // every later call fails -- and the first to notice, on 2026-09-15 on a Quadro M4000,
        // was egui-wgpu asking for a staging buffer for its own index data, which panicked
        // about buffer sizes that were fine. That report went looking at the wrong crate. This
        // writes the real cause first, so the next one reads "GPU lost" rather than "egui".
        // And it says where the work is, since the next thing the reader of that note wants is
        // their work rather than the reason.
        //
        // **It does not write the document itself.** wgpu runs this on whichever thread made
        // the call that found the device gone -- a shader compile on a worker as readily as a
        // submit here -- and the document belongs to the interface thread, behind no lock. So
        // it raises `gpu_lost` and asks for a frame, and the interface thread keeps the
        // document (`App::keep_the_document_if_the_gpu_was_lost`) and adds where to this note.
        let gpu_lost = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let raised = std::sync::Arc::clone(&gpu_lost);
        let interface = cc.egui_ctx.clone();
        wgpu_render_state.device.set_device_lost_callback(move |reason, message| {
            let said = format!(
                "GPU device lost ({reason:?}): {message}\n\
                 Windows resets a display driver that takes longer than about 2 s on one \
                 frame, and a heavy document on a slow or old GPU can do that. Every GPU call \
                 after this fails, so the application aborts on the next one.\n\
                 The interface keeps the document as it stands in <name>.recovered.\
                 {DOCUMENT_EXTENSION} beside its file, or beside this log if it was never saved, \
                 and says where below if a frame is left to do it in. Failing that, the document \
                 as of the last minute is at {}, which `open file=` reads.",
                autosave_path().display()
            );
            log::error!("{said}");
            note_in_crash_log("fieldform GPU lost", &said);
            raised.store(true, std::sync::atomic::Ordering::Release);
            interface.request_repaint();
        });

        // **Which GPU is actually doing the work, said once at startup.**
        //
        // This machine has six adapters — an AMD 780M and an NVIDIA RTX A500, each through both
        // Vulkan and DX12, plus a software rasteriser and a GL fallback — and wgpu's choice
        // among them is not guaranteed to be the intended one. Windows' own graphics preference
        // can override it silently, which means a timing that halves after a driver update looks
        // like a regression in this code and is not.
        //
        // `shading-spec.md` §10.8 asks for exactly this and calls it the cheapest of three
        // things worth doing, because every number measured on this application is ambiguous
        // without it. egui_wgpu already logs the adapters it *found*; nothing said which one it
        // took.
        // **Every button wears an outline, so a button looks like one before it is clicked.**
        //
        // egui draws an inactive widget with no stroke by default, which makes a toggle read as
        // a label until the pointer is over it — reported of the box's Ratio toggle, and true of
        // every mode switch in every block. The rule is one grey outline when idle and the
        // accent when the thing it names is on, said here once rather than per widget: a style
        // set at the widget is a style the next widget will not have.
        //
        // `inactive` is the resting state and `open` is a menu button whose menu is up; both get
        // the grey. `hovered` and `active` keep egui's own brighter strokes, which are the
        // feedback that the pointer is on it. `selected` is the blue.
        let mut style = (*cc.egui_ctx.global_style()).clone();
        {
            let w = &mut style.visuals.widgets;
            let outline = egui::Stroke::new(1.0_f32, egui::Color32::from_gray(90));
            // **`inactive` and `open` only.** `noninteractive.bg_stroke` is not a button's at
            // all -- it is the stroke every *frame* uses, so the outline around each block, the
            // divider between the viewport and the block list, and the one between Display and
            // the console are all drawn with it. Clearing it to make buttons the only outlined
            // thing removed every one of those instead. Reported as the UI losing its lines.
            w.inactive.bg_stroke = outline;
            w.open.bg_stroke = outline;
            // **`selection.stroke` is left alone: it is the selected *text*, not an outline.**
            // `Style::interact_selectable` assigns it to `fg_stroke`, so setting it to the
            // selection's own fill painted blue text on a blue background and every selected
            // row and toggle went blank. Reported as text disappearing on click. egui already
            // fills a selected widget with the accent, which is the whole of what "blue when it
            // is on" needs.
        }
        cc.egui_ctx.set_global_style(style);

        let info = wgpu_render_state.adapter.get_info();
        log::info!(
            "rendering on {} ({:?}, {:?}) — driver {} {}",
            info.name,
            info.device_type,
            info.backend,
            info.driver,
            info.driver_info,
        );
        // **And in the console, where somebody can see it.** The log line above goes to stderr,
        // which a windowed application launched from Explorer does not have — so the one fact
        // that makes every timing in this application interpretable was written somewhere nobody
        // would ever read. A laptop with two GPUs can be quietly drawing on the slower one, and
        // the symptom is "it feels sluggish", which is indistinguishable from the model being
        // heavy. Said once, at startup, beside everything else the application says.
        let adapter_said = format!(
            "Drawing on {} ({}), {:?} — driver {} {}; GPU timestamps {}",
            info.name,
            match info.device_type {
                wgpu::DeviceType::DiscreteGpu => "discrete",
                wgpu::DeviceType::IntegratedGpu => "integrated",
                wgpu::DeviceType::VirtualGpu => "virtual",
                wgpu::DeviceType::Cpu => "software — expect it to be very slow",
                wgpu::DeviceType::Other => "unknown kind",
            },
            info.backend,
            info.driver,
            info.driver_info,
            // Whether `perf` will report the card's own time or the wall clock's. Said here
            // because it is a fact about the device the line already describes, and the
            // ledger wants the two together.
            if wgpu_render_state
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
            {
                "on"
            } else {
                "unavailable, so `perf` times by the wall clock"
            },
        );

        // The same size as anything the Create menu makes, so the first thing on screen and
        // the second are comparable. It was a metre for a while — real parts arrive at real
        // sizes, and starting three orders of magnitude below anything importable made every
        // scale-relative default wrong on arrival — but that argument was about *not* starting
        // at 1 mm, and 100 mm satisfies it while matching everything else.
        let cage = Cage::cube_of(DEFAULT_BODY_MM);
        let cage_mesh = CageMesh::from_cage(&cage);
        let mut doc = Document::with_cage(cage.clone());
        let active_cage = *doc.cages.keys().next().expect("with_cage inserts one");
        let active_part = *doc.parts.keys().next().expect("with_cage inserts one");

        // Open on an empty part, not on a cube nobody asked for.
        //
        // The cage itself stays in the document; only the body that draws it is taken out. That
        // is the same state deleting the cage block already produces, and it is what keeps
        // `active_cage` naming something real — the app leans on that everywhere — while
        // leaving the viewport empty for the splash to sit over. Create > Cage puts *this*
        // cage back rather than making a second one, so nothing has been lost by not drawing
        // it.
        doc.parts
            .get_mut(&active_part)
            .expect("with_cage inserts one")
            .graph = Graph::empty();
        // Settled as every graph written past the history is (`block-ownership-spec.md` §4.1),
        // as `new_document` does after the same write.
        doc.settle_ownership();
        starting_look(&mut doc, active_part);

        // A test's `App` draws on Fast whatever the owner chose: the tests are about what the
        // interface does, and the owner's own rung would make them as slow as Enchilada is.
        let saved_quality = if cfg!(test) { Some(DisplayQuality::Fast) } else { display_quality::load_saved_quality() };
        let (start_quality, start_chosen) = starting_quality(
            saved_quality,
            info.device_type == wgpu::DeviceType::DiscreteGpu,
        );

        let echo_base = app::echoing::EchoBase {
            snapshot: echo::Snapshot::of(&doc, &doc.parts[&active_part].graph),
            part: active_part,
            moves: 0,
        };
        let mut app = Self {
            egui_ctx: cc.egui_ctx.clone(),
            console: ConsoleState {
                echo_base,
                echo_log: Vec::new(),
                progress: None,
                input: String::new(),
                log: vec![ConsoleLine {
                    ok: true,
                    text: "Type a command, or `help`. Everything here is the same operation a menu \
                           item runs."
                        .to_string(),
                    said: 1,
                }],
                reports: 0,
                warnings: 0,
                line_capture: None,
                line_error: None,
                cage_line: None,
                traced_error_said: None,
                // **Never from a test's `App`.** The file is one name in the temp folder and this
                // truncates it, so a test run would empty the log of whatever instance the owner has
                // open (`tests::headless_app`).
                file: if cfg!(test) { None } else { open_console_log() },
                file_bytes: 0,
                history: Vec::new(),
                history_at: None,
                complete: false,
            },
            tools: ToolState {
                session: None,
                last_values: std::collections::HashMap::new(),
                palette: None,
                recent: Vec::new(),
                marking: None,
                join_preview: None,
                last_command: None,
                shelf: Vec::new(),
                shelf_drawn: Vec::new(),
                floating_panel: false,
                focus_first_number: false,
                drawing_pending: false,
                keep_pending: false,
                pending_numbers: Vec::new(),
                palette_opens: 0,
            },
            keys: KeyState {
                bindings: Vec::new(),
                conflicts: Vec::new(),
                capturing: None,
                note: None,
                keymap: commands::default_keymap(),
                stolen_tab: None,
            },
            notebook: NotebookState {
                last_reference_field: None,
                renaming: None,
                reveal: None,
                followed: None,
                curve_picked: None,
                graph_followed: None,
                ref_edit: None,
                completing_a_move: None,
                hosted: None,
                naming_parameter: None,
                name_drafts: std::collections::HashMap::new(),
                size_locked: std::collections::HashSet::new(),
                collecting: None,
                expression_drafts: std::collections::HashMap::new(),
            },
            viewport_rect: None,
            doc,
            active_cage,
            control: ControlMesh::Cage,
            active_part,
            saving: SaveState {
                file: None,
                revision: 0,
                saved_revision: 0,
                closing: false,
                discard_prompt: None,
                discard_answered: false,
                file_picker: None,
                autosaved_revision: 0,
                // From launch rather than from the first edit, so a document is not written the
                // instant it is touched — which on a heavy one is a stall for no gain, the file
                // having just been opened.
                autosaved_at: std::time::Instant::now(),
            },
            document_moves: 0,
            kept: Kept {
                readings: Default::default(),
                list: Default::default(),
                statuses: Default::default(),
            },
            #[cfg(test)]
            drawn_frames: Vec::new(),
            #[cfg(test)]
            open_frames: Vec::new(),
            title: String::new(),
            splash: None,
            splash_shown: true,
            show_shortcuts: false,
            show_about: false,
            check: CheckState {
                window_open: false,
                only_errors: false,
                surface_check: None,
                amber_until: None,
                due: std::time::Instant::now(),
            },
            cage_tool: None,
            cage_session: None,
            infill_cavities: std::collections::HashMap::new(),
            drawing_plane: DrawingPlane::None,
            agent_link: AgentLink {
                state: Default::default(),
                host: None,
                window_hidden: false,
                allowed: false,
                endpoint: None,
            },
            scripting: ScriptState {
                script: Script::from_command_line(),
                running: false,
                expect_results: Vec::new(),
                tally: None,
            },
            pending_snap: None,
            images: image_block::ImageRuns::default(),
            comb_runs: comb_block::CombRuns::default(),
            written: WrittenState {
                verdicts: std::collections::HashMap::new(),
                drafts: std::collections::HashMap::new(),
                errors: std::collections::HashMap::new(),
                notices: std::collections::HashMap::new(),
            },
            started: std::time::Instant::now(),
            stack: CommandStack::default(),
            registry: OpRegistry::default(),
            wgpu_render_state,
            camera: OrbitCamera::default(),
            cage_mesh,
            imported_mesh: CageMesh::default(),
            view: ViewSettings {
                // **Field, because that is what the model looks like.** Polygon draws the
                // control cage and nothing else, which is right while editing a cage and wrong as
                // a starting point: the cage is scaffolding, and a part built from primitives has
                // no cage at all, so the first thing a new user saw was an empty viewport with no
                // hint that a mode was the reason. Every Create command already switches here, so
                // this only changes what is true before the first one.
                mode: ViewMode::Field,
                show_view_cube: true,
                idle_occlusion: true,
                idle_reflections: true,
                show_grid: true,
                // Horizon, with the starting part in chrome (`starting_look`): the owner's choice of
                // first impression, 2026-11-07.
                environment: fieldform_gpu::EnvironmentChoice::Horizon,
                draft: DraftAnalysis::default(),
                // The rung chosen last session if there is one; otherwise a guess from the adapter,
                // corrected once by the first draft it measures: a discrete card is offered
                // everything and demoted if its first draft misses the budget, and anything else
                // starts in the middle.
                quality: start_quality,
                quality_chosen: start_chosen,
            },
            perf: PerfState {
                stages: Vec::new(),
                adapter_line: adapter_said.clone(),
                file: None,
                frame_cpu_ms: std::collections::VecDeque::new(),
            },
            selection: Picked::default(),
            hover: Selection::None,
            move_state: None,
            curve_ui: CurveState {
                point_drag: None,
                pending: None,
                surface_axis_notes: std::collections::HashMap::new(),
                surface_axis_edit_before: None,
                other_points: Vec::new(),
                pending_narrow: None,
                marquee: None,
                show_control_points: true,
                edit_before: None,
                weight_edit_before: None,
                display: Vec::new(),
                revision: 0,
                profiles: ProfileRows::default(),
                loft_readings: None,
                pending_joins: Vec::new(),
                hidden: std::collections::HashSet::new(),
            },
            pending_hole: None,
            surface_points: SurfacePointState {
                states: std::collections::HashMap::new(),
                edit_before: None,
                slide: None,
            },
            presses: Presses {
                surface_point: false,
                gizmo: false,
                cage_tool: false,
                control_point: false,
                marquee: false,
                curve: false,
                selection: false,
            },
            sketch_projection_before: None,
            gizmo_drag: None,
            gizmo_hover: None,
            clipboard: None,
            viewport_hover: None,
            viewport_menu_at: None,
            selected_curve: None,
            graph_view: graph_view::GraphView::default(),
            selected_points: Vec::new(),
            next: NextBlock {
                // 5 mm is a plausible starting pipe for hardware-scale work and is visible at
                // the default framing; it is a starting point, not a constraint.
                pipe_radius: ParamRef::Literal(5.0),
                boolean_a: None,
                boolean_b: None,
                boolean_op: BooleanOp::Difference,
                blend_mode: None,
                blend_radius: ParamRef::Literal(2.0),
                export_detail: fieldform_scene::MeshDetail::Normal,
                export_simplify: false,
                export_relax: true,
                pattern_count: 6,
                pattern_radial: true,
                pattern_step_mm: DEFAULT_BODY_MM * 0.5,
                twist_axis: Axis::Z,
                mirror_axis: Axis::X,
                offset_distance: 1.0,
                shell_thickness: 2.0,
                // A shell's own starting wall, since it is the same question asked of a curve.
                wall_thickness: 2.0,
            },
            selected_body: None,
            selected_block: None,
            hidden_surfaces: std::collections::HashSet::new(),
            surface_display: Vec::new(),
            surface_prints: Vec::new(),
            traced: TracedView::default(),
            pick_field: None,
            // 64 cells across the longest axis: fast enough to feel interactive on this
            // brute-force field evaluator, fine enough to read the shape.
            mesh_resolution: 64,
            importing: ImportState {
                // Metres, because most DCC tools write most of the OBJs that reach a modeller and
                // treat one unit as one metre, and their default cube is 1 unit. A CAD exporter
                // more often means millimetres, which is why this is a visible setting and not a
                // constant.
                obj_unit: LengthUnit::Metre,
                prompt: None,
                obj_as: ObjReading::Cage,
                step_options: fieldform_io::StepOptions::default(),
                pending: None,
            },
            section_gesture: None,
            section_selected: None,
            gesture: None,
            verification: None,
            material_gesture: None,
            showing_original: std::collections::HashSet::new(),
            leaves: fieldform_field::LeafCache::default(),
            compile_error: None,
            draft_refused: None,
            parameter_edits: ParameterEdits::default(),
            outputs: Default::default(),
            gpu_lost,
            kept_after_loss: false,
        };
        app.start_agent_host(&cc.egui_ctx);
        if !cfg!(test) && app.agent_link.host.is_none() && display_quality::load_saved_allow_agent() {
            app.set_allow_agent(true, &cc.egui_ctx);
        }
        // Before anything else it says, because it is the frame of reference for all of it.
        app.say(adapter_said);
        // A kept rung is said once, because it is the one setting that outlives a session: a
        // still that looks coarse for a reason chosen weeks ago should say where the reason is.
        if let Some(saved) = saved_quality {
            app.say(format!(
                "Display quality: {}, as chosen in an earlier session. View > Display quality \
                 changes it.",
                saved.label()
            ));
        }
        // A test's `App` offers nothing it did not make: the temp folder is the owner's.
        if !cfg!(test) {
            app.offer_the_autosave();
        }
        app.rebuild_render_resources();
        // Frame the starting cage rather than trusting the default distance, which is fitted
        // to a default-sized body and so is right only for one. A real cage is whatever size
        // it is.
        let (center, radius) = cage_bounds(app.cage());
        app.camera.frame(center, radius);
        // And a starting pipe radius that suits the starting cage, for the same reason the
        // imported one is derived from the imported curve: a radius unrelated to the model is
        // a radius the mesher cannot resolve.
        app.next.pipe_radius = ParamRef::Literal(tidy_mm(radius as f64 * 0.05));
        app.next.blend_radius = ParamRef::Literal(tidy_mm(radius as f64 * 0.01));
        // The document a session opens on is said already: the echo starts from it.
        app.echo_rebase();
        // The keys: the registry's, with the person's file over them. Not in a test, whose keys
        // are the defaults whatever file the owner keeps.
        app.rebuild_keymap();
        if !cfg!(test) {
            app.load_bindings();
        }
        app.load_shelf();
        app
    }
}
