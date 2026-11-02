// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The properties `mcp-server-spec.md` §11 names for the agent host (T2, T3, T9, T10, T12 and the
//! path rules at the level of a line). Run on a headless `App`, so a missing GPU skips them as it
//! does the other `App` tests.

use super::*;
use crate::app_tests::headless_app;

fn run(app: &mut App, lines: &str) -> RunResponse {
    app.agent_run(&RunRequest { lines: lines.to_string(), ..Default::default() })
}

fn root_in_temp(app: &mut App, tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("fieldform-agent-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let dir = dir.canonicalize().unwrap();
    app.agent.roots = vec![dir.clone()];
    dir
}

#[test]
fn a_refused_line_takes_the_whole_call_back() {
    let Some(mut app) = headless_app() else { return };
    run(&mut app, "sphere radius=5 'keep'");
    let before_nodes = app.part_graph().nodes().count();
    let history = app.stack.history();
    let revision = app.revision;
    let out = run(&mut app, "sphere radius=1\ncube size=2,2,2\ncube side=3\nsphere radius=9\nsphere radius=8");
    assert!(!out.ok && !out.kept);
    assert_eq!(out.lines.len(), 3, "it stops at the refusal");
    assert_eq!(app.part_graph().nodes().count(), before_nodes);
    assert_eq!(app.stack.history(), history, "the history, redo list included, is as found");
    assert_eq!(app.revision, revision);
}

#[test]
fn one_call_is_one_undo_entry() {
    let Some(mut app) = headless_app() else { return };
    let before = app.stack.history().0.len();
    let out = run(&mut app, "sphere radius=1 'a'\nsphere radius=2 'b'\nsphere radius=3 'c'\nsphere radius=4 'd'");
    assert!(out.ok && out.kept, "{:?}", out.error);
    let (done, _) = app.stack.history();
    assert_eq!(done.len(), before + 1);
    assert!(done.last().unwrap().starts_with("agent: sphere"));
    assert_eq!(app.part_graph().nodes().count(), 4);
    let (names, _) = app.agent_undo(1, false, false).unwrap();
    assert_eq!(names.len(), 1);
    assert_eq!(app.part_graph().nodes().count(), 0, "one undo takes the whole call back");
}

#[test]
fn a_stale_revision_is_refused() {
    let Some(mut app) = headless_app() else { return };
    run(&mut app, "sphere radius=1");
    let nodes = app.part_graph().nodes().count();
    let out = app.agent_run(&RunRequest {
        lines: "sphere radius=2".into(),
        expected_revision: Some(app.revision.wrapping_add(7)),
        ..Default::default()
    });
    assert_eq!(out.error.map(|e| e.code), Some("stale_revision"));
    assert_eq!(app.part_graph().nodes().count(), nodes, "a stale call edits nothing");
}

#[test]
fn an_agent_does_not_undo_a_persons_edit_unasked() {
    let Some(mut app) = headless_app() else { return };
    app.run_command("sphere radius=3");
    let err = app.agent_undo(1, false, false).unwrap_err();
    assert_eq!(err.code, "not_agents_step");
    assert_eq!(app.part_graph().nodes().count(), 1, "refused, so nothing was undone");
    assert!(app.agent_undo(1, false, true).is_ok(), "force takes it back");
}

#[test]
fn a_cancelled_call_is_taken_back() {
    let Some(mut app) = headless_app() else { return };
    app.agent.cancel.store(false, std::sync::atomic::Ordering::SeqCst);
    // A zero timeout expires before the first line, which is the same path a cancellation takes.
    let out = app.agent_run(&RunRequest {
        lines: "sphere radius=1\nsphere radius=2".into(),
        timeout: Some(std::time::Duration::ZERO),
        ..Default::default()
    });
    assert!(!out.ok && !out.kept);
    assert_eq!(out.error.map(|e| e.code), Some("timeout"));
    assert_eq!(app.part_graph().nodes().count(), 0);
}

#[test]
fn a_verb_that_ends_the_application_is_refused_to_an_agent() {
    let Some(mut app) = headless_app() else { return };
    let out = run(&mut app, "exit");
    assert_eq!(out.lines[0].outcome.error.as_ref().map(|e| e.code), Some("agent_refused"));
}

#[test]
fn a_line_naming_a_file_outside_the_roots_is_refused_and_inside_is_not() {
    let Some(mut app) = headless_app() else { return };
    let root = root_in_temp(&mut app, "guard");
    run(&mut app, "sphere radius=5 'ball'");
    let out = run(&mut app, "saveas file=../escape.fform");
    assert_eq!(out.lines[0].outcome.error.as_ref().map(|e| e.code), Some("path_outside_root"), "{:?}", out.lines[0].outcome);
    assert!(!root.parent().unwrap().join("escape.fform").exists());
    let out = run(&mut app, "saveas file=part.fform");
    assert!(out.ok, "{:?}", out.lines[0].outcome);
    assert!(root.join("part.fform").exists());
    let again = run(&mut app, "saveas file=part.fform");
    assert_eq!(again.lines[0].outcome.error.as_ref().map(|e| e.code), Some("would_overwrite"));
    let allowed = app.agent_run(&RunRequest {
        lines: "saveas file=part.fform".into(),
        allow_overwrite: true,
        ..Default::default()
    });
    assert!(allowed.ok);
    let _ = std::fs::remove_dir_all(&root);
}

fn call(app: &mut App, tool: &str, args: serde_json::Value) -> tools::Reply {
    app.agent_handle(tool, args.as_object().expect("arguments are an object"))
}

#[test]
fn every_console_verb_is_listed_and_described_to_an_agent() {
    let Some(mut app) = headless_app() else { return };
    let listed = call(&mut app, "list_commands", serde_json::json!({}));
    let verbs: Vec<String> = listed.json["commands"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["verb"].as_str().unwrap().to_string())
        .collect();
    for (word, _) in crate::commands::console_words() {
        let described = call(&mut app, "describe_command", serde_json::json!({ "verb": word }));
        assert!(!described.is_error, "`{word}` is typed but cannot be described: {}", described.json);
        assert!(verbs.contains(&described.json["verb"].as_str().unwrap().to_string()), "`{word}` is not listed");
    }
    let sphere = call(&mut app, "describe_command", serde_json::json!({ "verb": "sphere" }));
    let radius = sphere.json["keys"].as_array().unwrap().iter().find(|k| k["key"] == "radius").unwrap();
    assert_eq!((radius["type"].as_str(), radius["default"].as_f64()), (Some("number"), Some(50.0)));
    let refused = call(&mut app, "describe_command", serde_json::json!({ "verb": "sphre" }));
    assert!(refused.is_error);
    assert_eq!(refused.json["error"]["did_you_mean"], "sphere");
}

#[test]
fn run_answers_with_what_each_line_did_and_a_failure_is_a_normal_result() {
    let Some(mut app) = headless_app() else { return };
    let ok = call(&mut app, "run", serde_json::json!({ "lines": "sphere radius=10 'ball'" }));
    assert!(!ok.is_error, "{}", ok.json);
    assert_eq!(ok.json["lines"][0]["created"][0]["name"], "ball");
    let revision = ok.json["revision_after"].as_u64().unwrap();
    let bad = call(&mut app, "run", serde_json::json!({ "lines": "cube sise=3", "expected_revision": revision }));
    assert!(bad.is_error);
    assert_eq!(bad.json["lines"][0]["error"]["code"], "unknown_key");
    assert_eq!(bad.json["lines"][0]["error"]["did_you_mean"], "size");
    let info = call(&mut app, "session_info", serde_json::json!({}));
    assert_eq!(info.json["document"]["revision"].as_u64(), Some(revision), "the refused call left the revision alone");
    let doc = call(&mut app, "get_document", serde_json::json!({}));
    assert_eq!(doc.json["parts"][0]["blocks"].as_array().unwrap().len(), 1);
}

#[test]
fn an_unknown_tool_is_a_tool_failure_not_a_crash() {
    let Some(mut app) = headless_app() else { return };
    let r = call(&mut app, "frobnicate", serde_json::json!({}));
    assert!(r.is_error);
    assert_eq!(r.json["error"]["code"], "not_available");
}

/// One JSON-RPC request per line, as the stdio transport frames them, answered per id.
#[cfg(windows)]
async fn client_connect(endpoint: &str) -> impl tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static {
    tokio::net::windows::named_pipe::ClientOptions::new().open(endpoint).expect("the endpoint exists")
}
#[cfg(not(windows))]
async fn client_connect(endpoint: &str) -> impl tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static {
    tokio::net::UnixStream::connect(endpoint).await.expect("the endpoint exists")
}

#[test]
fn a_client_on_the_socket_lists_tools_and_runs_a_line() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
    let Some(mut app) = headless_app() else { return };
    let endpoint = {
        #[cfg(windows)]
        {
            format!(r"\\.\pipe\fieldform-agent-test-{}", std::process::id())
        }
        #[cfg(not(windows))]
        {
            std::env::temp_dir().join(format!("fieldform-agent-test-{}.sock", std::process::id())).display().to_string()
        }
    };
    let host = server::start(endpoint.clone(), app.agent.cancel.clone(), std::sync::Arc::new(|| {}), server::HostMode::Launch).expect("listens");
    app.agent_host = Some(host);

    let client = std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async move {
            let stream = client_connect(&endpoint).await;
            let (read, mut write) = tokio::io::split(stream);
            let mut lines = tokio::io::BufReader::new(read).lines();
            let mut ask = |id: u64, method: &str, params: serde_json::Value| {
                serde_json::json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }).to_string() + "\n"
            };
            let mut answers = Vec::new();
            write
                .write_all(
                    ask(1, "initialize", serde_json::json!({
                        "protocolVersion": "2025-06-18", "capabilities": {},
                        "clientInfo": { "name": "test", "version": "0" }
                    }))
                    .as_bytes(),
                )
                .await
                .unwrap();
            answers.push(lines.next_line().await.unwrap().expect("initialize answered"));
            write
                .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
                .await
                .unwrap();
            write.write_all(ask(2, "tools/list", serde_json::json!({})).as_bytes()).await.unwrap();
            answers.push(lines.next_line().await.unwrap().expect("tools listed"));
            write
                .write_all(
                    ask(3, "tools/call", serde_json::json!({
                        "name": "run", "arguments": { "lines": "sphere radius=10 'ball'" }
                    }))
                    .as_bytes(),
                )
                .await
                .unwrap();
            answers.push(lines.next_line().await.unwrap().expect("run answered"));
            for (id, method, params) in [
                (4, "resources/list", serde_json::json!({})),
                (5, "resources/read", serde_json::json!({ "uri": "fieldform://guide" })),
                (6, "resources/read", serde_json::json!({ "uri": "fieldform://document" })),
                (7, "resources/read", serde_json::json!({ "uri": "fieldform://examples/nonesuch" })),
            ] {
                write.write_all(ask(id, method, params).as_bytes()).await.unwrap();
                answers.push(lines.next_line().await.unwrap().expect("resource call answered"));
            }
            let _ = &mut ask;
            answers
        })
    });

    // The frame loop's part: answer jobs until the client has what it came for.
    let started = std::time::Instant::now();
    while !client.is_finished() {
        app.drive_agent();
        assert!(started.elapsed() < std::time::Duration::from_secs(60), "the client never finished");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let answers = client.join().expect("the client ran");
    let init: serde_json::Value = serde_json::from_str(&answers[0]).unwrap();
    assert_eq!(init["result"]["serverInfo"]["name"], "fieldform");
    let tools: serde_json::Value = serde_json::from_str(&answers[1]).unwrap();
    let names: Vec<&str> = tools["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
    for want in ["run", "undo", "describe_command", "list_commands", "get_document", "session_info"] {
        assert!(names.contains(&want), "{want} missing from {names:?}");
    }
    let run: serde_json::Value = serde_json::from_str(&answers[2]).unwrap();
    assert_eq!(run["result"]["isError"], false, "{run}");
    assert_eq!(run["result"]["structuredContent"]["lines"][0]["created"][0]["name"], "ball");
    assert_eq!(app.part_graph().nodes().count(), 1, "the line reached the document over the socket");

    let listed: serde_json::Value = serde_json::from_str(&answers[3]).unwrap();
    let uris: Vec<&str> = listed["result"]["resources"].as_array().unwrap().iter().map(|r| r["uri"].as_str().unwrap()).collect();
    for want in ["fieldform://guide", "fieldform://reference", "fieldform://document", "fieldform://examples/phase-a"] {
        assert!(uris.contains(&want), "{want} missing from {uris:?}");
    }
    let guide: serde_json::Value = serde_json::from_str(&answers[4]).unwrap();
    assert!(guide["result"]["contents"][0]["text"].as_str().unwrap().contains("millimetres"), "{guide}");
    let document: serde_json::Value = serde_json::from_str(&answers[5]).unwrap();
    let text = document["result"]["contents"][0]["text"].as_str().unwrap();
    assert!(text.contains("sphere radius=10"), "the document resource says the live part: {text}");
    let missing: serde_json::Value = serde_json::from_str(&answers[6]).unwrap();
    assert!(missing["error"].is_object(), "an unknown resource is an error: {missing}");
}

/// What a document is, with the ids that differ between two builds of it taken out.
fn shape(app: &App) -> Vec<String> {
    let mut rows: Vec<String> = app
        .agent_document()["parts"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|p| p["blocks"].as_array().unwrap().iter())
        .map(|b| format!("{} {} body={}", b["kind"], b["name"], b["body"]))
        .collect();
    rows.sort();
    rows
}

#[test]
fn a_script_through_run_builds_what_the_same_script_through_the_console_builds() {
    let Some(mut by_run) = headless_app() else { return };
    let Some(mut by_console) = headless_app() else { return };
    let text = include_str!("../../../benchmarks/smoke/phase-a.ffc");
    // A file write is the one thing the guard answers differently, and not part of the shape.
    let lines: String = text.lines().filter(|l| !l.trim_start().starts_with("saveas")).collect::<Vec<_>>().join("\n");

    let out = by_run.agent_run(&RunRequest { lines: lines.clone(), atomic: false, ..Default::default() });
    assert!(out.kept, "{:?}", out.error);

    by_console.script_running = true;
    for l in lines.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        by_console.gesture = None;
        by_console.run_command(l);
        by_console.drain_ownership_changes(false);
    }
    assert_eq!(shape(&by_run), shape(&by_console));
    let parameters = |app: &App| -> Vec<String> {
        app.agent_document()["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| format!("{} = {} from {}", p["name"], p["value"], p["source"]))
            .collect()
    };
    assert_eq!(parameters(&by_run), parameters(&by_console));
}

#[test]
fn the_text_form_rebuilds_its_document() {
    let Some(mut original) = headless_app() else { return };
    let Some(mut rebuilt) = headless_app() else { return };
    let built = run(
        &mut original,
        "param name=h value=12\n\
         sphere radius=h centre=1,2,3 'ball'\n\
         cube size=10,20,30 centre=0,0,0 turn=0,0,30 'box'\n\
         cylinder radius=5 from=0,0,-40 to=0,0,-20 'rod'\n\
         boolean -u in=ball,box 'joined'\n\
         move target=joined by=0,0,20\n\
         sphere radius=4 centre=9,9,9",
    );
    assert!(built.ok, "{:?}", built.error);
    let script = original.agent_script();
    assert!(script.omitted.is_empty(), "{:?}\n{}", script.omitted, script.text);
    let out = run(&mut rebuilt, &script.text);
    assert!(out.ok, "{:?}\n{}", out.error, script.text);
    assert_eq!(shape(&original), shape(&rebuilt), "{}", script.text);
    // Ids are random, so the order independent blocks come in is not; the lines are.
    let lines = |app: &App| {
        let mut l: Vec<String> = app.agent_script().text.lines().map(str::to_string).collect();
        l.sort();
        l
    };
    assert_eq!(lines(&original), lines(&rebuilt), "the text form is stable under a rebuild");
}

#[test]
fn what_the_text_form_cannot_write_it_names() {
    let Some(mut app) = headless_app() else { return };
    run(&mut app, "tpms cell=gyroid cell_size=10 'g'\nsphere radius=3 'ok'");
    let script = app.agent_script();
    assert!(script.text.contains("sphere radius=3"), "{}", script.text);
    assert_eq!(script.omitted.len(), 1, "{:?}", script.omitted);
    assert!(script.text.contains("# omitted: "), "{}", script.text);
}

fn read(app: &mut App, tool: &str, args: serde_json::Value) -> serde_json::Value {
    app.agent_handle(tool, args.as_object().unwrap()).json
}

#[test]
fn measure_agrees_with_the_closed_form_and_states_its_resolution() {
    let Some(mut app) = headless_app() else { return };
    run(&mut app, "sphere radius=10 centre=0,0,0 'ball'");
    let m = read(&mut app, "measure", serde_json::json!({ "target": "ball" }));
    assert_eq!(m["ok"], true, "{m}");
    let v = m["volume_mm3"].as_f64().unwrap();
    let want = 4.0 / 3.0 * std::f64::consts::PI * 1000.0;
    assert!((v - want).abs() / want < 0.005, "{v} vs {want}");
    assert_eq!(m["resolution"], "standard");
    assert!(m["cell_size_mm"].as_f64().unwrap() > 0.0);
    let only = read(&mut app, "measure", serde_json::json!({ "quantities": ["volume"] }));
    assert!(only.get("surface_area_mm2").is_none() && only.get("centroid_mm").is_none());
}

#[test]
fn sample_is_negative_inside_and_says_what_the_magnitude_means() {
    let Some(mut app) = headless_app() else { return };
    run(&mut app, "sphere radius=10 centre=0,0,0 'ball'");
    let s = read(&mut app, "sample", serde_json::json!({ "points": [[0, 0, 0], [20, 0, 0]], "gradient": true }));
    assert_eq!(s["ok"], true, "{s}");
    let a = s["samples"][0]["value"].as_f64().unwrap();
    let b = s["samples"][1]["value"].as_f64().unwrap();
    assert!((a + 10.0).abs() < 1e-3 && (b - 10.0).abs() < 1e-3, "{a} {b}");
    assert_eq!(s["normalization"], "Distance");
    assert!(s["samples"][1]["gradient"].is_array());
    let bad = read(&mut app, "sample", serde_json::json!({ "points": [[0, 0]] }));
    assert_eq!(bad["error"]["code"], "bad_value");
}

#[test]
fn check_reports_a_problem_and_inspect_names_a_blocks_readers() {
    let Some(mut app) = headless_app() else { return };
    let clean = read(&mut app, "check", serde_json::json!({}));
    assert_eq!(clean["ok"], true, "{clean}");
    run(&mut app, "sphere radius=10 'a'\nsphere radius=10 centre=5,0,0 'b'\nboolean -u in=a,b 'u'");
    let i = read(&mut app, "inspect", serde_json::json!({ "block": "a" }));
    assert_eq!(i["ok"], true, "{i}");
    assert_eq!(i["body"], false, "a consumed block is not a body");
    assert_eq!(i["read_by"][0]["name"], "u");
    assert_eq!(i["exactness"], "Distance");
    let gone = read(&mut app, "inspect", serde_json::json!({ "block": "nope" }));
    assert_eq!(gone["error"]["code"], "not_found");
}

#[test]
fn a_view_that_is_not_one_is_refused_with_the_views_there_are() {
    let Some(mut app) = headless_app() else { return };
    let (reply, mut answer) = tokio::sync::oneshot::channel();
    let mut args = serde_json::Map::new();
    args.insert("view".into(), "sideways".into());
    app.begin_snapshot(server::Job { tool: "snapshot".into(), args, reply });
    let got = answer.try_recv().expect("a refused view is answered at once");
    assert!(got.reply.is_error && got.image.is_none());
    assert_eq!(got.reply.json["error"]["code"], "bad_value");
    assert!(got.reply.json["error"]["accepted"].as_array().unwrap().iter().any(|v| v == "iso"));
    assert!(app.agent.snapshot.is_none());
}


// --- Phase D: attach mode ---------------------------------------------------------------------

fn can_connect(endpoint: &str) -> bool {
    #[cfg(windows)]
    {
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let _enter = runtime.enter();
        tokio::net::windows::named_pipe::ClientOptions::new().open(endpoint).is_ok()
    }
    #[cfg(not(windows))]
    {
        std::os::unix::net::UnixStream::connect(endpoint).is_ok()
    }
}

fn attach_endpoint(tag: &str) -> String {
    let id = std::process::id();
    if cfg!(windows) {
        format!(r"\\.\pipe\fieldform-attach-test-{tag}-{id}")
    } else {
        std::env::temp_dir().join(format!("fieldform-attach-test-{tag}-{id}.sock")).display().to_string()
    }
}

fn attached(tag: &str) -> Option<(App, String)> {
    let mut app = headless_app()?;
    let endpoint = attach_endpoint(tag);
    app.agent_endpoint = Some(endpoint.clone());
    app.set_allow_agent(true, &eframe::egui::Context::default());
    Some((app, endpoint))
}

/// A job whose answer the test can read, as the transport would hold one.
fn job(tool: &str, args: serde_json::Value) -> (server::Job, tokio::sync::oneshot::Receiver<server::Answer>) {
    let (reply, answer) = tokio::sync::oneshot::channel();
    (server::Job { tool: tool.into(), args: args.as_object().cloned().unwrap_or_default(), reply }, answer)
}

#[test]
fn allow_agent_publishes_an_endpoint_and_turning_it_off_takes_it_away() {
    let Some((mut app, endpoint)) = attached("toggle") else { return };
    assert!(can_connect(&endpoint), "on: the endpoint exists");
    assert_eq!(app.agent_session_info()["host"], "attach");
    app.set_allow_agent(false, &eframe::egui::Context::default());
    assert!(app.agent_host.is_none());
    // The listener ends on its own thread; give it the moment it needs.
    let started = std::time::Instant::now();
    while can_connect(&endpoint) && started.elapsed() < std::time::Duration::from_secs(5) {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(!can_connect(&endpoint), "off: nothing is published");
}

#[test]
fn an_agent_the_person_disconnects_is_dropped_and_not_counted() {
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};
    let Some((app, endpoint)) = attached("drop") else { return };
    let host_clients = |app: &App| app.agent_host.as_ref().unwrap().clients();
    let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
    let client = std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async move {
            let stream = client_connect(&endpoint).await;
            let (read, mut write) = tokio::io::split(stream);
            let mut lines = tokio::io::BufReader::new(read);
            let hello = serde_json::json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": { "name": "t", "version": "0" } } });
            write.write_all((hello.to_string() + "\n").as_bytes()).await.unwrap();
            let mut first = String::new();
            lines.read_line(&mut first).await.unwrap();
            go_rx.recv().unwrap();
            // Closed from the other side: the read ends instead of waiting.
            let mut rest = Vec::new();
            tokio::time::timeout(std::time::Duration::from_secs(10), lines.read_to_end(&mut rest)).await.is_ok()
        })
    });
    let started = std::time::Instant::now();
    while host_clients(&app) == 0 {
        assert!(started.elapsed() < std::time::Duration::from_secs(10), "the client never counted");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(host_clients(&app), 1);
    app.agent_host.as_ref().unwrap().disconnect_all();
    go_tx.send(()).unwrap();
    assert!(client.join().unwrap(), "the client saw the connection end");
    let started = std::time::Instant::now();
    while host_clients(&app) != 0 {
        assert!(started.elapsed() < std::time::Duration::from_secs(10), "the count never came down");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[test]
fn a_call_that_arrives_mid_gesture_waits_then_is_told_the_person_is_busy() {
    let Some((mut app, _)) = attached("busy") else { return };
    app.agent.pretend_busy = true;
    let (fresh, mut fresh_answer) = job("check", serde_json::json!({}));
    app.agent.held.push_back((fresh, std::time::Instant::now()));
    app.drive_agent();
    assert!(fresh_answer.try_recv().is_err(), "within the allowance it waits");
    assert_eq!(app.agent.held.len(), 1);

    let (late, mut late_answer) = job("check", serde_json::json!({}));
    app.agent.held.clear();
    app.agent.held.push_back((late, std::time::Instant::now() - std::time::Duration::from_secs(3)));
    app.drive_agent();
    let answer = late_answer.try_recv().expect("after the allowance it is answered");
    assert!(answer.reply.is_error);
    assert_eq!(answer.reply.json["error"]["code"], "busy");
    assert!(answer.reply.json["error"]["retry_after_ms"].as_u64().unwrap() > 0);

    // session_info is how an agent finds out, so it is never made to wait.
    let (info, mut info_answer) = job("session_info", serde_json::json!({}));
    app.agent.held.push_back((info, std::time::Instant::now()));
    app.drive_agent();
    assert_eq!(info_answer.try_recv().unwrap().reply.json["busy"], true);

    app.agent.pretend_busy = false;
    let (go, mut go_answer) = job("check", serde_json::json!({}));
    app.agent.held.push_back((go, std::time::Instant::now()));
    app.drive_agent();
    assert!(!go_answer.try_recv().unwrap().reply.is_error, "the person is done, so it runs");
}

#[test]
fn a_snapshot_leaves_the_persons_view_alone() {
    let Some((mut app, _)) = attached("snap") else { return };
    let (quality, mode) = (app.display_quality, app.view_mode);
    let (snap, mut answer) = job("snapshot", serde_json::json!({ "view": "top" }));
    app.agent.held.push_back((snap, std::time::Instant::now()));
    app.drive_agent();
    let answer = answer.try_recv().expect("refused at once, not left waiting for a picture");
    assert_eq!(answer.reply.json["error"]["code"], "not_available");
    assert!(answer.image.is_none());
    assert!(app.pending_snap.is_none(), "no still was begun");
    assert!(app.display_quality == quality && app.view_mode == mode, "the person's view is as they left it");
    assert_eq!(app.agent_session_info()["capabilities"]["snapshot"], false);
}

#[test]
fn an_attached_agent_reaches_only_the_documents_own_folder() {
    let Some((mut app, _)) = attached("roots") else { return };
    let (info, mut answer) = job("session_info", serde_json::json!({}));
    app.agent.held.push_back((info, std::time::Instant::now()));
    app.drive_agent();
    assert_eq!(answer.try_recv().unwrap().reply.json["roots"], serde_json::json!([]), "an unsaved document has no folder");
    let refused = run(&mut app, "saveas file=\"elsewhere.fform\"");
    assert!(!refused.ok, "with no folder there is nowhere to write");
}

/// The seed stands 2 mm proud of the plate and the row's first hole is where it stands, so the
/// whole part's field reads all three faults a row can have: the pin outside, the plug inside, a
/// hole short.
#[test]
fn a_row_holds_its_seed_so_a_cut_cuts_the_whole_count() {
    let Some(mut app) = headless_app() else { return };
    let seed_and_row = "cube size=70,20,10 'plate'\n\
                        sphere radius=4 centre=-15,0,4 'seed'\n\
                        pattern target=seed kind=linear count=3 step=15,0,0 'row'";
    let made = run(&mut app, seed_and_row);
    assert!(made.ok, "{:?}", made.lines.last().map(|l| (&l.line, &l.outcome.error)));
    let seed = app.part_graph().nodes().map(|(id, _)| id).find(|id| app.body_name(*id) == Some("seed")).unwrap();
    assert!(!app.drawn_bodies().contains(&seed), "inside the row, so not drawn beside it");

    assert!(run(&mut app, "boolean -s in=plate,row 'cut'").ok);
    let drawn: Vec<String> = app.drawn_bodies().into_iter().map(|id| app.body_label(id)).collect();
    assert_eq!(drawn, ["cut"], "the cut draws, not its parts");
    assert!(app.agent_check().json["ok"].as_bool().unwrap());
    let value = |app: &App, p: [f64; 3]| {
        let args = serde_json::json!({ "points": [p] });
        app.agent_sample(args.as_object().unwrap()).json["samples"][0]["value"].as_f64().unwrap()
    };
    assert!(value(&app, [-15.0, 0.0, 7.0]) > 0.0, "no pin above the face");
    for x in [-15.0, 0.0, 15.0] {
        assert!(value(&app, [x, 0.0, 3.0]) > 0.0, "a row of three cuts three: none at {x}");
    }
    assert!(value(&app, [-7.5, 0.0, 3.0]) < 0.0, "and the plate is there between them");

    // An ingredient's eye turned on is somebody asking, and is told apart from a body.
    app.set_visible(seed, true);
    let inspected = app.agent_inspect(serde_json::json!({ "block": "seed" }).as_object().unwrap());
    assert_eq!(inspected.json["body"], false);
    assert_eq!(inspected.json["drawn"], true, "no body, and drawn: the two are different questions");
    assert_eq!(app.agent_check().json["problems"][0]["code"], "drawn_ingredient");
}

/// A small section bakes where a large one does not, so the two are what tells a reading of the
/// part from a reading of its display grid: at 0.05 mm the 17 mm square read 0.0125 mm proud on
/// every wall and the 100 mm one read nought.
#[test]
fn a_small_extrude_samples_on_its_own_wall_as_a_large_one_does() {
    let Some(mut app) = headless_app() else { return };
    let lines = "line p0=0,0,0 p1=17,0,0 'a0'\n\
                 line p0=17,0,0 p1=17,13,0 'a1'\n\
                 line p0=17,13,0 p1=0,13,0 'a2'\n\
                 line p0=0,13,0 p1=0,0,0 'a3'\n\
                 extrude profile=a0 height=2 corners=sharp 'small'\n\
                 line p0=0,50,0 p1=100,50,0 'b0'\n\
                 line p0=100,50,0 p1=100,120,0 'b1'\n\
                 line p0=100,120,0 p1=0,120,0 'b2'\n\
                 line p0=0,120,0 p1=0,50,0 'b3'\n\
                 extrude profile=b0 height=2 corners=sharp 'large'";
    let made = run(&mut app, lines);
    assert!(made.ok, "{:?}", made.lines.last().map(|l| (&l.line, &l.outcome.error)));
    let walls = [[8.5, 0.0, 1.0], [17.0, 6.5, 1.0], [8.5, 13.0, 1.0], [50.0, 50.0, 1.0], [100.0, 85.0, 1.0]];
    let read = app.agent_sample(serde_json::json!({ "points": walls }).as_object().unwrap());
    for (p, sample) in walls.iter().zip(read.json["samples"].as_array().unwrap()) {
        let value = sample["value"].as_f64().unwrap();
        assert!(value.abs() < 1e-3, "{value:.5} mm on the wall at {p:?}");
    }
}

/// What `join=false` is for, and the one way to ask for a row one short.
#[test]
fn a_row_of_copies_only_leaves_its_first_out() {
    let Some(mut app) = headless_app() else { return };
    let lines = "cube size=70,20,10 'plate'\n\
                 sphere radius=4 centre=-15,0,4 'seed'\n\
                 pattern target=seed kind=linear count=3 step=15,0,0 join=false 'row'\n\
                 boolean -s in=plate,row 'cut'";
    let made = run(&mut app, lines);
    assert!(made.ok, "{:?}", made.lines.last().map(|l| (&l.line, &l.outcome.error)));
    let args = serde_json::json!({ "points": [[-15.0, 0.0, 3.0], [0.0, 0.0, 3.0]] });
    let read = app.agent_sample(args.as_object().unwrap());
    assert!(read.json["samples"][0]["value"].as_f64().unwrap() < 0.0, "no hole where the first would be");
    assert!(read.json["samples"][1]["value"].as_f64().unwrap() > 0.0, "and the copies still cut");
}

#[test]
fn a_reflection_is_its_own_shape_and_its_original_stays_beside_a_cut_of_it() {
    let Some(mut app) = headless_app() else { return };
    let lines = "cube size=70,20,10 'plate'\n\
                 sphere radius=4 centre=10,0,12 'half'\n\
                 mirror target=half axis=x 'other'\n\
                 boolean -s in=plate,other 'cut'";
    assert!(run(&mut app, lines).ok);
    let mut drawn: Vec<String> = app.drawn_bodies().into_iter().map(|id| app.body_label(id)).collect();
    drawn.sort();
    assert_eq!(drawn, ["cut", "half"]);
}

#[test]
fn session_info_tells_the_viewport_from_the_person() {
    let Some(mut app) = headless_app() else { return };
    assert!(run(&mut app, "sphere radius=4 'ball'").ok);
    let info = app.agent_session_info();
    assert_eq!(info["compiling"], app.traced.is_compiling().is_some());
    assert_eq!(info["busy"], false, "no person was here, whatever the shader is doing");
}

/// **Unsaved work stops an agent's `new` with a reason, not with a dialog.** A dialog here is the
/// failure itself: it waits for a person, so this test would hang rather than fail. The block is
/// still there afterwards because the document may be somebody's, and saving is what lets the same
/// line through, which is the control.
#[test]
fn an_agent_is_told_to_save_rather_than_asked_about_unsaved_work() {
    let Some(mut app) = headless_app() else { return };
    root_in_temp(&mut app, "unsaved");
    assert!(run(&mut app, "sphere radius=5 'ball'").ok);
    for verb in ["new", "open"] {
        let out = run(&mut app, verb);
        let error = out.lines[0].outcome.error.as_ref().expect("refused");
        assert_eq!(error.code, "refused", "{verb}: {error:?}");
        assert!(error.message.contains("saveas file="), "{verb}: {}", error.message);
        assert!(app.node_named("ball").is_some(), "{verb} took the block with it");
    }
    let saved = run(&mut app, "saveas file=kept.fform\nnew");
    assert!(saved.ok, "{:?}", saved.lines.iter().map(|l| &l.outcome).collect::<Vec<_>>());
    assert!(app.node_named("ball").is_none(), "saved, so it may be replaced");
}

/// **Another document in the same place is a change, though nothing is dirty.** A saved `new` used
/// to leave the revision where it was and say `kept: false`, so `expected_revision` let a call
/// through onto a document its author had never read. Both verbs, because they replace it apart.
#[test]
fn replacing_the_document_moves_the_revision() {
    let Some(mut app) = headless_app() else { return };
    root_in_temp(&mut app, "replace");
    assert!(run(&mut app, "sphere radius=5 'ball'\nsaveas file=first.fform").ok);
    for line in ["new", "open file=first.fform"] {
        let read = app.revision;
        let out = run(&mut app, line);
        assert!(out.ok && out.kept, "{line}: {:?}", out.lines.iter().map(|l| &l.outcome).collect::<Vec<_>>());
        assert_ne!(out.revision_after, read, "{line}");
        assert!(!app.is_dirty(), "{line}: what was just read or started has nothing to save");
        let late = app.agent_run(&RunRequest {
            lines: "sphere radius=2".into(),
            expected_revision: Some(read),
            ..Default::default()
        });
        assert_eq!(late.error.map(|e| e.code), Some("stale_revision"), "{line}");
    }
}

/// **A call that starts a document owns what it adds to it.** The history is cleared under the
/// mark, so a mark taken at depth 1 found nothing past it and the new block was left as a step of
/// its own, which the agent's `undo` then called somebody else's.
#[test]
fn a_block_made_after_new_is_undone_with_the_call() {
    let Some(mut app) = headless_app() else { return };
    root_in_temp(&mut app, "after-new");
    assert!(run(&mut app, "sphere radius=5 'ball'\nsaveas file=first.fform").ok);
    let out = run(&mut app, "new\nsphere radius=2 'fresh'");
    assert!(out.ok && out.kept);
    assert_eq!(app.stack.history().0.len(), 1, "{:?}", app.stack.history());
    app.agent_undo(1, false, false).unwrap();
    assert!(app.node_named("fresh").is_none());
}

/// **All or nothing holds across a `new`.** The document alone is not enough to put back: the path,
/// the active part and cage and the history are all replaced with it, and a part id left over from
/// the abandoned document points at nothing.
#[test]
fn a_refused_line_takes_back_a_new_in_the_same_call() {
    let Some(mut app) = headless_app() else { return };
    root_in_temp(&mut app, "take-back");
    assert!(run(&mut app, "sphere radius=5 'ball'\nsaveas file=first.fform").ok);
    let history = app.stack.history();
    let path = app.file.as_ref().map(|(p, _)| p.clone());
    let revision = app.revision;
    let out = run(&mut app, "new\ncube side=3");
    assert!(!out.ok && !out.kept);
    assert!(app.node_named("ball").is_some(), "the block is back");
    assert_eq!(app.stack.history(), history);
    assert_eq!(app.file.as_ref().map(|(p, _)| p.clone()), path, "Save still means the same file");
    assert_eq!(app.revision, revision);
    assert!(!app.is_dirty());
    assert!(run(&mut app, "sphere radius=1 'after'").ok, "the active part is this document's");
    assert!(app.node_named("after").is_some());
}

/// **No verb that takes a path may ask for one**, which is the whole of the rule: each is refused
/// with the key that replaces the dialog. Script mode reached these through a test of the script
/// file rather than of the mode, so every one of them was open to an agent.
#[test]
fn every_verb_that_would_open_a_file_dialog_is_refused_for_an_agent() {
    let Some(mut app) = headless_app() else { return };
    root_in_temp(&mut app, "asking");
    assert!(run(&mut app, "sphere radius=5 'ball'").ok);
    assert!(run(&mut app, "saveas file=first.fform").ok);
    for verb in ["saveas", "open", "import", "export"] {
        let out = run(&mut app, verb);
        let error = out.lines[0].outcome.error.as_ref().unwrap_or_else(|| panic!("{verb} ran"));
        assert!(error.message.contains("file="), "{verb}: {}", error.message);
    }
}
