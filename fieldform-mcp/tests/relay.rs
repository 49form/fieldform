// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! The relay's choices: attach, launch, or either (`mcp-server-spec.md` §3.1). None of these needs
//! a FieldForm application: a stand-in endpoint takes its place, so the relay is tested for what it
//! does and not for what it carries.

use std::io::{Read, Write};
use std::process::{Command, Stdio};

fn relay() -> Command {
    Command::new(env!("CARGO_BIN_EXE_fieldform-mcp"))
}

fn endpoint(tag: &str) -> String {
    let id = std::process::id();
    if cfg!(windows) {
        format!(r"\\.\pipe\fieldform-relay-test-{tag}-{id}")
    } else {
        std::env::temp_dir().join(format!("fieldform-relay-test-{tag}-{id}.sock")).display().to_string()
    }
}

fn stderr_of(mut c: Command) -> (bool, String) {
    let out = c.stdin(Stdio::null()).output().expect("the relay runs");
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn attach_with_nothing_listening_refuses_and_does_not_launch() {
    let mut c = relay();
    c.args(["--attach", "--endpoint", &endpoint("none")]);
    let (ok, why) = stderr_of(c);
    assert!(!ok);
    assert!(why.contains("no application is listening"), "{why}");
}

#[test]
fn launch_without_an_application_beside_it_says_where_it_looked() {
    // Copied somewhere with no application next to it, since a build tree has one.
    let dir = std::env::temp_dir().join(format!("fieldform-relay-alone-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let alone = dir.join(std::path::Path::new(env!("CARGO_BIN_EXE_fieldform-mcp")).file_name().unwrap());
    std::fs::copy(env!("CARGO_BIN_EXE_fieldform-mcp"), &alone).unwrap();
    let mut c = Command::new(&alone);
    c.args(["--launch", "--endpoint", &endpoint("launch")]);
    let (ok, why) = stderr_of(c);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!ok);
    assert!(why.contains("--app"), "{why}");
}

#[test]
fn launch_with_an_application_that_cannot_start_names_the_failure() {
    let mut c = relay();
    c.args(["--launch", "--app", "no-such-application", "--endpoint", &endpoint("bad-app")]);
    let (ok, why) = stderr_of(c);
    assert!(!ok);
    assert!(why.contains("cannot start the application"), "{why}");
}

#[test]
fn an_agent_timeout_that_is_not_a_positive_number_is_refused() {
    for bad in ["soon", "0", "-5"] {
        let mut c = relay();
        c.args(["--attach", "--agent-timeout", bad]);
        let (ok, why) = stderr_of(c);
        assert!(!ok);
        assert!(why.contains("--agent-timeout"), "{bad}: {why}");
    }
}

#[test]
fn an_argument_it_does_not_know_is_refused() {
    let mut c = relay();
    c.arg("--frobnicate");
    let (ok, why) = stderr_of(c);
    assert!(!ok);
    assert!(why.contains("unknown argument"), "{why}");
}

/// A stand-in application: accepts one connection and sends back, upper-cased, what it is sent.
fn stand_in(name: String) -> std::thread::JoinHandle<()> {
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            #[cfg(windows)]
            let mut stream = {
                let server =
                    tokio::net::windows::named_pipe::ServerOptions::new().first_pipe_instance(true).create(&name).unwrap();
                ready_tx.send(()).unwrap();
                server.connect().await.unwrap();
                server
            };
            #[cfg(not(windows))]
            let mut stream = {
                let _ = std::fs::remove_file(&name);
                let listener = tokio::net::UnixListener::bind(&name).unwrap();
                ready_tx.send(()).unwrap();
                listener.accept().await.unwrap().0
            };
            let mut buf = [0u8; 256];
            let n = stream.read(&mut buf).await.unwrap();
            stream.write_all(&buf[..n].to_ascii_uppercase()).await.unwrap();
            stream.flush().await.unwrap();
            // Held open until the relay has read the answer and closed its end.
            let _ = stream.read(&mut buf).await;
        });
    });
    ready_rx.recv().unwrap();
    handle
}

#[test]
fn attach_and_the_default_both_carry_bytes_to_a_listening_endpoint_and_back() {
    for mode in [Some("--attach"), None] {
        let name = endpoint("echo");
        let server = stand_in(name.clone());
        let mut c = relay();
        c.args(mode).args(["--endpoint", &name]);
        let mut child = c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(b"hello\n").unwrap();
        stdin.flush().unwrap();
        let mut out = child.stdout.take().unwrap();
        let mut got = Vec::new();
        let mut buf = [0u8; 64];
        while !got.ends_with(b"\n") {
            let n = out.read(&mut buf).unwrap();
            assert!(n > 0, "the relay closed before answering: {:?}", String::from_utf8_lossy(&got));
            got.extend_from_slice(&buf[..n]);
        }
        assert_eq!(got, b"HELLO\n");
        drop(stdin);
        assert!(child.wait().unwrap().success());
        server.join().unwrap();
    }
}
