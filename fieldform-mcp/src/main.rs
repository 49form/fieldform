// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 49form
//! `fieldform-mcp`: a byte relay between an MCP client's stdio and a FieldForm application's
//! local agent endpoint (`mcp-server-spec.md` §3.1).
//!
//! **It parses nothing.** The MCP server lives in the application; this exists because a windowed
//! application is not a dependable stdio process on Windows, and so one client configuration can
//! either attach to a running session or start a private one.
//!
//! ```text
//! fieldform-mcp [--attach | --launch] [--endpoint NAME] [--app PATH] [--root DIR]... [--agent-timeout SECS]
//! ```
//!
//! With neither flag it attaches if an application is listening and launches one otherwise.
//! `--attach` never launches; `--launch` never attaches. Everything the application says on
//! stderr is left alone: stdout carries the protocol and nothing else.

use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncWrite};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Either,
    Attach,
    Launch,
}

struct Options {
    mode: Mode,
    endpoint: String,
    app: Option<PathBuf>,
    roots: Vec<String>,
    /// Passed on to the application it launches: how long one `run` call may take.
    timeout: Option<String>,
}

fn default_endpoint() -> String {
    let user = std::env::var("USERNAME").or_else(|_| std::env::var("USER")).unwrap_or_else(|_| "user".into());
    let user: String = user.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
    #[cfg(windows)]
    {
        format!(r"\\.\pipe\fieldform-agent-{user}")
    }
    #[cfg(not(windows))]
    {
        let dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
        format!("{dir}/fieldform-agent-{user}.sock")
    }
}

fn parse() -> Result<Options, String> {
    let mut o = Options { mode: Mode::Either, endpoint: default_endpoint(), app: None, roots: Vec::new(), timeout: None };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--attach" => o.mode = Mode::Attach,
            "--launch" => o.mode = Mode::Launch,
            "--endpoint" => o.endpoint = args.next().ok_or("--endpoint wants a name")?,
            "--app" => o.app = Some(args.next().ok_or("--app wants a path")?.into()),
            "--root" => o.roots.push(args.next().ok_or("--root wants a folder")?),
            "--agent-timeout" => {
                let secs = args.next().ok_or("--agent-timeout wants seconds")?;
                secs.parse::<f64>().ok().filter(|s| s.is_finite() && *s > 0.0).ok_or("--agent-timeout wants a positive number")?;
                o.timeout = Some(secs);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(o)
}

/// The application, found beside this executable and nowhere else unless told: a different
/// build's application would be speaking a different version of the tools.
fn find_app(o: &Options) -> Result<PathBuf, String> {
    if let Some(p) = &o.app {
        return Ok(p.clone());
    }
    let me = std::env::current_exe().map_err(|e| format!("cannot find myself: {e}"))?;
    let name = if cfg!(windows) { "fieldform.exe" } else { "fieldform" };
    let beside = me.with_file_name(name);
    beside.is_file().then_some(beside).ok_or_else(|| format!("no {name} beside {} (use --app)", me.display()))
}

#[cfg(windows)]
async fn connect(endpoint: &str) -> std::io::Result<impl AsyncRead + AsyncWrite + Unpin + Send + 'static> {
    tokio::net::windows::named_pipe::ClientOptions::new().open(endpoint)
}

#[cfg(not(windows))]
async fn connect(endpoint: &str) -> std::io::Result<impl AsyncRead + AsyncWrite + Unpin + Send + 'static> {
    tokio::net::UnixStream::connect(endpoint).await
}

async fn run() -> Result<(), String> {
    let o = parse()?;
    let mut child = None;
    let stream = match (o.mode, connect(&o.endpoint).await) {
        (Mode::Launch, _) | (Mode::Either, Err(_)) => {
            let app = find_app(&o)?;
            let mut cmd = tokio::process::Command::new(app);
            cmd.arg("--agent-host").arg(&o.endpoint);
            for root in &o.roots {
                cmd.arg("--root").arg(root);
            }
            if let Some(secs) = &o.timeout {
                cmd.arg("--agent-timeout").arg(secs);
            }
            // The application's own output is not the protocol's.
            cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).kill_on_drop(true);
            child = Some(cmd.spawn().map_err(|e| format!("cannot start the application: {e}"))?);
            let started = Instant::now();
            loop {
                match connect(&o.endpoint).await {
                    Ok(s) => break s,
                    Err(e) if started.elapsed() > Duration::from_secs(90) => {
                        return Err(format!("the application did not open {} in 90 s: {e}", o.endpoint));
                    }
                    Err(_) => tokio::time::sleep(Duration::from_millis(250)).await,
                }
            }
        }
        (_, Ok(s)) => s,
        (Mode::Attach, Err(e)) => {
            return Err(format!("no application is listening on {} ({e}); start one with Allow agent on, or drop --attach", o.endpoint));
        }
    };

    let (mut from_app, mut to_app) = tokio::io::split(stream);
    let mut stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    // Either side closing ends the relay: a client that goes away ends the session, and an
    // application that goes away is told to the client as end of stream.
    tokio::select! {
        r = tokio::io::copy(&mut stdin, &mut to_app) => { r.map_err(|e| format!("client to application: {e}"))?; }
        r = tokio::io::copy(&mut from_app, &mut stdout) => { r.map_err(|e| format!("application to client: {e}"))?; }
    }
    drop(child);
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(why) = run().await {
        eprintln!("fieldform-mcp: {why}");
        std::process::exit(1);
    }
}
