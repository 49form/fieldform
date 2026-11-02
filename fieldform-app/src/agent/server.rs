// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! The MCP server and the socket it listens on (`mcp-server-spec.md` §3).
//!
//! **A thread that only carries.** The listener runs on a tokio runtime of its own; every tool call
//! becomes a [`Job`] on a channel that the frame loop drains between frames, so the document is only
//! ever touched by the thread that owns it. The tools are in `tools.rs`; nothing here knows what
//! `run` means.

use super::tools::{self, Reply};
use super::tool_error;
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler,
    model::*,
    service::{RequestContext, ServiceExt},
};
use serde_json::{Map, Value as J};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc,
};
use tokio::sync::Notify;

/// One tool call on its way to the UI thread.
pub(crate) struct Job {
    pub(crate) tool: String,
    pub(crate) args: Map<String, J>,
    pub(crate) reply: tokio::sync::oneshot::Sender<Answer>,
}

/// A tool's reply, and the picture that goes with it when it has one (PNG bytes).
pub(crate) struct Answer {
    pub(crate) reply: Reply,
    pub(crate) image: Option<Vec<u8>>,
}

/// What the app keeps of a running listener.
pub(crate) struct Host {
    pub(crate) jobs: mpsc::Receiver<Job>,
    pub(crate) endpoint: String,
    pub(crate) wake: Waker,
    pub(crate) mode: HostMode,
    clients: Arc<AtomicUsize>,
    disconnect: Arc<Notify>,
    stop: Arc<Notify>,
}

/// Whose session this is (`mcp-server-spec.md` §3.2): one the agent started and owns, or a
/// person's, which the agent is a guest in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HostMode {
    Launch,
    Attach,
}

impl HostMode {
    pub(crate) fn name(self) -> &'static str {
        match self {
            HostMode::Launch => "launch",
            HostMode::Attach => "attach",
        }
    }
}

impl Host {
    /// How many clients are connected now.
    pub(crate) fn clients(&self) -> usize {
        self.clients.load(Ordering::SeqCst)
    }

    /// Drops every connection and leaves the endpoint open: the indicator's click.
    pub(crate) fn disconnect_all(&self) {
        self.disconnect.notify_waiters();
    }
}

impl Drop for Host {
    /// Taking the host away takes the endpoint with it, so Allow agent off publishes nothing.
    fn drop(&mut self) {
        self.stop.notify_one();
        self.disconnect.notify_waiters();
    }
}

/// How long a call waits for the person to finish a gesture before it is told they are busy.
const BUSY_ALLOWANCE: std::time::Duration = std::time::Duration::from_secs(2);
const BUSY_RETRY_MS: u64 = 500;

/// Wakes the frame loop when a job arrives, so a hidden or idle window still answers.
pub(crate) type Waker = Arc<dyn Fn() + Send + Sync>;

#[derive(Clone)]
struct Handler {
    jobs: mpsc::Sender<Job>,
    wake: Waker,
    cancel: Arc<AtomicBool>,
    clients: Arc<AtomicUsize>,
    disconnect: Arc<Notify>,
    stop: Arc<Notify>,
}

impl Handler {
    /// Sends one job to the UI thread and waits for its answer.
    async fn ask(
        &self,
        tool: &str,
        args: Map<String, J>,
        context: &RequestContext<RoleServer>,
    ) -> Result<Answer, McpError> {
        let (reply, answer) = tokio::sync::oneshot::channel();
        if self.jobs.send(Job { tool: tool.to_string(), args, reply }).is_err() {
            return Err(McpError::internal_error("the application is closing", None));
        }
        (self.wake)();
        tokio::select! {
            r = answer => r.map_err(|_| McpError::internal_error("the application dropped the call", None)),
            _ = context.ct.cancelled() => {
                // Taken back between lines by the run loop; the answer, if it comes, goes nowhere.
                self.cancel.store(true, Ordering::SeqCst);
                Err(McpError::internal_error("cancelled", None))
            }
        }
    }
}

impl ServerHandler for Handler {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().enable_resources().build())
            .with_server_info(Implementation::new("fieldform", env!("CARGO_PKG_VERSION")))
            .with_instructions(tools::INSTRUCTIONS)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let tools = tools::definitions()
            .into_iter()
            .map(|(name, description, schema, read_only)| {
                let schema = schema.as_object().cloned().unwrap_or_default();
                // Nothing here reaches outside the document and its folders (`openWorldHint`).
                let hints = ToolAnnotations::new().read_only(read_only).open_world(false);
                Tool::new(name, description, schema).with_annotations(hints)
            })
            .collect();
        Ok(ListToolsResult::with_all_items(tools))
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult::with_all_items(super::resources::list()))
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        Ok(ListResourceTemplatesResult::with_all_items(super::resources::templates()))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, McpError> {
        let uri = request.uri;
        let (mime, text) = if uri == "fieldform://document" {
            // The one resource that is the live document, so the UI thread has to say it.
            let mut args = Map::new();
            args.insert("format".into(), J::String("text".into()));
            let answer = self.ask("get_document", args, &context).await?;
            let text = answer.reply.json.get("text").and_then(J::as_str).unwrap_or_default().to_string();
            ("text/plain", text)
        } else {
            super::resources::fixed(&uri).ok_or_else(|| {
                McpError::resource_not_found(format!("`{uri}` is not a resource of this host"), None)
            })?
        };
        let contents = ResourceContents::text(text, uri).with_mime_type(mime);
        Ok(ReadResourceResult::new(vec![contents]).into())
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let Answer { reply, image } =
            self.ask(request.name.as_ref(), request.arguments.unwrap_or_default(), &context).await?;
        let text = serde_json::to_string(&reply.json).unwrap_or_else(|_| "{}".into());
        let mut result = if reply.is_error {
            CallToolResult::error(vec![ContentBlock::text(text)])
        } else {
            CallToolResult::success(vec![ContentBlock::text(text)])
        };
        if let Some(png) = image {
            result.content.push(ContentBlock::image(super::snapshot::encode_base64(&png), "image/png"));
        }
        if let J::Object(map) = reply.json {
            result.structured_content = Some(J::Object(map));
        }
        Ok(CallToolResponse::Complete(result))
    }
}

/// The name a user-private endpoint has: the account is in it, so another account's relay does not
/// find this one.
pub(crate) fn default_endpoint() -> String {
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

/// Starts listening on `endpoint`. Returns once the endpoint exists, so a relay that starts the
/// application can connect as soon as this does.
pub(crate) fn start(endpoint: String, cancel: Arc<AtomicBool>, wake: Waker, mode: HostMode) -> Result<Host, String> {
    let (jobs_tx, jobs_rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let (clients, disconnect, stop) = (Arc::new(AtomicUsize::new(0)), Arc::new(Notify::new()), Arc::new(Notify::new()));
    let handler =
        Handler { jobs: jobs_tx, wake: wake.clone(), cancel, clients: clients.clone(), disconnect: disconnect.clone(), stop: stop.clone() };
    let name = endpoint.clone();
    std::thread::Builder::new()
        .name("fieldform-agent".into())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build() {
                Ok(r) => r,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("cannot start the agent runtime: {e}")));
                    return;
                }
            };
            runtime.block_on(listen(name, handler, ready_tx));
        })
        .map_err(|e| format!("cannot start the agent thread: {e}"))?;
    ready_rx.recv().map_err(|_| "the agent thread ended before it was ready".to_string())??;
    Ok(Host { jobs: jobs_rx, endpoint, wake, mode, clients, disconnect, stop })
}

async fn serve_one<S>(stream: S, handler: Handler)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Send + Unpin + 'static,
{
    let (read, write) = tokio::io::split(stream);
    let (clients, disconnect) = (handler.clients.clone(), handler.disconnect.clone());
    // Registered before the handshake, so the indicator is up while a client is still introducing
    // itself and not only once it has asked for something.
    clients.fetch_add(1, Ordering::SeqCst);
    struct Leave(Arc<AtomicUsize>);
    impl Drop for Leave {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let _leave = Leave(clients);
    let gone = disconnect.notified();
    tokio::pin!(gone);
    // `enable` so a disconnect sent during the handshake is not missed.
    gone.as_mut().enable();
    match handler.serve((read, write)).await {
        Ok(running) => {
            let token = running.cancellation_token();
            tokio::select! {
                _ = running.waiting() => {}
                _ = &mut gone => token.cancel(),
            }
        }
        Err(e) => log::warn!("an agent connection did not start: {e}"),
    }
}

#[cfg(windows)]
async fn listen(endpoint: String, handler: Handler, ready: mpsc::Sender<Result<(), String>>) {
    use tokio::net::windows::named_pipe::ServerOptions;
    // `first_pipe_instance` fails if another process already owns the name, so a squatter cannot
    // sit on it and receive what is meant for this one. Remote clients are refused.
    let mut server = match ServerOptions::new().first_pipe_instance(true).reject_remote_clients(true).create(&endpoint) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready.send(Err(format!("cannot create {endpoint}: {e}")));
            return;
        }
    };
    let _ = ready.send(Ok(()));
    loop {
        let stop = handler.stop.clone();
        let connected = tokio::select! {
            r = server.connect() => r,
            _ = stop.notified() => break,
        };
        if connected.is_err() {
            break;
        }
        let next = match ServerOptions::new().reject_remote_clients(true).create(&endpoint) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("agent endpoint: {e}");
                break;
            }
        };
        let connected = std::mem::replace(&mut server, next);
        tokio::spawn(serve_one(connected, handler.clone()));
    }
}

#[cfg(not(windows))]
async fn listen(endpoint: String, handler: Handler, ready: mpsc::Sender<Result<(), String>>) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::remove_file(&endpoint);
    let listener = match tokio::net::UnixListener::bind(&endpoint) {
        Ok(l) => l,
        Err(e) => {
            let _ = ready.send(Err(format!("cannot bind {endpoint}: {e}")));
            return;
        }
    };
    // Only this account may connect.
    let _ = std::fs::set_permissions(&endpoint, std::fs::Permissions::from_mode(0o600));
    let _ = ready.send(Ok(()));
    loop {
        let stop = handler.stop.clone();
        tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => { tokio::spawn(serve_one(stream, handler.clone())); }
                Err(_) => break,
            },
            _ = stop.notified() => break,
        }
    }
    let _ = std::fs::remove_file(&endpoint);
}

impl crate::App {
    /// `--agent-host <endpoint> [--root DIR]...`: listen for an MCP relay on `endpoint`. With no
    /// root, files go in a fresh folder under the temp directory. Nothing is listening without the
    /// flag (`mcp-server-spec.md` §9.1: attach mode is opt-in).
    pub(crate) fn start_agent_host(&mut self, ctx: &eframe::egui::Context) {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let Some(at) = args.iter().position(|a| a == "--agent-host") else { return };
        let endpoint = args.get(at + 1).cloned().unwrap_or_else(default_endpoint);
        let mut roots: Vec<std::path::PathBuf> = args
            .windows(2)
            .filter(|w| w[0] == "--root")
            .filter_map(|w| std::fs::create_dir_all(&w[1]).ok().and_then(|_| std::path::Path::new(&w[1]).canonicalize().ok()))
            .collect();
        if roots.is_empty() {
            let fresh = std::env::temp_dir().join(format!("fieldform-agent-{}", std::process::id()));
            if std::fs::create_dir_all(&fresh).is_ok()
                && let Ok(real) = fresh.canonicalize()
            {
                roots.push(real);
            }
        }
        self.agent.roots = roots;
        self.agent.run_timeout = args
            .iter()
            .position(|a| a == "--agent-timeout")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|s| s.is_finite() && *s > 0.0)
            .map(std::time::Duration::from_secs_f64);
        let ctx = ctx.clone();
        let wake: Waker = Arc::new(move || ctx.request_repaint());
        match start(endpoint, self.agent.cancel.clone(), wake, HostMode::Launch) {
            Ok(host) => {
                self.say(format!("Agent host listening on {}", host.endpoint));
                self.agent_host = Some(host);
            }
            Err(why) => self.warn(why),
        }
    }

    /// Answers every job that has arrived, between frames. The document is only touched here.
    pub(crate) fn drive_agent(&mut self) {
        let Some(mode) = self.agent_host.as_ref().map(|h| h.mode) else { return };
        self.finish_snapshot();
        // A picture in flight holds everything else: the next call could aim the camera elsewhere.
        if self.agent.snapshot.is_some() {
            self.wake_agent();
            return;
        }
        if mode == HostMode::Attach {
            // A person's session reaches only the folder their document is in.
            self.agent.roots = self.attach_roots();
        }
        while let Some(job) = self.agent_host.as_ref().and_then(|h| h.jobs.try_recv().ok()) {
            self.agent.held.push_back((job, std::time::Instant::now()));
        }
        while let Some((_, since)) = self.agent.held.front() {
            let since = *since;
            // **The person's input wins** (D5): a call that arrives mid-gesture waits, in order, and
            // never runs in the middle of it. After the allowance it is told so rather than left.
            if mode == HostMode::Attach && self.person_is_busy() && self.agent.held.front().is_some_and(|(j, _)| j.tool != "session_info") {
                if since.elapsed() >= BUSY_ALLOWANCE {
                    let Some((job, _)) = self.agent.held.pop_front() else { break };
                    let mut e = tool_error("busy", "the person at this session is in the middle of something");
                    e.verb = Some(job.tool.clone());
                    let mut reply = Reply::err(&e);
                    reply.json["error"]["retry_after_ms"] = serde_json::json!(BUSY_RETRY_MS);
                    let _ = job.reply.send(Answer { reply, image: None });
                    continue;
                }
                self.wake_agent();
                return;
            }
            let Some((job, _)) = self.agent.held.pop_front() else { break };
            if job.tool == "snapshot" {
                self.begin_snapshot(job);
                break;
            }
            let reply = self.agent_handle(&job.tool, &job.args);
            let _ = job.reply.send(Answer { reply, image: None });
        }
    }

    fn wake_agent(&self) {
        if let Some(host) = &self.agent_host {
            (host.wake)();
        }
    }

    /// Whether a gesture or a tool session of the person's is open.
    pub(crate) fn person_is_busy(&self) -> bool {
        #[cfg(test)]
        if self.agent.pretend_busy {
            return true;
        }
        self.session.is_some()
            || self.cage_session.is_some()
            || self.cage_tool.is_some()
            || self.point_drag.is_some()
            || self.gizmo_drag.is_some()
            || self.material_gesture.is_some()
            || self.gesture.is_some()
    }

    /// The one folder an attached agent may use: the one the open document is in.
    fn attach_roots(&self) -> Vec<std::path::PathBuf> {
        self.file
            .as_ref()
            .and_then(|(path, _)| path.parent())
            .and_then(|dir| dir.canonicalize().ok())
            .into_iter()
            .collect()
    }

    /// Allow agent (`mcp-server-spec.md` §9.1): publishes the endpoint, or takes it away and
    /// drops whoever is connected. Does nothing when the host was started by `--agent-host`,
    /// which is a session the agent owns.
    pub(crate) fn set_allow_agent(&mut self, allow: bool, ctx: &eframe::egui::Context) {
        if self.agent_host.as_ref().is_some_and(|h| h.mode == HostMode::Launch) {
            return;
        }
        self.allow_agent = allow;
        if !allow {
            if self.agent_host.take().is_some() {
                self.say("Agent access is off: the endpoint is closed.");
            }
            return;
        }
        if self.agent_host.is_some() {
            return;
        }
        let endpoint = self.agent_endpoint.clone().unwrap_or_else(default_endpoint);
        let ctx = ctx.clone();
        let wake: Waker = Arc::new(move || ctx.request_repaint());
        match start(endpoint, self.agent.cancel.clone(), wake, HostMode::Attach) {
            Ok(host) => {
                self.say(format!("Agent access is on, listening on {}", host.endpoint));
                self.agent_host = Some(host);
            }
            Err(why) => {
                self.allow_agent = false;
                self.warn(why);
            }
        }
    }
}
