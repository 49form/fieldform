// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! `snapshot` (`mcp-server-spec.md` §5.11): the viewport's picture, as an image.
//!
//! **A tool that outlives the frame it began in.** A still converges over several frames, so the
//! job is begun by the same `snap` verb a typed line uses and answered when that verb has written
//! its file. Nothing else is run while one is in flight, since the next line could aim the camera
//! somewhere else.
//!
//! **What it does not do yet, said rather than approximated.** The picture is the viewport's own
//! size (`snap size=` is refused by the verb for the same reason), and a view moves the one camera
//! there is. That is right for a host the agent owns and wrong for a person's session, so attach
//! mode must give this its own render target before it offers the tool (Phase D, question 5 of
//! the discovery report). `size`, `camera`, `show`, `hide` and `section` are therefore not offered.

use super::server::{Answer, Job};
use super::tools::Reply;
use super::*;
use serde_json::json;

pub(crate) struct SnapshotWait {
    pub(crate) job: tokio::sync::oneshot::Sender<Answer>,
    pub(crate) path: std::path::PathBuf,
    pub(crate) view: String,
}

const VIEWS: [&str; 8] = ["iso", "front", "back", "left", "right", "top", "bottom", "current"];

fn fail(error: LineError) -> Answer {
    Answer {
        reply: Reply { json: json!({ "ok": false, "error": super::tools::error_json(&error) }), is_error: true },
        image: None,
    }
}

/// A PNG's width and height, from its header.
fn png_size(bytes: &[u8]) -> Option<(u32, u32)> {
    let w = bytes.get(16..20)?;
    let h = bytes.get(20..24)?;
    Some((u32::from_be_bytes(w.try_into().ok()?), u32::from_be_bytes(h.try_into().ok()?)))
}

impl App {
    pub(crate) fn begin_snapshot(&mut self, job: Job) {
        let view = job.args.get("view").and_then(|v| v.as_str()).unwrap_or("iso").to_lowercase();
        if !VIEWS.contains(&view.as_str()) {
            let mut e = tool_error("bad_value", format!("`{view}` is not a view"));
            e.key = Some("view".into());
            e.accepted = VIEWS.iter().map(|v| v.to_string()).collect();
            let _ = job.reply.send(fail(e));
            return;
        }
        if self.agent_host.as_ref().is_some_and(|h| h.mode == super::server::HostMode::Attach) {
            let e = tool_error(
                "not_available",
                "a snapshot would move the camera of the person at this session; it is offered only to a session the agent started",
            );
            let _ = job.reply.send(fail(e));
            return;
        }
        let dir = self.agent.roots.first().cloned().unwrap_or_else(std::env::temp_dir);
        self.agent.snapshots += 1;
        let path = dir.join(format!("snapshot-{}-{}.png", std::process::id(), self.agent.snapshots));
        let file = path.to_string_lossy().replace('\\', "/");
        let line = if view == "current" {
            format!("snap file=\"{file}\"")
        } else {
            format!("snap file=\"{file}\" view={view}")
        };
        self.commit_session();
        let was_running = std::mem::replace(&mut self.script_running, true);
        let outcome = self.run_line(&line);
        self.script_running = was_running;
        if !outcome.ok || self.pending_snap.is_none() {
            let e = outcome.error.unwrap_or_else(|| {
                tool_error("refused", outcome.said.unwrap_or_else(|| "the snapshot could not start".into()))
            });
            let _ = job.reply.send(fail(e));
            return;
        }
        self.agent.snapshot = Some(SnapshotWait { job: job.reply, path, view });
    }

    /// Answers the snapshot in flight once the verb has finished with it.
    pub(crate) fn finish_snapshot(&mut self) {
        if self.agent.snapshot.is_none() || self.pending_snap.is_some() {
            return;
        }
        let Some(wait) = self.agent.snapshot.take() else { return };
        let bytes = std::fs::read(&wait.path).ok();
        let _ = std::fs::remove_file(&wait.path);
        let answer = match bytes {
            None => fail(tool_error("timeout", "the view did not finish a still, so no picture was written")),
            Some(bytes) => {
                let (w, h) = png_size(&bytes).unwrap_or((0, 0));
                let bodies: Vec<String> = self.drawn_bodies().into_iter().map(|id| self.body_label(id)).collect();
                Answer {
                    reply: Reply {
                        json: json!({
                            "ok": true,
                            "view": wait.view,
                            "width": w,
                            "height": h,
                            "bodies_drawn": bodies,
                            "note": "the picture is the viewport's own size; this host's camera was moved to take it",
                        }),
                        is_error: false,
                    },
                    image: Some(bytes),
                }
            }
        };
        let _ = wait.job.send(answer);
    }
}

pub(crate) fn encode_base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_the_standard_vectors() {
        for (plain, coded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(encode_base64(plain.as_bytes()), coded);
        }
    }
}
