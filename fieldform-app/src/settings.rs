// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 49form
//! What a person's settings file holds beside the display quality: the shelf and the count of
//! palette openings the first-run hint goes by (`interaction-spec.md` §10.2, §10.3).
//!
//! One JSON object at `display_quality::settings_path`, every entry kept when one is written, so a
//! setting added later never erases another. A file that cannot be read is an empty one: a setting
//! is never a reason not to start.

/// The shelf's command ids, in order.
pub const SHELF: &str = "shelf";
/// How many times the palette has been opened, until the hint stops showing.
pub const PALETTE_OPENS: &str = "palette_opens";

/// The palette openings after which the first-run hint is gone for good (§10.3).
pub const HINT_UNTIL: u64 = 3;

/// The settings file's entries, or none.
pub fn entries(text: Option<&str>) -> serde_json::Map<String, serde_json::Value> {
    match text.and_then(|t| serde_json::from_str(t).ok()) {
        Some(serde_json::Value::Object(map)) => map,
        _ => serde_json::Map::new(),
    }
}

/// `text` with `key` set to `value` and every other entry kept.
pub fn with_entry(text: Option<&str>, key: &str, value: serde_json::Value) -> String {
    let mut map = entries(text);
    map.insert(key.to_string(), value);
    serde_json::to_string_pretty(&serde_json::Value::Object(map)).expect("a JSON map always serializes")
}

/// The shelf a settings file holds: every id the registry still has, in order, once each.
pub fn shelf_of(text: Option<&str>) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    if let Some(serde_json::Value::Array(items)) = entries(text).get(SHELF) {
        for id in items.iter().filter_map(|v| v.as_str()) {
            if let Some(c) = crate::commands::by_id(id)
                && !out.contains(&c.id)
            {
                out.push(c.id);
            }
        }
    }
    out
}

/// How many times a settings file says the palette has been opened.
pub fn palette_opens_of(text: Option<&str>) -> u64 {
    entries(text).get(PALETTE_OPENS).and_then(|v| v.as_u64()).unwrap_or(0)
}

/// The settings file as it stands, if there is one.
pub fn read() -> Option<String> {
    std::fs::read_to_string(crate::display_quality::settings_path()?).ok()
}

/// Sets one entry in the settings file, keeping the rest.
pub fn write(key: &str, value: serde_json::Value) -> Result<(), String> {
    let path = crate::display_quality::settings_path().ok_or("this platform names no per-person settings directory")?;
    let before = std::fs::read_to_string(&path).ok();
    if let Some(folder) = path.parent() {
        std::fs::create_dir_all(folder).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, with_entry(before.as_deref(), key, value)).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writing the shelf keeps the display quality beside it, and reading it drops what the
    /// registry no longer has and anything listed twice.
    #[test]
    fn the_shelf_is_kept_beside_the_other_settings_and_read_back_clean() {
        let start = r#"{ "display_quality": "Full" }"#;
        let text = with_entry(Some(start), SHELF, serde_json::json!(["create.sphere", "no.such", "create.sphere", "create.extrude"]));
        assert!(text.contains("\"display_quality\": \"Full\""), "{text}");
        assert_eq!(shelf_of(Some(&text)), vec!["create.sphere", "create.extrude"]);
        assert!(shelf_of(None).is_empty(), "empty by default: no toolbar until one is built");
        let text = with_entry(Some(&text), PALETTE_OPENS, serde_json::json!(2));
        assert_eq!(palette_opens_of(Some(&text)), 2);
        assert_eq!(shelf_of(Some(&text)).len(), 2, "kept");
        assert_eq!(palette_opens_of(Some("not json")), 0, "a file that cannot be read is an empty one");
    }
}
