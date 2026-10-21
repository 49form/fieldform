#!/usr/bin/env python3
"""Prepend an SPDX licence identifier and copyright line to every first-party source file.

Idempotent: a file that already carries an SPDX line is left alone. Vendored code under
`vendor/` is never touched; it keeps its own notices.

    python tools/release/add_spdx_headers.py [--year 2026] [--holder 49form] [--check]

--check changes nothing and exits 1 if any file lacks a header (used by CI).
"""
import argparse, pathlib, subprocess, sys

LAYER = {
    "fieldform-cage": "MPL-2.0", "fieldform-subd": "MPL-2.0", "fieldform-field": "MPL-2.0",
    "fieldform-scene": "MPL-2.0", "fieldform-ops": "MPL-2.0", "fieldform-io": "MPL-2.0",
    "fieldform-gpu": "MPL-2.0", "fieldform-app": "GPL-3.0-only", "fieldform-import": "GPL-3.0-only",
}
EXTS = {".rs", ".wgsl"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2026")
    ap.add_argument("--holder", default="49form")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    missing = changed = 0
    for rel in subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0"):
        p = pathlib.PurePosixPath(rel)
        if not rel or p.suffix not in EXTS or p.parts[0] not in LAYER:
            continue
        path = root / rel
        raw = path.read_bytes()
        bom = raw.startswith(b"\xef\xbb\xbf")
        text = raw[3:].decode("utf-8") if bom else raw.decode("utf-8")
        if "SPDX-License-Identifier" in text[:400]:
            continue
        missing += 1
        if a.check:
            print("missing header:", rel)
            continue
        nl = "\r\n" if "\r\n" in text else "\n"
        head = (f"// SPDX-License-Identifier: {LAYER[p.parts[0]]}{nl}"
                f"// Copyright (c) {a.year} {a.holder}{nl}")
        path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + (head + text).encode("utf-8"))
        changed += 1
    print(f"{missing} without header, {changed} written")
    return 1 if (a.check and missing) else 0

sys.exit(main())
