#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 49form
"""Checks the block-family colour tokens (the block-family colour notes, section 5).

Pure Python, no dependencies. Run from the repository root:

    python tools/design/family_palette_check.py            # checks the table below
    python tools/design/family_palette_check.py tokens.json  # or a {"dark": {...}, "light": {...}} file

For each theme it simulates every token under protanopia, deuteranopia and tritanopia
(Machado, Oliveira & Fernandes 2009, severity 1.0, applied in linear sRGB), measures every pair
with CIEDE2000, and measures each token's WCAG 2.x contrast against the theme's panel colour.
It exits 1 when a threshold in THRESHOLDS is not met, and prints the table either way, so the
numbers in the spec can be copied from here rather than typed.

The neutral `reference` token is reported but never fails the pair test: its border is dashed,
and the dash is what tells it apart (the notes, section 4.4). Every chromatic pair must pass on
colour alone.
"""
import json
import math
import sys

TOKENS = {
    "dark": {
        "combine": "#9dc9f7",
        "create": "#9d92de",
        "modify": "#85b286",
        "curve": "#4f9c8e",
        "cage": "#d79dbd",
        "file": "#9c76a4",
        "reference": "#7a7a7a",
    },
    "light": {
        "combine": "#437eb9",
        "create": "#5e5197",
        "modify": "#568157",
        "curve": "#176c5f",
        "cage": "#9d6785",
        "file": "#603e68",
        "reference": "#8c8c8c",
    },
}

PANEL = {"dark": "#1b1b1b", "light": "#f8f8f8"}

THRESHOLDS = {
    # minimum CIEDE2000 between any two chromatic tokens, under normal vision and every simulation
    "pair_min": {"dark": 10.0, "light": 9.5},
    # minimum WCAG contrast of every token against the panel (3:1 is the floor for UI graphics)
    "contrast_min": {"dark": 4.0, "light": 3.0},
}

CVD = {
    "protanopia": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "deuteranopia": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "tritanopia": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}
CONDITIONS = ["normal"] + list(CVD)


def hex_to_linear(hx):
    hx = hx.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(hx[i:i + 2], 16) / 255
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return out


def matmul(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def simulate(lin, cond):
    if cond == "normal":
        return lin
    return [min(1.0, max(0.0, c)) for c in matmul(CVD[cond], lin)]


def linear_to_lab(lin):
    m = [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
    white = [0.95047, 1.0, 1.08883]
    xyz = [v / w for v, w in zip(matmul(m, lin), white)]

    def f(t):
        return t ** (1 / 3) if t > (6 / 29) ** 3 else t / (3 * (6 / 29) ** 2) + 4 / 29

    fx, fy, fz = (f(t) for t in xyz)
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]


def ciede2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cbar ** 7 / (Cbar ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)

    def hp(a, b):
        if a == 0 and b == 0:
            return 0.0
        h = math.degrees(math.atan2(b, a))
        return h + 360 if h < 0 else h

    h1p, h2p = hp(a1p, b1), hp(a2p, b2)
    dLp, dCp = L2 - L1, C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dhp = h2p - h1p
        if dhp > 180:
            dhp -= 360
        elif dhp < -180:
            dhp += 360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbp = h1p + h2p
    else:
        s = h1p + h2p
        hbp = s / 2 if abs(h1p - h2p) <= 180 else ((s + 360) / 2 if s < 360 else (s - 360) / 2)
    T = (1 - 0.17 * math.cos(math.radians(hbp - 30)) + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6)) - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dtheta = 30 * math.exp(-((hbp - 275) / 25) ** 2)
    RC = 2 * math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / math.sqrt(20 + (Lbp - 50) ** 2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -math.sin(math.radians(2 * dtheta)) * RC
    return math.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH))


def contrast(lin_a, lin_b):
    def lum(l):
        return 0.2126 * l[0] + 0.7152 * l[1] + 0.0722 * l[2]

    la, lb = lum(lin_a), lum(lin_b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def check(tokens):
    ok = True
    for theme, table in tokens.items():
        names = list(table)
        lins = {n: hex_to_linear(table[n]) for n in names}
        panel = hex_to_linear(PANEL[theme])
        print(f"== {theme} (panel {PANEL[theme]})")
        for n in names:
            c = contrast(lins[n], panel)
            flag = "" if c >= THRESHOLDS["contrast_min"][theme] else "   <-- below contrast floor"
            if flag:
                ok = False
            print(f"  {n:10} {table[n]}  contrast {c:4.1f}:1{flag}")
        chromatic = [n for n in names if n != "reference"]
        for cond in CONDITIONS:
            labs = {n: linear_to_lab(simulate(lins[n], cond)) for n in names}
            worst, pair = 1e9, None
            for i, a in enumerate(chromatic):
                for b in chromatic[i + 1:]:
                    d = ciede2000(labs[a], labs[b])
                    if d < worst:
                        worst, pair = d, (a, b)
            grey = min(ciede2000(labs["reference"], labs[n]) for n in chromatic) if "reference" in names else float("nan")
            flag = "" if worst >= THRESHOLDS["pair_min"][theme] else "   <-- below pair floor"
            if flag:
                ok = False
            print(f"  {cond:13} closest chromatic pair {worst:5.1f} dE00 ({pair[0]}/{pair[1]}){flag};"
                  f" neutral to nearest family {grey:4.1f} (dashed, informational)")
    return ok


if __name__ == "__main__":
    tokens = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else TOKENS
    sys.exit(0 if check(tokens) else 1)
