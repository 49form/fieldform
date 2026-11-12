# Testing against real CAD

Three machined parts, rebuilt in fieldform from their engineering drawings and measured against
the manufacturer's own STEP model. The parts come from the U.S. National Institute of Standards
and Technology (NIST), which publishes them with drawings for exactly this kind of test. They are
public domain, and are fetched from NIST rather than kept here.

The closed-form benchmarks test what can be worked out by hand. These test what cannot: whether a
field model built the way a person would build it lands on a real part.

| Part | What it exercises | Lines |
|---|---|---|
| **FTC-07**, an enclosure, 311 × 223 × 136 mm | Drafted walls, filleted corners, pockets, bosses, counterbores, ports | 172 |
| **D2MI 905**, a shouldered spacer, 25 mm long | Turned geometry, a small fillet, a bore and a drilled hole with a 118° point | 8 |
| **D2MI 906**, a heatsink top, 103 × 70 mm | Chamfered plate, filleted standoffs, pads, fins with a lead-in, round-topped pins, counterbored drill points | 75 |

## Running it

```bash
python benchmarks/reference-cad/fetch.py
cargo build -p fieldform-app
cargo build -p fieldform-import
```

`fetch.py` downloads two archives from NIST (14 MB together) and keeps six files in `models/`, each
checked against the SHA-256 the results below were measured on. It refuses a file that differs,
because a comparison against a later revision is not this comparison.

Build each part from its script, which saves the document beside it:

```bash
target/debug/fieldform --run benchmarks/reference-cad/d2mi-905.ffc --exit
```

Then compare it with NIST's model:

```bash
cargo run -p fieldform-io --example compare_to_step -- benchmarks/reference-cad/d2mi-905.fform benchmarks/reference-cad/models/827-9999-905.stp map=x,y,z cell=0.1 tolerance=0.019
```

| Part | `map` | `cell` | `tolerance` |
|---|---|---|---|
| FTC-07 | `x,-z,y` | `1.0` | `0.76` |
| D2MI 905 | `x,y,z` | `0.1` | `0.019` |
| D2MI 906 | `x,y,z` | `0.25` | `0.203` |

`map` says where NIST's axes went: FTC-07 is modelled Z-up, and NIST's file is Y-up. `tolerance` is
the drawing's own limit, explained per part below. The comparison reads NIST's surface through the
importer, so both models are signed distance fields, and measures in both directions: a feature
lost shows in one, and a feature added only in the other. It then lists the worst places at least
5 mm apart, so a miss can be traced to a feature. Its output for each part is in `results/`.

## Limits

| Limit | Value | Source |
|---|---|---|
| Drawing | The drawing's own tolerance, halved where it is a profile | The drawing. Missing it means the wrong part |
| Worst point | 0.1 mm | Proposed |
| 99% of points | 0.05 mm | Proposed |
| Solid that differs | 0.1% of the volume | Proposed, and wrong for thin parts (see 905) |

Nothing under 0.02 mm is measurable: NIST's model arrives as triangles at that deflection, so it is
how far the reference sits from its own true surface.

A real profile tolerance is measured from datums. Comparing distance to the nominal surface against
half of one is a rough check, not an inspection.

## Results, 2026-11-02

### FTC-07

| Limit | Result | Verdict |
|---|---|---|
| Drawing, ±0.76 mm | Worst 0.446 mm | Pass |
| Worst point, 0.1 mm | 0.446 mm | Fail |
| 99% of points, 0.05 mm | 0.034 mm | Pass |
| Solid that differs, 0.1% | 0.013% | Pass |

The worst point is NIST's model, not the kernel. Three of the four bosses have a 1.5 mm fillet
where the top meets the angled wall, and the fourth, at (+X, +Y), has none. The rebuild rounds all
four with one blend, so that one place differs by the fillet. The next worst are the bottom
corners, where three fillets meet, and the model reports those blends as approximate.

### D2MI 905

| Limit | Result | Verdict |
|---|---|---|
| Drawing, tightest band ±.0015 in as a radius, ±0.019 mm | Worst 0.0035 mm | Pass |
| Worst point, 0.1 mm | 0.0035 mm | Pass |
| 99% of points, 0.05 mm | 0.0035 mm | Pass |
| Solid that differs, 0.1% | 0.68% | Fail, and the limit is what is wrong |

All nine of NIST's vertices lie on the rebuilt surface to within 0.000007 mm. The volume is
272.25 mm³ against 272.27 worked by hand.

The failed row measures NIST's triangles. Each chord cuts inside a round surface, so NIST's
solid is 0.2% short of the exact part, and chord sag averaging 0.0018 mm over 777 mm² of surface is
the 1.4 mm³ found. A finer grid does not shrink it. A limit as a share of volume favours thick
parts, and this one's walls are 0.26 to 0.74 mm. Dividing the differing volume by the surface area
gives a mean thickness, 0.0017 mm here, which can be held to the 0.02 mm deflection instead.

### D2MI 906

| Limit | Result | Verdict |
|---|---|---|
| Drawing, profile .016 in halved, ±0.203 mm | Worst 0.009 mm | Pass |
| Worst point, 0.1 mm | 0.009 mm | Pass |
| 99% of points, 0.05 mm | 0.005 mm | Pass |
| Solid that differs, 0.1% | 0.025% | Pass |

The worst places are the four round pin tops, inside NIST's 0.02 mm deflection. The drawing leaves
the pin diameter, the pad depths and the drill points to the model, so those were read from the
STEP file.

## What the tests found

Each was found by these comparisons or by looking, and fixed before these results were taken.

- **A loft cut small fillets off long outlines,** up to 3.9 mm, with no warning. Its outline was
  resampled evenly by length. It now keeps the ends of long straight runs and spaces samples by
  turn as well as length.
- **The measuring tools read a small extrude 0.0125 mm proud.** They compiled the field at
  0.05 mm, where every profile under about 100 mm across answers from a display grid. They now
  compile at 0.001 mm, the tolerance `expect` already used. The picture was never wrong, because the
  viewport finishes on the exact outline.
- **Imported meshes were drawn blurred,** from a 64-cell grid. Every number was clean, and a person
  looking at the part saw it. The grid is now 256 cells.

Still open:

- A STEP file of triangles alone, with no exact surfaces, does not import.
- The volume limit above is unfair to thin parts.

The console's `expect volume` used to refuse all three parts at any tolerance: a fillet sized its
first grid finer than two levels of it could fit in one measurement, so there was nothing to
estimate an error from. The first grid now coarsens until both levels fit, giving way for a
fillet's radius but never for a thin wall, which could vanish from a coarse mesh. Each part now
gets a verdict:

| Part | Volume measured | Error estimate | Grid |
|---|---|---|---|
| FTC-07 | 1,726,723 mm³ | ± 85 mm³ | 1 mm |
| D2MI 906 | 22,869.4 mm³ | ± 2.7 mm³ | 0.25 mm |
| D2MI 905 | 272.271 mm³ | ± 0.008 mm³ | 0.031 mm |

The scripts still only build and save, and the comparison above is the check.

## Adding a part

Rebuild from the drawing, and take from the STEP file only what the drawing leaves to the model.
Sample the STEP file's vertices as you go: a model can pass every structural check and be the wrong
shape. Add its archive and hashes to `fetch.py`, its script here with an `expect`, and its row to
the tables. Check the source's terms first: public domain or a licence that allows redistribution,
attributed.
