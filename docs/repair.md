# Contiguous F0 gap repair

The measured `f0-contiguous-gap-repair-v1` profile recovers more local gaps than
the conservative preset used in the original F0 report. It keeps the frozen
**F0/200k model and T=.35 unchanged**. This is a separate additive postprocessor,
not a training recipe, threshold change, model release or automatic deployment.

## Evidence and scope

Two disjoint PHerc1447 blocks were predicted with real neighboring context:
27 cubes / 384³ voxels each, at **8.640 µm** pitch. Four presets were compared
on development; only the old preset and the development-selected profile were
run on the reserved control. Both use exactly the same raw F0 predictions.

| Block | Per-plane joins, old → new | Connection tracks, old → new | Added voxels, old → new | Central cube additions |
| --- | ---: | ---: | ---: | ---: |
| Development | 96 → 286 | 23 → 195 | 1,364 → 3,970 | 0 → 42 |
| Reserved control | 59 → 97 | 12 → 55 | 961 → 1,174 | 0 → 25 |

No original F0 foreground was erased. An independent audit of all 768 planes
found no detached repair strokes and no strokes touching more than two raw
2D components. CT-aligned representative and risk-first orthogonal views
support useful short continuations. These checks **do not establish anatomical
ground-truth precision or certify global cross-wrap topology**.

The tighter radial limit also drops 319 development and 309 control voxels
that the old filler added. Additivity is relative to **raw F0**, not to the old
filled result. All repaired output remains separate from the original masks.

This remains a local line-gap repair. Across both blocks, 5,144 added voxels
are only about 0.025% of the original foreground. It does not reconstruct broad
missing sheets, remove existing blobs, or claim a new official Dice score.

See the [repair evidence record](../recipes/f0_repair_20260907/README.md),
including the local HTML report, all 250 accepted-track cards and risk-first
Z/Y/X views. The [training review](../recipes/final_c3_250k_20260907/review.md)
is preserved unchanged as the pre-repair analysis.

## What changed

| Parameter | Old report preset | Measured profile |
| --- | ---: | ---: |
| Maximum reach, coarse-grid pixels | 12 | 21 |
| Radial displacement limit, pixels | 4 | 3 |
| Hard minimum connection support | 3 | 0 |
| Near reach / minimum score / paint radius | 6 / .30 / 1 | unchanged |
| Tracking and geometric pair gates | on | on |
| Bridge cutting / slab splitting | off | off |

Support zero removes the hard final persistence veto, **not tracking**.
An explicit umbilicus arms the radial guard. The native matching algorithm
and binary were not changed. These settings follow the existing radial-armed
source defaults, with F0-specific checked packaging and new measured evidence.
The legacy internal preset name `extended_safe` is not a safety certification.

## Checked command

After installing this repository, `socratic-repair` is available. The equivalent
module entry point is `python -m socratic_method.repair`. Existing
`socratic-train` and `socratic-export` still describe historical v31.

First check the input and print a plan; without `--run` no native worker or
output directory is created:

```powershell
socratic-repair INPUT_GRID NEW_OUTPUT_GRID `
  --binary D:/work/vesuvius-c/build/c3-human-line-fitter_20260905/extracted/Release/pred_fixup.exe `
  --voxel-size-um 8.64
```

Add `--run` to execute the checked plan. Choose a **new output path** and keep
the input predictions. The command launches one native CPU worker, with
1–16 threads (default 16), and writes a SHA-256-bearing `repair_receipt.json`.
It independently checks the output cube inventory, binary encoding, input
identity and zero-erasure contract before publishing the output directory.

Required input:

- `cubes_PRED/zZZZZZ_yYYYYY_xXXXXX.tif`: uint8 binary 0/255, 128³, with world
  origins aligned to 128. All coordinates in the bounding box must be present;
  the initial wrapper requires at least three cubes per axis.
- `manifest.json` or `source_manifest.json`: `chunk_size: 128` and finite
  `umbilicus_yx` in those same world-pixel coordinates.
- `provenance.json`, or `manifest.json` when provenance is absent: `threshold:
  0.35` and either `checkpoint_sha256` or `checkpoint.sha256` matching the frozen
  model. When `target_cube_ids` is present, the TIFF inventory must match it.

The explicit pitch argument is an operator assertion, not a registration
measurement. The coarse profile rejects pitches outside 8–10 µm; **the new
comparison is at 8.640 µm only**. Verify the actual volume pitch and umbilicus;
do not transfer these pixel settings to a 2.399 µm high-resolution grid.
Outer block edges still lack external context. Use/evaluate interior regions
with genuine neighbors rather than padding absent predictions with zeros.

The initial profile pins the measured Windows binary:

```text
Model  54db9a59cb602ff08e3dda0acc1195ad980ab05fb660332d32fb436a0d49a175
Binary 54fe434f76beb8c1eff58cdb1bd72200e4630497e23539cd467e4e64360f1e13
```

The binary is an external local artifact, not committed to this repository.
A rebuilt or other-platform binary needs a separately verified profile; the
command deliberately rejects a silent executable substitution.

## Resource and interruption behavior

The wrapper estimates native working memory and refuses plans above
`--max-work-gib` (default 32). This is a preflight estimate, **not an OS memory
limit** or automatic whole-scroll tiler. Plan bounded contiguous regions.

An exclusive output lock prevents concurrent writes to the same destination.
On ordinary cancellation the wrapper stops only its own native child. A power
failure can still leave a lock, PID record and `.partial-PID` directory. These
are never reclaimed by timeout or overwritten automatically: inspect process
ownership and diagnostics first. Failed runs stay uncommitted for recovery.

No repair is automatically deployed. Successful completion proves execution
and additivity, not anatomical correctness. Review accepted connections and
cross-wrap risks before applying repaired masks downstream.

## Next boundary

For larger missing stretches, the next intervention should use cached
probabilities plus CT/orientation/radial evidence to propose longer continuations
between confident sheet segments. That mechanism is **not implemented by this
profile**. Judge recovered useful paths alongside wrong-sheet connections on
new regions; do not simply widen every gate or lower the global threshold.
