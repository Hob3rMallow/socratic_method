# Probability-guided continuity for frozen F0

The separate, opt-in `socratic-continuity` command adds longer, curved paths on
top of an existing checked `socratic-repair` output. It does not change F0/200k,
T=.35, the native repair default, or any historical training/export launcher.

The development-selected v2 policy added 764 per-plane corridors (357 longer
than 21 pixels) on a 27-cube development block. Without control-based tuning,
it added 515 corridors (210 longer than 21 pixels) on a fresh disjoint block.
Central-cube additions were 825 and 2,004 voxels over the native-filled masks.
All existing foreground was preserved. These are continuity/change measures,
not anatomical correctness rates or new Dice scores.

## Inputs and check-only usage

Use the original contiguous inference grid, its checked native-filled grid,
and a new separate output directory:

```powershell
python -m socratic_method.continuity INFERENCE_GRID NATIVE_REPAIRED_GRID NEW_OUTPUT `
  --voxel-size-um 8.64
# Checks only. Add --run to create the repaired output.
```

After installing this repository, `socratic-continuity` is the equivalent console
entry point. No installation or automatic deployment was performed by the study.
From a checkout, set `PYTHONPATH` to its `src` directory when needed. Use the
same NumPy/SciPy/scikit-image environment recorded in the study for exact parity.

Required: frozen F0 checkpoint identity, T=.35, halo32, mirror TTA, BF16;
128-cube binary masks and matching float16/float32 probability TIFFs; dense real
context with at least three cubes per axis; explicit coarse voxel pitch in
[8,10] um; explicit umbilicus; a completed unchanged native-repair receipt with
the measured pinned binary identity. The old native binary must remain available
for identity validation, but this command does not execute it.

The measured pitch was 8.640 um, not the fine-grid 2.399 um or training 9.362 um.
Probability dtypes must be consistent; stored precision is preserved. A gross
probability/raw-mask disagreement is rejected, allowing a narrow .349-.351 band
for float16 quantization around the original float32 threshold decision.

## Policy and boundaries

`f0-probability-continuity-v2` uses seeded paths at P>=.20, maximum path length
96 pixels, mean gap P>=.25, narrow continuous-probability route refinement,
endpoint direction/anisotropy/outward checks, ridge separation, limited detour
and local turn, and neighboring probability support. It never simply thresholds
the whole volume at .20. Existing native-filled foreground is protected.

The native global radial endpoint limit of 3 pixels **does not apply to this
new stage**. Folded sheets change radius along legitimate local continuations;
local checks replace that constraint. Do not claim the old radial certificate.

Only Z planes 8 through depth-9 are processed. Skeleton proposals have an
8-pixel XY margin; route refinement can move by 3 pixels and painting by 1.
Use interior cubes with real neighboring context. The checked loader supports
dense rectangular grids; qualification here covers two 384-cubed blocks from
one scroll, not large-volume tiling equivalence or independent-scroll validation.

## Execution and recovery

One CPU process, no child workers, inference, training, cutting or uploads.
Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to `1` for the
measured single-thread environment. The default 8 GiB memory preflight is an
estimate, not an OS-enforced process limit. Large workloads need planned regions.

Output is first written to `NEW_OUTPUT.partial-PID` under an exclusive
`NEW_OUTPUT.continuity.lock`. Inputs and code are rehashed before publication;
cubes, a complete corridor/refusal JSONL ledger and a receipt are retained.
The independent audit checks zero erasure, two original component endpoints
becoming face-connected, third-component/detached paint, and actual TIFF counts.
Single one-voxel brush aprons must be immediately adjacent to declared anchors;
they are counted separately from two-contact bridge strokes.

Failure or cancellation preserves the partial and diagnostics. There is no
timeout-based lock recovery and no overwrite/resume. Inspect the recorded PID
and artifacts before choosing a new output name. Do not remove unknown locks.

## Interpretation

The CT-aligned review supports useful local recovery, including roughly 54- and
65-pixel central-cube connections. Ambiguous packed folds and single-plane
patches remain. The neighboring-P guard rejected no real proposals; it is
self-consistency, not independently demonstrated anatomical safety. The
implementation is locally verified for opt-in review, not an automatic
production default or a global sheet reconstruction guarantee.

See [the evidence record](../recipes/f0_continuity_20260907/README.md) and its local
HTML report for every accepted card, refusal ledger, risk-first orthogonal views,
development v1 failures, fresh-control results and exact package replay.
