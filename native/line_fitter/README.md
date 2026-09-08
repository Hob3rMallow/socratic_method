# 2D line fitter

This directory is the standalone extraction of the calibrated prediction
postprocessor formerly embedded in `vesuvius-c`. The recommended path is an
additive-only slice fitter: trace skeleton endpoints, gate symmetric candidate
pairs, score cross-plane connection-track evidence, and paint cubic Bezier
joins. A hard support cutoff is conditional or explicitly configured. The radial gate should be armed with an umbilicus whenever one is known.

## Measured contiguous F0 profile

The separate [socratic-repair wrapper](../../docs/repair.md) now checks and runs
the radial-armed 21 / 3 / support-0 profile on dense frozen-F0 prediction grids.
Development per-plane joins rose 96 to 286; the reserved control rose 59 to 97,
with zero original foreground erased. These are local repair improvements,
not anatomical precision scores or universal topology certification.
The [evidence record](../../recipes/f0_repair_20260907/README.md) includes all
accepted-track cards, independent contact checks, control results and limits.
Native source/defaults are unchanged; the wrapper pins the measured Windows
binary, keeps tracking and geometry gates on, and never enables cutting.

## Build the dependency-free core tests

```bash
cmake -S native/line_fitter -B build/line-fitter
cmake --build build/line-fitter --config Release
ctest --test-dir build/line-fitter -C Release --output-on-failure
```

The default target does not require libtiff or OpenMP. To build the production
grid command, install TIFF and an OpenMP-capable C toolchain, then configure:

```bash
cmake -S native/line_fitter -B build/line-fitter \
  -DSOCRATIC_BUILD_PRED_FIXUP=ON
cmake --build build/line-fitter --config Release
```

The historical conservative example below runs on a cube grid whose predictions
are under `cubes_PRED`. These explicit flags are **not the implementation's
automatic defaults** and are not a newly validated F0 production preset:

```bash
pred_fixup INPUT_GRID OUTPUT_GRID \
  --umb-y UMBILICUS_Y --umb-x UMBILICUS_X \
  --reach-safe 6 --reach-max 12 --radial-dr 4 --min-support 3
```

## Source defaults versus the F0 report preset

| Parameter | Explicit conservative example / F0 report | Repository source default |
| --- | ---: | ---: |
| Maximum reach (pixels) | 12 | 21 |
| Maximum radial displacement (pixels) | 4 | 3 |
| Minimum connection support | 3 | 0 with radial guard armed; 3 otherwise |

The radial guard is armed by supplying both umbilicus coordinates. Automatic
support zero removes the hard final support rejection; it does **not** disable
tracking or the remaining geometry gates. The report preset is stricter on
persistence and reach, but looser on radial displacement. Source history records
a full-turn fusion that motivated tightening the radial default to 3.

The completed F0 report tested the explicit conservative flags above on six
sparse cubes. At 200k, 26 of 31 proposed per-plane joins were dropped by support,
and 7,608 of 9,328 traced endpoints were boundary/absent-region exclusions.
Only 81 voxels were added. The
[full review](../../recipes/final_c3_250k_20260907/review.md) explains why this is
not a production-scale recovery result.

Verify input-grid pitch, umbilicus and topology behavior before transferring
pixel settings. High-resolution adjudication mentioned in source comments
does not itself establish the voxel pitch of the fitter input. No defaults,
native code or report parameters were changed by this documentation update.

Use `--dry-run` for manifest/overlay review before writing TIFF output. Avoid
`--no-tracks` except for diagnosis. `--cut-bridges` is intentionally not a
recommended default because it erases predictions and breaks the additive
contract.

## Retained research modules

`bridge_scan` reports thin-neck weld candidates, but cutting remains off.
`slab_split` and visualization code are retained for provenance; enable
`SOCRATIC_BUILD_RESEARCH_TESTS` to compile their tests. Slab splitting measured
0/246 fused-run separations in its mesh-stage evaluation and is not part of the
default pipeline.

The extracted source history begins with commits `37314428ed09a4fde3f5159159bcadf5bf9deaee`
(initial fitter), `784bd282` (large-grid tiling and fixes), and `71b907a`
(negative slab-split result). See `docs/line_fitter.md` for measured evidence and
safety interpretation.
