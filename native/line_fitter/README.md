# 2D line fitter

This directory is the standalone extraction of the calibrated prediction
postprocessor formerly embedded in `vesuvius-c`. The recommended path is an
additive-only slice fitter: trace skeleton endpoints, gate symmetric candidate
pairs, require cross-plane connection-track support, and paint cubic Bezier
joins. The radial gate should be armed with an umbilicus whenever one is known.

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

Run it on a cube grid whose predictions are under `cubes_PRED`:

```bash
pred_fixup INPUT_GRID OUTPUT_GRID \
  --umb-y UMBILICUS_Y --umb-x UMBILICUS_X \
  --reach-safe 6 --reach-max 12 --radial-dr 4 --min-support 3
```

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
