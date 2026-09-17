# Opt-in 3D sheet patches

The frozen F0/200k model at T=.35 is unchanged. This separate local CPU
postprocessor extends accepted v2 curves into explicit joint 3D triangle patches
and additive masks. It is not a new training recipe, a replacement default, or
an anatomical identity guarantee. See the [evidence record](../recipes/f0_sheet_patches_20260907/README.md).

## Checked invocation

From an installed/checkout environment with `socratic_method` on the Python path:

```powershell
python -m socratic_method.sheet_patches CONTINUITY_DIR RAW_GRID NEW_OUTPUT --voxel-size-um 8.64
```

The default checks inputs and prints a plan; it does not create output. Add
`--run` to execute one CPU process. A newly installed package exposes the
equivalent `socratic-sheet-patches` entry point. Existing editable environments
may need their entry points refreshed; the `python -m` form works directly.
No inference, native executable or training process is started by this command.

Inputs must be a completed checked `socratic-continuity` v2 directory, the
same real uint8 raw CT grid used by inference, and an entirely new output path.
The original frozen-F0 probability cache and native-repair inputs referenced by
the receipt must still exist and match their hashes. Only 8.640 um, halo32,
eight-way mirror TTA, BF16 inference and a dense 3x3x3 block of 128-cubed targets
are qualified. This is not a generic-resolution or whole-scroll executor. The
qualified probability cache is the T=0.35 profile; the raw model's shipped
operating point moved to T=0.30 with TTA on 9 September 2026
(`recipes/f0_inference_20260909`) without re-qualifying this stage.
Use interior cubes; the surrounding context is not a seam-free tiled product.

[`socratic-predict`](predict.md) writes a compatible grid when it is run with `--threshold 0.35`; its default T=0.30 output is refused by this profile.

## Outputs and interruption handling

- `cubes_PRED/`: additive binary TIFFs; all prior foreground is protected.
- `meshes/*.npz`: exact vertices in local ZYX, triangle indices and normals.
- `meshes/*.ply`: world voxel XYZ coordinates. Multiply by 8.64 for micrometers,
  or 0.00864 for millimeters. PLY values themselves are not millimeters.
- `ledger.jsonl`: flushed after each accepted/refused seed; `ledger.json` and
  `patches.json` collect the completed decisions.
- `request.json`, `summary.json`, `state.json`: hashes, policy, geometry and counts.

Execution claims an exclusive output-specific lock, builds a separate partial
directory, checks inputs again, verifies written masks, then publishes by rename.
Failures preserve the partial directory and an explicit failure/state record.
An unknown lock or existing partial is never automatically reclaimed or erased.
Inspect the owner and files before deciding how to recover. The 4 GiB working-set
estimate is an estimate, not an enforced OS memory cap; no worker pool is created.

## Fixed local policy and evidence boundary

A monotone parallel chart fits a seeded surface jointly over Z and path distance.
CT-profile similarity, probability and displacement costs are optimized using an
integer-capacity height-field graph cut. The target searches +/-8 pixels around
CT-derived motion; +/-24 pixels are used to find strongly supported neighbors.
Neighbor surfaces are fitted together with at least 5 pixels normal separation.
Hard displacement steps are 2 along Z and 1 along the curve. Regularization is
.20 and .10; the selected cost scale is 1000.

Only contiguous supported rows around the seed survive: at least 5 planes,
40 square pixels of valid triangle area, 24 additional voxels, and paint P>=.16.
The full row requires >=90% P>=.16 coverage, mean P>=.23 and mean CT correlation
>=.15; actually missing vertices need mean P>=.20 and >=85% coverage. Paint
contacts at most two prior binary components per plane. Interacting new patches
must agree in local normal position and orientation. Up to three longest seeds
are tried per prior added component; the first passing patch is retained.

This extends supported geometry but does not resolve every crowded fold. CT
appearance repeats, probabilities share one model, and large cost margins are
not calibrated anatomical confidence. Genuine interior fold-back charts refuse;
fragmented true sheets may fail the two-component contact count. Existing wrong
joins are preserved, not repaired by this additive stage. Do not use the output
as trusted training labels without adjudication.

Development: 107 patches / 75,661 additions over v2. New target-disjoint local
assessment: 82 / 46,780. Neither erases foreground. These are geometric counts,
not new Dice, independent-scroll validation or true-sheet precision. All exact
meshes and masks are available in the local HTML report and checked inventories.
