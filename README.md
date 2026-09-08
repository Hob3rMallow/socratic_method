# The Socratic Method

Repository: <https://github.com/ubc-nvining/socratic_method>

> In the dialogues Socrates presents himself as a simple man who confesses that
> he has little knowledge. With this ironic approach he manages to confuse the
> other who boasts that he is an expert in the domain they discuss. The outcome
> of the dialogue is that Socrates demonstrates that the other person's views
> are inconsistent. In this way Socrates tries to show the way to real wisdom.
>
> - Wikipedia

This repository isolates the ScrollFiesta! team's scroll-training work from
the geometry tools repository. The current frozen model is **C3-F0 at 200,000
sample exposures, T=0.35**, selected after completion of the full 250,000-exposure
training run.

## A substantial advance over released M7

On the same frozen 689-row benchmark, with no TTA and T=0.35 for both models,
F0 reaches **0.6808 macro Dice versus M7's 0.5541**: **+0.1266 Dice,
12.7 percentage points, or 22.9% relative improvement**. It also exceeds
M7 with eight-way TTA at its calibrated threshold (0.5926 Dice). The geometry
review shows cleaner separation through several broad, merged-looking regions;
improving the continuity of the remaining sheet geometry is the next priority.

The current official record is
[recipes/final_c3_250k_20260907](recipes/final_c3_250k_20260907/README.md),
with a machine-readable pointer in [CURRENT_MODEL.json](CURRENT_MODEL.json).
The raw student checkpoint has SHA-256
`54db9a59cb602ff08e3dda0acc1195ad980ab05fb660332d32fb436a0d49a175`.
It is frozen as a separate, hash-verified, read-only local artifact. The selected
weights are **200k**, not the exact 250k endpoint, which remains retained.

The winning recipe starts fresh from released M7 on the wide15 anti-aliased
corpus: CE 1 + Dice 1 + M7-KL 0.5 with known/confident agreement and an
unknown-corridor radius of 2; SGD 1e-3 with the full-horizon poly schedule;
**no trust ball**. Inference uses one ordinary M7-architecture student, with no
teacher or M7 blend. “250k” counts repeated sample exposures, not unique rows.

See the [full post-run review](recipes/final_c3_250k_20260907/review.md) for
all 25 checkpoints, matched M7 comparisons, completed native metrics and the
postprocessor diagnosis. The gain is substantial without claiming every metric
wins: operating-point topology narrowly misses its advisory target, and the
six-cube conservative filler has not demonstrated broad sheet reconstruction.

## More useful local gap repair

The new [checked repair command](docs/repair.md) leaves F0/200k and T=.35
fixed while recovering more local continuations. In two contiguous 27-cube
blocks, development joins increased from 96 to 286 and reserved-control joins
from 59 to 97, with zero original foreground erased. Tracking remains on and
the radial guard is tighter, not looser. The [repair record](recipes/f0_repair_20260907/README.md)
includes comparisons, CT-aligned views and explicit limits: this is still
small-gap repair, not broad sheet reconstruction or global topology certification.
It is a separate opt-in postprocessor; no production deployment is implied.

## Longer, curved continuity

The additional [probability-guided continuity tool](docs/continuity.md) now
recovers gaps beyond the native filler's 21-pixel reach, while keeping the frozen
model and existing foreground unchanged. Development and fresh-control blocks
gain 764 and 515 per-plane corridors; 357 and 210 exceed 21 pixels. The central
cubes gain 825 and 2,004 voxels beyond native repair. CT-aligned views show useful
long continuations, with ambiguous folds retained for review. This separate
opt-in stage is not a global threshold reduction or anatomical certification.
See [the continuity evidence record](recipes/f0_continuity_20260907/README.md).

## Frozen recipe reproduction

The exact F0 configuration, run identity, copied evaluation evidence and 63
sealed runtime files are preserved in the record, and the engine snapshot under
`src/crossres_pred/voxel` is byte-verified against those sealed pins
(`socratic-engine-pins --check`). The executable recipe
[recipes/f0/recipe.json](recipes/f0/recipe.json) transcribes the sealed F0
command with placeholders for the three staged inputs, and `socratic-train`,
`socratic-export` and the `huggingface/f0` templates now default to it:

```powershell
socratic-engine-pins --check
socratic-train --paths recipes/f0/paths.local.json --check --print-command
socratic-train --paths recipes/f0/paths.local.json --run
crossres-voxel audit-checkpoint --checkpoint <run>/checkpoint_milestone_00200000.pt --patches <frozen v14p2 manifest> --output <audit dir> --split val
socratic-export <run>/checkpoint_milestone_00200000.pt huggingface/export/f0
```

`--check` verifies the SHA-256 of every staged input, the wide15 corpus scope
(16,200 rows, 15,432 train / 768 val, twelve training scrolls, 26 record ids)
and prints a command whose tokens equal the sealed F0 argv. See
[docs/reproduction.md](docs/reproduction.md) for the full walkthrough and its
limits (same-seed recipe repeat, not bitwise reproduction).

## Coherent local 3D patches

For the newer opt-in post-processing stage, see [3D sheet patches](docs/sheet_patches.md)
and its [separate evidence record](recipes/f0_sheet_patches_20260907/README.md).
It extends supported v2 curves into explicit 3D meshes: 107 development patches
and 82 patches on a new target-disjoint block, with no existing foreground erased.
Crowded-fold identity remains unresolved; no model or production default changes.

### Historical v31 quick start

The commands below reproduce the older v31 method, not the current frozen model.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[baseline,zarr,dev]"
```

Copy [recipes/v31/paths.example.json](recipes/v31/paths.example.json), stage the
pinned inputs, then verify identities before any historical reproduction:

```powershell
socratic-train --recipe recipes/v31/recipe.json --paths recipes/v31/paths.local.json --check --print-command
```

A new training run is a separate explicit action. See
[docs/reproduction.md](docs/reproduction.md) for the F0 walkthrough and
the preserved v31 instructions. The old
[v31 selection](recipes/v31/selection.json) and
[qualification](recipes/v31/release_qualification.json) remain intact.

## Repository map

- `recipes/final_c3_250k_20260907/`: current frozen model identity, complete
  post-run review, observed evidence and sealed F0 runtime-file snapshot.
- `recipes/f0/`: the executable F0 recipe (sealed argv with placeholders,
  artifact pins, observed milestones, selection and qualification records,
  environment lock, paths example).
- `recipes/v31/`: historical executable recipe, environment lock and release evidence.
- `recipes/f0_repair_20260907/`: measured contiguous repair profile and evidence,
  implemented by the separate `socratic-repair` command.
- `src/crossres_pred/`: the engine snapshot, byte-synced to the 63 sealed F0
  code pins (with the documented read-time path-relocation hook);
  `src/socratic_method/`: the portable recipe/export/postprocessor wrappers.
- `native/line_fitter/`: additive gap-joining postprocessor. Its
  [README](native/line_fitter/README.md) distinguishes source defaults from
  the conservative preset used in the F0 report.
- `huggingface/`: model-card/export templates per recipe (`f0/` current,
  `v31/` historical); weights are not committed to Git.
- `provenance/source/`: preserved earlier plans, run records and research drivers.
- `submission.pdf` and `submissions/2026-09/`: existing paper/submission
  artifacts, predating this F0 freeze; not rewritten by this record update.

## Release and artifact boundary

The **current model record is frozen locally**; this update does not claim a
production deployment or perform a public upload. Large corpora, released M7
weights, student weights and full cached image reports remain external artifacts.
Their identities and local locations are recorded. Public distribution still
requires a license and resolution of upstream data/teacher/M7 terms.

See [docs/model.md](docs/model.md), [docs/reproduction.md](docs/reproduction.md),
and the [post-run review](recipes/final_c3_250k_20260907/review.md).
