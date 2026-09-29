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
sample exposures**, selected after completion of the full 250,000-exposure
training run, shipped since 9 September 2026 with **eight-way mirror TTA at
T=0.30** on the unchanged weights (macro Dice 0.7130 on the frozen benchmark).
The September 2026 progress video and paper present it as **Socratic Method
September 2026**; paths, record ids and code keep the name F0.

## Head-to-head, 24 September 2026

The same frozen 689-row benchmark (PHerc0814 and PHerc1451, which no compared
model trained on), the same patches for every model, and the challenge's own
Kaggle metric computed by one binary:

| | macro Dice | Kaggle composite |
|---|---:|---:|
| **F0, shipped** (eight-way mirror TTA, T=0.30) | **0.7130** | **0.6521** |
| M7, as the organisers publish it (no TTA, T=0.2) | 0.5607 | 0.6012 |
| HercUNet v0, best of every setting on this benchmark | 0.4394 | 0.5447 |

Paired over the same rows, F0 leads M7 as published by +0.1524
[+0.1413, +0.1636] and HercUNet at its best by +0.2736 [+0.2548, +0.2924]. The
harness reproduces the sealed F0 and M7 audits to within 5e-8 before scoring
anything new, and HercUNet's own command-line inference on the live scroll
scores at or below the harness on every patch checked.

The published M7 does make blobs, and they sit where the scroll is compressed.
On 256 randomly drawn interior cubes from eight scrolls, the share of its
predicted voxels inside a solid mass rises from 2.0% in the most open third of
cubes to 6.0% in the most compressed third; rerunning the released weights at
the published setting reproduces it, and F0 does not rise with compression
(2.9%, 1.5% and 2.8% from the most open third to the most compressed). On the
human-labelled PHerc0500P2, which released M7 trained on, the models are close.
Evidence, the fairness rules, every script and all 256 cubes are in
[recipes/f0_benchmark_20260924](recipes/f0_benchmark_20260924/README.md).

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

## The objective is what survived ablation

The earlier v31 recipe had **seven** weighted loss terms and a parameter-space
trust region. The shipped one has three and no projection, and the difference is
measured rather than stylistic: on a corpus of adequate breadth, with the
schedule run to its declared length and the trust ball off, only the M7 function
anchor clears its own seed-noise floor. Medial crest recall, the separation
shell, M7 positive preservation and dynamic connectivity are retired; the last
of them, preservation, was tested directly on 2026-09-10 in three preregistered
arms and all three were killed on rim thickening.

The full ledger, the single-term ablation table and the preregistered
preservation evidence are in
[recipes/f0_ablations_20260910](recipes/f0_ablations_20260910/README.md).
Two terms are recorded as **untested at breadth rather than refuted**: the
medial tail floor, which needs a crest-complete corpus that the mixed wide15
corpus cannot be, and dynamic connectivity, whose events only exist where a
full-volume M7 prediction does. The trust region gets its own cautionary note in
the paper, because while it was active it made every other ablation
uninformative.

See the [full post-run review](recipes/final_c3_250k_20260907/review.md) for
all 25 checkpoints, matched M7 comparisons, completed native metrics and the
postprocessor diagnosis. The gain is substantial without claiming every metric
wins: operating-point topology narrowly misses its advisory target, and the
six-cube conservative filler has not demonstrated broad sheet reconstruction.

## Inference update, 9 September 2026: 0.713 macro Dice with TTA at T=0.30

The same 200k weights, scored with eight-way mirror test-time augmentation and
the preregistered secondary operating point T=0.30, reach **0.7130** macro Dice
[0.7012, 0.7248] on the frozen benchmark (0.7038 at the primary endpoint
T=0.35; paired cluster-bootstrap delta over the no-TTA reference +0.0230
[+0.0195, +0.0265], Holm pass, both scrolls positive). Released M7 scored the
same way reaches 0.5875 (TTA, T=0.30) and 0.5926 at its calibrated T=0.20. The
Kaggle-style composite on the same rows moves from 0.6401 to 0.6521 with
topology 0.4016 (advisory 0.3776, now met), and the six-cube gates and frontier
veto pass at T=0.30. The preregistered seed replication ran the same night and
passed: the independent seed-1204 run with the same recipe reaches 0.7089
[0.6974, 0.7204], +0.0351 [+0.0302, +0.0400] over its own no-TTA reference with
both scrolls positive
([record](recipes/f0_c3r_replication_20260909/README.md)). One item stays open
and is recorded, not waived: re-qualification of the postprocessors below, which
keep their T=0.35 pins. Evidence, receipts and the preregistration are in
[recipes/f0_inference_20260909](recipes/f0_inference_20260909/README.md).

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

## Run the model on a scroll region

`socratic-predict` runs the frozen weights on a region of a scroll with the
shipped inference recipe (eight-way mirror TTA, T=0.30, a 32-voxel halo, BF16)
and writes an OME-NGFF prediction store in the scroll's own coordinates plus a
ScrollFiesta cube grid. The volume is streamed anonymously from the open-data
bucket; the model is not a flag, it is the frozen record.

```powershell
socratic-predict --scroll PHerc1447 --region 12032:12416,4096:4480,2816:3200 `
  --out .\infer-demo --run
```

That writes `infer-demo\prediction.zarr`, in the same form as the published M7
surface predictions so a viewer opens it straight over the CT, and
`infer-demo\cubes_PRED\`, which `grid_pipeline` and `socratic-repair` read
directly. Without `--run` nothing is written and the checked plan is printed.
`--volume` takes any local or `s3://` store instead of a scroll name, `--grid`
re-predicts an existing carved grid, and `--threshold 0.35` produces the
training-record operating point the T=0.35 postprocessors require. The pins come
from the `inference` section of [recipes/f0/recipe.json](recipes/f0/recipe.json)
and the scroll names from [recipes/f0/scrolls.json](recipes/f0/scrolls.json).
See [docs/predict.md](docs/predict.md). This is a research runner, not a
deployment.

The weights need no configuration: they are committed under
`releases/c3-f0-200k-20260907/` and resolved automatically, then checked against
the SHA-256 the recipe pins before anything runs. They are stored with Git LFS,
so clone with `git lfs install` in place, or run `git lfs pull` in an existing
clone; otherwise the checkpoint is a pointer stub and `socratic-predict` says
so. `--checkpoint` and `--model-hf` still override.

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
crossres-voxel audit-checkpoint --checkpoint <run>/checkpoint_milestone_00200000.pt --patches <frozen v14p2 manifest> --output <audit dir>/tta --split val --tta
socratic-export <run>/checkpoint_milestone_00200000.pt huggingface/export/f0
```

`--check` verifies the SHA-256 of every staged input, the wide15 corpus scope
(16,200 rows, 15,432 train / 768 val, twelve training scrolls, 26 record ids)
and prints a command whose tokens equal the sealed F0 argv. The first audit
reproduces the training record (0.680787 at T=0.35, no TTA); the second is the
shipped inference number (0.713026 at T=0.30 with `--tta`). See
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
- `recipes/f0_inference_20260909/`: the shipped inference recipe's evidence
  (preregistration, decision table, TTA audits, bootstrap receipts, Kaggle and
  six-cube receipts, hash inventory).
- `recipes/f0/`: the executable F0 recipe (sealed argv with placeholders,
  artifact pins, observed milestones, selection and qualification records,
  environment lock, paths example), the pinned inference parameters its
  `inference` section holds, and the `scrolls.json` name registry.
- `recipes/f0_benchmark_20260924/`: the head-to-head against the published
  M7 and HercUNet v0, the random-cube blob audit of the published M7, and
  every script, per-row count and figure behind them.
- `recipes/f0_c3r_replication_20260909/`: the passed seed replication of the
  shipped inference recipe.
- `recipes/f0_ablations_20260910/`: the ablation ledger behind the reduced
  objective — single-term arms, the preregistered preservation experiment, and
  what remains untested rather than refuted.
- `recipes/v31/`: historical executable recipe, environment lock and release evidence.
- `recipes/f0_repair_20260907/`: measured contiguous repair profile and evidence,
  implemented by the separate `socratic-repair` command.
- `src/crossres_pred/`: the engine snapshot, byte-synced to the 63 sealed F0
  code pins (with the documented read-time path-relocation hook);
  `src/socratic_method/`: the portable recipe, export, prediction and
  postprocessor wrappers (`socratic-predict` runs the frozen model on a
  scroll region).
- `native/line_fitter/`: additive gap-joining postprocessor. Its
  [README](native/line_fitter/README.md) distinguishes source defaults from
  the conservative preset used in the F0 report.
- `huggingface/`: model-card/export templates per recipe (`f0/` current,
  `v31/` historical).
- `releases/c3-f0-200k-20260907/`: the frozen F0 checkpoint itself, tracked
  with Git LFS and byte-verified against the SHA-256 in `recipes/f0/recipe.json`.
- `provenance/source/`: preserved earlier plans, run records and research drivers.
- `submission.pdf` and `submissions/2026-09/`: the paper. Rewritten on
  2026-09-10 around the shipped F0 model and its reduced objective; the retired
  loss terms, the trust region and the additive postprocessors are now
  appendices. Updated 2026-09-24: the teaser now shows the shipped model beside
  the published M7 and HercUNet v0 on seeded-random cubes, and the results add
  the head-to-head and the blob audit. The other rendered figures still show
  the earlier v31 student and are labelled as such.

## Release and artifact boundary

The **F0 student weights are committed to this repository** under
`releases/c3-f0-200k-20260907/`, tracked with Git LFS and byte-verified against
the SHA-256 pinned in `recipes/f0/recipe.json`. They carry the same Apache-2.0
license as the code. This is a source release, not a production deployment.

Large corpora, released M7 weights and full cached image reports remain external
artifacts; their identities and local locations are recorded. Rebuilding from
them is still governed by the upstream Vesuvius Challenge data, Villa teacher
and M7 terms, which this repository's license does not extend. See
[NOTICE](NOTICE).

See [docs/model.md](docs/model.md), [docs/reproduction.md](docs/reproduction.md),
the [post-run review](recipes/final_c3_250k_20260907/review.md), and the
[ablation ledger](recipes/f0_ablations_20260910/README.md).

## License

Apache License 2.0 - see [LICENSE](LICENSE). It covers the original code in this
repository and the frozen F0 weights under `releases/`.

Upstream artifacts are not relicensed here and keep their own terms: the
Vesuvius Challenge scroll data, the released M7 model the student was
initialised from, the approximately 2.399 um Villa fine teacher, and the
vendored `stb_image_write.h`. [NOTICE](NOTICE) lists them.
