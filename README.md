# The Socratic Method

Repository: <https://github.com/ubc-nvining/socratic_method>

> In the dialogues Socrates presents himself as a simple man who confesses that
> he has little knowledge. With this ironic approach he manages to confuse the
> other who boasts that he is an expert in the domain they discuss. The outcome
> of the dialogue is that Socrates demonstrates that the other person's views
> are inconsistent. In this way Socrates tries to show the way to real wisdom.
>
> - Wikipedia

This repository isolates the ScrollFiesta! team's scroll-training work that does **not**
belong in the geometry tools repository. It contains the full training implementation
of our improved, ScrollFiesta-targeting segmentation method and a frozen recipe:

1. use a native-fine approximately 2.399 um Villa teacher to supervise the
   official released 9.362 um M7 segmentation network;
2. add medial-crest recall, de-blob separation, M7 preservation, and dynamic
   widest-path connectivity terms during training;
3. emit one ordinary M7 nnU-Net student, with no teacher and no M7 blend at
   inference; and
4. optionally repair short gaps with the additive 2D line fitter before the
   geometry pipeline consumes the prediction.

The canonical recipe is [recipes/v31/recipe.json](recipes/v31/recipe.json).
It describes the completed 8,192-sample duration ladder and the selected release
candidate exactly. The deployed artifact is the raw 8,192-sample student at
threshold 0.45, checkpoint SHA-256
`8de376f8a3ad1b14e25a57db1f8dd20e8c505ceb169a49bc006b2903d1ccb3c1`.
The 0.25 Dice-calibration boundary remains a useful censoring warning, but it
fails the independent anti-blob gate and is not the operating threshold.
The concise decision record is
[recipes/v31/selection.json](recipes/v31/selection.json); its complete locked,
blind-corpus, and FLIP evidence is
[recipes/v31/release_qualification.json](recipes/v31/release_qualification.json).

## Quick start

Create an environment and install the training stack:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[baseline,zarr,dev]"
```

Copy [recipes/v31/paths.example.json](recipes/v31/paths.example.json), point it
at the four pinned inputs, then verify their identities and print the exact
command:

```powershell
socratic-train --paths recipes/v31/paths.local.json --check --print-command
```

Start a fresh run only after the checks pass:

```powershell
socratic-train --paths recipes/v31/paths.local.json --run
```

Use `--resume` only for the same output directory after an interrupted run.
Embedded paths in the frozen manifests can be relocated without rewriting their
bytes by setting `original_root` and `artifact_root` in the paths file; see
[docs/data.md](docs/data.md).

## Repository map

- `src/crossres_pred/`: frozen training, teacher, medial-axis, evaluation, and
  inference engine needed by v31.
- `src/socratic_method/`: portable recipe and artifact-export wrappers.
- `recipes/v31/`: executable recipe, path template, environment lock, measured
  milestones, concise selection record, and detailed release qualification.
- `native/line_fitter/`: isolated additive 2D gap-joining postprocessor.
- `huggingface/`: model-card and export templates; weights are intentionally not
  committed to Git.
- `provenance/source/`: verbatim plans, live-run records, and research drivers
  copied from `vesuvius-c`.
- `submission.pdf`: the current paper at a stable, top-level path.
- `submissions/2026-09/`: the new paper/submission workspace and generated draft.

## Reproducibility boundary

The code and identities are here. The large CT/teacher atlases, 31 GB validation
corpus, released M7 weights, and student weights are not Git payloads. They need
separate artifact hosting (for example, a Hugging Face dataset/model repository)
under compatible licenses. The recipe fails closed on SHA-256 mismatches.

See [docs/reproduction.md](docs/reproduction.md), [docs/model.md](docs/model.md),
and [docs/line_fitter.md](docs/line_fitter.md) for the full contracts.

## Release status

The model artifact and operating point are selected: raw student only, 8,192
training samples, threshold 0.45. On the locked PHerc0139 set it passes all 16
anti-blob checks and matches the teacher component count on 15 of 16 slices; the
rank-26 scalar mismatch is documented. On the blind six-cube PHerc1447 audit it
passes the anti-blob gate with no interior or thickness regressions. See
[recipes/v31/release_qualification.json](recipes/v31/release_qualification.json)
for the full evidence and FLIP-selection nuance.

The weights can now be exported locally with `socratic-export`. Public upload is
still blocked on choosing a license and resolving upstream data, teacher, and M7
artifact terms; the exporter never uploads anything.
