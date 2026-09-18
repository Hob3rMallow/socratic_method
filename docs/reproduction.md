# Reproduction and frozen artifacts

## Current C3-F0 record — 7 September 2026

The current selected model is F0/200k at T=.35, from the naturally completed
250k run. Its [manifest](../recipes/final_c3_250k_20260907/manifest.json)
pins the 409,674,095-byte checkpoint and the exact operating point.

The local frozen artifact is:

```text
D:/work/vesuvius-c/output/crossres_data/releases/c3-f0-200k-20260907/checkpoint_f0_00200000.pt
SHA256: 54db9a59cb602ff08e3dda0acc1195ad980ab05fb660332d32fb436a0d49a175
```

The official record preserves the exact
[configuration](../recipes/final_c3_250k_20260907/provenance/recipe.json),
[run identity](../recipes/final_c3_250k_20260907/provenance/run.json),
[training history](../recipes/final_c3_250k_20260907/provenance/history.jsonl),
[evaluation summary](../recipes/final_c3_250k_20260907/provenance/evaluation_summary.json),
and [hash inventory](../recipes/final_c3_250k_20260907/files.json).
The record's `runtime/` tree contains all 63 preregistered source/configuration
files in their original relative layout, verified against the sealed pins.

The exact 250k endpoint, final optimizer checkpoint, all earlier checkpoints
and full cached HTML/image reports remain retained in the original workspace.
The freeze does not authorize another training run, threshold change, filler
experiment, production deployment or public upload.

## Reproduce F0

The executable recipe is [recipes/f0/recipe.json](../recipes/f0/recipe.json)
(`c3-f0-200k-20260907`). Its `training.argv` is the sealed F0 command from
[provenance/recipe.json](../recipes/final_c3_250k_20260907/provenance/recipe.json)
with placeholders for exactly the three staged inputs; a test asserts the two
are token-identical after substitution. This is a **same-seed recipe repeat**
(seed 1203 on the same corpus), not a promise of bitwise reproduction across
software or hardware.

0. **Environment.** `recipes/f0/environment.lock.txt` records the observed
   packages (Python 3.12.9, torch 2.11.0+cu128, nnU-Net v2 2.8.1). The engine
   needs `nvidia-smi` on PATH, a CUDA device whose power limit is at most
   600.5 W, and at least 100 GiB free under the output directory. Run
   `socratic-engine-pins --check` once: it verifies the snapshot against the
   63 sealed code pins.
1. **Stage the three training artifacts** named in
   `recipes/f0/paths.example.json` (the file also carries the optional
   `selected_checkpoint` and `scroll_mirrors` entries that `socratic-predict`
   reads): the
   wide15 anti-aliased manifest with its archive trees (110.1 GB), the released
   M7 checkpoint and, for the audit only, the frozen v14p2 manifest with its
   patch tree (32.7 GB). Every wide15 row path is absolute under the original
   workspace, so set `original_root`/`artifact_root` (see [data.md](data.md)).
2. **Verify before spending GPU time:**

   ```bash
   socratic-train --paths recipes/f0/paths.local.json --check --print-command
   ```

   The check verifies SHA-256 and byte size of the staged inputs, the corpus
   scope (16,200 rows, 15,432 train / 768 val, twelve training scrolls, 26
   record ids) and the frozen validation scope when staged, then prints the
   command.
3. **Train:** `socratic-train --paths recipes/f0/paths.local.json --run`.
   Expect about 22 h on an RTX PRO 6000 at 450 W for 250,000 exposures with a
   milestone every 10,000. After an interruption repeat with `--resume`; never
   use `--resume` to start a new candidate from an older student. The
   selected artifact is `<output>/checkpoint_milestone_00200000.pt`, not
   `checkpoint_best.pt` or `checkpoint_last.pt`.
4. **Audit on the frozen harness.** The training record's number is the no-TTA
   audit read at T=0.35; the shipped inference number is the same audit with
   `--tta`, read at T=0.30:

   ```bash
   crossres-voxel audit-checkpoint --checkpoint <output>/checkpoint_milestone_00200000.pt --patches <frozen_validation_manifest> --output <audit dir> --split val --device cuda
   ```

   ```bash
   crossres-voxel audit-checkpoint --checkpoint <output>/checkpoint_milestone_00200000.pt --patches <frozen_validation_manifest> --output <audit dir>/tta --split val --device cuda --tta
   ```

   For the sealed weights the no-TTA sweep row at T=0.35 reads
   `macro_scroll_dice = 0.680787428136385`, and the `--tta` sweep reads
   `0.713025708376333` at T=0.30 and `0.7038261039837073` at T=0.35
   (see `recipes/f0_inference_20260909/evidence/audits/`). A fresh training
   run lands in the seed band, not on those exact values.
5. **Export:** `socratic-export <output>/checkpoint_milestone_00200000.pt huggingface/export/f0`
   fails closed unless the checkpoint bytes, SHA-256, sample counters and the
   recipe's selection/qualification records agree; it writes
   `model.safetensors`, `config.json`, preprocessing metadata, the recipe,
   observed milestones, qualification, selection and the model card from
   `huggingface/f0/README.md`. Publishing remains a separate `hf upload`.

6. **Predict a region.** `socratic-predict --scroll <name> --region
   z0:z1,y0:y1,x0:x1 --out <dir>` runs these same weights on a scroll with the
   shipped inference recipe and writes an OME-NGFF prediction store plus a cube
   grid. It verifies the checkpoint against `release.selected_checkpoint` and the
   engine against the sealed code pins before it fetches anything, and without
   `--run` it writes nothing. No setup is required: the weights ship in this
   repository under `releases/c3-f0-200k-20260907/` and are picked up
   automatically, so `paths.local.json` is needed for training, not prediction.
   Clone with Git LFS (`git lfs install`, then `git lfs pull` in an existing
   clone) or the checkpoint arrives as a pointer stub. See
   [predict.md](predict.md).

## Historical v31 reproduction

Everything below documents the earlier v31 environment, recipe and exporter.
It is retained for reproducibility and does not describe F0; pass
`--recipe recipes/v31/recipe.json` to both wrappers.

## 1. Environment

The live run used Python 3.12.9, PyTorch 2.13.0+cu130, CUDA 13, nnU-Net v2
2.8.1, and dynamic-network-architectures 0.4.4. The complete observed package
set is in `recipes/v31/environment.lock.txt`. Install this project editable so
the `crossres_pred` checkpoint module names remain stable.

## 2. Stage the artifacts

Mirror the original `output/crossres_data` layout under any local artifact root.
Copy `recipes/v31/paths.example.json` to an untracked local file and edit the
five paths. Do not edit `recipe.json` for machine-specific locations.

## 3. Verify before spending GPU time

```bash
socratic-train --recipe recipes/v31/recipe.json --paths recipes/v31/paths.local.json --check --print-command
```

The check verifies regular-file existence, SHA-256, M7 byte size, manifest row
count, PHerc0139-only training scope, held-out scroll scope, and the dynamic
state's declared event/step counts. A failed check is not overridable by the
runner.

## 4. Train

```bash
socratic-train --recipe recipes/v31/recipe.json --paths recipes/v31/paths.local.json --run
```

This creates checkpoints after 1,024, 2,048, 4,096, and 8,192 cumulative
samples and validates every 1,024 samples. To recover the same run after a
crash, repeat with `--resume`. Never use `--resume` to initialize a new candidate
from an older student: fresh candidates must begin from the exact released M7.

## 5. Verify and export the selected checkpoint

The selected checkpoint is the 8,192-sample raw student with SHA-256
`8de376f8a3ad1b14e25a57db1f8dd20e8c505ceb169a49bc006b2903d1ccb3c1`
and byte size `409675375`. Its operating threshold is 0.45. The complete
registered, blind anti-blob, and FLIP record is
`recipes/v31/release_qualification.json`; the shorter human decision record is
`recipes/v31/selection.json`.

```bash
socratic-export --recipe recipes/v31/recipe.json path/to/checkpoint_milestone_00008192.pt huggingface/export/v31
```

The exporter fails closed unless checkpoint size, SHA-256, sample counters,
recipe selection, and qualification selection all agree. It writes
`model.safetensors`, `config.json`, preprocessing metadata, checkpoint metadata,
the recipe, observed metrics, release qualification, concise selection record,
and model card. Publishing remains a separate, explicit `hf upload` action.

Do not substitute the 7,168-sample best-Dice interval or the censored threshold
0.25. Those answer the ordinary validation ranking, not the independent
morphology and anti-blob release criterion.
