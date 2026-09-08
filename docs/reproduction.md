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

This archive is not a freshly tested portable F0 build, environment lock,
launcher or exporter. Embedded original D:/G: artifact paths remain evidence,
not automatically relocatable inputs. A portable F0 release requires a separate
packaging task and verification against the frozen model identity. **Do not use
the existing v31 wrappers as an F0 launcher or exporter.** They still enforce
v31's model hash, counters and recipe.

The exact 250k endpoint, final optimizer checkpoint, all earlier checkpoints
and full cached HTML/image reports remain retained in the original workspace.
The freeze does not authorize another training run, threshold change, filler
experiment, production deployment or public upload.

## Historical v31 reproduction

Everything below documents the earlier v31 environment, recipe and exporter.
It is retained for reproducibility and does not describe F0.

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
socratic-train --paths recipes/v31/paths.local.json --check --print-command
```

The check verifies regular-file existence, SHA-256, M7 byte size, manifest row
count, PHerc0139-only training scope, held-out scroll scope, and the dynamic
state's declared event/step counts. A failed check is not overridable by the
runner.

## 4. Train

```bash
socratic-train --paths recipes/v31/paths.local.json --run
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
socratic-export path/to/checkpoint_milestone_00008192.pt huggingface/export
```

The exporter fails closed unless checkpoint size, SHA-256, sample counters,
recipe selection, and qualification selection all agree. It writes
`model.safetensors`, `config.json`, preprocessing metadata, checkpoint metadata,
the recipe, observed metrics, release qualification, concise selection record,
and model card. Publishing remains a separate, explicit `hf upload` action.

Do not substitute the 7,168-sample best-Dice interval or the censored threshold
0.25. Those answer the ordinary validation ranking, not the independent
morphology and anti-blob release criterion.
