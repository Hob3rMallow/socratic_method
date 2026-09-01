# Reproduction recipe

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

## 5. Qualify and export

Do not select solely from the threshold sweep. The completed review of
registered held-out metrics, PHerc1447 blind anti-blob behavior, shrink-side
coverage, and false mergers is recorded in `recipes/v31/selection.json`; the 2D
fitter remains a separate qualification. Export the selected raw checkpoint:

```bash
socratic-export path/to/checkpoint.pt huggingface/export
```

The exporter fails closed unless the checkpoint SHA-256 matches the selection
record. It writes `model.safetensors`, `config.json`, checkpoint metadata, the
selected threshold, recipe, metrics, selection record, and model card.
Publishing is intentionally a separate, explicit `hf upload` action.
