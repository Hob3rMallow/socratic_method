---
library_name: pytorch
pipeline_tag: image-segmentation
license: other
tags:
  - vesuvius-challenge
  - 3d-segmentation
  - nnunet
  - knowledge-distillation
  - cross-resolution
metrics:
  - dice
---

# Socratic Method M7-XR v31

This is the artifact template for a 3D papyrus-surface segmentation student
trained from the released 9.362 um M7 model with soft supervision from an
approximately 2.399 um Villa teacher. Training adds de-blob separation,
medial-crest recall, M7 preservation, and dynamic widest-path connectivity.

**Release status:** a release candidate is selected, but weights are not yet
approved for redistribution. Locked and blinded morphology review selects the
raw 8,192-sample checkpoint at threshold 0.45. It passes the PHerc1447 hard
anti-blob gate with no interior- or thickness-regression cubes. All eight
held-out intervals improve both calibration scrolls over released M7; the best
calibrated macro-scroll Dice occurs at 7,168 samples (0.58293), while the
8,192-sample result is 0.58037. Every overlap calibration selects the tested
lower boundary of 0.25, so that censored value is retained as a duration
diagnostic rather than an operating recommendation.

Source repository: <https://github.com/ubc-nvining/socratic_method>

## Model description

- Architecture: one 3D residual-encoder nnU-Net, 102,349,770 parameters.
- Input: one normalized 9.362 um CT channel in `NCDHW` order.
- Output: two-class logits; use the softmax probability at class index 1.
- Inference: raw student only. The fine teacher and M7 blend are not used.
- Training scroll: PHerc0139, 4,096 deterministic atlas rows, two passes.
- Held-out validation: PHerc0814 and PHerc1451.
- Selected operating point: raw 8,192-sample student at threshold 0.45.
- Morphology gates: 16 locked PHerc0139 slices and six blinded PHerc1447 cubes.

The model definition lives in
`crossres_pred.voxel.model.VoxelNNUNet`; install the accompanying Socratic Method
repository before loading the state dictionary. `config.json` contains the exact
network configuration and `preprocessor_config.json` contains CT normalization.

```python
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from crossres_pred.voxel.model import NNUNetConfig, VoxelNNUNet

root = Path(".")
config = json.loads((root / "config.json").read_text())
model = VoxelNNUNet(NNUNetConfig.from_dict(config["model_config"]))
model.load_state_dict(load_file(root / "model.safetensors"), strict=True)
model.eval()
```

## Training and evaluation

`training_recipe.json` is machine-readable and contains the complete v31
objective, optimizer, thresholds, sample milestones, artifact SHA-256 values,
and exact command arguments. `observed_metrics.json` contains the duration
ladder, while `selection.json` records the independent morphology-gated choice
and its evidence. See the source repository for the frozen implementation and
the additive 2D line-fitter postprocess.

This model was not selected solely on held-out Dice. Registered per-scroll
behavior, blinded PHerc1447 anti-blob review, two-sided morphology/coverage,
and human inspection overrode the censored overlap optimum. The optional
postprocessor remains independently qualified and is not included in the model
gate measurements.

## Limitations

The training schedule is PHerc0139-only and may not represent other scrolls,
acquisition conditions, or geometry near the umbilicus. Fine-teacher labels are
model-derived soft supervision, not ground truth. The current calibration band
does not bracket its optimum. Output errors can create false sheet mergers or
gaps and require downstream geometric safeguards.

## Artifact provenance

- Source checkpoint SHA-256: `{{CHECKPOINT_SHA256}}`
- Exported safetensors SHA-256: `{{MODEL_SHA256}}`
- Export date (UTC): `{{EXPORT_DATE}}`

## License and citation

No release license has yet been selected. Do not upload or redistribute the
weights until the repository owner resolves the source, data, teacher, and M7
artifact terms. Ownership and citation fields are still to be completed.
