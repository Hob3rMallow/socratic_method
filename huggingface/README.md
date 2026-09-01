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

**Release status:** the raw {{CHECKPOINT_SAMPLES}}-sample student is selected at
an operating threshold of {{OPERATING_THRESHOLD}}. The fine teacher and M7 are
training-time references only; inference does not blend either one into the
student. Weight publication remains subject to the repository's pending license
and upstream artifact terms.

All eight duration intervals improve both ordinary validation scrolls over
released M7. Their Dice-calibrated optimum remains censored at the tested lower
boundary, 0.25, but that value fails the independent PHerc1447 anti-blob gate
and is not deployed. Registered morphology review instead selected 0.45. At
that operating point, the 8,192-sample model passes the six-cube PHerc1447
anti-blob audit with foreground ratio 0.951 and recall 0.911 relative to the v15
comparison model, and matches the teacher component count on 15 of 16 locked
PHerc0139 slices. The remaining rank-26 mismatch is documented as a scalar
topology exception rather than hidden.

Source repository: <https://github.com/ubc-nvining/socratic_method>

## Model description

- Architecture: one 3D residual-encoder nnU-Net, 102,349,770 parameters.
- Input: one normalized 9.362 um CT channel in `NCDHW` order.
- Output: two-class logits; use the softmax probability at class index 1.
- Operating threshold: `{{OPERATING_THRESHOLD}}` for the selected release.
- Inference: raw student only. The fine teacher and M7 blend are not used.
- Training scroll: PHerc0139, 4,096 deterministic atlas rows, two passes.
- Held-out validation: PHerc0814 and PHerc1451.

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
ladder, `selection.json` records the concise release decision, and
`release_qualification.json` records the independent morphology, anti-blob,
and FLIP evidence. See the source repository for the frozen implementation and
the additive 2D line-fitter postprocess.

The release decision is deliberately not the best held-out Dice row. FLIP mean
also narrowly prefers 4,096 samples at threshold 0.42 for literal teacher
resemblance. Human morphology review, PHerc1447 de-blob behavior, and the
asymmetric preference for undergrowth over foreground inflation select 8,192 at
0.45; FLIP weighted-median pooling supports that longer checkpoint.

## Limitations

The training schedule is PHerc0139-only and may not represent other scrolls,
acquisition conditions, or geometry near the umbilicus. Fine-teacher labels and
the PHerc1447 v15 comparison are model outputs, not ground truth. The ordinary
Dice calibration band still does not bracket its optimum, and locked rank 26
remains a component-count exception. Output errors can create false sheet
mergers or gaps and require downstream geometric safeguards.

## Artifact provenance

- Source checkpoint SHA-256: `{{CHECKPOINT_SHA256}}`
- Exported safetensors SHA-256: `{{MODEL_SHA256}}`
- Export date (UTC): `{{EXPORT_DATE}}`

## License and citation

No release license has yet been selected. Do not upload or redistribute the
weights until the repository owner resolves the source, data, teacher, and M7
artifact terms. Ownership and citation fields are still to be completed.
