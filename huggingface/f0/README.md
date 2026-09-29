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

# Socratic Method C3-F0

This is the artifact template for the C3-F0 3D papyrus-surface segmentation
student: one residual-encoder nnU-Net initialised from the released 9.362 um M7
model and trained on the wide15 anti-aliased mixed corpus (soft native
fine-teacher targets plus hard human passthrough rows over twelve training
scrolls) with cross-entropy 1, Dice 1 and an M7 function-space KL anchor 0.5.
No trust ball, no separation shell, no medial or connectivity terms.
The September 2026 progress video and paper present it as **Socratic Method
September 2026**.

**Release status:** the raw {{CHECKPOINT_SAMPLES}}-exposure student is the
frozen model, chosen by a preregistered late-checkpoint rule after the complete
250,000-exposure run. Shipped inference is {{TEST_TIME_AUGMENTATION}} at an
operating threshold of {{OPERATING_THRESHOLD}}, preregistered before the
2026-09-08 inference ladder and promoted on 9 September 2026 on the unchanged
weights. The fine teacher and M7 are training-time references only; inference
does not blend either one into the student. The weights are released under the
Apache License 2.0; the upstream data, teacher and M7 artifacts keep their own
terms.

On the frozen 689-row benchmark (PHerc0814 and PHerc1451, registered
fine-teacher labels) the shipped recipe reaches macro Dice 0.7130 at T=0.30
with eight-way mirror TTA (0.7038 at T=0.35), against released M7's 0.5875 with
the same TTA at the same threshold and 0.5926 at M7's calibrated T=0.20. The
training record's matched no-TTA comparison at T=0.35 is 0.6808 versus M7's
0.5541 (+0.1266, 22.9% relative). The Kaggle-style composite on the same rows
is 0.6521 with topology 0.4016 (advisory 0.3776 met); the six-cube PHerc1447
anti-blob gates and the frontier veto pass at T=0.30, and all 25 checkpoints of
the run passed the gate at T=0.35.

Source repository: <https://github.com/ubc-nvining/socratic_method>

## Model description

- Architecture: one 3D residual-encoder nnU-Net, 102,349,770 parameters.
- Input: one normalized 9.362 um CT channel in `NCDHW` order.
- Output: two-class logits; use the softmax probability at class index 1.
- Operating threshold: `{{OPERATING_THRESHOLD}}`, applied to the class-1 softmax
  probability after test-time augmentation.
- Test-time augmentation: {{TEST_TIME_AUGMENTATION}}; average the softmax over
  the eight axis-flip passes (each flip undone before averaging), then threshold.
- Inference: raw student only. The fine teacher and M7 blend are not used.
- Inference window: 128-voxel target cubes with a 32-voxel raw halo (192 cubed
  context), sliding-window overlap 0.5 with Gaussian importance weighting, and
  bfloat16 autocast on CUDA. These are pinned in `training_recipe.json` under
  `inference`.
- Reference runner: `socratic-predict` in the source repository takes a scroll
  name and a region and applies all of the above.
- Training data: wide15 anti-aliased corpus, 15,432 training rows over twelve
  scrolls (10,072 native soft rows and 6,128 human hard rows across the corpus),
  250,000 sample exposures.
- Held-out validation: PHerc0814 and PHerc1451 (frozen v14p2 harness).

The model definition lives in
`crossres_pred.voxel.model.VoxelNNUNet`; install the accompanying Socratic Method
repository before loading the state dictionary. `config.json` contains the exact
network configuration and `preprocessor_config.json` contains CT normalization.
`model.safetensors` stores each parameter once: the nnU-Net state dictionary
exposes the encoder under several key prefixes, and the `tied_keys` entry in
the safetensors metadata maps every alias back to its stored tensor, which the
loader below re-expands before strict loading.

```python
import json
from pathlib import Path

import torch

from crossres_pred.voxel.model import NNUNetConfig, VoxelNNUNet
from socratic_method.hf_export import load_exported_state_dict

root = Path(".")
config = json.loads((root / "config.json").read_text())
model = VoxelNNUNet(NNUNetConfig.from_dict(config["model_config"]))
model.load_state_dict(load_exported_state_dict(root / "model.safetensors"), strict=True)
model.eval()
```

## Training and evaluation

`training_recipe.json` is machine-readable and contains the complete objective,
optimizer, snapshot schedule, artifact SHA-256 values and exact command
arguments. `observed_metrics.json` lists all 25 frozen-harness milestone
audits, `selection.json` records the release decision, and
`release_qualification.json` records the frozen benchmark, native scores,
six-cube gates and frontier review. See the source repository for the frozen
implementation, the sealed runtime files and the optional postprocessors.

The release decision is the preregistered rule, not the best observed row: the
global fixed-threshold maximum is the 110k checkpoint (0.6816) and the
calibrated optimum sits at T=0.30; both are training observations that the
frozen record reports beside the selected 200k checkpoint. The inference recipe
(eight-way mirror TTA, T=0.30) was likewise preregistered before its ladder ran;
its evidence and paired bootstraps are in `recipes/f0_inference_20260909/`.
The preregistered seed replication passed (`recipes/f0_c3r_replication_20260909/`);
postprocessor re-qualification at T=0.30 is still pending.

## Limitations

Validation labels are registered fine-teacher outputs, not human ground truth,
and cover two scrolls. Held-out human-labelled PHerc0500P2 scores are absolute
only because released M7 saw those labels. One seed and correlated checkpoints
do not establish that the long schedule beats a short one. Output errors can
create false sheet mergers or gaps and require downstream geometric safeguards;
the opt-in repair, continuity and sheet-patch postprocessors are separate,
gated tools.

## Artifact provenance

- Source checkpoint SHA-256: `{{CHECKPOINT_SHA256}}`
- Exported safetensors SHA-256: `{{MODEL_SHA256}}`
- Export date (UTC): `{{EXPORT_DATE}}`

## License and citation

Apache License 2.0. The same license covers the source repository's original
code and these weights, and the frozen checkpoint is committed there under
`releases/c3-f0-200k-20260907/`.

Apache-2.0 does not extend to the upstream artifacts this model was built from -
the Vesuvius Challenge scroll data, the released M7 model used for
initialisation, and the Villa fine teacher - each of which keeps its own terms.
Citation fields are still to be completed.
