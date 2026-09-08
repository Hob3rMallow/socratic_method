# Model and objective

## Architecture

The deployed model is a single 3D residual-encoder nnU-Net with one CT channel,
two output classes, six stages, features `(32, 64, 128, 256, 320, 320)`, and
102,349,770 trainable parameters. It is initialized strictly from the released
M7 checkpoint and all parameters are retrained. The fine teacher appears only
in corpus construction; inference uses the raw student probability, not an M7
blend and not the teacher.

## Cross-resolution supervision

The current training schedule uses 4,096 deterministic PHerc0139 patches from a
native-fine-teacher atlas, traversed twice for 8,192 samples. Fine predictions
are pulled back into the 9.362 um grid as soft occupancy targets. The actual
manifest contains only PHerc0139 even though an inherited field in the original
v31 recipe lists four scrolls. The corrected executable recipe follows the
hashed manifest bytes; the verbatim original remains under `provenance/source`.

The objective contract is
`soft-occupancy-ce-dice-villa-medial-crest-shell-kl-corridor-m7-preservation-dynamic-widest-path-v9`:

- soft cross-entropy: 1.0;
- soft Dice: 0.25;
- Villa medial-crest recall: 1.0;
- teacher-background separation: 2.0, radius 2, `q <= 0.1`;
- M7 function KL: 0.5 on known, teacher-confident agreement and an unknown
  corridor of radius 2;
- one-sided M7 preservation: 1.0, radius 2, anchor threshold 0.5, with no soft
  floor; and
- dynamic medial connectivity: 0.03125, probability floor 0.2, 96 widest-path
  propagation steps.

The connectivity atlas contains 149 fully owned events. Ninety-nine manifest
rows encounter an event in the exact schedule, 477 rows are eligible, and the
largest event needs 44 of the configured 96 propagation steps. It is constructed
from training-manifest boxes only and does not use a held-out gate.

## Optimization

Adam runs at a constant learning rate of `2e-5`, betas `(0.9, 0.999)`, epsilon
`1e-8`, zero weight decay, batch size 3, accumulation 8, bfloat16 autocast,
seed 1203, two workers, and at most 16 CPU threads. After every optimizer update,
the student is projected into a global relative-L2 ball of radius
`0.0027535421730275947` around released M7. The projection is active essentially
all of the observed schedule, so it is part of the effective method rather than
an inactive guardrail.

## Evaluation and release selection

Ordinary validation uses held-out PHerc0814 and PHerc1451. A checkpoint must
retain the minimum per-scroll M7 gain and is ranked by calibrated macro-scroll
Dice. Every observed v31 milestone selects 0.25 at the lower edge of the sweep.
That censored optimum is retained as a training observation, but it fails the
PHerc1447 anti-blob gate and is not a valid operating point.

The release candidate is the raw 8,192-sample checkpoint at threshold 0.45.
Selection combines the registered 16-slice PHerc0139 review, the blind six-cube
PHerc1447 anti-blob audit, and the deliberate preference for mild undergrowth
over foreground inflation. At 0.45, all 16 locked slices pass the anti-blob
check, 15 match the teacher component count, and mean ASSD is 0.626 voxels. The
PHerc1447 aggregate has foreground ratio 0.951 and recall 0.911 relative to the
v15 comparison model, with no interior or thickness regressions.

The component-count exception is locked rank 26: the teacher really has four
components, while two top blobs and touching student strokes make the scalar
count five. Several other legacy failures are visually correct but rejected by
the one-voxel M7-preservation proxy because the student smooths or shifts a
surface. Consequently, the legacy `learned_growth` scalar is recorded for
diagnosis but is not the release gate. A future replacement should decompose
displacement along and normal to the teacher medial center/radius field.

FLIP mean narrowly prefers 4,096 samples at threshold 0.42 for literal teacher
resemblance. The selected 8,192/0.45 candidate ranks 18 of 45 anti-blob-eligible
settings by that statistic, but improves 12 of 16 slices over 2,048 samples and
reduces total erosion and additions. Weighted-median FLIP, blind morphology, and
human review support the longer checkpoint. The complete, non-sanitized record
is in `recipes/v31/release_qualification.json`.
