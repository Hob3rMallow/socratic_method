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

The medial representation is LSMAT-inspired: slice-wise centers and physical
2D radii separate longitudinal continuation from radial thickness. The released
Villa center extraction is retained; this recipe does not claim to run LSMAT's
continuous least-squares solver. Connectivity pins are the contact sets between
an eligible thin teacher component and disconnected released-M7 components.
The admissible region is the complete minimum-dilation teacher-medial corridor
that joins those pins, rather than one prescribed centerline.

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

## Evaluation contract

Validation uses held-out PHerc0814 and PHerc1451. A checkpoint must retain the
minimum per-scroll M7 gain and is ranked by calibrated macro-scroll Dice. The
threshold sweep is 0.25 through 0.60 in 0.01 steps. Every observed v31 milestone
selects 0.25, the lower boundary, so this score is used only as a duration
diagnostic. Independent review of 16 locked PHerc0139 slices and six blinded
PHerc1447 cubes selects the raw 8,192-sample checkpoint at threshold 0.45. It
passes the hard anti-blob gate with zero interior- and thickness-regression
cubes. Exact candidate identity and qualification measurements are in
[`recipes/v31/selection.json`](../recipes/v31/selection.json).
