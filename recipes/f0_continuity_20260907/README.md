# Longer continuity, same frozen F0

F0/200k at T=.35 remains frozen. The separate `f0-probability-continuity-v2`
postprocessor recovers longer curved, probability-supported connections on top
of the measured native-filled baseline; it does not change the native default.

| Measure | Development | Fresh control |
|---|---:|---:|
| New per-plane corridors | 764 | 515 |
| Paths longer than 21 px | 357 | 210 |
| Added voxels over native baseline | 48,337 | 31,951 |
| Central-cube additions over native | 825 | 2,004 |
| Existing foreground erased | 0 | 0 |

The policy was selected on development before evaluating a fresh disjoint
PHerc1447 block. Both blocks are 384-cubed, 27 contiguous cubes, at 8.640 um.
The control was not used to tune the algorithm. These counts are not unique
3D anatomical joins, precision labels, or new official Dice scores.

The local report is
`D:/work/vesuvius-c/output/crossres_data/f0_continuity_20260907/index.html`.
It includes CT-gray / baseline-magenta / additions-green comparisons,
all accepted corridor galleries, 40 inspected risk-first Z/Y/X cards,
the v1 development failures, explicit ambiguous cases and verification.

Use [the checked command](../../docs/continuity.md) for opt-in review.
The original `selected_policy.json` is preserved verbatim with its pre-control
pending status; `decision.json` records the subsequent completed assessment.
`files.json` hashes the copied evidence and source. Large image arrays and
weights remain local external artifacts. Nothing was uploaded or deployed.

Important limits: global umbilicus radius is no longer the new stage's hard
endpoint constraint; local evidence replaces it. Neighboring model probability
is not independent anatomical validation. Packed folds remain ambiguous, and
some additions occupy only one plane. Further work should target coherent 3D
sheet extension and ambiguous-pair adjudication, not indiscriminate widening.
