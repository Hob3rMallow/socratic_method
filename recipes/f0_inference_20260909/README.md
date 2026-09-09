# C3-F0 inference update: eight-way mirror TTA at T = 0.30

This record changes the **shipped inference recipe** of the frozen C3-F0 model and
nothing else. The weights are the unchanged 200,000-exposure checkpoint
(`54db9a59cb602ff08e3dda0acc1195ad980ab05fb660332d32fb436a0d49a175`,
409,674,095 bytes). Inference now applies eight-way mirror test-time augmentation
(mean softmax over the 2^3 axis flips) and thresholds the surface probability at
**T = 0.30** instead of the training record's no-TTA T = 0.35.

On the frozen 689-row benchmark (PHerc0814 + PHerc1451, registered fine-teacher
labels) the same weights now reach **macro Dice 0.713026** [0.701206, 0.724822]
at T = 0.30 and **0.703826** at the pre-registered primary endpoint T = 0.35.
Against the no-TTA reference (0.680787 at T = 0.35) the paired cluster-bootstrap
delta is **+0.023039 [+0.019517, +0.026546]** at the same threshold (P(delta <= 0)
below 1/2000, Holm pass over the three family primaries, both scrolls positive) and
+0.032238 [+0.028063, +0.036361] at the operating point. Released M7 scored the same
way sits at 0.5575 (flat, T = 0.30), 0.5875 (with TTA, T = 0.30) and 0.5926 (with TTA
at its calibrated T = 0.20).

## What was decided, and by which rule

Everything was pre-registered before any scoring in
[`evidence/prereg/F0_FAMILY_PREREG_20260908.md`](evidence/prereg/F0_FAMILY_PREREG_20260908.md)
(SHA-256 in `manifest.json`): the family primary is the frozen 200k weights with
eight-way mirror TTA; the primary endpoint is fixed T = 0.35; T = 0.30 is the single
secondary operating point; the operating point of a candidate is the argmax of the
fixed-threshold macro Dice over {0.30, 0.35}; a "breaks 0.70" claim needs the point
above 0.70 at the declared operating point, a positive paired lower bound after
Holm, and a non-negative delta on each scroll; six-cube gates and the frontier veto
are evaluated at that operating point. The decision table
([`evidence/decision_table.json`](evidence/decision_table.json)) records the verdict
`breaks-0.70-candidate`; the operator promoted it on 9 September 2026 with the two
pending items below recorded rather than waived.

| | flat M7 | M7 + TTA | F0 200k, no TTA (reference) | **F0 200k + TTA (shipped)** |
|---|---:|---:|---:|---:|
| frozen macro Dice, T 0.30 | 0.5575 | 0.5875 | 0.6881 | **0.7130** |
| frozen macro Dice, T 0.35 | 0.5541 | 0.5799 | 0.6808 | **0.7038** |
| v14p2 Kaggle composite, T 0.30 | 0.5972 | 0.6095 | 0.6401 | **0.6521** |
| v14p2 TopoScore, T 0.30 | 0.3575 | 0.3709 | 0.3880 | **0.4016** |
| v14p2 VOI merge, T 0.30 (lower is better) | 1.242 | 1.197 | 1.075 | **1.044** |
| PHerc0500P2 composite, T 0.30 (human GT, absolute) | 0.4886 | 0.4942 | 0.4890 | **0.4917** |

Six cubes (blind PHerc1447, halo 32, TTA export, thresholded at 0.30): foreground
ratio 0.799, interior and thickness gates pass, 4,945 inter-wrap bridge pixels
against an interpolated maximum of 7,983 at skeleton recall 0.782 (frontier pass).
Readiness advisories at the operating point: composite 0.6521 against the 0.6212
target and topology 0.4016 against the 0.3776 target both pass; the topology target
was narrowly missed by the no-TTA T = 0.35 record.

## Pending, recorded not waived

- **C3r replication** (pre-registration rule 8): the promoted inference recipe must
  beat the seed-1204 replicate's own no-TTA reference with a positive paired lower
  bound. C3r was at 160,000 of 250,000 samples when this record was written; the
  family coordinator evaluates it unattended.
- **Postprocessor re-qualification**: `socratic-repair`, `socratic-continuity` and
  `socratic-sheet-patches` pin the frozen F0 prediction at T = 0.35 (halo 32, mirror
  TTA, BF16) and remain qualified for that input only. They were not re-qualified at
  T = 0.30 and their pins were deliberately left unchanged.

Two exploratory variants score at or slightly above the shipped recipe on every
metric (F0 110k + TTA: 0.7138 / 0.7082; the 100k-250k weight average + TTA: 0.7128 /
0.7050). They are reported in the all-metrics ledger and not promoted, because the
family primary was fixed before scoring.

## Files

- [`manifest.json`](manifest.json): every number above, read from the evidence copies.
- [`files.json`](files.json): SHA-256 inventory of the 71 copied files and their
  original locations.
- `evidence/audits/`: the TTA audit of the shipped weights and the no-TTA re-audit of
  the reference (which reproduced the sealed 0.680787428136385 exactly), each with
  its per-patch counts sidecar.
- `evidence/bootstrap/`: cluster-bootstrap receipts (levels, calibration optimism,
  paired deltas).
- `evidence/kaggle/`: Kaggle-metric summaries and per-case rows for the shipped
  recipe at T 0.30 and 0.35, the no-TTA reference at both thresholds, and the sealed
  M7 baselines at their calibrated T = 0.20.
- `evidence/geometry/`: six-cube receipts at both thresholds, the frontier curve and
  one review view (cyan = published M7, yellow = incumbent student, magenta = F0).
- `evidence/all_metrics/`: the flat-M7-and-every-variant table (JSON, Markdown, and
  the ledger page).
- `runtime/`: the ladder scripts and the audit-side modules that produced the
  evidence. `checkpoint_audit.py` differs from the sealed engine pin only by the
  additive per-patch sidecar; the pinned engine reproduces the same sweep, so the
  shipped number needs no engine change.

## Reproduce the shipped number

```bash
crossres-voxel audit-checkpoint --checkpoint <run>/checkpoint_milestone_00200000.pt --patches <frozen_validation_manifest> --output <audit dir> --split val --device cuda --tta
```

For the sealed weights the sweep rows read `macro_scroll_dice = 0.713025708376333`
at T = 0.30 and `0.7038261039837073` at T = 0.35. The no-TTA audit without `--tta`
still reproduces the training record's `0.680787428136385` at T = 0.35.
