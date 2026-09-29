# C3r seed replication of the shipped inference recipe: passed

The inference record of 9 September 2026 left one preregistered check open: the
shipped inference recipe (eight-way mirror TTA at T = 0.30) had to be replicated
on the independent seed-1204 run C3r. That check ran unattended on
**9 September 2026 at 23:43 UTC**, after the record was written, and it
**passed**. This record copies its receipts into the repository. The dated
[inference record](../f0_inference_20260909/README.md) is left unchanged: it
correctly describes what was known when it was written.

## The rule

[F0 family preregistration](../f0_inference_20260909/evidence/prereg/F0_FAMILY_PREREG_20260908.md),
rule 9 (the replication rule; earlier files call it rule 8): *the winning
inference recipe (member index set, TTA, T) applied to C3r must beat C3r's own
reference with paired lower bound > 0.*

C3r is the sealed F0 argv with `--seed 1204`, trained from the released M7 to
the full 250,000 exposures. Its own checkpoint was chosen by the identical F0
late-selection rule: **180,000 exposures**
([C3r summary](evidence/family/c3r_evaluation_summary.json)).

## The result

| | macro Dice | 95% interval |
|---|---:|---|
| C3r 180k, no TTA, T = 0.35 (C3r's reference) | 0.6738 | |
| C3r 180k + eight-way mirror TTA, T = 0.30 | **0.7089** | [0.6974, 0.7204] |
| paired delta | **+0.0351** | [+0.0302, +0.0400], P(delta <= 0) = 0 of 2,000 |
| paired delta, PHerc0814 | +0.0419 | [+0.0347, +0.0497] |
| paired delta, PHerc1451 | +0.0282 | [+0.0216, +0.0347] |

The paired cluster bootstrap uses the preregistered design: clusters are
`support_anchor_chunk_zyx` within scroll (256 on PHerc0814, 142 on PHerc1451),
B = 2,000, seed 20260908, identical resample weights for both audits
([receipt](evidence/bootstrap/c3r_sel_tta.json)).

The shipped F0 recipe reaches 0.7130 on the same rows; the seed replicate of the
same recipe reaches 0.7089. The inference gain is therefore not a property of
one seed.

## One labelling detail

The family coordinator's [final summary](evidence/family/final_summary.json)
names `f0_110k_tta` as the "winner" whose recipe was replicated. That label comes
from a plain argmax of the T = 0.35 macro over all shippable rows
([`best_shippable`](evidence/family/evaluate_f0_cross_run.py)), not from the
preregistered promotion, which fixed `f0_200k_tta` as the family primary before
scoring. It does not change the test: `c3r_analogue` maps any single-milestone
TTA recipe to C3r's own rule-selected milestone with TTA, at the recipe's
operating threshold, and both candidates operate at T = 0.30. The shipped recipe
would receive the identical comparison.

## Still open

Re-qualification of `socratic-repair`, `socratic-continuity` and
`socratic-sheet-patches` at T = 0.30. No work on it was found; the postprocessors
keep their T = 0.35 pins.

## Files

| file | source |
|---|---|
| `evidence/bootstrap/c3r_sel_tta.json` | `output/crossres_data/f0_family_20260908/cross_run/bootstrap/c3r_sel_tta.json` |
| `evidence/bootstrap/c3r_sel.json` | `.../cross_run/bootstrap/c3r_sel.json` (no-TTA level intervals) |
| `evidence/audits/c3r_sel_tta/report.json` | `.../cross_run/audits/c3r_sel_tta/report.json` |
| `evidence/audits/c3r_sel_180k_reference/report.json` | `.../evaluation/C3r/milestone_00180000/audit/report.json` |
| `evidence/family/final_summary.json` | `.../final/summary.json` |
| `evidence/family/c3r_evaluation_summary.json` | `.../evaluation/C3r/summary.json` |
| `evidence/family/evaluate_f0_cross_run.py` | `vesuvius-c/crossres_pred/scripts/evaluate_f0_cross_run.py` |

SHA-256 of every copy is in [files.json](files.json).
