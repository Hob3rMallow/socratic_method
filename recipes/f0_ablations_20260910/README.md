# Ablation ledger: what the objective lost, and why

This record holds the measurements behind Appendix A of the paper. It is an
evidence record, not a recipe: nothing here changes the frozen model, and the
shipped configuration remains [recipes/f0](../f0/recipe.json) exactly as
recorded in [recipes/final_c3_250k_20260907](../final_c3_250k_20260907/README.md).

The shipped objective is CE 1 + Dice 1 + M7-KL 0.5. An earlier configuration
(`recipes/v31`) carried seven weighted terms and a parameter-space trust
region. This record is why the other four terms and the trust region are gone.

## Ledger

| Term | v31 weight | Status | Decisive measurement |
|---|---|---|---|
| Soft cross-entropy | 1.0 | **retained** | — |
| Soft Dice | 0.25 | **retained**, reweighted to 1.0 | — |
| M7 function anchor (KL) | 0.5 | **retained** | only term to clear the seed band in single-term ablation |
| Medial crest recall | 1.0 | retired | its triggering diagnostic does not fire on the wide corpus; where measured, bought length by buying width |
| Separation shell | 2.0 | retired | single-term arm below baseline; behaves as a threshold dial, not a curve shift |
| M7 positive preservation | 1.0 | retired | three preregistered arms, all killed on thickening (below) |
| Dynamic connectivity | 0.03125 | retired | events exist only where a full-volume M7 prediction exists; ~3% of wide-corpus rows |
| Pinned-axial precursor | 0.0 | superseded | degraded ordinary validation monotonically with weight |
| Hardened Dice | — | rejected | lost at both matched milestones, intervals excluding zero |
| Trust region | r = 0.0027535422 | retired | the successful recipe sits 67–82x outside it; see paper Appendix B |

## Single-term ablations

Wide corpus, no trust region, arms differing only in the named flag, best
frozen milestone, macro-scroll Dice on the frozen 689-row benchmark:

| Arm | Macro Dice |
|---|---|
| Occupancy only (CE + Dice) | 0.654226 |
| Occupancy only, seed replicate | 0.645472 |
| + separation shell 0.5 | 0.634897 |
| + M7 function anchor 0.5 | **0.676853** |

The two occupancy-only arms differ by 0.0088 on nothing but a seed. Every
verdict in this ledger is expressed against a measured seed band for that
reason.

## Preservation: the last candidate (2026-09-10)

One-sided M7 preservation was the only term never measured in isolation at
scale, and the one with the best surviving motivation: our students beat M7 on
overlap and on merge behaviour but lose to it on topology on the one
human-labelled holdout. Three arms were preregistered, trained from released M7
on the shipped recipe's corpus and optimizer, each declaring the full
250,000-exposure horizon and terminated at the durable 50,000-exposure
optimizer commit.

| Arm | Delta | Pooled paired delta vs shipped (T=.35) | vs seed replicate | fg ratio @50k | Verdict |
|---|---|---|---|---|---|
| `P_hard` | preservation 1.0, r 2, thr .5 | −0.0480 [−0.0567, −0.0383] | −0.0379 | 1.54x | killed |
| `P_soft` | soft floor 4.0 | −0.1117 [−0.1214, −0.1014] | −0.1016 | 1.82x | killed |
| `P_sep` | `P_hard` + separation 0.4 | −0.0228 [−0.0311, −0.0140] | −0.0127 | 1.34x | killed |

The shipped recipe runs at a foreground ratio of 0.709 on the same six blind
cubes. Every arm failed the fixed-threshold six-cube gate at two consecutive
milestones on interior fraction and maximum thickness, and every arm tripped the
preregistered rim-thickening trigger. The frontier veto could not be evaluated:
at the skeleton coverage these models reach (0.86–1.00) they lie beyond the end
of the reference curve, and the protocol forbids extrapolation. None was inert —
the term contributed 12.9% to 21.9% of the committed training loss — so these
are measured negatives, not absences.

`P_sep` is the informative arm. Adding back a reduced separation shell halves
the damage on every axis and still fails every gate: preservation's cost is paid
in girth, and the v31 shell existed to pay it.

The conditional medial tail-floor lane never triggered, because its trigger
requires a surviving preservation arm. That term therefore remains untested at
scale rather than refuted; testing it requires a crest-complete corpus, which
the wide corpus cannot be, since its human-labelled rows can never carry a
medial sidecar.

## Contents

| File | What it is |
|---|---|
| `evidence/preservation_preregistration.md` | the preregistration, fixed before any arm ran: arms, kill criteria, seed-derived tie margin, thickening and tail-lane triggers |
| `evidence/preservation_config.json` | the sealed machine-readable config, including comparator milestone tables and decision numbers |
| `evidence/preservation_kill_table.json` | per-arm verdicts, per-milestone gate/frontier receipts, continuation selection |
| `evidence/preservation_P_*_summary_kill.json` | per-arm: five milestone audits, paired and pooled cluster bootstraps against both comparators at both operating thresholds, loss-share inertness check, thickening trigger |
| `evidence/preservation_SUMMARY.md` | the coordinator's own end-of-run summary |
| `manifest.json` | SHA-256 and byte size for every file above |

Source experiment root: `output/crossres_data/f0_rescue_20260909/` in the
geometry repository. The five comparator audits used here reproduced the sealed
milestone sweep exactly (maximum absolute macro difference 0.00e+00 at both
operating thresholds), so the comparisons are against the same numbers the
release record reports.

## What this record does not establish

These are negative results under one corpus, one optimizer and one horizon.
They establish that these terms do not earn their place in this configuration.
They do not establish that no medial, separation or connectivity objective could
help under a different corpus or schedule, and two terms are explicitly
untested rather than refuted: the medial tail floor, and dynamic connectivity at
breadth.
