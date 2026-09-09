# Pre-registration: the F0 family ladder toward macro Dice 0.70 (2026-09-08)

Operator decisions (2026-09-08): the harness stays the frozen v14p2 689-row
macro-scroll Dice (mean of the per-scroll pooled Dice over PHerc0814 and
PHerc1451); the shipped inference recipe may change (operating threshold,
eight-way mirror TTA, a weight-averaged checkpoint) when pre-registered here;
GPU work is authorized for the inference-variant ladder, the C3r seed
replicate at the full 250k horizon, and two 50k hypothesis arms (C3h hardened
Dice, C3n native-only rows). Configs: `configs/f0_variants_20260908.json`,
`configs/f0_family_20260908.json`. Experiment identity:
`output/crossres_data/f0_family_20260908/`.

## A. Variants ladder

1. **Reference R** = F0@200k (sha `54db9a59…a175`), no TTA, fixed T=.35,
   frozen macro 0.680787428136385 (PHerc0814 .682172 / PHerc1451 .679403).
   R is re-audited first with per-patch counts; the pooled sweep must
   reproduce the sealed sweep within 1e-4 macro at every threshold or the
   ladder stops.
2. **Ledger before scoring.** Variants are declared in the config and the
   journal ledger before any audit; undeclared variants cannot be reported.
   Families: S singles {200k, 110k, 250k} x {noTTA, TTA}; W uniform weight
   averages within F0 {150k–250k (11), 200k–250k (6), top-3 {200k, 220k,
   250k}, last-3 {230k–250k}, 100k–250k (16)}; E mean-probability ensembles
   {200k+110k, 200k+250k, 200k+110k+250k}, **diagnostic only, never shipped**
   (single-network rule); conditional: the two best W by fixed-.35 macro get
   a TTA audit; X after C3r: E{F0-sel, C3r-sel}, E{F0-W200–250, C3r-W200–250},
   exploratory soup W{F0-sel, C3r-sel}.
3. **Family primaries (Holm, alpha .05):** `f0_200k_tta`, `avg_200k_250k`,
   `ens_200k_110k`, `x_ens_f0sel_c3rsel`. Everything else is exploratory.
4. **Endpoints.** Primary: fixed T=.35 macro. Secondary: fixed T=.30, the
   single pre-registered alternative operating point (calibrated argmax .30
   at 21/25 F0 milestones). Calibrated T is diagnostic. No per-variant
   calibration, no T from the wide15 768-row val, no half-split of the rows.
5. **Uncertainty.** Paired cluster bootstrap: clusters =
   `support_anchor_chunk_zyx` within scroll, B=2000, seed 20260908,
   identical resample weights for candidate and reference, percentile 95%
   intervals. It describes patch sampling only, never seed noise.
6. **"Breaks 0.70"** = point macro >= 0.700000 at the declared operating T
   AND paired 95% lower bound of (variant − R) > 0 AND Holm pass (primaries)
   AND per-scroll delta >= 0 on both scrolls. The level interval is printed
   beside every claim. Exploratory variants can be reported, not promoted, on
   v14p2 alone.
7. **Geometry at the promoted operating point** (six cubes, halo 32, TTA,
   bf16): foreground ratio <= 1.25 (.75 advisory), interior delta <= .10 and
   max-thickness delta <= 2 per cube; frontier veto if bridge pixels exceed
   1.20 x the C3 curve interpolated at matched coverage (no extrapolation);
   v14p2 TopoScore >= .375646 − .005 at the operating point.
8. **PHerc0500P2 (absolute only):** composite at the operating point
   >= .481767 − .005. M7+TTA .5015 is tabled, never a baseline.
9. **Replication:** the winning inference recipe (member index set, TTA, T)
   applied to C3r must beat C3r's own reference with paired lower bound > 0.
10. **Averages** whose fixed-.35 macro < min(members) − .005 are logged as
    barrier failures, not weak variants.
11. **Any new weights, T or ensemble breaks the repair provenance pins**
    (`run_f0_repair_context.py`: sha, .35, halo 32, TTA, bf16). "Promoted"
    means raw model only until the repair profile is re-qualified.
12. **TTA legitimacy:** deployment already exports with eight-way TTA; the
    honest matched-inference pair is student+TTA at fixed .35 vs M7+TTA at
    its in-sample calibrated .20. No-TTA remains the harness number.

## B. Training arms (family coordinator)

13. **C3r** = the sealed F0 argv with `--seed 1204`, full 250k, 25 snapshots,
    new identity `arms/C3r`, fresh released M7, three launches max with strict
    `--resume`. It defines the seed band: RMS of (F0 − C3r) fixed-.35 macro
    over the 25 matched milestones (also RMS over >= 150k and <= 50k). Tie
    margin for any cross-run claim = max(.005, 2 x RMS). The historical
    B0/B0r .0715 band is a labelled placeholder until C3r reaches 50k, never a
    confidence interval. C3r's own checkpoint is chosen by the identical F0
    rule; the honest same-recipe comparison is late-milestone mean vs mean.
14. **C3h** = F0 argv + `--loss-hard-dice-weight 1.0` (CE 1 + soft Dice 1 +
    hard Dice on q >= 0.5 + M7-KL .5), declared 250k for LR identity, killed
    at the durable 50k milestone. Documented confound: human rows carry a
    binary q, so their Dice weight doubles.
15. **C3n** = F0 argv on the native-only manifest (wide15 rows whose
    `supervision_source` starts with `official-native-fine-teacher/`, val
    split intact; expected 10,072 rows = 9,304 train / 768 val, 12 scrolls),
    same 50k kill.
16. **Promotion (C3h, C3n):** primary = mean of the fixed-.35 macro deltas at
    30k and 50k vs F0 (.676853 / .638938, re-audited with per-patch counts)
    and vs C3r; all five matched milestones tabled, never best-of-5. "Beats"
    = delta > tie margin AND paired lower bound > 0 at both milestones AND
    the .35 six-cube gates + scroll guard pass at 50k. Promotion to 250k =
    strict same-run resume from the 50k optimizer commit; judged afterwards
    by the full F0 rule; then its own replicate before it can ship.
17. **Cross-run:** within-run weight averages are shippable candidates;
    cross-seed probability ensembles are the X primary; cross-seed weight
    soups are exploratory with the barrier check.
18. **Fixed:** never resume old C3 or F0; never write into the F0 run dir or
    the sealed release; the sealed evaluator is not re-run (its code pins
    drift once `voxel/*.py` changes; logged, never re-pinned); no selection
    on `history.jsonl` or the in-run 768-row val; no threshold or checkpoint
    choice on PHerc1447; PNGs veto, never pick; no concurrent GPU jobs; the
    450 W cap is a wall-time matter only and is logged if changed.
