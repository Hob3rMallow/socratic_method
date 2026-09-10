# Pre-registration: the F0 rescue ladder (2026-09-09)

Operator decisions (2026-09-09): of the seven loss terms in the 2026-08-31
Socratic Method submission (v31), only one-sided M7 preservation survives the
evidence review as a rescue candidate; run its two forms on the exact F0
recipe to the durable 50k optimizer commit, then PAUSE for review. The 250k
continuation, the seed replicate and the shipped-recipe check are
pre-registered below but gated behind an operator authorization file. The
medial tail floor is a conditional lane on the only crest-complete corpus,
gated behind a trigger AND a second authorization file. Config:
`configs/f0_rescue_20260909.json`; experiment identity
`output/crossres_data/f0_rescue_20260909/`; coordinator
`scripts/run_f0_rescue.py`; evaluator `scripts/evaluate_f0_rescue.py`;
launcher `scripts/launch_f0_rescue.ps1`.

## A. Question and arms

1. **Question.** Can one-sided M7 preservation be re-added to F0 (CE 1 +
   Dice 1 + M7-KL 0.5, wide15, SGD 1e-3 poly, no ball) with a late-band macro
   gain of at least 0.012 beyond seed noise, without rim thickening, and does
   it restore PHerc0500P2 topology (students lose topology to M7 there,
   .10-.12 against .146)?
2. **Why only this term.** Separation shell: a threshold dial; within the
   same lane C1 (B0 + shell 0.5) had a WORSE bridge-vs-coverage frontier than
   B0 (25,597 vs 16,928 bridges at matched coverage). Crest recall: refilled
   blobs at 1.5 and its ceiling hypothesis was measured absent on wide15 (C5
   trigger band fraction 0.60 > 0.5). Medial tail floor: needs `teacher_crest`
   on every training row, impossible with wide15's 6,128 human rows.
   Dynamic connectivity: events only where a full-volume M7 prediction exists
   (PHerc0139, 3% of rows). Trust ball: 88.7x/113.9x free displacement.
   Hard Dice: -0.032 [-0.040, -0.024] vs F0 at 30k. Pinned axial: superseded.
   Preservation was never isolated at scale; its soft-floor variant was swept
   at 1,024 samples on 2026-08-31 and never scored; the changelog post-mortem
   after R_sep3 named it the right next axis.
3. **Arms** (all: sealed F0 argv, fresh released M7, seed 1203 unless a
   replicate, `--max-train-samples 250000`, the FULL 25-snapshot list because
   the snapshot list is run identity, trust radius 0):
   - **P_hard** = F0 + `--loss-m7-preservation-weight 1.0 --loss-m7-preservation-radius 2 --loss-m7-preservation-anchor-threshold 0.5` (the v31 setting; contract v5).
   - **P_soft** = the same at weight **4.0** + `--loss-m7-preservation-soft-floor` (contract v7). The soft-floor KL acts only where the student has dropped below M7 and is second-order small; the v22 sweep labelled 1.0 "unit control", 4.0 "moderate", 16 "strong", so 0.5 would likely trip the inertness predicate.
   - **P_sep** (conditional C1) = the thickening parent + `--loss-separation-weight 0.4`.
   - **P_hard_r / P_soft_r** (conditional, after promotion) = the promoted arm with `--seed 1204`, full 250k.
   - **T_base / T_tail** (conditional C2 + authorization) = F0 / F0 + `--loss-medial-tail-floor-weight 0.015625 --loss-medial-tail-floor-probability 0.3 --loss-medial-tail-bottom-fraction 0.2` on the native v19 medial corpus (249,836 rows, 4 scrolls, no val split; `--validation-patches` = the frozen v14p2 manifest). The corpus change is the known confound (C3n's -0.018), which is why T_tail is judged against T_base only.
4. **Kill mechanics.** The kill fires on the durable 50k optimizer commit:
   `checkpoint_last.pt` at `cumulative_samples == 50000` (epoch 4) with the
   optimizer state, history row 5 present, snapshot index = 10k..50k. The
   family ladder killed on the milestone file, which the trainer writes
   before validation and the commit; C3h's committed history stops at 40k,
   so the family's "same-run resume from the 50k commit" was never
   executable. Continuation resumes the same identity with `--resume`.

## B. Kill stage (authorized)

5. Per arm, five milestones (10k..50k): frozen v14p2 audit with per-patch
   counts at 17 thresholds, fixed T .35 and .30 macro with level cluster
   bootstrap, six-cube gates at {.25, .30, .35, calibrated}, frontier at .35,
   paired cluster bootstraps vs F0 and vs C3r at each milestone and a POOLED
   paired delta over the five milestones (one resample matrix shared by every
   audit of both runs).
6. **Comparators.** F0 per-patch audits at 10k/20k/40k are produced once
   under the rescue root; 30k/50k are reused from the family evaluation and
   110k/200k/250k from the variants ladder when their checkpoint sha matches;
   every F0 re-audit must reproduce the sealed sweep within 1e-4 at .35 and
   .30 or the ladder stops. C3r audits are reused from the family root after
   sha and macro verification.
7. **Kill predicates** (any kills): K1 `.35` six-cube gate fails at BOTH 40k
   and 50k, or the scroll guard fails at both; K2 frontier veto (bridges >
   1.20 x the C3 curve at matched coverage, resolved) at BOTH; K3 pooled
   delta at .35 with point < -0.030 AND upper bound < 0 vs F0 AND vs C3r;
   K4 technical (nonfinite loss, identity mismatch, three attempts). **I**
   (inert): weighted share of the rescued term in the committed loss below
   0.01 = unmeasured, no resume, no re-weighting inside this ladder.
   Single-milestone deltas are tabled, never decisive.
8. **C1 rim thickening** (both 40k and 50k): `.35` six-cube foreground ratio
   >= 1.10 x F0's sealed value at the same milestone, OR thickness gate
   fail, OR interior gate fail, OR frontier veto fail. Then P_sep runs at the
   kill stage on the thickening parent with the better pooled delta. A
   killed parent is replaced by P_sep; a surviving parent competes with
   P_sep by pooled delta and only the better one resumes.
9. **C2 tail lane**: a surviving arm at 50k with fixed-18 coverage >=
   F0@50k + 0.02, a resolved passing frontier and foreground <= 1.10 x
   F0@50k, AND `authorizations/T_lane.json`. Verdict: T_tail vs T_base pooled
   5-milestone delta at .35 with point > 0 and lower bound > 0 and no K1/K2
   = eligible for its own 250k prereg. A 50k pooled mean cannot resolve
   +0.02 against seed noise; this is never a ship claim.
10. **STOP.** `SUMMARY.md` + `decision/kill_table.json`, journal
    `operator_review_pause`, task disabled.

## C. Late-band promotion (gated on `authorizations/continue_250k.json`)

11. **Power.** C3r vs F0: RMS 0.029532 over 25 milestones (0.0299 at <=50k,
    0.0191 at >=150k); the family tie margin 0.059 cannot resolve +0.02.
    Late-band (>=150k, 11 milestones) means: F0 0.661331 / C3r 0.655322 at
    .35 (0.670448 / 0.665517 at .30). The 11 paired late deltas have SD
    0.0191, so the SE of an 11-milestone mean is 0.0058; **tie margin =
    max(0.005, 2 x |0.006008|) = 0.012** (~2 SE). +0.020 is ~3.4 SE.
12. Survivors resume to 250k; all 25 milestones audited with geometry;
    pooled late paired deltas vs F0 and vs C3r at .35 and .30; the selected
    milestone by the F0 rule (best eligible >= 150k, ties later); Kaggle
    composite on v14p2 (689) and PHerc0500P2 (320) at .35 and .30 without
    TTA; PHerc0500P2 TopoScore paired cluster bootstrap vs sealed F0@200k.
13. **D1** late mean (.35) exceeds BOTH F0 and C3r late means by >= 0.012.
    **D2** pooled late lower bounds > 0 vs both. **D3** no reversal at .30.
    **D4** scroll guard >= 10/11, .35 gates >= 9/11 and at the selected
    milestone, frontier resolved + passed there. **D5** v14p2 composite >=
    .6304 - .005 (.35) and .6401 - .005 (.30), v14p2 TopoScore >= .375646 -
    .005, PHerc0500P2 composite >= .481767 - .005. **D6** PHerc0500P2
    TopoScore paired lower bound > -0.010. `promote` = D1..D6;
    `unresolved-positive` = D2 with point > 0 but not D1 (inside the margin
    is unresolved, never equivalent); else `reject`. **H_topo** (reported,
    never required): TopoScore delta >= +0.010 with lower bound > 0.
14. **Replicate.** The promoted arm's seed-1204 run must pass D1, D2 and D4
    on its own; the reported effect is the mean of both seeds' late deltas;
    the two seeds' late means define this family's seed band. D2 with
    point > 0 but not D1 = unreplicated positive, not shippable.
15. **Ship check.** Best seed's selected milestone + eight-way mirror TTA at
    T .30: macro >= the shipped F0 200k + TTA value with paired lower bound
    > -0.005 vs `f0_200k_tta`; TTA gates + frontier at .30; TTA Kaggle
    floors as D5; family rules 6 ("breaks 0.70") and 11 (repair-profile
    re-qualification) apply unchanged.

## D. Fixed

16. Never resume old C3, F0 or the family arms; never write into the F0 run
    dir, the family root or the sealed release; the sealed evaluator is not
    re-run; no `voxel/*.py` edit (the rescue preflight asserts the voxel-tree
    pins equal the family config's); no selection on `history.jsonl` or the
    in-run 768-row val; no threshold or checkpoint choice on PHerc1447; PNGs
    veto, never pick; no concurrent GPU jobs; the 450 W cap is a wall-time
    matter only. `run.json` omits `m7_preservation_soft_floor` when false, so
    P_hard's identity must NOT carry the key and P_soft's must carry `true`.
17. **Budget** (observed at 450 W): kill stage ~10 GPU-h (comparators 0.25 +
    2 x (4.3 + 0.5)), ~15 with P_sep, +9.5 if the tail lane fires and is
    authorized; continuation ~20 GPU-h per survivor, replicate ~25, ship
    ~0.7.
