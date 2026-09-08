# Authorized: C3 / F0 through the full 250k schedule

Corrected 2026-09-05 after operator review. Config:
`configs/final_c3_250k_20260905.json`. The operator explicitly authorized:
“Correct these instructions. Launch the corpus. heartbeat/status update every
30 minutes.” This authorizes **F0 only**, its evaluation, and an HTML report.
C3r and N0 are deferred proposals, not prerequisites or executable arms.

## Fixed training contract

Fresh released M7, SHA256
`17465b77591b794638e671f1a9f79c4cf1e79821f302e6fc235e3725e5da7d7e`.
Do not resume the old C3. Its last optimizer commit is 40k; the new snapshot
list is also part of a different sealed run identity.

Wide15 anti-aliased manifest, SHA256
`f18c4efbb0de1e2576f6d3900d1e0994b0159dbddb0f3ba7305ea22724031128`:
16,200 rows, 15,432 train, 768 val, 12 training plus 2 validation-only scrolls;
10,072 native soft-target rows
plus 6,128 human hard-label passthrough rows. PHerc0500P2 is excluded;
PHerc0814 and PHerc1451 are validation only. **250k is sample exposures, not
250k unique rows.** Do not substitute the four-scroll native corpus.

CE 1 + Dice 1 + M7-KL .5; known-agreement and confident-agreement enabled,
unknown-corridor radius 2. This is the recorded C3 flag triple; without crest
sidecars its corridor seed remains truth-only, as in the original C3.
All other loss weights zero; **trust radius zero**. SGD .001, momentum .99,
weight decay 3e-5, poly exponent .9 over all 250,000 exposures, zero warmup
and LR floor. Batch 3, accumulate 1, BF16, augmentation on, unstratified
sampling, seed 1203. Keep the recorded C3 defaults (including gradient clip
12). Only output and the additional snapshots differ from the recorded argv.

Save every 10k milestone through 250k. Allow natural trainer exit, including
the final validation, optimizer checkpoint and history commit. A model-only
250k snapshot alone is NOT completion. Never apply the old ladder's
kill-at-milestone mechanism to F0.

## Why this choice is defensible

C3@30k's frozen no-TTA macro is .676853 at T=.35; v14p2 Kaggle composite is
.635015. Its .25 blob gate fails interior/thickness; the soft-arm .35 gate
passes (the foreground lower bound is advisory).

On the fixed 18 views, C3@30k at .35 has 5,011 reference-relative bridge pixels,
.758198 reference-line coverage and 151 meaningful components. The human
pilot at .25 has 5,854, .730176 and 238 respectively, at similar but not
identical foreground (.697x versus .682x). This supports C3 as the preferred
observed tradeoff; it does not establish universal frontier dominance.
The jackpot reference is a model exemplar, not human ground truth.

The 8x-ball lane is excluded by the operator. Boundary projection does not
prove further learning is impossible: human25's first-interval projection
fraction was .76555, then 1.0. Likewise, correlated milestones from one seed
replicate cannot establish superiority beyond seed noise. No seed replicate
is required for this authorized run.

## Fixed operating threshold and evaluation

**Operating T=.35.** It is not replaced by a validation-calibrated threshold.
The .25 gate remains advisory. Never tune thresholds or choose the winning
checkpoint from the PHerc1447 images.

After training, sequentially evaluate all 25 milestones on the unchanged
689-row v14p2 validation manifest (SHA256
`a995787032eb446f71759a4dbd510396273b93c65f65863bac8fe9587e12b714`).
Use the frozen 17-threshold no-TTA audit; record fixed-.35 macro and per-scroll
gains, with calibrated values as diagnostics. Measure M7 displacement only;
no projection is applied.

For each milestone, export all six PHerc1447 cubes using halo32, BF16 and
8-way TTA at .35. Reuse the frozen morphology/gate functions at .35, advisory
.25, and calibrated T. The same fixed18 and named3 bridge/coverage/component
frontier is measured at .25/.30/.35/.40. The promoted CPU measurement must
reproduce saved C3/human baseline results and pass tests before launch.
No baseline model is retrained or reinferred for that check.

Eligibility requires the .35 six-cube gate: foreground ratio <=1.25,
each cube's interior delta <=.10 and max-thickness delta <=2. The .75 lower
foreground bound is advisory. Both frozen validation scrolls must have
nonnegative released-M7 Dice gain at .35. The in-run wide15 hard-target
validation is telemetry only and cannot select or stop F0.

Propose the highest fixed-.35 macro among eligible milestones >=150k,
ties to the later milestone. Evaluate every milestone BEFORE applying this
rule. Report exact 250k, best eligible early (<=50k), global best eligible,
global best unrestricted, and original C3@30k alongside. No eligible late
checkpoint means no proposed replacement, not an undocumented fallback.
H1 is late-best >= early-best, separately from the delta against C3@30k.
The historical B0/B0r .0715 variability band is descriptive, not a confidence
interval or permission to silently replace C3 with a worse model.

On the proposed candidate (or exact 250k as explicitly labeled diagnostic
if none is eligible), score v14p2 and PHerc0500P2 at fixed .35, calibrated T,
and legacy .45, deduplicating equal thresholds. At most TWO metric workers.
Readiness checks at .35 are advisory: v14p2 composite >=.6212 and topology
>=.3776. Label no-TTA student versus M7+TTA as a stronger inference benchmark,
not identical inference. PHerc0500P2 scores are absolute only because released
M7 saw those labels. Retain raw-model and any future filler evidence separately.

For structural review, interpolate C3's recorded fixed18 bridge curve to the
candidate's .35 line coverage. More than +20% bridges is a review veto;
never extrapolate outside measured overlap (report unresolved). This does not
tune T or automatically select a different checkpoint. Generate HTML with
all milestone evidence and stop for operator review: no automatic deployment,
checkpoint deletion or new experiment.

## Interruption, ownership and status

No score-based early stopping. The former .5926 two-audit guard is removed:
no-TTA student versus M7+TTA is not an apples-to-apples divergence test, and
post-training audits cannot be an in-training guard. Low scores can raise
retrospective alerts only. Technical failures (including nonfinite loss) are
reported separately. Strict resume of this SAME sealed run after a crash,
with at most three training launches TOTAL across coordinator restarts.
Never change inputs/code, relax identity checks or fresh-restart over a run.

Use isolated `scripts/run_final_c3_250k.py` and
`scripts/launch_final_c3_250k.ps1`; leave historical launchers unchanged.
One Hidden/IgnoreNew task, bound to the config SHA256, with an AtLogOn resume
trigger and a singleton lock. The existing GPU-lock protocol serializes
training/evaluation. The old UnifiedLadder really DOES have an enabled
AtLogOn trigger: disable it before F0. All obsolete 8x-ball tasks remain
Disabled. Wait for unrelated existing evaluation workers; do not kill them.
No concurrent GPU evaluation while F0 trains.

Use only RTX PRO 6000 UUID GPU-304375e6-2d2f-b771-b8be-a0c095fe209e at the
existing 450 W cap. No power-limit increase. Constrain the owned process tree
to 16 logical CPUs. Verify disk space immediately before launch. Training
estimate is about 22 hours at historical C3 throughput; the larger explicit
all-milestone evaluation and two-worker scoring may take longer than the
old 3.5-hour estimate. Update ETA from observed progress.

Write a durable status receipt every 30 seconds and a heartbeat in the run
log every 30 minutes; the active assistant provides chat updates every 30
minutes. Report validated samples separately from saved-but-not-yet-validated
snapshots, training loss/LR, failures/retries and evaluation stage. Stop the
scheduled task after the final report reaches operator review.
