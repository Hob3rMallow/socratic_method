# Current model: C3-F0

As of 7 September 2026, the current frozen model is **F0 at 200,000 sample
exposures, operating threshold 0.35**, selected after the complete 250k run.
It improves matched frozen macro Dice from released M7's **0.554140 to
0.680787**: **+0.126647 Dice / 12.7 percentage points / 22.9% relative**.

The authoritative [manifest](../recipes/final_c3_250k_20260907/manifest.json)
and [full evidence review](../recipes/final_c3_250k_20260907/review.md) distinguish
this selected milestone from the retained exact 250k endpoint.

## Architecture and training

The model is one 3D residual-encoder nnU-Net, initialized from the exact released
M7 checkpoint: one CT channel, two classes, six stages, features
`(32, 64, 128, 256, 320, 320)`, and 102,349,770 parameters. Inference uses the
raw student, not a teacher ensemble or an M7 blend.

Training uses the wide15 anti-aliased mixed corpus: 15,432 training rows and
768 validation rows, over 12 training and two validation scrolls. Across all
16,200 rows, 10,072 native rows carry soft targets and 6,128 human rows are
hard passthrough. The 250k budget counts sample exposures, not unique rows.

The objective is CE 1 + Dice 1 + M7-KL 0.5, with known-agreement,
confident-agreement and unknown-corridor radius 2. All additional loss terms
are zero. Optimization is SGD 1e-3, momentum 0.99/Nesterov, weight decay 3e-5,
gradient clip 12, and a poly exponent of 0.9 over the full 250k horizon.
Batch size is 3, accumulation 1, BF16, seed 1203 and augmentation enabled.
**There is no trust ball or projection.**

The exact settings and identities are in the byte-preserved
[preregistered recipe](../recipes/final_c3_250k_20260907/provenance/recipe.json)
and [run record](../recipes/final_c3_250k_20260907/provenance/run.json).

## Selection and evidence

All 25 checkpoints completed their frozen audits and were eligible at T=.35.
The preregistered late-selection rule chooses 200k (macro Dice 0.680787);
110k is the global maximum (0.681556), and the exact 250k endpoint is 0.669299.
The late-versus-early hypothesis is met, but the +0.003935 over original
C3/30k is not evidence of duration superiority beyond seed noise.

At the selected operating point, v14p2 native composite is 0.630391,
topology 0.375646 and surface Dice 0.855518. Composite improves over the
stronger calibrated M7+TTA baseline; topology narrowly trails it. PHerc0500P2
results are absolute-only evidence, not an unseen-M7 gain claim.

Visual review supports cleaner paths through broad merged regions. The
selected 200k weights retain more weak geometry than the exact endpoint.
Postprocessing remains separate: the conservative six-cube filler added only
81 voxels through one tracked connection, and is not a validated production
repair policy. The [review](../recipes/final_c3_250k_20260907/review.md)
details its context, support-filter and safety constraints.

## Historical model

The earlier 8,192-sample v31 model at T=.45 is preserved as historical evidence,
not erased or relabeled. Its complete former model documentation is
[archived here](archive/v31_model.md), and its recipe/selection remain under
[recipes/v31](../recipes/v31/recipe.json). Existing portable training/export
wrappers still target that historical recipe. Freezing F0 does not silently
retarget those commands or claim public deployment.
