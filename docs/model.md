# Current model: C3-F0

The current frozen model is **F0 at 200,000 sample exposures**, selected on
7 September 2026 after the complete 250k run. Since 9 September 2026 it ships
with **eight-way mirror TTA at operating threshold 0.30** on the unchanged
weights: **0.713026** macro Dice on the frozen benchmark, against released M7's
0.5575 flat or 0.5875 with the same TTA at the same threshold. The training
record's matched no-TTA comparison at T=0.35 stands: **0.554140 to 0.680787**,
**+0.126647 Dice / 12.7 percentage points / 22.9% relative**.
The September 2026 progress video and paper present this model as **Socratic
Method September 2026**.

The authoritative [manifest](../recipes/final_c3_250k_20260907/manifest.json)
and [full evidence review](../recipes/final_c3_250k_20260907/review.md) distinguish
this selected milestone from the retained exact 250k endpoint.

## Inference recipe update (9 September 2026)

The 2026-09-08 variants ladder was preregistered before scoring
([record](../recipes/f0_inference_20260909/README.md)): the family primary was the
frozen 200k weights with eight-way mirror TTA, the primary endpoint fixed T=0.35,
T=0.30 the single secondary operating point, and a 0.70 claim required a positive
paired cluster-bootstrap lower bound after Holm plus non-negative deltas on both
scrolls. The primary reached 0.703826 at T=0.35 (+0.023039 [+0.019517, +0.026546]
over the no-TTA reference) and 0.713026 [0.701206, 0.724822] at T=0.30, the
declared argmax. Six-cube gates and the frontier veto pass at T=0.30 (foreground
ratio 0.799; 4,945 bridge pixels against a 7,983 maximum at skeleton recall
0.782). The Kaggle-style composite on the frozen rows rises from 0.6401 to 0.6521
and topology from 0.3880 to 0.4016, so the 0.3776 topology advisory missed by the
no-TTA record is now met. On the human-labelled PHerc0500P2 the composite is flat
across every variant (0.476 to 0.497) and released M7 with TTA at its calibrated
T=0.20 stays ahead there (0.5015 against 0.4917); those scores are absolute only.
Weight averaging over late milestones only reduced variance; the exploratory
F0 110k + TTA and the 100k-250k average + TTA score at or slightly above the
shipped recipe and are reported, not promoted. The C3r seed replication of the
inference recipe passed (seed 1204: 0.7089 at T=0.30 with TTA, +0.0351
[+0.0302, +0.0400] over its own no-TTA reference;
[record](../recipes/f0_c3r_replication_20260909/README.md)). Open:
re-qualification of the repair, continuity and sheet-patch postprocessors,
which keep their T=0.35 pins.

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

"All additional loss terms are zero" is a measured choice, not an omission. The
earlier v31 objective carried four further terms and a parameter-space trust
region; on a corpus of adequate breadth, with the trust ball off and the
schedule run to its declared length, only the M7 function anchor clears its own
seed-noise floor. The single-term ablation table, the preregistered preservation
experiment that retired the last candidate on 2026-09-10, and the two terms that
remain untested rather than refuted are in the
[ablation ledger](../recipes/f0_ablations_20260910/README.md).

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
[recipes/v31](../recipes/v31/recipe.json). The portable training, export and
prediction wrappers (`socratic-train`, `socratic-export`, `socratic-predict`)
default to the executable F0 recipe
([recipes/f0/recipe.json](../recipes/f0/recipe.json)), whose `inference` section
pins the shipped runner: eight-way mirror TTA, T=0.30, a 32-voxel halo and BF16.
Pass `--recipe recipes/v31/recipe.json` for the historical one; it carries no
`inference` section, and `socratic-predict` refuses it for that reason. Neither
implies public deployment. See [predict.md](predict.md).
