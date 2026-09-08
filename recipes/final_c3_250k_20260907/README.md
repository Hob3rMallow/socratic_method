# C3-F0: frozen 200k model from the completed 250k run

The Socratic Method has a substantial new segmentation result: **0.6808 macro Dice versus released M7's 0.5541**, measured on the same frozen 689-row benchmark at T=0.35 without TTA. That is **+0.1266 Dice (12.7 percentage points; 22.9% relative improvement)**. It also exceeds the stronger M7 + eight-way-TTA baseline's calibrated 0.5926.

The current model is **F0 at 200,000 sample exposures, T=0.35**, selected by the preregistered late-checkpoint rule after training completed all 250,000 exposures. It is not the exact 250k endpoint.

## Frozen identity

- Checkpoint: `checkpoint_f0_00200000.pt`, 409,674,095 bytes.
- SHA-256: `54db9a59cb602ff08e3dda0acc1195ad980ab05fb660332d32fb436a0d49a175`.
- Local release directory: `D:/work/vesuvius-c/output/crossres_data/releases/c3-f0-200k-20260907`.
- Fresh released M7; wide15 anti-aliased corpus; CE 1 + Dice 1 + M7-KL 0.5 with the shipped flag triple; SGD 1e-3, poly over the full 250k horizon; **no trust ball**.
- Raw student only: no teacher or M7 blend at inference. Model weights are copied separately, hash-verified and read-only; they are not committed to the official Git repository.

## Record

- [Frozen model manifest](manifest.json): selected identity, operating point, comparisons and boundaries.
- [Full post-run review and next steps](review.md): the win, all 25 checkpoints, geometry tradeoffs and the filler diagnosis.
- [Machine-readable analysis](analysis.json): exact values reduced from existing evidence.
- [Original sealed recipe](provenance/recipe.json), [run identity](provenance/run.json), [all-milestone evaluation](provenance/evaluation_summary.json).
- [Copied-file hashes](files.json): 63 sealed runtime files plus the existing run, baseline and filler evidence.

The exact 250k checkpoint and all earlier checkpoints remain intact in the original run directory. Freezing this model does not change the operating threshold, replace raw metrics with filled metrics, or authorize another experiment.

The existing `socratic-train` / `socratic-export` wrappers still implement historical **v31**, not this F0 recipe. The `runtime/` directory preserves the 63 preregistered source/configuration files; it is provenance, not a newly tested portable distribution. Large input corpora, M7 weights and the full cached HTML/image reports remain external artifacts. The HTML identities and original locations are in the manifest. No public upload or production deployment was performed.
