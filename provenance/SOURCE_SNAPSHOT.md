# Source snapshot provenance

The implementation was copied on 2026-09-01 from
`D:\work\vesuvius-c` while the v31 duration run was active.

- Source branch: `experiment/crossres-v15-relaxed-trust`
- Source HEAD: `7bf25e7` (`checkpoint: preserve M7-XR v25 9-of-16 winner`)
- Source worktree: dirty; the v27-v31 implementation and plans were not all
  committed at that HEAD.
- Canonical live run:
  `output/crossres_data/m7_xr_v31_pherc0139_dynamic_medial_duration_8192_20260831/candidates/dynconn_w0p03125_n8192_duration`
- Source recipe SHA-256:
  `3d42f3268ee8fa6d25a4949d9ae12145ff9447eccd795900fa02e857e659d0de`

`src/crossres_pred` is the full Python package snapshot so checkpoint module
names and research utilities remain available. Selected original plans, run
records, and drivers are verbatim under `provenance/source`. The native line
fitter is a verbatim extraction of its C sources plus a new standalone build.

One deliberate post-copy change was made to the Python engine: the new
`crossres_pred.pathmap` hook remaps an original artifact-root prefix at read time.
Calls in voxel I/O and manifest/state resolution use that hook. With the mapping
environment variables absent, the original behavior is unchanged. This avoids
rewriting provenance-bound JSON files just to move them to artifact hosting.

The original recipe is retained verbatim because it contains an inherited
metadata inconsistency: `corpus.train_scrolls` names four scrolls, while the
hashed manifest has 4,096 PHerc0139 rows and no other training scroll. The
executable recipe corrects the scope based on the actual bytes.

Run `python scripts/build_source_manifest.py` to regenerate
`provenance/SHA256SUMS` after intentionally updating a source snapshot.

## Release qualification update (2026-09-01)

The public release now selects the raw 8,192-sample v31 checkpoint at an
operating threshold of 0.45.  It does not blend the checkpoint with M7 or use
the fine teacher at inference.  The exact selected checkpoint is identified by
byte size and SHA-256 in `recipes/v31/recipe.json`; the locked-gate, blind
PHerc1447, and FLIP evidence used for that decision is recorded in
`recipes/v31/release_qualification.json`.

Verbatim machine-readable evaluation outputs and the audit/report programs are
preserved under `provenance/source/evaluation/v31_release`.  The release and
paper figures are regenerated from those outputs and the original registered
report imagery by `scripts/generate_release_figures.py`; each generated figure
and every discrete source it reads is hashed in the adjacent figure manifest.

## F0 engine sync (2026-09-08)

The current model is C3-F0 (`recipes/final_c3_250k_20260907`).  Its sealed
configuration pins 63 source files by SHA-256 (`provenance/recipe.json`,
`execution.code_pins`) and keeps byte-exact copies under `runtime/`.  The engine
snapshot in `src/crossres_pred/voxel` was re-synchronised to those pins by
`socratic-engine-pins --apply` (`src/socratic_method/engine_pins.py`), copying
the 48 pinned `voxel` modules from the record's `runtime/` tree, never from the
private working tree.  Eight modules changed content relative to the 2026-09-01
snapshot (`antialias_corpus.py`, `cli.py`, `io.py`, `loss.py`, `patches.py`,
`resources.py`, `train.py`; `blob_diagnostics.py` was added).  The synced tests
`tests/test_voxel_{antialias_targets,img2img,medial,train_resume}.py` and
`tests/test_blob_diagnostics.py` follow the same sources.  The live source date
of the pinned code is 2026-09-05 (vesuvius-c worktree, sealed with the F0
launch); the observed environment was PyTorch 2.11.0+cu128.

The pathmap deviation is now a reversible table in `engine_pins.HOOKS`:
`io.py` gains two inserted lines (the `remap_volume_spec` import and its call at
the top of `split_volume_spec`); `patches.py` gains the `remap_embedded_path`
import and six line substitutions replacing `Path(str(...))` reads of embedded
artifact paths (`path`, `catalog`, `training_manifest` twice, `atlas_state`,
`medial_state`, and the anti-aliased lineage `source_manifest`, which is new in
the pinned code).  `socratic-engine-pins --check` verifies every pin against the
record and against the snapshot modulo that table, and refuses unpinned modules.

Pinned bytes must never be normalised: the root `.gitattributes` marks
`src/crossres_pred/**` as `-text`, so the index stores the exact pinned bytes and
`provenance/SHA256SUMS` is reproducible from a clone on any platform.  The 14
pinned scripts and the ladder configuration are provenance only; they live
hash-verified in the record's `runtime/` tree and are not duplicated under
`provenance/source`.
