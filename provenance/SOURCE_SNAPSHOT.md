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
