# Data and artifact layout

## Required replay inputs

The v31 training command needs four pinned files/bundles:

| Key | Expected identity | Purpose |
|---|---|---|
| `m7_checkpoint` | `17465b...d7d7e`, 820,473,701 bytes | released M7 initializer and function anchor |
| `train_manifest` | `dd2d3d...f0177`, 4,096 rows | PHerc0139 fine-teacher atlas schedule |
| `validation_manifest` | `a995787...714`, PHerc0814 + PHerc1451 | registered held-out validation |
| `dynamic_medial_connectivity_state` | `5489a6...a8f8` | 149-event widest-path atlas |

The manifest and state files refer to atlas arrays and patch archives. Publish
the complete directory trees, not just the named JSON/JSONL files. The current
validation tree is about 31 GB and therefore belongs in dataset/artifact storage,
not ordinary Git.

## Relocating frozen bytes

The source experiment intentionally recorded absolute Windows paths. Rewriting
those JSON files would change their hashes. This repository instead supports a
single prefix mapping at read time:

```json
{
  "original_root": "D:/work/vesuvius-c",
  "artifact_root": "/mnt/vesuvius-c"
}
```

Keep the directory suffix below that root unchanged, for example
`output/crossres_data/...`. `socratic-train` exports the mapping to the child
process, and the copied engine remaps embedded paths before opening them. With no
mapping configured, behavior is byte-for-byte the source behavior.

## Rebuilding from raw inputs

For a from-raw rebuild, start from the pinned native-teacher pairs, materialize
the approximately 2.399 um Villa teacher predictions, build the coarse teacher
atlas and medial crest, construct the 4,096-row PHerc0139 schedule, audit medial
geometry, and build the dynamic connectivity atlas. The exact research drivers
and plans are preserved under `provenance/source/scripts` and
`provenance/source/configs`.

Those drivers are evidence rather than a one-command public downloader: public
volume locations, Villa source/checkpoint identity, storage capacity, and data
licenses must be supplied by the reproducer. A future dataset release should
ship the prepared replay bundles plus a dataset card documenting those upstream
terms.
