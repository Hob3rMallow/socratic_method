# Predicting a scroll region with the frozen model

`socratic-predict` runs the frozen C3-F0 weights on a region of a scroll and
writes two things: an OME-NGFF prediction store in the scroll's own coordinates,
and a ScrollFiesta cube grid. One command, and the model is not a parameter.

```powershell
socratic-predict --scroll PHerc1447 --region 12032:12416,4096:4480,2816:3200 `
  --out .\infer-demo --run
```

```text
.\infer-demo\prediction.zarr    attach this to a viewer over the CT
.\infer-demo\cubes_PRED\        grid_pipeline and socratic-repair read this directly
```

Without `--run` nothing is written and the checked plan is printed instead. That
is the default, so the command above is safe to try before you mean it.

This is a research runner. It does not claim a production deployment, and the
predictions it writes are model output, not certified geometry.

## Install

The repository has no environment of its own. Either build one:

```powershell
uv venv --python 3.12 .venv          # or: py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[zarr]"
.\.venv\Scripts\python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
socratic-predict --help
```

or borrow the engine environment that already has CUDA torch, zarr and s3fs:

```powershell
$env:PYTHONPATH = 'D:\work\socratic_method\src'
& D:\work\vesuvius-c\crossres_pred\.venv\Scripts\python.exe -m socratic_method.predict --help
```

Either way `nvidia-smi` must be on `PATH` and the card's power limit at most
600.5 W: the engine refuses to run otherwise. `--model-hf` additionally needs the
`publish` extra. The exact package set the frozen run observed is in
[recipes/f0/environment.lock.txt](../recipes/f0/environment.lock.txt).

## Three ways to say where

**By scroll name.** The name is resolved through
[recipes/f0/scrolls.json](../recipes/f0/scrolls.json), which supplies the public
volume, the voxel pitch, the level-0 shape and, where the value is unambiguous,
the umbilicus. Volume data is streamed anonymously from the open-data bucket;
nothing is downloaded up front and only the cubes the region needs are fetched.

```powershell
socratic-predict --scroll PHerc1447 --region 12032:12416,4096:4480,2816:3200 --out .\demo --run
```

Names are forgiving: `pherc1447`, `PHerc-1447` and `PHerc1447` all resolve, and
`PHerc139` resolves to `PHerc0139`. An unknown name lists the valid ones.

**By volume.** Any local `.zarr`, `.npy` or `.tif`, or an `s3://` OME-Zarr. Add
`::2` to a zarr spec to read a pyramid level other than 0.

```powershell
socratic-predict --volume D:\work\vesuvius-c\PHerc0139-full\20250728140407-9.362um-1.2m-113keV-masked.zarr `
  --bbox 4352 4864 3072 3712 2560 3200 --umbilicus 3405 2878 --out .\pherc0139 --run
```

**By existing grid.** A directory with `manifest.json`, `cubes_RAW/` and
`cubes_PRED/present.json`. The published cube list is the target inventory and
`--region` narrows it. Raw cubes are hard-linked, so nothing is copied twice.

```powershell
socratic-predict --grid D:\work\vesuvius-c\PHerc1447-10x10x10-overnight `
  --region 12032:12416,4096:4480,2816:3200 --out .\from-grid --run
```

If the grid records which volume it was carved from, the scroll is recognised
from that and the prediction store gets the full scroll frame.

## Regions

`--region z0:z1,y0:y1,x0:x1` is half-open, in level-0 voxels. `--bbox Z0 Z1 Y0
Y1 X0 X1` means the same thing in the older six-token form.

A region need not be cube-aligned. It is expanded outward to the 128-voxel
lattice, and both the request and the expansion are printed and recorded.

Every predicted cube needs a full 32-voxel halo of real CT on all six sides, so
the command stages the cube plus its 26 neighbours. A target whose halo runs off
the end of the volume, or whose neighbours are absent from an input grid, is
**skipped and listed with the reason**. Missing context is never padded with
zeros. Ask for one cube more than you need on each side and use the interior.

## What is pinned, and where it comes from

| Parameter | Value | Where it is recorded |
|---|---|---|
| test-time augmentation | eight-way mirror | `recipes/f0/recipe.json` → `inference` |
| operating threshold | 0.30 | the shipped inference record, 2026-09-09 |
| halo | 32 voxels around a 128 cube (192³ context) | the postprocessor evidence records |
| autocast | bfloat16 | same |
| composition | the raw student, no teacher, no M7 blend | the release record |

The checkpoint is verified before anything else happens: byte size first, then
SHA-256, against `release.selected_checkpoint` in the recipe. The engine snapshot
is checked against the sealed code pins in the same step.

The only model choice left to you is `--threshold`, and only between the two
values the inference preregistration considered:

- **0.30**, the shipped operating point. The default.
- **0.35**, the training-record operating point, which is the *only* input
  `socratic-repair`, `socratic-continuity` and `socratic-sheet-patches` accept.
  They were qualified at T=0.35 and deliberately not re-qualified at 0.30.

Any other value is refused. Passing `--print-command` prints the equivalent
`crossres-voxel infer-grid` invocation, so nothing about the run is hidden.

Do not reach for `crossres-pred infer-grid`, without the `voxel`. That is the
older two-stage engine and it cannot load an F0 checkpoint at all.

## The prediction store

`prediction.zarr` is written in the same shape as the M7 surface predictions the
challenge publishes, which is the form viewers already open over the CT:

| | |
|---|---|
| format | OME-NGFF 0.4 multiscale group, Zarr v2 |
| levels | 6, nearest downsampling, factor 2 per level |
| values | `uint8`, 0 or 255 |
| codec | blosc / zstd, clevel 1, byte shuffle |
| chunks | 128³ |
| coordinates | the scroll's own; level 0 has the volume's shape |

It is **sparse**: only chunks containing predictions exist, everything else reads
back as zero. So the store spans the whole scroll while holding only your region,
and it overlays the CT with no offset arithmetic. The published M7 stores are
sparse the same way.

Two differences from those stores, both recorded in the sidecar `metadata.json`:
chunks are 128 rather than 192, so one prediction cube is exactly one chunk and
nothing is ever re-chunked; and when the volume's shape cannot be determined, the
array stops at the region's upper bound instead of the scroll's. Coordinates are
global either way.

`metadata.json` follows the upstream `surface-inference` shape and records the
model, the checkpoint hash, the region as requested and as aligned, and the
threshold.

## Output layout

| Path | What | Who reads it |
|---|---|---|
| `prediction.zarr/` | the OME-NGFF store above | viewers |
| `cubes_PRED/*.tif` + `present.json` | 128³ `uint8` masks, 0 or 255 | `grid_pipeline`, `socratic-repair` |
| `probability/*.tif` | float16 surface probability | `socratic-continuity`, `socratic-sheet-patches` |
| `cubes_RAW/*.tif` | the CT context that was staged | the above; safe to delete afterwards |
| `manifest.json`, `source_manifest.json` | grid identity, umbilicus, pitch | the postprocessors |
| `provenance.json` | the engine's own record of the inference | the postprocessors |
| `predict_plan.json`, `predict_receipt.json` | what was planned and what happened | audit |

The receipt is the last thing written and carries a SHA-256 inventory of every
output. **A directory without `predict_receipt.json` is incomplete.**

## Umbilicus and pitch

Both are recorded in `manifest.json` and both are operator assertions, not
measurements. `socratic-repair` needs a finite `umbilicus_yx` later, so the plan
says plainly when it does not have one.

The umbilicus comes from `--umbilicus Y X`, else the input grid's manifest, else
the registry. PHerc1447 has two recorded values; the registry carries the one the
frozen repair profile was qualified against and names the other, so pass
`--umbilicus` if you are working in the other box.

The pitch comes from `--voxel-size-um`, else the input manifest, else the volume
name (`…-8.640um-…`), else nothing. F0 is a coarse-resolution model: the corpus
spans 7.910 to 9.366 µm, values outside 7.5 to 10 µm are refused, and anything
below 4 µm is refused outright. `socratic-repair` separately pins 8 to 10 µm and
was measured only at 8.640 µm.

## Was this scroll in the training data

The plan says so. Twelve scrolls trained F0 and two were its validation set;
predicting on those is not an honest demonstration, and the summary warns.
PHerc1447 was deliberately kept out of the corpus, which is why it is the
scroll the frozen model's gates were evaluated on.

## Interruption

The output directory is built in place under an exclusive `.predict.lock`. If a
run stops, the lock is released, the staged raw cubes survive, and no receipt is
written. Repeat the same command with `--resume` to reuse the cubes that verify
against their recorded hashes and finish the job.

A lock or an engine `f0.partial-…` directory that is still present is never
reclaimed automatically, by timeout or otherwise. Look at what owns it first.

## Why there is no `--passes`

F0 is a single-pass model. There is no second pass in this recipe and no evidence
for one, so the flag does not exist; passing it says exactly that.

## Why `--model-hf` exists

`--model-hf <repo-id>` fetches the checkpoint from a Hugging Face repository and
verifies it against the recipe like any other. The frozen weights now ship in
this repository under `releases/c3-f0-200k-20260907/` via Git LFS, so the flag is
not required to run F0; it remains for a Hugging Face mirror, public or private.

## Limits

The model was qualified on one scroll at 8.640 µm and one frozen 689-row
benchmark. Output errors can merge sheets or open gaps, and the geometry
safeguards are separate, opt-in tools. Expect roughly one second per cube with
test-time augmentation on an RTX PRO 6000. Nothing here certifies topology, and
nothing here is deployed.

See also [model.md](model.md), [reproduction.md](reproduction.md), and the
shipped inference record in
[recipes/f0_inference_20260909](../recipes/f0_inference_20260909/README.md).
