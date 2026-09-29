# F0 head-to-head: the official M7 and HercUNet v0, and an audit of published M7 blobs

Measured 24 September 2026 on the unchanged shipped model (C3-F0 200k, eight-way
mirror TTA, T = 0.30, sha `54db9a59…a175`). Nothing here changes the model, its
threshold or its recipe.

## Summary

- **On the frozen benchmark F0 is ahead of both on every measure.** It scores
  0.7130 macro Dice. The organisers' M7, run exactly as they publish it, scores
  0.5607, and HercUNet v0 scores at most 0.4394 (0.4481 with the dilation its
  write-up proposes) even with every setting chosen in its favour on this
  benchmark. On the challenge's own Kaggle metric the order is the same: 0.6521,
  0.6012 and at most 0.5447.
- **On the one human-labelled holdout the models are level.** On PHerc0500P2, which
  released M7 trained on, the Kaggle composite is 0.4917 for F0, 0.4938 for M7 as
  published, 0.5015 for M7 with TTA and at most 0.4795 for HercUNet. The benchmark
  advantage does not carry over to these labels, and we say so.
- **The published M7 does make blobs, and they are concentrated where the scroll is
  compressed.** On 256 randomly drawn interior cubes from eight scrolls, the share
  of M7's predicted voxels buried inside a solid mass rises from 2.0% in the most
  open third of cubes to 6.0% in the most compressed third. Our own run of the
  released M7 weights at the published setting reproduces this (2.1% to 6.5%), so it
  is the model, not the publication pipeline. F0 stays flat (2.9%, 1.5%, 2.8%).
- **Blobs are local, not everywhere.** A whole cube is rarely a solid slab (1 of 256
  for published M7, none for F0). In the most open third F0 carries slightly more solid
  mass than M7 (2.9% against 2.0%), and one of the eight seeded-random open cubes shows
  F0 fusing two wraps. M7's failure is fused, unstructured masses in tight regions; the
  same seeded-random figures show it without selection.
- **The measurement machinery was verified before it was used.** The harness
  reproduces the sealed F0 and M7 audits to within 5e-8 and the sealed Kaggle
  composite to within 1e-6. HercUNet's own command-line inference, run on the live
  scroll around four benchmark patches, scores the same as or below our harness.

{{STATUS_LINE}}

## What was compared

| model | identity | how it was run |
|---|---|---|
| **F0** (ours) | `releases/c3-f0-200k-20260907/checkpoint_f0_00200000.pt`, sha `54db9a59…a175` | shipped recipe: eight-way mirror TTA (mean softmax), T = 0.30, bf16 |
| **M7, as published** | released `surface_m7_nnunet` in the student container (`…/unified_ladder_20260902/baselines/m7_raw/checkpoint_milestone_00000000.pt`, sha `35462a25…6e4677`); for the blob audit, the published `…-surface-m7-L0-th0.2.zarr` masks themselves | the organisers' own setting from their `metadata.json`: `disable_tta: true`, threshold 0.2 |
| **M7 + our TTA** | same weights | eight-way mirror TTA at its best threshold on this benchmark (0.20) |
| **HercUNet v0** | `jimmylomro/hercunet-v0` `fold_0/checkpoint_best.pth`, sha `7cb36f9a5abd…2c946`; code at `jimmylomro/hercUNet` commit `100b828` (2026-09-17) | its own inference functions; see below |
| soup (exploratory) | uniform weight average of F0 200k and the seed-1204 replicate C3r 180k, sha `164ca9d4…19311b5` | eight-way mirror TTA |

## Data

- **Frozen v14p2 benchmark, 689 rows** (`voxel_balanced_native_v14p2_registered_antialias_softtarget_corpus`,
  manifest sha `a995787032eb…e12b714`): 192³ patches from PHerc0814 (256) and PHerc1451 (433), two
  scrolls no model here trained on. Labels are the official 2.4 µm fine teacher,
  registered onto the ~9 µm scans and hardened at q ≥ 0.5; label 2 is ignored.
  442 of the rows are byte-for-byte crops of the public open-data volumes at the
  recorded origin (checked against S3); the other 247 are 4×-downsampled fine scans.
  The manifest dates from 2026-08-24, two weeks before HercUNet's training log begins.
  Neither PHerc0814 nor PHerc1451 appears in HercUNet's training corpus (4,031 pseudo-label
  windows plus 100 M7 rehearsal windows from 22 scrolls).
- **Human-labelled PHerc0500P2, 320 rows** (from `voxel_all_sources_patches_5856`): the one
  held-out set with human labels. Released M7 was trained on these labels and HercUNet
  starts from M7, so scores here are absolute only, not a fair gain claim.
- **Random-cube blob audit, 256 cubes**: see the audit section.

## How it was kept fair

Rules fixed after a 41-row pilot and before any full HercUNet number existed
(`runtime/analyze.py` docstring):

1. F0 is scored only at its preregistered operating point (TTA, T = 0.30).
2. M7 is scored at the organisers' own published setting, and separately with our
   TTA at its best threshold on this benchmark.
3. HercUNet publishes no threshold. It gets the best of: normalisation as shipped
   or training-matched, one to four passes, TTA off or eight-way mirror, every
   threshold 0.10–0.90 and, for overlap metrics, a 0–3 voxel dilation, **each chosen
   on this benchmark**. That selection flatters it and so makes any claim that F0 is
   ahead conservative.
4. HercUNet predicts a thin **medial** surface rather than the sheet face. We therefore
   also report offset-tolerant surface F1 at 2 and at 4 voxels and dilated Dice, the
   remedies its own write-up proposes.
5. Every model sees the same 192³ patch and is scored against the same target by the
   same code. Uncertainty is a paired cluster bootstrap over support-anchor chunks
   within scroll (B = 2,000, seed 20260908, the family preregistration's design).

### HercUNet details that matter

- **Protocol.** A 192³ benchmark patch is exactly one HercUNet window, so we run its
  Jacobi recurrence in-window: pass 0 reads `prev = 0`, pass *p* reads the uint8-quantised
  output of pass *p−1*, as its buffer does. This is also its DAgger training recurrence.
- **Fidelity check.** On four registered-real patches we ran HercUNet's own CLI
  (`hercunet infer single-instance`, 4 passes, overlap 0.25, half-stride shifts, Gaussian
  blend) on the live scroll with a 96-voxel margin, pinned to the exact coarse volume.
  The CT it read equals the benchmark image byte for byte. Pooled over the four patches its
  best Dice was 0.364 through its own CLI against 0.398 in our harness, so the harness,
  if anything, flatters it.
- **Normalisation.** Its training CT was stretched per window to the 1st–99th percentile
  and normalised with M7's statistics; its shipped inference normalises raw CT with the
  stretched-data fingerprint (mean 138.8, std 64.9). We ran both. They score the same to
  within 0.001 Dice, most likely because instance normalisation absorbs the difference.
- **Passes and TTA.** More passes lower its Dice (0.4394 after one pass with TTA, 0.4170
  after four), but with eight-way mirror TTA the four-pass output has the best topology, so
  its best Kaggle composite (0.5447) is four passes with TTA at T 0.30. The blob audit and
  figures use that setting, by the fixed rule of showing its best challenge-metric
  configuration.

## Results: frozen benchmark (689 rows)

{{BENCHMARK_TABLE}}

Paired deltas (cluster bootstrap, 95%):

{{DELTA_TABLE}}

{{REAL_ONLY_LINE}}

## Results: human-labelled PHerc0500P2 (320 rows, absolute only)

{{P0500_TABLE}}

## Blob audit on randomly drawn cubes

**Rule, fixed in `runtime/sample_cubes.py` before any cube was read.** Eight scrolls with an
official published M7 L0 prediction and not in F0's training data: PHerc1447, PHerc0814,
PHerc0125, PHerc0191, PHerc0343, PHerc0800, PHerc1218, PHerc0826. Unit: a 128³ cube on
the CT's own chunk grid whose 192³ context lies inside the scroll (all nonzero at pyramid
level 5). 32 cubes per scroll drawn uniformly without replacement, numpy
`default_rng(SeedSequence(20260924).spawn(8))`. 67,717 to 250,951 cubes were eligible
per scroll. Seven of the eight scrolls (all but PHerc0814) contribute sparse windows to
HercUNet's training corpus, which can only help it here. For each cube we cached the CT context and the organisers' published M7 mask
unchanged; every model then predicted the central 128³ with a 32-voxel halo, F0's shipped
geometry.

**Measures** (`runtime/blob_metrics.py`): the share of predicted voxels at six-connected
depth ≥ 5 from background, i.e. inside a solid mass at least ~9 voxels thick, which a
single sheet at ~9 µm does not produce; the canonical ScrollFiesta verdicts (thick mass,
solid slab; contract of 2026-06-03); and the share of predicted voxels on CT darker than
the cube's Otsu threshold. **Compression strata** are terciles, within each scroll, of the
share of CT darker than that scroll's Otsu threshold, fixed before any model comparison.

{{BLOB_TABLE}}

What this does and does not show:

- The published M7's solid-mass share triples from the open to the compressed third, and
  our rerun of the released weights at the same setting reproduces it. F0 does not rise
  with compression.
- M7 with eight-way TTA at T = 0.30 has far fewer deep voxels, but in the compressed cubes
  it produces fragments, not sheets: it fills less of the cube than F0 (17.5% against
  23.0%) yet breaks into more pieces (33.9 per cube against 24.0, counting 26-connected
  pieces of at least 50 voxels; `runtime/fragments.py`, added after the audit and not
  preregistered). On the benchmark it is still 0.12 Dice below F0. Fewer deep voxels
  alone can be bought by predicting less.
- HercUNet at its best challenge-metric setting (four passes with TTA) is the thickest
  output measured. Its single pass without TTA at T = 0.50 is thin (0.9%) and has a
  higher best Dice (0.4327 against 0.4170), but a lower Kaggle composite (0.5240
  against 0.5447) because its topology score is lower.
- Whole-cube failures are rare for everyone. The canonical thick-mass predicate fires on
  2.3% of cubes for published M7, 1.2% for F0 and 4.3% for HercUNet at its best setting.

**Figures.** `evidence/figures/showcase_compressed.jpg` and `showcase_open.jpg` show one cube
per scroll and stratum, chosen by a seeded random index (seed 20260925) that never looks at
a prediction. `evidence/figures/contact_*.jpg` show every one of the 256 cubes for CT,
published M7, F0 and HercUNet, with no selection at all. An earlier "first four drawn per
scroll" rule was dropped: the drawn origins are stored sorted by position, so it favoured
the scroll ends.

## Exploratory findings (reported, not promoted)

- **The cross-seed soup is the strongest network measured**: 0.7248 at T = 0.30 with TTA,
  +0.0118 [+0.0079, +0.0159] over the shipped recipe on the same rows, positive on both
  scrolls. Soups were declared exploratory in the family preregistration, and its threshold
  and TTA reading come from this benchmark, so it needs its own preregistered confirmation
  before it can ship. {{SOUP_EXTRA}}
- Averaging logits instead of probabilities over the eight mirrors (HercUNet's TTA
  arithmetic) does not help F0: 0.7124 against 0.7130.

## Ideas worth taking from HercUNet

Ranked by expected value for our failure modes; none has been trained yet.

1. **A self-conditioned refiner** (a `prev` input channel initialised to zero, octant
   composition and fragmentation of `prev`, online DAgger on the model's own output).
   It removes by construction the synthetic-versus-real shortcut that sank our R1/R2
   repair networks, and it targets the thin-continuation and cross-window failures. On
   this benchmark HercUNet's later passes lower its Dice and lift its composite only with
   TTA (0.5349 to 0.5447 at T = 0.30), so a trial must show that a later pass beats pass
   one before anything else.
2. **A separation penalty only inside real gaps** (closing of the surface minus the
   surface, restricted to label background), unlike our retired shell, which also hit
   sheet rims and acted as a threshold dial.
3. **Sheet-instance supervision** (an affinity head trained with constrained MALIS), using
   instance labels from our 2.4 µm teacher. It is the only term aimed directly at merge
   paths between wraps.
4. **Six CT orientation channels** (a unit-trace structure tensor recomputed from the
   augmented CT). Cheap, and the model starts identical to M7.
5. **A soft-skeleton crest on every row**, which would finally make the untested medial tail
   floor testable on the mixed corpus.

## Limits

- The frozen benchmark's labels come from the fine teacher, not from people; the only
  human-labelled holdout is flat across models and was seen by M7.
- The benchmark rows sit in fairly clean regions, so they show accuracy more than blobs;
  the blob evidence comes from the random-cube audit, which has no ground truth.
- The HercUNet fidelity check covers four patches, not all 442 registered rows.
- "Solid mass" is a geometric proxy. It cannot see two wraps fused where the CT itself shows
  no gap.

## Reproduce

Environment: the engine venv (`D:\work\vesuvius-c\crossres_pred\.venv`, torch 2.11 cu128) with
`PYTHONPATH=src` for F0, M7 and the soup; a separate venv with HercUNet installed
(`nnunetv2==2.8.1`, `zarr<3`) for HercUNet.

```powershell
# frozen benchmark, one family at a time (writes exact per-row counts and uint8 probabilities)
python runtime/patch_benchmark.py --family engine --label f0 --checkpoint releases/c3-f0-200k-20260907/checkpoint_f0_00200000.pt --manifest <v14p2 patches.jsonl> --work <work> --store f0_tta
python runtime/patch_benchmark.py --family engine --label m7 --checkpoint <released M7 weights> --manifest <v14p2 patches.jsonl> --work <work> --store m7_flat m7_tta
python runtime/patch_benchmark.py --family hercunet --label hshipped --model-dir <hercunet-v0> --norm shipped --passes 4 --manifest <v14p2 patches.jsonl> --work <work> --store hshipped_p0 hshipped_p3
python runtime/patch_benchmark.py --family hercunet --label hshtta --model-dir <hercunet-v0> --norm shipped --passes 4 --tta mirror --manifest <v14p2 patches.jsonl> --work <work> --store hshtta_p3
# the human-labelled holdout: the same commands with --scroll PHerc0500P2 and a separate work root
# the challenge metric (same metric.exe, ground truth and sharded runner as the sealed evidence)
python runtime/kaggle_score.py --probs <work>/f0/probs/f0_tta --threshold 0.30 --out <kaggle>/f0_tta_t030
python runtime/kaggle_score.py --probs <work>/hshtta/probs/hshtta_p3 --threshold 0.30 --out <kaggle>/hshtta_p3_t030
python runtime/kaggle_score.py --probs <work_p0500>/f0/probs/f0_tta --threshold 0.30 --set p0500p2_val --out <kaggle>/p0500_f0_tta_t030
# HercUNet through its own CLI on the live scroll
python runtime/insitu_check.py --manifest <v14p2 patches.jsonl> --model-dir <hercunet-v0> --work <insitu> --patch-ids <ids>
# random-cube audit
python runtime/sample_cubes.py --out <audit>
python runtime/cube_infer.py --family engine --label f0 --checkpoint <f0> --cubes <audit>/cubes --out <audit>/preds --keep f0_flat f0_tta
python runtime/cube_infer.py --family engine --label m7 --checkpoint <released M7 weights> --cubes <audit>/cubes --out <audit>/preds --keep m7_flat m7_tta
python runtime/cube_infer.py --family hercunet --label hshipped --model-dir <hercunet-v0> --norm shipped --passes 4 --cubes <audit>/cubes --out <audit>/preds --keep hshipped_p0 hshipped_p3
python runtime/cube_infer.py --family hercunet --label hshtta --model-dir <hercunet-v0> --norm shipped --passes 4 --tta mirror --cubes <audit>/cubes --out <audit>/preds --keep hshtta_p0 hshtta_p3
python runtime/blob_metrics.py --audit <audit> --preds <audit>/preds --source m7_published=published --source m7_rerun_flat=m7_flat:0.20 --source m7_rerun_tta=m7_tta:0.30 --source f0=f0_tta:0.30 --source hercunet=hshtta_p3:0.30 --source hercunet_4pass=hshipped_p3:0.30 --source hercunet_1pass=hshipped_p0:0.50 --out <audit>/final
python runtime/fragments.py --audit <audit> --preds <audit>/preds --source m7_published=published --source m7_rerun_flat=m7_flat:0.20 --source m7_rerun_tta=m7_tta:0.30 --source f0=f0_tta:0.30 --source hercunet=hshtta_p3:0.30 --out <audit>/final/fragments.json
python runtime/analyze.py --work <work> --kaggle <kaggle> --insitu <insitu> --out evidence/analysis.json
# numbers, figures, this README and the show-and-tell page
python runtime/collect_numbers.py --work <work root> --out evidence/numbers.json
python runtime/render_figures.py --analysis evidence/analysis.json --frontier <frontier.json> --audit <audit> --preds <audit>/preds --blob-summary <audit>/final/summary.json --blob-metrics <audit>/final/cube_metrics.jsonl --herc-config hshtta_p3 --herc-threshold 0.30 --out <figures>
python runtime/fill_readme.py --numbers evidence/numbers.json --template runtime/README.template.md --out README.md
python runtime/build_page.py --numbers evidence/numbers.json --analysis evidence/analysis.json --frontier <frontier.json> --out <page>/index.html
```

## Files

`runtime/` holds every script that produced a number here. `evidence/` holds the analysis
tables, the per-row counts of every run (gzipped JSON lines), the Kaggle receipts, the
fidelity rows, the blob-audit sample plan, per-cube metrics and summaries, and the figures.
[files.json](files.json) lists the SHA-256 of each.
