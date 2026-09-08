# C3-F0 post-run review — 7 September 2026

## Verdict

**Freeze this model and move the next engineering effort to usable sheet continuity.** The C3 recipe produced a substantial advance over released M7: 0.680787 versus 0.554140 macro-scroll Dice on the same frozen v14p2 benchmark, at the same threshold and without TTA. The visual review also supports cleaner separation through several previously broad, merged-looking regions. This is a meaningful result, not merely a marginal leaderboard movement.

The verified matched gain is **+0.126647 Dice / 12.7 percentage points / 22.9% relative**, not an exact +0.15. The approximately 0.53 figure is not the matched frozen macro baseline in these artifacts. No stronger claim is needed to present this result positively.

This review reads the existing run histories, all 25 frozen checkpoint audits, threshold sweeps, six completed native scoring passes, drift measurements, geometry reports, cached filler output and relevant filler implementation. It launches no experiments and changes no model, threshold or postprocessor.

## What is frozen, and why

F0 completed its declared **250,000 sample exposures** naturally. The chosen weights are the **200k milestone**, the highest fixed-T=0.35 macro Dice among eligible checkpoints at or after 150k. All 25 milestones were eligible. The exact 250k endpoint is retained, not renamed or substituted. The global-best milestone is 110k, but changing the late-selection rule after seeing it would be a different decision.

The recipe is fresh released M7, wide15 anti-aliased supervision, CE 1 + Dice 1 + M7-KL 0.5, known-agreement and confident-agreement flags, unknown-corridor radius 2, SGD 1e-3 with momentum 0.99/Nesterov and a full-horizon poly schedule. Other loss weights and the trust radius are zero. This is **not the old 8x-ball/v31 lane**.

The corpus has 16,200 rows (15,432 train, 768 validation), spanning 12 training and two validation scrolls; it mixes 10,072 native soft rows and 6,128 hard human passthrough rows. The run did not use 250k unique rows or the deferred native 249,836-row corpus. Human annotation was already present in the winning corpus; another generic “add human labels” experiment would not describe a new intervention precisely enough.

The source and frozen 409,674,095-byte checkpoint were independently SHA-256 checked. A separate read-only model copy, sealed configuration, all 63 pinned runtime files and selected raw evidence are archived. See [manifest](manifest.json), [inventory](files.json) and [original recipe](provenance/recipe.json).

## Comparison with M7

| Evaluation / operating point | Frozen macro Dice | Native composite | Native topology |
| --- | ---: | ---: | ---: |
| Released M7, no TTA, T=.35 | 0.554140 | not scored at this T | not scored at this T |
| Released M7, no TTA, calibrated T=.20 | 0.560669 | 0.601235 | 0.357756 |
| Released M7, eight-way TTA, calibrated T=.20 | 0.592577 | 0.616214 | 0.377645 |
| **F0 200k, no TTA, operating T=.35** | **0.680787** | **0.630391** | **0.375646** |
| F0 200k, no TTA, diagnostic T=.30 | 0.688065 | 0.640064 | 0.387968 |

The strongest M7 inference baseline still trails F0 by 0.0882 Dice. At T=.35, F0 improves composite over M7+TTA by 0.0142, while topology is approximately 0.0020 lower. It clears the composite readiness advisory (0.6212), narrowly misses the topology advisory (0.3776), and is not a universal win on every metric.

The gain is present on both validation scrolls: F0 Dice is 0.682172 on PHerc0814 and 0.679403 on PHerc1451. Their precision/recall are 0.7289/0.6411 and 0.6800/0.6788. On PHerc0814 especially, missing foreground remains a useful improvement target.

These are separate comparisons: the frozen macro score is not ordinary wide15 training telemetry, and the cyan M7 panels are published M7 masks at T=.20, not this matched no-TTA/T=.35 audit. F0 blob exports use eight-way TTA; quantitative frozen audits use no TTA. Reference-mask agreement is not ground-truth Dice. Per-scroll published-mask gains in the audit additionally use matched rows only; they must not be combined with the re-inferred M7 macro baseline as if they were the same statistic.

Sources: [M7 raw audit](provenance/m7_raw_audit.json), [M7 raw summary](provenance/m7_raw_summary.json), [M7 TTA summary](provenance/m7_tta_summary.json), [F0 audit](provenance/candidate_audit.json), [full evaluation](provenance/evaluation_summary.json).

## Training behaved normally; longer was not monotonically better

Training survived two reboot interruptions through strict same-run resume: three launches total, 25 validated intervals, 25 retained snapshots, and a verified final optimizer commit. The final learning rate is zero and final loss is finite, 0.791600. Previous integrity checks found no checkpoint or run-identity mismatches; the freeze rechecked the selected model and all 63 sealed source pins. Nothing in this evidence requires a corruption explanation for the metric oscillation.

Average training loss falls from 1.0173 at 10k to 0.7916 at 250k. Cross-entropy falls from 0.3574 to 0.2604 and Dice loss from 0.5804 to 0.4329. KL rises early and then stays around 0.15–0.16 rather than exploding. Relative parameter displacement from M7 reaches 0.3401 at 200k and 0.3442 at 250k; these are measurements, not an active trust constraint.

Frozen validation remains non-monotonic despite this training-loss improvement. The sharp 110k→120k drop (0.681556→0.624299) recovers at 130k; the late 200k→210k dip also recovers. These observations establish checkpoint variability, not its precise causal mechanism.

| Milestone | Train loss | End-of-interval LR | Frozen macro @.35 | Historical-reference bridge alerts | Historical-reference line coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10k | 1.0173 | 9.639e-4 | 0.661700 | 5,440 | 69.83% |
| 20k | 0.9748 | 9.277e-4 | 0.644223 | 2,584 | 66.56% |
| 30k | 0.9399 | 8.913e-4 | 0.676853 | 5,011 | 75.82% |
| 40k | 0.9236 | 8.548e-4 | 0.671969 | 2,672 | 76.36% |
| 50k | 0.9113 | 8.181e-4 | 0.638938 | 2,018 | 77.03% |
| 60k | 0.9074 | 7.811e-4 | 0.665072 | 984 | 69.85% |
| 70k | 0.8893 | 7.440e-4 | 0.645228 | 2,358 | 72.90% |
| 80k | 0.8875 | 7.067e-4 | 0.666177 | 1,905 | 75.00% |
| 90k | 0.8777 | 6.692e-4 | 0.662892 | 1,576 | 71.68% |
| 100k | 0.8684 | 6.314e-4 | 0.676606 | 1,346 | 70.76% |
| 110k | 0.8659 | 5.934e-4 | 0.681556 | 1,557 | 73.71% |
| 120k | 0.8578 | 5.551e-4 | 0.624299 | 671 | 68.24% |
| 130k | 0.8506 | 5.166e-4 | 0.669440 | 1,777 | 74.82% |
| 140k | 0.8476 | 4.776e-4 | 0.634920 | 191 | 67.29% |
| 150k | 0.8440 | 4.384e-4 | 0.665874 | 932 | 68.80% |
| 160k | 0.8327 | 3.987e-4 | 0.660016 | 1,381 | 74.10% |
| 170k | 0.8261 | 3.586e-4 | 0.665468 | 2,337 | 74.07% |
| 180k | 0.8205 | 3.180e-4 | 0.647746 | 799 | 69.27% |
| 190k | 0.8207 | 2.768e-4 | 0.660461 | 884 | 64.98% |
| 200k | 0.8109 | 2.349e-4 | 0.680787 | 2,273 | 70.73% |
| 210k | 0.8046 | 1.922e-4 | 0.641901 | 1,336 | 66.44% |
| 220k | 0.8010 | 1.483e-4 | 0.668358 | 1,863 | 65.97% |
| 230k | 0.7955 | 1.030e-4 | 0.659249 | 915 | 66.35% |
| 240k | 0.7886 | 5.519e-5 | 0.655476 | 1,741 | 66.95% |
| 250k | 0.7916 | 0 | 0.669299 | 1,149 | 66.43% |

The last two columns use the originally preregistered model exemplar on fixed18 views, not human labels or the later M7 display reference. They are retained for historical audit, not restored as report image columns.

The preregistered H1 is met: the best eligible late result exceeds the best early result. But 200k exceeds original C3/30k by only **0.003935**; 110k is 0.000769 higher than 200k, and the exact 250k endpoint is 0.011489 lower than 200k. One seed and correlated checkpoints do not establish that the long schedule intrinsically beats a short one. The historical B0/B0r variation band is not a C3 confidence interval. The durable win is **the recipe versus M7**, with a useful late geometry tradeoff—not proof that still more training will help.

The in-run wide15 validation uses different rows and targets, and different threshold summaries. It is telemetry, not the selection metric. For example, 200k gives 0.516601 at the telemetry default .50 and 0.593369 at its calibrated threshold, while the frozen 689-row macro at operating .35 is 0.680787. Those numbers do not contradict one another.

Sources: [training history](provenance/history.jsonl), [training integrity](provenance/training_integrity_review.json), [run journal](provenance/run_journal.json), [machine-readable trajectory](analysis.json).

## Geometry: less merged foreground, with recoverable and unrecoverable-looking gaps

The fixed-view review shows clearer paths through several broad C3 regions, preservation of prominent long tracks on controls, and fewer fork-like connections. The 200k output generally retains more weak/short geometry than the exact 250k endpoint. Packed folds still contain broad areas, breaks and some isolated fragments. Speckle has not been counted systematically.

Against the original fixed18 model reference, 200k has 2,273 bridge-alert pixels versus C3's 5,011 (54.6% fewer), while reference-line coverage falls from 75.82% to 70.73%. The exact endpoint lowers alerts further to 1,149 but coverage to 66.43%. These are **not percentages of true blob volume removed**. At matched coverage, the C3 interpolation predicts about 3,959 alerts versus 2,273 observed, so the preregistered frontier review passes.

The six-cube upper-foreground, interior and thickness checks pass at .35. The lower-foreground advisory does not: the candidate is deliberately sparse relative to that exemplar. Notably, the late 200k checkpoint also passes the coarse .25 diagnostic gate, unlike the earlier C3 concern. That does **not** establish that every visible merge is solved or authorize lowering the operating point.

The existing threshold sweep shows exploitable probability signal: moving .35→.30 raises aggregate recall from 0.6540 to 0.7159, while precision falls from 0.7107 to 0.6638 and macro Dice rises from 0.6808 to 0.6881. This supports investigating selective recovery of low-confidence continuations. It does not prove that a globally lower threshold or any particular path is topologically safe.

All selected-model native scoring passes are complete:

| Dataset | T | Composite | Topology | Surface Dice |
| --- | ---: | ---: | ---: | ---: |
| v14p2, 689 rows | .30 | .640064 | .387968 | .872889 |
| v14p2, 689 rows | .35 | .630391 | .375646 | .855518 |
| v14p2, 689 rows | .45 | .600807 | .341761 | .803633 |
| PHerc0500P2, 320 rows | .30 | .489025 | .125681 | .716894 |
| PHerc0500P2, 320 rows | .35 | .481767 | .120846 | .706840 |
| PHerc0500P2, 320 rows | .45 | .464464 | .105939 | .678828 |

PHerc0500P2 is absolute-only evidence: released M7 saw its labels, so this is not an unseen-M7 improvement claim. Native official scores for the exact 250k endpoint were not preregistered or run. No filled-mask official scores are claimed.

Sources: [raw geometry review](provenance/raw_geometry_review.json), [visual/filler review](provenance/operator_visual_review.json), [candidate frontier](provenance/candidate_frontier.json), [six-cube gates](provenance/candidate_six_cube_gates.json).

## The filler: the concern is real, and the bottlenecks are now specific

| Cached model @.35 | Per-plane proposed joins | Kept | Support-dropped | Added voxels | Erased |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical C3 | 68 | 18 | 50 | 318 | 0 |
| **F0 200k** | **31** | **5** | **26** | **81** | **0** |
| F0 250k | 50 | 11 | 39 | 183 | 0 |

F0 200k gained only **0.0030%** of its foreground. All five kept per-plane joins belong to **one connection track**, not five independent 3D gap repairs. None of the three named blob planes changed; fixed18 line coverage did not improve for either late model. The present filler has not demonstrated the broad recovery the geometry pipeline needs.

The complete 200k funnel, from existing logs:

| Stage | Count | Interpretation |
| --- | ---: | --- |
| Traced per-plane endpoints | 9,328 | Includes clipped cube/absent-region boundaries |
| Boundary/absent-region exclusions | 7,608 (81.6%) | Deliberate two-pixel boundary exclusion, not evidence of 7,608 missed real gaps |
| Endpoints eligible for matching | 1,720 | Remaining interior endpoints |
| Unordered pairs within 12 pixels | 173 | Final matcher considers only in-range, non-excluded pairs |
| Rejected by pair gates | 142 | Tangent 100; same component 25; radial 10; cross-sheet 4; merger 3 |
| Geometrically accepted per-plane joins | 31 | 24 connection tracks |
| Survived support >=3 | 5 | One connection track; 81 painted voxels |

Zero pairs were rejected by score, adjacent-plane evidence, arc ratio, crossing or endpoint occupancy in this final pass. Therefore “lower min-score” is not the first fix for the measured refusal pattern. Increasing paint radius would thicken accepted strokes, not discover missing paths.

There is a **documentation/preset mismatch**. The report follows the documented conservative example (`reach-max=12, radial-dr=4, min-support=3`). The repository implementation instead defaults to `reach-max=21, radial-dr=3`, with automatic `min-support=0` when the umbilicus/radial guard is armed, or 3 without it. Tracking still operates with support zero: it is the hard final cutoff that is removed. The report was stricter on persistence and reach, but **looser on radial displacement** than those source defaults; “aggressiveness” is not one scalar.

The support distribution is 20 joins with support 1, six with support 2, and five with support 5. By inspection of these existing proposals, a cutoff of 2 would admit 11 per-plane joins across four tracks; removing the hard cutoff would admit all 31 across 24 tracks. These are static filter counts, **not measured new masks, useful sheet connections or safety results**. Eighteen support-rejected proposals are at most six pixels long; for example, a three-pixel proposal at z12224 scored 0.836 but had support 1.

The sparse six-cube grid is a major context limitation: most endpoints are crop/absent-region boundaries. The boundary exclusion must not simply be disabled—outside those cubes there is no evidence. A contiguous block with real neighbors is the correct next test bed.

Finally, source comments document a previous full-turn fusion that motivated tightening the radial limit from 4 to 3. They also cite high-resolution adjudication of longer joins and loss of precision at long reach. This is reason to recover more true joins while preserving or strengthening topology guards, not to widen everything. “2.4 um” in those comments describes the high-resolution evidence; the actual fitter-grid pitch must be checked before interpreting or transferring pixel distances.

Sources: [recorded filler command/results](provenance/filler_manifest.json), [per-plane census](provenance/filler_200k_planes.csv), [join records](provenance/filler_200k_joins.jsonl), [matching and cutoff implementation](provenance/native/pred_fixup.c), [boundary exclusions](provenance/native/slice_trace.c), [gate order](provenance/native/slice_match.c), [default parameters and historical safety notes](provenance/native/pipeline_constants.h). The original hashed [line-fitter document](provenance/line_fitter_documentation.md) is preserved as evidence, not silently rewritten.

## Recommended next steps — proposals, not launched work

1. **Keep F0 200k/T=.35 as the fixed baseline.** Stop changing the training recipe for now. Preserve continuous probability fields as well as binary masks. The operator-approved model win is secure; downstream repair can be evaluated independently.

2. **Make the first repair comparison about context and support.** Use a bounded contiguous block with real neighboring slices/cubes, plus clean controls. Hold topology-sensitive gates fixed, verify umbilicus and units, and compare the present support cutoff against a less restrictive policy. A near/high-confidence exception or the source's radial-armed automatic policy are concrete candidates; neither is yet validated on F0. Report both new useful connections and new cross-sheet/cross-wrap failures, including multi-join chains.

3. **Then improve proposal recall where a genuine gap was never proposed.** Separate boundary exclusions, out-of-reach gaps, noisy tangents and missing endpoint geometry. Do not assume every rejection is a defect: same-component and cross-sheet rejections protect against false loops and merges. Inspect examples before changing tangent estimation or reach.

4. **For larger missing stretches, use probabilities rather than blind binary dilation.** A proposed next mechanism is seeded, orientation/radial-constrained recovery through the cached lower-confidence field, connecting confident .35 sheet segments without opening every low-threshold blob. The .30 sweep provides a reason to investigate; it does not yet validate this mechanism. Evaluate local recovered length/area and usable continuous paths alongside cross-wrap merges and official metrics.

5. **Revisit learning only if needed signal is genuinely absent.** If safe paths cannot be found even in the probability field with proper context, collect focused annotations of those failures and consider a targeted continuity/recall intervention. A generic longer run, restoration of the 8x ball, a broad human-label mix change, or another threshold sweep is not justified as the immediate next step.

The success criterion should match the intended geometry use: substantially more usable sheet continuity without reintroducing destructive merges. Dice remains useful evidence, not the sole optimization target. Use new holdout repair regions instead of repeatedly optimizing the same 18 views. No new training, inference, scoring or filler run was started for this review.
