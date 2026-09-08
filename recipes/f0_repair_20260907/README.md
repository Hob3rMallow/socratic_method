# F0 contiguous repair — 7 September 2026

**A measured improvement to local gap recovery, with the model left frozen.**
The checked `f0-contiguous-gap-repair-v1` profile uses reach 21, radial limit 3,
hard support cutoff 0, tracking on, paint radius 1 and cutting off. It operates
on the existing frozen F0/200k predictions at T=.35.

| Block | Old → new per-plane joins | Old → new tracks | Old → new added voxels |
| --- | ---: | ---: | ---: |
| Development | 96 → 286 | 23 → 195 | 1,364 → 3,970 |
| Reserved control | 59 → 97 | 12 → 55 | 961 → 1,174 |

Both comparisons use the same dense 27-cube inputs. Four presets were measured
on development; the control was reserved for old-versus-selected verification.
Zero original foreground was erased. The additions are still small (0.025%
of pooled original foreground); this is not broad missing-sheet reconstruction.
The stricter radial guard declines some repairs made by the old preset.

The machine-readable [profile](profile.json), [results](results.json),
[decision](decision.json), [visual review](visual_review.json),
[request](request.json), [pre-control selection](selected_policy.json), and
[verification](verification.json) preserve the evidence and limitations.
The selection file intentionally retains its original pre-control status;
`decision.json` records the subsequent control result.

Local visual report (images remain outside Git):

`D:/work/vesuvius-c/output/crossres_data/f0_repair_20260907/index.html`

It links all 250 accepted-track cards and 17 risk-first orthogonal views.
Visual assessment is qualitative, not new ground-truth join labels. Independent
contact diagnostics do not establish global wrap identity. The report and
receipt explicitly retain these boundaries; no production promotion is claimed.

Use the [checked repair command](../../docs/repair.md). This record supplements
the [frozen model record](../final_c3_250k_20260907/README.md); it does not rewrite
that training snapshot, old reports, source pins or historical v31 wrappers.

The first wrapper integration attempt finished native processing but exposed
an unescaped Windows-path issue in native JSON. Forward-slash native arguments
fixed it. The failed partial output was preserved; the successful integration
produced 27 TIFFs byte-identical to the selected control result. Transactional
and cancellation checks were then hardened without changing the native profile.

This repository update is local and uncommitted. No weights or data were
uploaded, no new training was launched, and no global threshold was changed.
