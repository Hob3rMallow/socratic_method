# From found paths to coherent 3D sheet patches

The frozen C3-F0/200k model remains a substantial advance over M7: 0.680787
versus 0.554140 matched macro Dice, a 12.7-point improvement. This follow-on
work moves the engineering effort from the model to usable continuity. It does
not retrain, lower T=.35, erase source foreground, or change production defaults.

A separate opt-in CPU stage now extends found v2 curves into connected triangle
meshes and additional mask geometry across neighboring Z planes. The local
height-field objective combines probability, CT-profile identity and motion,
hard smoothness bounds, and jointly ordered neighboring sheets when visible.

| Measure | Development | New target-disjoint assessment |
| --- | ---: | ---: |
| Explicit accepted mesh patches | 107 | 82 |
| Added voxels beyond v2 continuity | 75,661 | 46,780 |
| Central-cube additional voxels | 1,512 | 2,893 |
| Median / maximum Z planes | 11 / 33 | 16.5 / 33 |
| Patches modeling neighboring sheets | 36 | 76 |
| Foreground erased | 0 | 0 |

Both blocks contain 27 contiguous 128-cubed targets at 8.640 um. The new block
was predeclared and the candidate frozen before inference. It is target-disjoint
from all earlier blocks but remains PHerc1447, not independent-scroll or
anatomical-ground-truth validation. No new official Dice is claimed.

All output additions reconstruct exactly from the saved meshes. Independent
checks cover face connectivity, winding, boundary structure, area, attachments
and interacting surfaces. The packaged loop reproduces all 27 development
TIFFs and 107 meshes exactly, including refusal counts. All 151 combined tests
pass: 75 new sheet-patch tests plus 76 existing repair/continuity tests.

## Crowded folds: what was learned

Joint ordering prevents a known brighter-neighbor branch switch in a repeated-
texture synthetic case. In the real ambiguous component 301, modeling neighbors
changes the proposed geometry but does not establish anatomical identity. The
final construction refuses all three tried seeds there on multi-component
contact. Alternative-cost margins also fail to flag that case as uncertain.
A smooth, high-margin optimum is not an anatomical label.

The two-component contact count still rejects useful-looking fragmented sheets.
The next priority is explicit 3D sheet ownership and sparse manual same-sheet /
different-sheet anchors at ambiguous junctions, followed by overlapping valid
charts for genuine fold-backs. Those are proposed next steps, not completed work.
The existing additive stage cannot undo an erroneous old join. Whole-scroll
tiling, seam qualification and unreviewed deployment are outside this result.

## Use and evidence

See [the checked command](../../docs/sheet_patches.md). Default is preflight;
`--run` opts in to one local CPU process. A new output is required. Input hashes,
original inference profile, grid, CT units, masks and seed ledgers are checked.
Interrupted partial work is preserved rather than silently restarted.

Full HTML with the established CT-gray / baseline-magenta / additions-green
palette, all 189 patch galleries and exact mesh links:
`D:/work/vesuvius-c/output/crossres_data/f0_sheet_patches_20260907/index.html`.
The visual-review record distinguishes inspected images from the complete
numerical audit. Large CT/mask/mesh/image arrays remain local; copied text/source
evidence and external locations are hashed by `files.json`.

`selected_policy.json` retains its pre-assessment pending status verbatim;
`decision.json` records the completed assessment. Diagnostic foldovers, refused
cases, the corrected metadata-serialization failure and the initial diagnostic
Z-step statistic bug are retained. No prior frozen evidence was overwritten.
No Git commit, public upload or automatic deployment was performed.
