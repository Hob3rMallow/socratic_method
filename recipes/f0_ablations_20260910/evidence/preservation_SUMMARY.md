# F0 rescue ladder summary (2026-09-10T19:57:48.381053+00:00)

Config `f0_rescue_20260909.json` sha `b9be36cbfbe1471e88928ce9e1a46a01f898918078deea7ecbca224615131e23`; prereg `crossres_pred/F0_RESCUE_PREREG_20260909.md`.

## Stages

| stage | status | at (UTC) |
|---|---|---|
| comparators:kill | complete | 2026-09-10T03:45:32.779595+00:00 |
| conditional:P_sep | triggered | 2026-09-10T13:39:19.944116+00:00 |
| conditional:tail_lane | skipped:not_triggered | 2026-09-10T19:57:48.151084+00:00 |
| evaluation:P_hard:kill | complete | 2026-09-10T08:44:50.945609+00:00 |
| evaluation:P_hard_kill | running | 2026-09-10T08:17:45.511686+00:00 |
| evaluation:P_sep:kill | complete | 2026-09-10T19:57:47.996122+00:00 |
| evaluation:P_sep_kill | running | 2026-09-10T19:22:02.713019+00:00 |
| evaluation:P_soft:kill | complete | 2026-09-10T13:39:19.875635+00:00 |
| evaluation:P_soft_kill | running | 2026-09-10T13:12:12.169641+00:00 |
| evaluation:comparators_kill | running | 2026-09-10T03:28:32.399382+00:00 |
| live_identity:P_hard | verified | 2026-09-10T03:46:10.558032+00:00 |
| live_identity:P_sep | verified | 2026-09-10T13:39:57.816990+00:00 |
| live_identity:P_soft | verified | 2026-09-10T08:45:28.837007+00:00 |
| operator_review_pause | complete | 2026-09-10T19:57:48.289648+00:00 |
| preflight | complete | 2026-09-10T03:28:30.701286+00:00 |
| training:P_hard:kill | complete | 2026-09-10T08:17:38.087489+00:00 |
| training:P_sep:kill | complete | 2026-09-10T19:21:51.571473+00:00 |
| training:P_soft:kill | complete | 2026-09-10T13:12:04.831549+00:00 |

## P_hard (kill stage)

Verdict: **KILLED** - K1 gates/guard fail at 40k and 50k; K3 pooled delta below the kill limit vs both comparators

| samples | macro .35 | level CI .35 | macro .30 | eligible | gate .35 | fg ratio | frontier |
|---|---|---|---|---|---|---|---|
| 10,000 | 0.5836 | [0.5705, 0.5971] | 0.5681 | False | False | 1.806 | unresolved |
| 20,000 | 0.6174 | [0.6053, 0.6300] | 0.6007 | False | False | 1.584 | unresolved |
| 30,000 | 0.6244 | [0.6112, 0.6382] | 0.6125 | False | False | 1.425 | unresolved |
| 40,000 | 0.6184 | [0.6066, 0.6309] | 0.6045 | False | False | 1.577 | unresolved |
| 50,000 | 0.6101 | [0.5982, 0.6233] | 0.5912 | False | False | 1.542 | unresolved |

- pooled 10k-50k delta vs F0 at T 0.35: -0.0480 [-0.0567, -0.0383]
- pooled 10k-50k delta vs F0 at T 0.30: -0.0620 [-0.0706, -0.0524]
- pooled 10k-50k delta vs C3r at T 0.35: -0.0379 [-0.0476, -0.0274]
- pooled 10k-50k delta vs C3r at T 0.30: -0.0522 [-0.0618, -0.0415]
- weighted share of `loss_m7_preservation_loss` in the committed loss: 0.1839 (floor 0.01)
- rim thickening trigger: True

## P_soft (kill stage)

Verdict: **KILLED** - K1 gates/guard fail at 40k and 50k; K3 pooled delta below the kill limit vs both comparators

| samples | macro .35 | level CI .35 | macro .30 | eligible | gate .35 | fg ratio | frontier |
|---|---|---|---|---|---|---|---|
| 10,000 | 0.5071 | [0.4950, 0.5198] | 0.4935 | False | False | 2.258 | unresolved |
| 20,000 | 0.5480 | [0.5352, 0.5609] | 0.5318 | False | False | 2.037 | unresolved |
| 30,000 | 0.5661 | [0.5526, 0.5806] | 0.5540 | False | False | 1.867 | unresolved |
| 40,000 | 0.5503 | [0.5367, 0.5647] | 0.5374 | False | False | 1.962 | unresolved |
| 50,000 | 0.5638 | [0.5500, 0.5783] | 0.5515 | False | False | 1.820 | unresolved |

- pooled 10k-50k delta vs F0 at T 0.35: -0.1117 [-0.1214, -0.1014]
- pooled 10k-50k delta vs F0 at T 0.30: -0.1238 [-0.1331, -0.1139]
- pooled 10k-50k delta vs C3r at T 0.35: -0.1016 [-0.1119, -0.0908]
- pooled 10k-50k delta vs C3r at T 0.30: -0.1140 [-0.1238, -0.1032]
- weighted share of `loss_m7_preservation_loss` in the committed loss: 0.1287 (floor 0.01)
- rim thickening trigger: True

## P_sep (kill stage)

Verdict: **KILLED** - K1 gates/guard fail at 40k and 50k

| samples | macro .35 | level CI .35 | macro .30 | eligible | gate .35 | fg ratio | frontier |
|---|---|---|---|---|---|---|---|
| 10,000 | 0.6231 | [0.6108, 0.6361] | 0.6104 | False | False | 1.460 | unresolved |
| 20,000 | 0.6384 | [0.6267, 0.6505] | 0.6270 | False | False | 1.183 | FAIL |
| 30,000 | 0.6403 | [0.6276, 0.6531] | 0.6301 | False | False | 1.237 | unresolved |
| 40,000 | 0.6441 | [0.6325, 0.6566] | 0.6326 | False | False | 1.324 | FAIL |
| 50,000 | 0.6338 | [0.6219, 0.6466] | 0.6182 | False | False | 1.337 | unresolved |

- pooled 10k-50k delta vs F0 at T 0.35: -0.0228 [-0.0311, -0.0140]
- pooled 10k-50k delta vs F0 at T 0.30: -0.0338 [-0.0422, -0.0249]
- pooled 10k-50k delta vs C3r at T 0.35: -0.0127 [-0.0219, -0.0028]
- pooled 10k-50k delta vs C3r at T 0.30: -0.0240 [-0.0333, -0.0138]
- weighted share of `loss_m7_preservation_loss` in the committed loss: 0.2185 (floor 0.01)
- rim thickening trigger: True

## Conditionals

- P_sep: {"arm_id": "P_sep", "at_utc": "2026-09-10T13:39:19.913599+00:00", "parent": "P_hard", "reason": "thickening parent with the better pooled delta vs F0", "thickening_parents": ["P_hard", "P_soft"]}
- tail_lane: {"at_utc": "2026-09-10T19:57:48.101823+00:00", "coverage_delta_min": 0.02, "fg_ratio_growth_max": 1.1, "reference": {"coverage": 0.7702921386593987, "foreground_ratio": 0.7085286284714847}, "triggered": false}

## Verdicts

```json
{
 "kill_stage": {
  "arms": {
   "P_hard": {
    "inert": false,
    "k1_gates": true,
    "k2_frontier": false,
    "k3_pooled_delta": true,
    "killed": true,
    "pooled_vs_c3r_035": {
     "half_width": 0.010095467776703011,
     "high": -0.02742545469251785,
     "low": -0.04761639024592387,
     "mean": -0.038011775679280976,
     "point": -0.03793874978333385,
     "probability_delta_le_zero": 1.0,
     "std": 0.005029443865059155
    },
    "pooled_vs_f0_035": {
     "half_width": 0.009219337317841626,
     "high": -0.038263154113506885,
     "low": -0.056701828749190136,
     "mean": -0.04803942370277107,
     "point": -0.047966831768326834,
     "probability_delta_le_zero": 1.0,
     "std": 0.004687229726619767
    },
    "reason": "K1 gates/guard fail at 40k and 50k; K3 pooled delta below the kill limit vs both comparators",
    "survivor": false
   },
   "P_sep": {
    "inert": false,
    "k1_gates": true,
    "k2_frontier": false,
    "k3_pooled_delta": false,
    "killed": true,
    "pooled_vs_c3r_035": {
     "half_width": 0.009573582496424793,
     "high": -0.0027811161476217574,
     "low": -0.021928281140471344,
     "mean": -0.012874374930948231,
     "point": -0.012748903906771924,
     "probability_delta_le_zero": 0.9975,
     "std": 0.004800128202890739
    },
    "pooled_vs_f0_035": {
     "half_width": 0.008583516967941435,
     "high": -0.013957496119733768,
     "low": -0.031124530055616637,
     "mean": -0.022902022954438318,
     "point": -0.02277698589176491,
     "probability_delta_le_zero": 1.0,
     "std": 0.004425835317579304
    },
    "reason": "K1 gates/guard fail at 40k and 50k",
    "survivor": false
   },
   "P_soft": {
    "inert": false,
    "k1_gates": true,
    "k2_frontier": false,
    "k3_pooled_delta": true,
    "killed": true,
    "pooled_vs_c3r_035": {
     "half_width": 0.010570041057791559,
     "high": -0.09076360427952233,
     "low": -0.11190368639510545,
     "mean": -0.10168384300565264,
     "point": -0.10164660331566577,
     "probability_delta_le_zero": 1.0,
     "std": 0.005385561179722182
    },
    "pooled_vs_f0_035": {
     "half_width": 0.0100064424353297,
     "high": -0.10139620793973836,
     "low": -0.12140909281039776,
     "mean": -0.11171149102914274,
     "point": -0.11167468530065876,
     "probability_delta_le_zero": 1.0,
     "std": 0.005115277935494435
    },
    "reason": "K1 gates/guard fail at 40k and 50k; K3 pooled delta below the kill limit vs both comparators",
    "survivor": false
   }
  },
  "at_utc": "2026-09-10T19:57:48.197447+00:00",
  "selection": {
   "notes": {
    "P_sep": "killed at the kill stage"
   },
   "resume": []
  }
 }
}
```

## Training attempts

- P_hard:kill: 1
- P_sep:kill: 1
- P_soft:kill: 1

## Wall time

- conditional:P_sep: reached 2026-09-10T13:39:19.944116+00:00
- conditional:tail_lane: reached 2026-09-10T19:57:48.151084+00:00
- evaluation:P_hard:kill: reached 2026-09-10T08:44:50.945609+00:00
- evaluation:P_hard_kill: reached 2026-09-10T08:17:45.511686+00:00
- evaluation:P_sep:kill: reached 2026-09-10T19:57:47.996122+00:00
- evaluation:P_sep_kill: reached 2026-09-10T19:22:02.713019+00:00
- evaluation:P_soft:kill: reached 2026-09-10T13:39:19.875635+00:00
- evaluation:P_soft_kill: reached 2026-09-10T13:12:12.169641+00:00
- evaluation:comparators_kill: reached 2026-09-10T03:28:32.399382+00:00
- live_identity:P_hard: reached 2026-09-10T03:46:10.558032+00:00
- live_identity:P_sep: reached 2026-09-10T13:39:57.816990+00:00
- live_identity:P_soft: reached 2026-09-10T08:45:28.837007+00:00
- operator_review_pause: reached 2026-09-10T19:57:48.289648+00:00
- preflight: reached 2026-09-10T03:28:30.701286+00:00
- training:P_hard:kill: reached 2026-09-10T08:17:38.087489+00:00
- training:P_sep:kill: reached 2026-09-10T19:21:51.571473+00:00
- training:P_soft:kill: reached 2026-09-10T13:12:04.831549+00:00
