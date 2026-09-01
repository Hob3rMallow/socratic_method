# Reserved paper figures

The LaTeX paper detects these files automatically (PDF preferred, PNG accepted),
scales them into their reserved panels, and otherwise leaves a labeled box:

1. `teaser` - coarse CT, released M7, our raw student, and line-fitted geometry
   in one representative registered crop.
2. `figure_1_method_overview` - fine teacher, projection, student losses,
   trust region, raw inference, and postprocess boundary.
3. `figure_crossres_supervision` - fine-teacher mask, LSMAT-style medial
   centers with slice-wise radii, soft occupancy and binary crest projections,
   de-blob separation shell, and a dynamic widest-path event with pins, free
   anchors, corridor, and selected bottleneck path.
4. `figure_2_registered_examples` - registered held-out examples comparing
   CT, M7, fine-teacher target, selected v31 milestone, and error overlay.
5. `figure_3_line_fitter_examples` - accepted short joins, rejected
   cross-wrap proposals, and before/after masks with zooms.
6. `figure_4_failure_cases` - remaining faint-sheet misses, blob/merger
   risks, and calibration-sensitive examples.

Prefer wide PNGs at 300 dpi. Keep text labels large enough to remain legible in
a 6.7-inch-wide paper column. Do not crop away registration coordinates or
scale bars.
