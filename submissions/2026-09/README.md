# September 2026 submission workspace

This directory is intentionally separate from the reproducibility and model
artifacts. It contains the paper-style account of the Socratic Method and is
marked as a working draft. Authorship, licensing, and artifact URLs remain
provisional.

**Rewritten 2026-09-10.** The paper now reports the shipped F0 model and the
three-term objective that survived ablation. The four retired loss terms are in
Appendix A, the parameter-space trust region in Appendix B, and the additive 2D
curve fitter and its successors in Appendix C. Section files were renumbered to
match; the appendices are `sections/A*.tex`. Numeric macros for the shipped run
live in `src/f0_run.tex` (transcribed from the sealed records), while
`src/generated_run.tex` is retained unchanged because Appendix A still uses its
historical duration table.

**Updated 2026-09-24.** The teaser shows the shipped model beside the published
M7 and HercUNet v0 on seeded-random cubes, and the results add the head-to-head
and the random-cube blob audit (`recipes/f0_benchmark_20260924/`) and the passed
seed replication (`recipes/f0_c3r_replication_20260909/`).

**Updated 2026-09-29.** The paper names the shipped model as the September 2026
progress video does, **Socratic Method September 2026** (F0 in the records),
states its release (Apache-2.0 weights, one-command inference), and adds the
open-cube fusion limitation shown in the video. `relabel_teaser.py` gives the
teaser's student row the same name; its image panels are unchanged.

The other rendered figures still show the earlier v31 student and are labelled
as such in every caption; regenerating them against the shipped weights is
outstanding release work, listed in the paper's limitations.

Canonical repository: <https://github.com/ubc-nvining/socratic_method>

The canonical source uses the same anonymous ACM TOG/SIGGRAPH review format as
the papers under `D:\papers` (`acmtog`, author-year citations, two columns, a
teaser, overview, method, results, limitations, conclusion, and appendices).

Build the PDF from the repository root:

```powershell
submissions/2026-09/build_paper.ps1
```

The build script runs `pdflatex`, `bibtex`, and the two required finishing
passes explicitly, so it does not depend on Perl-backed `latexmk` on Windows.

The canonical output is
`submissions/2026-09/output/paper.pdf`. Rendered page images used for visual QA
belong under `rendered/` and are ignored by Git.

Before submission, fill in authors/affiliations, venue metadata, remaining
upstream licenses, dataset/model URLs, and regenerated figures. The shipped
checkpoint identity, gate results, and operating thresholds are recorded in the
paper and in `recipes/f0/selection.json`; the ablation evidence behind Appendix A
is in `recipes/f0_ablations_20260910/`. The historical v31 identity remains in
`recipes/v31/selection.json`.

The LaTeX source reserves and captions each visual slot. It accepts either PDF
or PNG files under `figures/` without changing the surrounding paper.
