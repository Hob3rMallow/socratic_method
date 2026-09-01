# September 2026 submission workspace

This directory is intentionally separate from the reproducibility and model
artifacts. It contains the first paper-style account of the Socratic Method and
is marked as a working draft. The v31 raw-model selection is complete; figures,
authorship, licensing, and artifact URLs remain provisional.

Canonical repository: <https://github.com/ubc-nvining/socratic_method>

The canonical source uses the same anonymous ACM TOG/SIGGRAPH review format as
the papers under `D:\papers` (`acmtog`, author-year citations, two columns, a
teaser, overview, method, results, limitations, and conclusion).

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
upstream licenses, dataset/model URLs, and representative figures. The final
checkpoint identity, blinded gate results, and selected threshold are now
recorded in the paper and `recipes/v31/selection.json`.

The LaTeX source reserves and captions each visual slot. It accepts either PDF
or PNG files under `figures/` without changing the surrounding paper.
