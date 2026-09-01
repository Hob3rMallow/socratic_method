---
license: other
task_categories:
  - image-segmentation
tags:
  - vesuvius-challenge
  - 3d
  - teacher-student
  - cross-resolution
---

# Socratic Method v31 replay data

This is a dataset-card scaffold for the prepared v31 replay bundles. It is not
yet a published dataset.

The intended release contains the complete trees referenced by the pinned
training manifest (`dd2d3d...f0177`), held-out validation manifest
(`a995787...714`), and dynamic connectivity state (`5489a6...a8f8`). Preserve
the original directory suffixes so the read-time root mapping can relocate the
frozen absolute paths without changing JSON/JSONL hashes.

Training has 4,096 PHerc0139 schedule rows for one atlas record. Validation uses
PHerc0814 and PHerc1451. The teacher labels are soft predictions projected from
approximately 2.399 um to 9.362 um; they are not human ground truth.

Before publication, fill in upstream data sources and licenses, Villa teacher
version/checkpoint identity, download instructions, sizes/file counts, privacy
review, and citation. Do not publish under a guessed license.
