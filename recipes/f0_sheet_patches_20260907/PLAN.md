# Extend paths into coherent 3D sheet patches

Full objective: extend found paths into coherent 3D sheet patches; investigate
solutions for resolving crowded-fold connections. This is not satisfied by
thicker 2D paint, connecting only a few selected slices, or a synthetic-only demo.

The preceding continuity goal was verified progress: the opt-in v2 engine and
83 tests are installed, all 27 package/control masks match exactly, and 62
official evidence files were copied. Revalidate relevant inputs before use.

Keep frozen F0/200k, T=.35, inference calibration, native repair and v2 artifacts
unchanged. New source versions and outputs are separate. No training or automatic
deployment. Do not turn an ambiguous old connection into trusted 3D supervision.

Completion requirements:

1. Implement joint 3D surface growth from found curves, with explicit surface
   geometry and actual additional mask coverage across neighboring Z planes.
   Preserve source foreground; track provenance, contacts and overlapping patches.
2. Test against ground-truth synthetic bent/moving sheets and competing nearby
   surfaces, including missing evidence, crossings, branch switches, invalid
   local charts and interaction between separate repairs. Verify the optimizer
   with a small exact oracle, not just plausible images.
3. Compare candidate mechanisms on real contiguous cached development context,
   including core path 1934, useful long components, and crowded-fold component
   301. Explicitly investigate CT-profile identity, joint geometric constraints,
   and alternative-path uncertainty. Record failures, not just favorable cases.
4. Quantify meaningful surface extension: supported surface area/axial span,
   mesh regularity, recovered gaps, raw/v2 additions and wrong-sheet proxies.
   Audit full-block interacting patches; inspect CT-aligned Z/Y/X views.
5. Freeze the candidate before a new disjoint assessment block. Previous
   continuity control is now known transfer evidence, not a fresh blind control.
   Predeclare center (12800,4608,2688), origin (12672,4480,2560), 384-cubed,
   conditional on real context inventory. At most one GPU worker if required.
6. Deliver a usable opt-in checked implementation, full evidence and HTML report,
   with specific findings on crowded-fold disambiguation and unresolved limits.
   No unsupported claim of global anatomical correctness or full-scroll validity.

Initial direction: a local curved chart around each seed path; jointly choose
normal displacement over (Z, path-distance) with hard slope constraints and
surface regularization. A height-field graph cut can optimize this restricted
problem exactly after integer cost quantization. It cannot represent every fold;
chart validity and alternative sheet identity must be independently assessed.

Primary references consulted (not implementations of the entire solution):
- Li, Wu, Chen and Sonka (2006), Optimal Surface Segmentation in Volumetric
  Images—A Graph-Theoretic Approach:
  https://www.cs.cmu.edu/~kangli/doc/papers/optnet-pami.pdf
- Installed SciPy 1.18 maximum_flow API (integer capacities):
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.maximum_flow.html
- scikit-image 0.26 registration reference, for the optical-flow alternative:
  https://scikit-image.org/docs/stable/api/skimage.registration.html

The new surface objective and practical acceptance rules are our own hypotheses
to test here; the cited method's medical-image results do not validate scroll
anatomy or crowded-fold identity in this corpus.
