# Additive 2D line fitter

The postprocessor in `native/line_fitter` works independently on each z-plane,
then uses neighboring planes to reject unsafe joins and require persistent
support. Its default contract is additive: it may bridge a short model gap but
does not erase predicted foreground.

For each slice it:

1. runs Zhang-Suen thinning, prunes short spurs, and extracts curve endpoints
   with outward tangents;
2. constructs symmetric endpoint pairs and applies distance, facing,
   anti-parallel, radial, adjacent-plane, arc-ratio, anti-merger, cross-sheet,
   crossing, and endpoint-occupancy gates;
3. greedily accepts pairs in deterministic score order;
4. builds endpoint and connection tracks across z, then repeats matching with
   persistence evidence; and
5. paints accepted joins as cubic Bezier strokes into the binary mask.

The calibrated 4x5x5 study produced 481 joins and 12,697 added pixels, with
92.5% high-resolution precision overall, 98.4% for joins of at most six pixels,
no observed full-turn fusions, and 21% less atlas overlap. A larger 21-cube
census measured 98.3% weighted precision. These figures are development
evidence, not a claim that adjacent high-resolution connectivity alone proves a
join cannot cross wraps; the radial gate remains the primary cross-wrap guard.

The bridge scanner is report-only by default. Cutting breaks the additive
contract and remains off. The experimental slab splitter is retained for
negative-result provenance: at the mesh stage it separated 0 of 246 fused runs
and is not part of the recommended pipeline.

Build and test instructions are in `native/line_fitter/README.md`.
