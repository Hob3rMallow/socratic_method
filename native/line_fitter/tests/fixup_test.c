/* fixup_test.c — harness shim for the src/fixup/ module selftests.
 * The real tests live as Module_selftest() beside each module; the tool-level
 * end-to-end mini-grid test runs via `pred_fixup --selftest`. */
#include "common/ves_platform.h"
#include "fixup/bridge_scan.h"
#include "fixup/slab_split.h"
#include "fixup/fixup_viz.h"
#include "fixup/join_paint.h"
#include "fixup/join_tracks.h"
#include "fixup/slice_match.h"
#include "fixup/slice_trace.h"

#ifdef TEST_HARNESS
#define main fixup_test_main
#endif

int main(void)
{
    int rc = 0;
    rc |= SliceTrace_selftest();
    rc |= JoinPaint_selftest();
    rc |= SliceMatch_selftest();
    rc |= JoinTracks_selftest();
    rc |= BridgeScan_selftest();
    rc |= SlabSplit_selftest();
    rc |= FixupViz_selftest();
    return rc;
}
