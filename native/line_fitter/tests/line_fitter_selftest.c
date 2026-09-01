#include "fixup/bridge_scan.h"
#include "fixup/join_paint.h"
#include "fixup/join_tracks.h"
#include "fixup/slice_match.h"
#include "fixup/slice_trace.h"

int main(void)
{
    int rc = 0;
    rc |= SliceTrace_selftest();
    rc |= JoinPaint_selftest();
    rc |= SliceMatch_selftest();
    rc |= JoinTracks_selftest();
    rc |= BridgeScan_selftest();
    return rc;
}
