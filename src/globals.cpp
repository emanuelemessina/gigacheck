#include "globals.h"

namespace globals
{
    bool useIntValues = false;
    bool debugPrint = false;
    bool noEDC = false;
    int numStreams = 4;
    int tileSide = 32;
    size_t maxGlobalMem = 1073741824; // 1 GB

    namespace profiling
    {
        uint64_t flop_counter = 0;
        uint64_t transfer_counter = 0;
    }
}
