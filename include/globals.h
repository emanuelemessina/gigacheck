#pragma once

#include "timer.h"
#include <stddef.h>
#include <stdint.h>

namespace globals
{
    extern bool debugPrint;
    extern bool useIntValues;
    extern bool noEDC;

    // default tile side for squared block tiles
    extern int tileSide;

    extern size_t maxGlobalMem;

    // will be incremented synchronously by the cpu
    namespace profiling
    {
        extern uint64_t flop_counter;
        extern uint64_t transfer_counter;
        extern CudaAggregateTimer timer;
    }
}
