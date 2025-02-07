#pragma once

#include <stddef.h>

typedef enum
{
    simple,
    preloadAB,
    preloadAB_deferUnloadC,
    parallelMul
} Strategy;

namespace globals
{
    extern bool debugPrint;
    extern bool useIntValues;

    // default tile side for squared block tiles
    extern int tileSide;

    extern size_t maxGlobalMem;
}
