#pragma once

#include <stddef.h>

namespace globals
{
    extern bool printMatrices;
    extern bool useIntValues;
    extern int numStreams;

    // default tile side for squared block tiles
    extern int tileSide;

    extern size_t maxGlobalMem;
}
