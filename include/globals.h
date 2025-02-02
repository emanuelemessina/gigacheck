#pragma once

#include <stddef.h>

typedef enum
{
    noBuffer,
    bufferAB,
    bufferABC_forWriteback,
    bufferABC_for2muls
} Strategy;

namespace globals
{
    extern bool debugPrint;
    extern bool useIntValues;

    // default tile side for squared block tiles
    extern int tileSide;

    extern size_t maxGlobalMem;
}
