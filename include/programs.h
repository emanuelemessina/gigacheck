#pragma once

#include "cuda.cuh"

namespace programs
{
    int gigacheck(int, int, int, bool, int, bool, cuda::MulStrategy);
}
