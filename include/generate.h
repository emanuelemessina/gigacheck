#pragma once

#include <cstdlib>

#define MAX_RAND_INT 5

inline float random_float(bool intValue)
{
    return intValue ? (float)(rand() % MAX_RAND_INT) : (float)rand() / RAND_MAX;
}

inline float random_int(int limit)
{
    return std::rand() % limit;
}
