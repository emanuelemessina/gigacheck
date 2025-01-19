#include "generate.h"
#include "matrix.h"

float* matrix::alloc(int rows, int cols, bool initialize)
{
    if (!initialize)
        return (float*)calloc(rows * cols, sizeof(float));

    float* tmp = (float*)malloc(rows * cols * sizeof(float));

    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            tmp[r * cols + c] = random_float(globals::useIntValues);

    return tmp;
}
