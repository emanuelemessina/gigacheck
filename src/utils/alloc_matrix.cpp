#include "utils.h"

#include <stdlib.h>

float* utils::matrix::alloc(int rows, int cols, bool initialize)
{
    float* tmp = (float*)malloc(rows * cols * sizeof(float));

    if (initialize)
    {
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
#ifdef TEST_intValues
                tmp[r * cols + c] = (float)(rand() % 5);
#else
                tmp[r * cols + c] = (float)rand() / RAND_MAX;
#endif
    }

    return tmp;
}
