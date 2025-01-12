#include "utils/check_product.h"
#include "utils/inferred_matrix_sizes.h"

void check(float* A, float* B, float* C, int ra, int ca, int cb)
{
    for (int r = 0; r < rc; r++)
    {
        for (int c = 0; c < cc; c++)
        {
            float val = 0;
            for (int k = 0; k < ca; k++)
                val += A[r * ca + k] * B[c + k * cb];

            if (C[c + cb * r] != val)
            {
#ifdef TEST_intValues
                printf("Wrong product (first error at (%d, %d): %2.0f instead of %2.0f)\n", r, c, C[c + cb * r], val);
#else
                printf("Wrong product (first error at (%d, %d): %f instead of %f)\n", r, c, C[c + cb * r], val);
#endif
                return;
            }
        }
    }
}
