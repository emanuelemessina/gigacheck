#include "iomod.h"
#include "matrix.h"

bool matrix::check_product(float* A, float* B, float* C, int ra, int ca, int cb)
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
                std::cerr << RED;
                fprintf(stderr, "❌ Check failed @ (%d, %d): got ", r, c);
                fprintf(stderr, globals::useIntValues ? "%2.0f instead of %2.0f" : "%f instead of %f", C[c + cb * r], val);
                fprintf(stderr, "\n");
                std::cerr << RESET;
                return false;
            }
        }
    }

    std::cout << GREEN;
    printf("✔️  Check passed\n");
    std::cout << RESET;

    return true;
}
