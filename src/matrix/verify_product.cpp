#include "iomod.h"
#include "matrix.h"

bool matrix::verify_product(float* A, float* B, float* C, int ra, int ca, int cb)
{
    for (int r = 0; r < rc; r++)
    {
        for (int c = 0; c < cc; c++)
        {
            float val = 0;
            for (int k = 0; k < ca; k++)
                val += A[r * ca + k] * B[c + k * cb];

            if (abs(C[c + cb * r] - val) > 0.0001 * val)
            {
                CERR << RED;
                fprintf(stderr, "❌ Check failed @ (%d, %d): got ", r + 1, c + 1);
                CERR << FMT_FLOAT(C[c + cb * r]) << " instead of " << FMT_FLOAT(val) << ENDL;
                CERR << RESET;
                return false;
            }
        }
    }

    COUT << GREEN;
    printf("✔️  Check passed\n");
    COUT << RESET;

    return true;
}
