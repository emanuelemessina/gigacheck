#include "cuda.cuh"
#include "globals.h"
#include "matrix.h"
#include "programs.h"
#include "timer.h"

namespace programs
{
    int gigacheck(int ra, int ca, int cb, bool vanilla, bool check)
    {
        float* A = matrix::alloc(ra, ca, true);
        float* B = matrix::alloc(rb, cb, true);
        float* C = matrix::alloc(rc, cc, false);

        if (globals::printMatrices)
        {
            matrix::print(A, ra, ca, "A");
            matrix::print(B, rb, cb, "B");
        }

        {
            ScopedTimer timer("GPU mul", PRE);

            if (vanilla)
                cuda::matmul(A, B, C, ra, ca, cb);
            else
                cuda::matmul_ec(A, B, C, ra, ca, cb);
        }

        int result = 0;

        if (check)
        {
            ScopedTimer timer("CPU mul check", PRE);
            result = matrix::check_product(A, B, C, ra, ca, cb);
        }

        if (globals::printMatrices)
            matrix::print(C, rc, cc, "C");

        return result;
    }
}