#include "cuda.cuh"
#include "matrix.h"
#include "programs.h"

namespace programs
{
    int gigacheck(int ra, int ca, int cb)
    {
        auto rb = ca;
        auto rc = ra;
        auto cc = cb;

        float* A = matrix::alloc(ra, ca, true);
        float* B = matrix::alloc(rb, cb, true);
        float* C;

        cuda::matmul(A, B, C, ra, ca, cb);
        printf("Computation finished\n");

        matrix::check_product(A, B, C, ra, ca, cb);
        printf("Check finished\n");

        matrix::print(A, ra, ca);
        matrix::print(B, rb, cb);
        matrix::print(C, rc, cc);

        return 0;
    }
}