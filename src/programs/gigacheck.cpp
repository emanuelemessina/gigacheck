#include "kernels.cuh"
#include "programs.h"
#include "utils.h"

using namespace utils;

namespace programs
{
    int gigacheck(int ra, int ca, int cb)
    {
        auto rb = ca;
        auto rc = ra;
        auto cc = cb;

        float* A = matrix::alloc(ra, ca, true);
        float* B = matrix::alloc(rb, cb, true);
        float* C = matrix::alloc(rc, cc, false);

        cuda::tiled_matmul(A, B, C, ra, ca, cb);
        printf("Computation finished\n");

        check_product(A, B, C, ra, ca, cb);
        printf("Check finished\n");

        matrix::print(A, ra, ca);
        matrix::print(B, rb, cb);
        matrix::print(C, rc, cc);

        return 0;
    }
}