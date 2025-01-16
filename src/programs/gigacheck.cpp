#include "cuda.cuh"
#include "globals.h"
#include "iomod.h"
#include "matrix.h"
#include "memsize_string.h"
#include "programs.h"
#include "timer.h"

namespace programs
{
    int gigacheck(int ra, int ca, int cb, bool vanilla, bool check)
    {
        cuda::Info info = cuda::getInfo();

        // print info

        printf("GIGACHECK\n");
        printf("=========\n\n");

        std::cout << BOLD << "Params:" << RESET << std::endl;
        printf("\tA: %d x %d\n", ra, ca);
        printf("\tB: %d x %d\n", ca, cb);
        printf("\t-> C: %d x %d\n", rc, cc);
        std::cout << "\tValues type: " << (globals::useIntValues ? "int" : "float") << std::endl;
        std::cout << "\tGPU mul alg: " << (vanilla ? "vanilla" : "error corrected") << std::endl;
        std::cout << "\t# streams: " << globals::numStreams << std::endl;
        printf("\tTile side: %d", globals::tileSide);
        printf("\n\n");

        std::cout << BOLD << "Device info:" << RESET << std::endl;
        std::cout << "\tName: " << info.deviceName
                  << std::endl;
        std::cout << "\tMax Global Mem: " << humanReadableMemSize(globals::maxGlobalMem)
                  << std::endl;
        printf("\n\n");

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
