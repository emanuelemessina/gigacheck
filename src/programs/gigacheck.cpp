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

#define WOUT (std::cout << std::setw(18))

        std::cout << BOLD << "Params:" << RESET << std::endl;
        WOUT << "A: " << ra << " x " << ca << std::endl;
        WOUT << "B: " << rb << " x " << cb << std::endl;
        WOUT << "-> C: " << rc << " x " << cc << std::endl;
        WOUT << "Values type: " << (globals::useIntValues ? "int" : "float") << std::endl;
        WOUT << "GPU mul alg: " << (vanilla ? "vanilla" : "error corrected") << std::endl;
        WOUT << "# streams: " << globals::numStreams << std::endl;
        WOUT << "Tile side: " << globals::tileSide << std::endl;
        printf("\n\n");

        std::cout << BOLD << "Device info:" << RESET << std::endl;
        WOUT << "Name: " << info.deviceName
             << std::endl;
        WOUT << "Max Global Mem: " << humanReadableMemSize(globals::maxGlobalMem)
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

        free(A);
        free(B);
        free(C);

        return result;
    }
}
