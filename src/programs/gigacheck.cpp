#include "cuda.cuh"
#include "edc.cuh"
#include "globals.h"
#include "iomod.h"
#include "matrix.h"
#include "memsize_string.h"
#include "programs.h"
#include "timer.h"
#include <cstdlib>
#include <tuple>
#include <vector>

namespace programs
{
    int gigacheck(int ra, int ca, int cb, bool vanilla, bool check, int errors_count, bool collinear_errors)
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
        if (!vanilla)
            WOUT << "# errors: " << errors_count << (errors_count > 1 && collinear_errors ? "(collinear)" : "") << std::endl;
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
            {
                std::vector<int> error_xs, error_ys;
                std::vector<float> error_values;

                if (errors_count > 0)
                {
                    bool align_on_x = std::rand() % 2;
                    int fixed_coord = std::rand() % (align_on_x ? rc : cc);

                    auto random_coord = [](int limit)
                    { return std::rand() % limit; };
                    auto random_value = []()
                    { return static_cast<float>(std::rand()) / RAND_MAX; }; // TODO: make universal (float/int generator)

                    for (int i = 0; i < errors_count; ++i)
                    {
                        int x = collinear_errors ? (align_on_x ? random_coord(cc) : fixed_coord) : random_coord(cc);
                        int y = collinear_errors ? (align_on_x ? fixed_coord : random_coord(rc)) : random_coord(rc);

                        error_xs.push_back(x);
                        error_ys.push_back(y);
                        error_values.push_back(random_value());
                    }
                }

                cuda::EDCResult edc_res = cuda::matmul_ec(A, B, C, ra, ca, cb, error_values.size(), error_xs.data(), error_ys.data(), error_values.data());

                if (edc_res == cuda::UNCORRECTABLE_ERROR)
                {
                    std::cout << "😐 Uncorrectable error encountered, multiplication failed." << std::endl;
                    check = false;
                }
                else if (edc_res == cuda::CORRECTED_ERROR)
                {
                    std::cout << "😎 Corrected detected error(s)" << std::endl;
                }
            }
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
