#include "cuda.cuh"
#include "edc.cuh"
#include "generate.h"
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

#define WOUT (COUT << std::setw(18))

        COUT << BOLD << "Params:" << RESET << ENDL;
        WOUT << "A: " << ra << " x " << ca << ENDL;
        WOUT << "B: " << rb << " x " << cb << ENDL;
        WOUT << "-> C: " << rc << " x " << cc << ENDL;
        WOUT << "Values type: " << (globals::useIntValues ? "int" : "float") << ENDL;
        WOUT << "GPU mul alg: " << (vanilla ? "vanilla" : "error corrected") << ENDL;
        if (!vanilla)
            WOUT << "# errors: " << errors_count << (errors_count > 1 && collinear_errors ? "(collinear)" : "") << ENDL;
        WOUT << "# streams: " << globals::numStreams << ENDL;
        WOUT << "Tile side: " << globals::tileSide << ENDL;
        printf("\n\n");

        COUT << BOLD << "Device info:" << RESET << ENDL;
        WOUT << "Name: " << info.deviceName
             << ENDL;
        WOUT << "Max Global Mem: " << humanReadableMemSize(globals::maxGlobalMem)
             << ENDL;
        printf("\n\n");

        float* A = matrix::alloc(ra, ca, true);
        float* B = matrix::alloc(rb, cb, true);
        float* C = matrix::alloc(rc, cc, false);

        if (globals::debugPrint)
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

                    for (int i = 0; i < errors_count; ++i)
                    {
                        int x = collinear_errors ? (align_on_x ? random_int(cc) : fixed_coord) : random_int(cc);
                        int y = collinear_errors ? (align_on_x ? fixed_coord : random_int(rc)) : random_int(rc);

                        error_xs.push_back(x);
                        error_ys.push_back(y);
                        error_values.push_back(random_float(globals::useIntValues));
                    }
                }

                cuda::EDCResult edc_res = cuda::matmul_ec(A, B, C, ra, ca, cb, error_values.size(), error_xs.data(), error_ys.data(), error_values.data());

                if (edc_res == cuda::UNCORRECTABLE_ERROR)
                {
                    COUT << "😐 Uncorrectable error encountered, multiplication failed." << ENDL;
                    check = false;
                }
                else if (edc_res == cuda::CORRECTED_ERROR)
                {
                    COUT << "😎 Corrected detected error(s)" << ENDL;
                }
            }
        }

        int result = 0;

        if (check)
        {
            ScopedTimer timer("CPU mul check", PRE);
            result = matrix::check_product(A, B, C, ra, ca, cb);
        }

        if (globals::debugPrint)
            matrix::print(C, rc, cc, "C");

        free(A);
        free(B);
        free(C);

        return result;
    }
}
