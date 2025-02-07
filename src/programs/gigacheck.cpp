#include "cuda.cuh"
#include "edc.cuh"
#include "generate.h"
#include "globals.h"
#include "iomod.h"
#include "kernels.cuh"
#include "matrix.h"
#include "memsize_string.h"
#include "programs.h"
#include "stdio.h"
#include "timer.h"
#include <cstdlib>
#include <set>
#include <tuple>
#include <vector>

namespace programs
{
    int gigacheck(int ra, int ca, int cb, bool vanilla, bool check, int errors_count, bool collinear_errors, cuda::MulStrategy strategy)
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
        WOUT << "Tile side: " << globals::tileSide << ENDL;
        printf("\n\n");

        COUT << BOLD << "Device info:" << RESET << ENDL;
        WOUT << "Name: " << info.deviceName << ENDL;
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

            std::vector<int*> per_block_error_xs, per_block_error_ys;
            std::vector<float*> per_block_error_values;

            int splits_square, splits;
            matrix::calc_splits(strategy, ra, ca, cb, &splits, &splits_square);

            const int total_blocks = splits * splits_square * splits_square;
            const int limit_x = CEIL_DIV(ca, splits_square) + 1;
            const int limit_y = CEIL_DIV(ra, splits_square) + 1;

            for (int i = 0; i < total_blocks; i++)
            {
                int* error_xs = (int*)malloc(errors_count * sizeof(int));
                int* error_ys = (int*)malloc(errors_count * sizeof(int));
                float* error_values = (float*)malloc(errors_count * sizeof(float));

                if (errors_count > 0) // generate errors
                {
                    std::set<std::pair<int, int>> error_points;

                    bool align_on_x = std::rand() % 2;
                    int fixed_coord = std::rand() % (align_on_x ? limit_y : limit_x);

                    while (error_points.size() < errors_count)
                    {
                        int x = collinear_errors ? (align_on_x ? random_int(limit_x) : fixed_coord) : random_int(limit_x);
                        int y = collinear_errors ? (align_on_x ? fixed_coord : random_int(limit_y)) : random_int(limit_y);
                        std::pair<int, int> point(x, y);

                        if (error_points.insert(point).second)
                        {
                            float val;
                            do
                            {
                                val = random_float(globals::useIntValues);
                            } while (globals::useIntValues && std::find(error_values, error_values + error_points.size(), val) != error_values + error_points.size()); // avoid same val if using ints (debug)

                            int idx = error_points.size() - 1;
                            error_xs[idx] = x;
                            error_ys[idx] = y;
                            error_values[idx] = val;
                        }
                    }
                }

                per_block_error_values.push_back(error_values);
                per_block_error_xs.push_back(error_xs);
                per_block_error_ys.push_back(error_ys);
            }

            cuda::EDCResult edc_res = cuda::matmul_ec(A, B, C, ra, ca, cb, errors_count, per_block_error_xs.data(), per_block_error_ys.data(), per_block_error_values.data(), strategy, vanilla);

            if (edc_res == cuda::UNCORRECTABLE_ERROR)
            {
                COUT << "😐 Uncorrectable error encountered, multiplication failed." << ENDL;
                // choice: dont' check with cpu if we already know there's an error
                check = false;
            }
            else if (edc_res == cuda::CORRECTED_ERROR)
            {
                COUT << "😎 Corrected detected error(s)" << ENDL;
            }

            for (int i = 0; i < splits * splits_square * splits_square; i++)
            {
                free(per_block_error_values[i]);
                free(per_block_error_xs[i]);
                free(per_block_error_ys[i]);
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
