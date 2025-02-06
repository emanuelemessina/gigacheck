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
    int gigacheck(int ra, int ca, int cb, bool vanilla, bool check, int errors_count, bool collinear_errors, Strategy strategy)
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
            matrix::choose_division(ra, ca, cb, &splits, &splits_square, strategy);

            for (int i = 0; i < splits * splits_square * splits_square; i++)
            {

                int* error_xs = (int*)malloc(errors_count * sizeof(int));
                int* error_ys = (int*)malloc(errors_count * sizeof(int));
                float* error_values = (float*)malloc(errors_count * sizeof(float));

                std::set<std::pair<int, int>> error_points;
                int limit_y = CEIL_DIV(rc, splits_square) + 1;
                int limit_x = CEIL_DIV(cc, splits_square) + 1;

                if (errors_count > 0) // generate errors
                {
                    bool align_on_x = std::rand() % 2;
                    int fixed_coord = std::rand() % (align_on_x ? limit_y : limit_x);

                    while (error_points.size() < errors_count)
                    {
                        int x = collinear_errors ? (align_on_x ? random_int(limit_x) : fixed_coord) : random_int(limit_x);
                        int y = collinear_errors ? (align_on_x ? fixed_coord : random_int(limit_y)) : random_int(limit_y);
                        std::pair<int, int> point = std::make_pair(x, y);

                        float val;
                        bool already_exists = false;
                        do
                        {
                            already_exists = false;
                            val = random_float(globals::useIntValues);

                            for (int j = 0; j < error_points.size(); j++)
                            {
                                if (error_values[j] == val)
                                {
                                    already_exists = true;
                                    break;
                                }
                            }
                        } while (already_exists);

                        if (error_points.find(point) == error_points.end())
                        {
                            error_xs[error_points.size()] = x;
                            error_ys[error_points.size()] = y;
                            error_values[error_points.size()] = val;
                            error_points.insert(point);
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
