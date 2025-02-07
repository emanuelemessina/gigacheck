#include "edc.cuh"
#include "globals.h"
#include "inferred_matrix_sizes.h"
#include "matrix.h"
#include <math.h>

void matrix::choose_division(int rows_A, int cols_A, int cols_B, int* num_split_common_dim, int* num_split_other_dim, Strategy strategy)
{
    int factA = 1, factB = 1, factC = 1;

    switch (strategy)
    {
        case preloadAB_deferUnloadC:
        case parallelMul:
            factC = 2;

        case preloadAB:
            factA = factB = 2;
    }

    float required_mem = factA * (rows_A + 1) * cols_A;  // A with checksum
    required_mem += factB * ROWS_B * (cols_B + 1);       // B with checksum
    required_mem += factC * (ROWS_C + 1) * (COLS_C + 1); // C with checksum
    required_mem += factC * (COLS_C + 1);                // Column checksum buffer for C
    required_mem += factC * (ROWS_C + 1);                // Row checksum buffer for C
    required_mem *= sizeof(float);

    required_mem += 2 * EDC_MAX_ERRORS * sizeof(int); // Mismatches index buffer for error correction
    required_mem += 4 * sizeof(int);                  // Mismatch_count_x/y, error_x/y

    float exceeding_factor = required_mem / globals::maxGlobalMem;

    *num_split_common_dim = ceil(sqrt(exceeding_factor));
    *num_split_other_dim = ceil(exceeding_factor / (float)(*num_split_common_dim));
}
