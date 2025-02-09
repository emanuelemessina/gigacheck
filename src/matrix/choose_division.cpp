#include "cuda.cuh"
#include "edc.cuh"
#include "globals.h"
#include "inferred_matrix_sizes.h"
#include "matrix.h"
#include <math.h>

using namespace cuda;

bool matrix::calc_splits(MulStrategy strategy, int rows_A, int cols_A, int cols_B, int* num_split_common_dim, int* num_split_other_dim)
{
    int factA = 1, factB = 1, factC = 1, factEDC = 1;

    switch (strategy)
    {
        case parallelMul: // AB->C,A'B'->C'
            factEDC = 2;
        case preloadAB_deferUnloadC: // AB->C,A'B',C'
            factC = 2;
        case preloadAB: // AB->C,A'B'
            factA = factB = 2;
    }

    size_t available_mem = globals::maxGlobalMem; // global memory available to store the input data
    size_t data_mem = 0;                          // global memory required to store the input data
    size_t aux_mem = 0;                           // aux memory needed for checksums

    data_mem += factA * (rows_A * cols_A); // A input
    aux_mem += factA * (1 * cols_A);       // A col checksum

    data_mem += factB * (ROWS_B * cols_B); // B input
    aux_mem += factB * (ROWS_B * 1);       // B row checksum

    data_mem += factC * (ROWS_C * COLS_C);            // C output
    aux_mem += factC * (ROWS_C * 1 + 1 * COLS_C - 1); // C checksums

    aux_mem += factC * (1 * COLS_C + ROWS_C * 1); // control checksums buffers for C

    data_mem *= sizeof(float);
    aux_mem *= sizeof(float);

    aux_mem += factEDC * (2 * EDC_MAX_ERRORS) * sizeof(int); // Mismatches index buffers x/y for error correction
    aux_mem += factEDC * (4) * sizeof(int);                  // Mismatch_count_x/y, error_x/y

    if (aux_mem >= available_mem) // not enough space for aux memory
        return false;

    available_mem -= aux_mem;

    if (data_mem <= available_mem)
    {
        *num_split_common_dim = 1;
        *num_split_other_dim = 1;
        return true;
    }

    float exceeding_factor = data_mem / available_mem;

    *num_split_common_dim = ceil(sqrt(exceeding_factor));
    *num_split_other_dim = ceil(exceeding_factor / (float)(*num_split_common_dim));

    return true;
}
