#include "edc.cuh"
#include "globals.h"
#include "inferred_matrix_sizes.h"
#include "matrix.h"
#include <math.h>

void matrix::calc_splits(Strategy strategy, int rows_A, int cols_A, int cols_B, int* num_split_common_dim, int* num_split_other_dim)
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

    data_mem += factA * (rows_A * cols_A); // A input
    available_mem -= factA * (1 * cols_A); // A col checksum

    data_mem += factB * (ROWS_B * cols_B); // B input
    available_mem += factB * (ROWS_B * 1); // B row checksum

    data_mem += factC * (ROWS_C * COLS_C);              // C output
    available_mem -= factC * (ROWS_C * 1 + 1 * COLS_C); // C checksums

    available_mem -= factC * (COLS_C + 1); // control col checksum buffer for C
    available_mem -= factC * (ROWS_C + 1); // control row checksum buffer for C

    data_mem *= sizeof(float);
    available_mem *= sizeof(float);

    available_mem -= factEDC * (2 * EDC_MAX_ERRORS) * sizeof(int); // Mismatches index buffers x/y for error correction
    available_mem -= factEDC * (4) * sizeof(int);                  // Mismatch_count_x/y, error_x/y

    float exceeding_factor = data_mem / available_mem;

    *num_split_common_dim = ceil(sqrt(exceeding_factor));
    *num_split_other_dim = ceil(exceeding_factor / (float)(*num_split_common_dim));
}
