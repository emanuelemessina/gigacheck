#include "edc.cuh"
#include "kernels.cuh"
#include <stdio.h>

__global__ void kernels::find_checksum_mismatches(const float* ec_matrix, int rows, int cols, float* control_checksum, ChecksumCompareMode comparison_mode, int* mismatch_count, int* mismatch_indexes, int* error_flag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int limit = comparison_mode == ChecksumCompareMode::ROW ? cols + 1 : rows + 1; // also verify checksums themselves

    if (idx >= limit || *error_flag)
        return;

    float mat_checksum_item = comparison_mode == ChecksumCompareMode::ROW ? ec_matrix[idx * (cols + 1) + cols] : ec_matrix[rows * (cols + 1) + idx];
    float control_checksum_item = control_checksum[idx];

    if (mat_checksum_item != control_checksum_item)
    {
        // atomically update the mismatch count
        int mismatch_idx = atomicAdd(mismatch_count, 1);

        // Check if the mismatch array is full
        if (mismatch_idx >= EDC_MAX_ERRORS)
        {
            // mismatch array full: too many errors
            // set error flag and stop further processing
            *error_flag = 1;
            return;
        }

        mismatch_indexes[mismatch_idx] = idx;
    }
}
