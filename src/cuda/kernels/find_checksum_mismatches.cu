#include "edc.cuh"
#include "kernels.cuh"
#include <stdio.h>

#define CUDA_DEBUG_PRINT 0

__global__ void kernels::find_checksum_mismatches(const float* ec_matrix, int rows, int cols, float* control_checksum, ChecksumsToCompare checksums_to_compare, int* mismatch_count, int* mismatch_indexes, int* error_flag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int limit = checksums_to_compare == ChecksumsToCompare::ROW ? rows + 1 : cols + 1; // also verify checksums themselves

    if (idx >= limit || *error_flag)
        return;

    float mat_checksum_item = checksums_to_compare == ChecksumsToCompare::ROW ? ec_matrix[idx * (cols + 1) + cols] : ec_matrix[rows * (cols + 1) + idx];
    float control_checksum_item = control_checksum[idx];
    float diff = abs(mat_checksum_item - control_checksum_item);

#if CUDA_DEBUG_PRINT
    printf("%s thread %d: mat %f vs ctrl %f -> diff %f\n", checksums_to_compare == ChecksumsToCompare::ROW ? "row" : "col", idx, mat_checksum_item, control_checksum_item, diff);
#endif

    if (diff > 0.000001)
    {
        // atomically update the mismatch count
        int mismatch_idx = atomicAdd(mismatch_count, 1);

#if CUDA_DEBUG_PRINT
        printf("%s thread %d: current errors idx %d\n", checksums_to_compare == ChecksumsToCompare::ROW ? "row" : "col", idx, mismatch_idx);
#endif

        // Check if the mismatch array is full
        if (mismatch_idx >= EDC_MAX_ERRORS)
        {
            // mismatch array full: too many errors
            // set error flag and stop further processing
            *error_flag = 1;

#if CUDA_DEBUG_PRINT
            printf("%s thread %d: too many errors, raising error flag\n", checksums_to_compare == ChecksumsToCompare::ROW ? "row" : "col", idx + 1);
#endif

            return;
        }

        mismatch_indexes[mismatch_idx] = idx;
    }
}
