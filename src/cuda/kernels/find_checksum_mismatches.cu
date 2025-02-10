#include "edc.cuh"
#include "kernels.cuh"
#include <stdio.h>

#define CUDA_DEBUG_PRINT 0

__global__ void kernels::find_checksum_mismatches(const float* ec_matrix, int rows, int cols, float* control_checksum, ChecksumsToCompare checksums_to_compare, int* mismatch_count, int* mismatch_indexes, int* error_flag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                   // 2 op
    int limit = checksums_to_compare == ChecksumsToCompare::ROW ? rows + 1 : cols + 1; // also verify checksums themselves  // 1 op

    if (idx >= limit || *error_flag) // 1 transf
        return;

    float mat_checksum_item = checksums_to_compare == ChecksumsToCompare::ROW ? ec_matrix[idx * (cols + 1) + cols] : ec_matrix[rows * (cols + 1) + idx]; // 1 transf, 3 ops
    float control_checksum_item = control_checksum[idx];                                                                                                 // 1 transfer
    float diff = abs(mat_checksum_item - control_checksum_item);                                                                                         // 1 op

#if CUDA_DEBUG_PRINT
    printf("%s thread %d: mat %f vs ctrl %f -> diff %f\n", checksums_to_compare == ChecksumsToCompare::ROW ? "row" : "col", idx, mat_checksum_item, control_checksum_item, diff);
#endif

    if (diff > 0.0001 * mat_checksum_item) // suppose 1 error per kernel, 1 thread in N triggers this // 1 op
    {
        // atomically update the mismatch count
        int mismatch_idx = atomicAdd(mismatch_count, 1); // 1 transfer, 1 op

#if CUDA_DEBUG_PRINT
        printf("%s thread %d: current errors idx %d\n", checksums_to_compare == ChecksumsToCompare::ROW ? "row" : "col", idx, mismatch_idx);
#endif

        // Check if the mismatch array is full
        if (mismatch_idx >= EDC_MAX_ERRORS) // suppose this is not triggered
        {
            // mismatch array full: too many errors
            // set error flag and stop further processing
            *error_flag = 1; // 1 transfer

#if CUDA_DEBUG_PRINT
            printf("%s thread %d: too many errors, raising error flag\n", checksums_to_compare == ChecksumsToCompare::ROW ? "row" : "col", idx + 1);
#endif

            return;
        }

        mismatch_indexes[mismatch_idx] = idx; // 1 transfer
    }
}

// transfers = (N-1)*3 + 1*2
// transfer size = total transfers * sizeof(float)
// ops = (N-1)*8 + 1*1
// intensity [flops/byte] = ops / transfer size
