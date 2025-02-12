#pragma once

#include <cuda_runtime.h>

namespace cuda
{
    enum EDCResult
    {
        NO_ERROR = 0,
        CORRECTED_ERROR = 1,
        UNCORRECTABLE_ERROR = 2,
    };

#define EDC_EPSILON 1e-5f; // tolerance for error detection

#define EDC_MAX_ERRORS 3 // extremely rare

    void compare_checksums(const float* d_ec_matrix, int rows, int cols, float* d_cc_control, float* d_rc_control, cudaStream_t mainStream, cudaStream_t secondaryStream, int** mismatch_info_out, int** d_error_xs_out, int** d_error_ys_out);
    EDCResult errors_localize_correct(const float* d_ec_matrix, int rows, int cols, int* mismatch_info, int* d_error_xs, int* d_error_ys, float* d_cc_control, float* d_rc_control, cudaStream_t mainStream, cudaStream_t secondaryStream, bool* recompute_vertical_checksums, bool* recompute_horizontal_checksums);
}
