#pragma once

#include <cuda_runtime.h>

namespace cuda
{
    enum EDCResult
    {
        NO_ERROR = 0,
        CORRECTED_ERROR = 1,
        UNCORRECTABLE_ERROR = 2,
        ERROR_ONLY_IN_LAST_ROW_CHECKSUMS = 3,
        ERROR_ONLY_IN_LAST_COL_CHECKSUMS = 4,
    };

#define EDC_EPSILON 1e-5f; // tolerance for error detection

#define EDC_MAX_ERRORS 3 // extremely rare

    EDCResult errors_detect_correct(const float* d_ec_matrix, int rows, int cols, float* d_cc_control, float* d_rc_control, cudaStream_t mainStream, cudaStream_t secondaryStream);
}
