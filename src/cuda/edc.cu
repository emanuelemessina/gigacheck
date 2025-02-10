#include "edc.cuh"
#include "kernels.cuh"
#include "matrix.h"
#include <iomanip>
#include <iostream>
#include <vector>

namespace cuda
{
    EDCResult errors_detect_correct(const float* d_ec_matrix, int rows, int cols, float* d_cc_control, float* d_rc_control, cudaStream_t mainStream, cudaStream_t secondaryStream, bool* recompute_vertical_checksums, bool* recompute_horizontal_checksums)
    {
        EDCResult edc_res;

        // allocate mismatches index buffers

        int error_xs[EDC_MAX_ERRORS], error_ys[EDC_MAX_ERRORS];
        int *d_error_xs, *d_error_ys;
        cudaMalloc(&d_error_xs, EDC_MAX_ERRORS * sizeof(int));
        cudaMalloc(&d_error_ys, EDC_MAX_ERRORS * sizeof(int));
        cudaMemset(d_error_xs, 0, EDC_MAX_ERRORS * sizeof(int));
        cudaMemset(d_error_ys, 0, EDC_MAX_ERRORS * sizeof(int));

        // mismatch info array to avoid multiple allocs and copies
        int* mismatch_info;
        cudaMallocHost(&mismatch_info, 4 * sizeof(int));
        int* d_mismatch_info;
        cudaMalloc(&d_mismatch_info, 4 * sizeof(int)); // mismatch_count_x/y, error_x/y
        cudaMemset(d_mismatch_info, 0, 4 * sizeof(int));

        CUDA_CHECK

#define MISMATCH_COUNT_X 0
#define ERROR_X 1
#define MISMATCH_COUNT_Y 2
#define ERROR_Y 3

        // depth-first issuing to avoid consecutive kernel scheduling blocking kernel0 signal to copy queue

        kernels::find_checksum_mismatches<<<CEIL_DIV(cols + 1, tileDim.y), tileDim.y, 0, mainStream>>>(d_ec_matrix, rows, cols, d_cc_control, ChecksumsToCompare::COL, &d_mismatch_info[MISMATCH_COUNT_X], d_error_xs, &d_mismatch_info[ERROR_X]);

        cudaMemcpyAsync(mismatch_info, d_mismatch_info, 2 * sizeof(float), cudaMemcpyDeviceToHost, mainStream);

        kernels::find_checksum_mismatches<<<CEIL_DIV(rows + 1, tileDim.x), tileDim.x, 0, secondaryStream>>>(d_ec_matrix, rows, cols, d_rc_control, ChecksumsToCompare::ROW, &d_mismatch_info[MISMATCH_COUNT_Y], d_error_ys, &d_mismatch_info[ERROR_Y]);

        cudaMemcpyAsync(mismatch_info + 2, d_mismatch_info + 2, 2 * sizeof(float), cudaMemcpyDeviceToHost, secondaryStream);

        cudaStreamSynchronize(mainStream);
        cudaStreamSynchronize(secondaryStream);
        CUDA_CHECK

#define AXIS_X ReductionDirection::ALONG_COL
#define AXIS_Y ReductionDirection::ALONG_ROW

        ReductionDirection collinear_axis = mismatch_info[MISMATCH_COUNT_X] <= 1 ? AXIS_Y : AXIS_X; // only 1 (or 0 in case of collinear checksum corruption) mismatch found in x implies the collinear axis must be y and viceversa
        int num_errors = mismatch_info[MISMATCH_COUNT_X] <= 1 ? mismatch_info[MISMATCH_COUNT_Y] : mismatch_info[MISMATCH_COUNT_X];
        int non_discarded = 0;

        if ((mismatch_info[MISMATCH_COUNT_X] | mismatch_info[MISMATCH_COUNT_Y]) == 0)
        {
            // no errors
            edc_res = NO_ERROR;
            goto cleanup;
        }

        if (mismatch_info[ERROR_Y] || mismatch_info[ERROR_X])
        {
            // kernel error (more errors than max allowed)
            edc_res = UNCORRECTABLE_ERROR;
            if (globals::debugPrint)
            {
                COUT << "Kernel error flag raised: (x " << mismatch_info[ERROR_Y] << ", y " << mismatch_info[ERROR_X] << ") mismatches found (max allowed per axis" << EDC_MAX_ERRORS << ")" << ENDL;
            }
            goto cleanup;
        }

        if ((mismatch_info[MISMATCH_COUNT_X] >> 1 & mismatch_info[MISMATCH_COUNT_Y] >> 1) != 0) // one of these must be 1 (shifted right becomes 0) -> collinear axis exists
        {
            // non collinear errors, can't correct
            edc_res = UNCORRECTABLE_ERROR;
            goto cleanup;
        }

        // all errors collinear, or single error: can correct on axis
        edc_res = CORRECTED_ERROR;

        // overwrite d_ec_matrix with corrected vals

        // copy mismatch coords to host
        cudaMemcpyAsync(error_xs, d_error_xs, num_errors * sizeof(float), cudaMemcpyDeviceToHost, mainStream);
        cudaMemcpyAsync(error_ys, d_error_ys, num_errors * sizeof(float), cudaMemcpyDeviceToHost, mainStream);

        // allocate host correction checksums
        float* correction_checksums;
        cudaMallocHost(&correction_checksums, num_errors * sizeof(float));

        // allocate host control checksums
        float* control_checksums;
        cudaMallocHost(&control_checksums, num_errors * sizeof(float));

        // allocate host error values
        float* error_values;
        cudaMallocHost(&error_values, num_errors * sizeof(float));

        // allocate host corrected values
        float* corrected_vals;
        cudaMallocHost(&corrected_vals, num_errors * sizeof(float));

        cudaStreamSynchronize(mainStream);
        CUDA_CHECK

        for (int i = 0; i < num_errors; ++i)
        {
            // correct collinear coords (one kernel found only 1 mismatch, need to duplicate the single coord)
            if (collinear_axis == AXIS_X) // only 1 y
                error_ys[i] = error_ys[0];
            else // only 1 x
                error_xs[i] = error_xs[0];

            // discard errors on checksum vectors
            if (error_xs[i] == cols)
            {
                *recompute_horizontal_checksums = true;
                continue;
            }

            if (error_ys[i] == rows)
            {
                *recompute_vertical_checksums = true;
                continue;
            }

            non_discarded++;

            // calculate correction

            // copy correction checksum
            cudaMemcpyAsync(correction_checksums + i, (collinear_axis == AXIS_X ? d_ec_matrix + rows * (cols + 1) + error_xs[i] : d_ec_matrix + error_ys[i] * (cols + 1) + cols), sizeof(float), cudaMemcpyDeviceToHost, mainStream);
            // copy control checksum
            cudaMemcpyAsync(control_checksums + i, (collinear_axis == AXIS_X ? d_cc_control + error_xs[i] : d_rc_control + error_ys[i]), sizeof(float), cudaMemcpyDeviceToHost, mainStream);
            // copy error value
            cudaMemcpyAsync(error_values + i, (void*)(d_ec_matrix + error_ys[i] * (cols + 1) + error_xs[i]), sizeof(float), cudaMemcpyDeviceToHost, mainStream);

            cudaStreamSynchronize(mainStream);
            CUDA_CHECK

            corrected_vals[i] = correction_checksums[i] - control_checksums[i] + error_values[i];

            // write correction
            cudaMemcpyAsync((void*)(d_ec_matrix + error_ys[i] * (cols + 1) + error_xs[i]), corrected_vals + i, sizeof(float), cudaMemcpyHostToDevice, mainStream);

            if (globals::debugPrint)
            {
                printf("Found correctable error @ C(%d, %d):\n", error_ys[i] + 1, error_xs[i] + 1); // math notation (row, col)
                COUT << "  mul " << (collinear_axis == AXIS_X ? "col ↓" : "row →") << " checksum = " << FMT_FLOAT(correction_checksums[i]) << ENDL;
                COUT << "  corrected value = " << corrected_vals[i] << ENDL;
            }
        }

        cudaStreamSynchronize(mainStream);

        CUDA_CHECK

        cudaFreeHost(corrected_vals);
        cudaFreeHost(correction_checksums);
        cudaFreeHost(error_values);
        cudaFreeHost(control_checksums);

        CUDA_CHECK

        if (non_discarded == 0)
            edc_res = NO_ERROR;

    cleanup:

        cudaFreeHost(mismatch_info);
        cudaFree(d_mismatch_info);
        cudaFree(d_error_xs);
        cudaFree(d_error_ys);

        CUDA_CHECK

        return edc_res;
    }
}
