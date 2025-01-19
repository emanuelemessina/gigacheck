#include "edc.cuh"
#include "kernels.cuh"
#include <iomanip>
#include <iostream>

namespace cuda
{
    EDCResult errors_detect_correct(const float* d_ec_matrix, int rows, int cols, float* d_cc_control, float* d_rc_control, cudaStream_t* streams)
    {
        EDCResult edc_res;

        // allocate mismatches index buffers

        int error_xs[EDC_MAX_ERRORS], error_ys[EDC_MAX_ERRORS];
        int *d_error_xs, *d_error_ys;
        cudaMalloc(&d_error_xs, EDC_MAX_ERRORS * sizeof(int));
        cudaMalloc(&d_error_ys, EDC_MAX_ERRORS * sizeof(int));

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

        kernels::find_checksum_mismatches<<<CEIL_DIV(cols + 1, tileDim.y), tileDim.y, 0, streams[0]>>>(d_ec_matrix, rows, cols, d_cc_control, ChecksumCompareMode::COL, &d_mismatch_info[MISMATCH_COUNT_X], d_error_xs, &d_mismatch_info[ERROR_X]);

        cudaMemcpyAsync(mismatch_info, d_mismatch_info, 2 * sizeof(float), cudaMemcpyDeviceToHost, streams[0]);

        kernels::find_checksum_mismatches<<<CEIL_DIV(rows + 1, tileDim.x), tileDim.x, 0, streams[1]>>>(d_ec_matrix, rows, cols, d_rc_control, ChecksumCompareMode::COL, &d_mismatch_info[MISMATCH_COUNT_Y], d_error_ys, &d_mismatch_info[ERROR_Y]);

        cudaMemcpyAsync(mismatch_info + 2, d_mismatch_info + 2, 2 * sizeof(float), cudaMemcpyDeviceToHost, streams[1]);

        cudaDeviceSynchronize();
        CUDA_CHECK

#define AXIS_X ReductionDirection::ALONG_COL
#define AXIS_Y ReductionDirection::ALONG_ROW

        ReductionDirection collinear_axis = mismatch_info[MISMATCH_COUNT_X] == 1 ? AXIS_Y : AXIS_X; // only 1 mismatch found in x implies the collinear axis must be y and viceversa
        int num_errors = mismatch_info[MISMATCH_COUNT_X] == 1 ? mismatch_info[MISMATCH_COUNT_Y] : mismatch_info[MISMATCH_COUNT_X];

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
            goto cleanup;
        }

        if ((mismatch_info[MISMATCH_COUNT_X] >> 1 & mismatch_info[MISMATCH_COUNT_Y] >> 1) != 0)
        {
            // non collinear errors, can't correct
            edc_res = UNCORRECTABLE_ERROR;
            goto cleanup;
        }

        // all errors collinear, or single error: can correct on axis
        edc_res = CORRECTED_ERROR;

        // overwrite d_ec_matrix with corrected vals

        // copy mismatch coords to host
        cudaMemcpyAsync(error_xs, d_error_xs, num_errors * sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
        cudaMemcpyAsync(error_ys, d_error_ys, num_errors * sizeof(float), cudaMemcpyDeviceToHost, streams[1]);

        // allocate correction checksums

        float* correction_checksums;
        cudaMallocHost(&correction_checksums, num_errors * sizeof(float));

        // allocate host correction values
        float* h_edc_vals;
        cudaMallocHost(&h_edc_vals, num_errors * sizeof(float));

        // allocate device one-off sums
        float* d_correction_sum;
        cudaMalloc(&d_correction_sum, num_errors * sizeof(float));

        cudaDeviceSynchronize();
        CUDA_CHECK

        for (int i = 0; i < num_errors; ++i)
        {
            int streamId = i % globals::numStreams;

            // correct collinear coords (one kernel found only 1 mismatch, need to duplicate the single coord)
            if (collinear_axis == AXIS_X) // only 1 y
                error_ys[i] = error_ys[0];
            else // only 1 x
                error_xs[i] = error_xs[0];

            // one-off sum on opposite axis of the collinear one

            int exclude_index = collinear_axis == AXIS_X ? error_ys[i] : error_xs[i];
            int sum_axis_index = collinear_axis == AXIS_X ? error_xs[i] : error_ys[i];

            kernels::sum_axis_except<<<1, tileDim.x, linearDimToBytes(tileDim.x), streams[streamId]>>>(d_ec_matrix, rows, cols, collinear_axis, sum_axis_index, exclude_index, (float*)(d_correction_sum + i));

            cudaMemcpyAsync(h_edc_vals + i, d_correction_sum + i, sizeof(float), cudaMemcpyDeviceToHost, streams[streamId]);

            // calculate correction

            cudaMemcpyAsync(correction_checksums + i, (collinear_axis == AXIS_X ? d_ec_matrix + rows * (cols + 1) + error_xs[i] : d_ec_matrix + error_ys[i] * (cols + 1) + cols), sizeof(float), cudaMemcpyDeviceToHost, streams[streamId]);
            cudaStreamSynchronize(streams[streamId]);
            CUDA_CHECK

            h_edc_vals[i] = correction_checksums[i] - h_edc_vals[i];

            // write correction
            cudaMemcpyAsync((void*)(d_ec_matrix + error_ys[i] * (cols + 1) + error_xs[i]), h_edc_vals + i, sizeof(float), cudaMemcpyHostToDevice, streams[streamId]);

            if (globals::debugPrint)
            {
                printf("Found correctable error @ C(%d, %d):\n", error_ys[i] + 1, error_xs[i] + 1); // math notation (row, col)
                COUT << "  mul " << (collinear_axis == AXIS_X ? "col ↓" : "row →") << " checksum = " << FMT_FLOAT(correction_checksums[i]) << ENDL;
                COUT << "  corrected value = " << h_edc_vals[i] << ENDL;
            }
        }

        cudaDeviceSynchronize();

        CUDA_CHECK

        cudaFree(d_correction_sum);
        cudaFreeHost(h_edc_vals);
        cudaFreeHost(correction_checksums);

        CUDA_CHECK

    cleanup:

        cudaFreeHost(mismatch_info);
        cudaFree(d_mismatch_info);
        cudaFree(d_error_xs);
        cudaFree(d_error_ys);

        CUDA_CHECK

        return edc_res;
    }
}
