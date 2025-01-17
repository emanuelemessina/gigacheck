#include "edc.cuh"
#include "kernels.cuh"

namespace cuda
{
    EDCResult errors_detect_correct(const float* d_ec_matrix, int rows, int cols, float* d_cc_control, float* d_rc_control, cudaStream_t* streams)
    {
        EDCResult edc_res;

        // allocate mismatches index buffers

        int xs[EDC_MAX_ERRORS], ys[EDC_MAX_ERRORS];
        int *dXs, *dYs;
        cudaMalloc(&dXs, EDC_MAX_ERRORS * sizeof(int));
        cudaMalloc(&dYs, EDC_MAX_ERRORS * sizeof(int));

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

        kernels::find_checksum_mismatches<<<CEIL_DIV(cols + 1, tileDim.y), tileDim.y, 0, streams[0]>>>(d_ec_matrix, rows, cols, d_cc_control, COMPARE_CHECKSUM_COL, &d_mismatch_info[MISMATCH_COUNT_X], dXs, &d_mismatch_info[ERROR_X]);

        cudaMemcpyAsync(mismatch_info, d_mismatch_info, 2 * sizeof(float), cudaMemcpyDeviceToHost, streams[0]);

        kernels::find_checksum_mismatches<<<CEIL_DIV(rows + 1, tileDim.x), tileDim.x, 0, streams[1]>>>(d_ec_matrix, rows, cols, d_rc_control, COMPARE_CHECKSUM_ROW, &d_mismatch_info[MISMATCH_COUNT_Y], dYs, &d_mismatch_info[ERROR_Y]);

        cudaMemcpyAsync(mismatch_info + 2, d_mismatch_info + 2, 2 * sizeof(float), cudaMemcpyDeviceToHost, streams[1]);

        cudaDeviceSynchronize();
        CUDA_CHECK

#define AXIS_X REDUCE_ALONG_COL
#define AXIS_Y REDUCE_ALONG_ROW

        int collinear_axis = mismatch_info[MISMATCH_COUNT_X] == 1 ? AXIS_X : AXIS_Y;
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
        cudaMemcpyAsync(xs, dXs, num_errors * sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
        cudaMemcpyAsync(ys, dYs, num_errors * sizeof(float), cudaMemcpyDeviceToHost, streams[1]);

        // allocate host correction values
        float* h_edc_vals;
        cudaMallocHost(&h_edc_vals, num_errors * sizeof(float));

        // allocate device one-off sums
        float* dSum;
        cudaMalloc(&dSum, num_errors * sizeof(float));

        cudaDeviceSynchronize();
        CUDA_CHECK

        for (int i = 0; i < num_errors; ++i)
        {
            int streamId = num_errors % globals::numStreams;

            // one-off sum on opposite axis of the collinear one

            int exclude_index = collinear_axis == AXIS_X ? ys[i] : xs[i];
            int sum_axis_index = collinear_axis == AXIS_X ? xs[i] : ys[i];

            kernels::sum_axis_except<<<1, tileDim.x, linearDimToBytes(tileDim.x), streams[streamId]>>>(d_ec_matrix, rows, cols, collinear_axis, sum_axis_index, exclude_index, (float*)(dSum + i));

            cudaDeviceSynchronize();
            CUDA_CHECK

            cudaMemcpyAsync(h_edc_vals + i, dSum + i, sizeof(float), cudaMemcpyDeviceToHost, streams[streamId]);

            cudaDeviceSynchronize();
            CUDA_CHECK

            // calculate correction

            float checksum;
            cudaMemcpyAsync(&checksum, (collinear_axis == AXIS_X ? d_ec_matrix + rows * (cols + 1) + xs[i] : d_rc_control + ys[i] * (cols + 1) + cols), sizeof(float), cudaMemcpyDeviceToHost, streams[streamId]);

            cudaDeviceSynchronize();
            CUDA_CHECK

            h_edc_vals[i] = checksum - h_edc_vals[i];

            // write correction
            cudaMemcpyAsync((void*)(d_ec_matrix + ys[i] * (cols + 1) + xs[i]), h_edc_vals + i, sizeof(float), cudaMemcpyHostToDevice, streams[streamId]);

            cudaDeviceSynchronize();
            CUDA_CHECK
        }

        cudaDeviceSynchronize();

        CUDA_CHECK

        cudaFree(dSum);
        cudaFreeHost(h_edc_vals);

        CUDA_CHECK

    cleanup:

        cudaFreeHost(mismatch_info);
        cudaFree(d_mismatch_info);
        cudaFree(dXs);
        cudaFree(dYs);

        CUDA_CHECK

        return edc_res;
    }
}
