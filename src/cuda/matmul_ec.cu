#include "cuda.cuh"
#include "edc.cuh"
#include "globals.h"
#include "inferred_matrix_sizes.h"
#include "kernels.cuh"
#include "matrix.h"
#include "timer.h"

namespace cuda
{
    EDCResult matmul_ec(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B, int errors_count, int* error_xs, int* error_ys, float* error_values)
    {
        // register host pointers as pinned

        cudaHostRegister(A, SIZE_A_BYTES, cudaHostRegisterDefault);
        cudaHostRegister(B, SIZE_B_BYTES, cudaHostRegisterDefault);
        cudaHostRegister(C, SIZE_C_BYTES, cudaHostRegisterDefault);

        CUDA_CHECK

        // allocate device matrices with extra space for checksums A(m+1xn)B(nxp+1) = C(m+1xp+1)

        int size_A_ec = SIZE_A_BYTES + cols_A * sizeof(float);
        int size_B_ec = SIZE_B_BYTES + ROWS_B * sizeof(float);
        int size_C_ec = SIZE_C_BYTES + (COLS_C + ROWS_C + 1) * sizeof(float);

        float *dA, *dB, *dC;

        cudaMalloc(&dA, size_A_ec);
        cudaMalloc(&dB, size_B_ec);
        cudaMalloc(&dC, size_C_ec);

        // allocate checksums buffers

        float *d_cc_control, *d_rc_control;

        cudaMalloc(&d_cc_control, (COLS_C + 1) * sizeof(float));
        cudaMalloc(&d_rc_control, (ROWS_C + 1) * sizeof(float));

        CUDA_CHECK

        // create streams for parallel executions

        cudaStream_t* streams = new cudaStream_t[globals::numStreams];
        for (int i = 0; i < globals::numStreams; ++i)
        {
            cudaStreamCreate(&streams[i]);
        }

        // define threads organization
        dim3 gridDim;
        dim3 blockDim;
        int sharedMemSize;

        // send matrices to device and calculate checksums in parallel on different streams

        // stream1: copyA, checkA
        // stream2: copyB, checkB
        // copies are sent to possibly the same copy queue, kernels to the same kernel queue (especially if only one queue per category exists)
        // we use depth-first issue order: copyA - checkA - copyB - checkB
        // breadth-first issue order would be: copyA - copyB - checkA - checkB
        // we found that depth-first gives better performance

        {
            ScopedTimer timer("A,B to device + calc input checksums", POST);

            // copy A to device

            cudaMemcpyAsync(dA, A, SIZE_A_BYTES, cudaMemcpyHostToDevice, streams[0]);

            // calculate col checksums for A

            gridDim = dim3(cols_A);
            blockDim = dim3(1, tileDim.y);
            sharedMemSize = linearDimToBytes(tileDim.y);
            kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, streams[0]>>>(dA, rows_A, cols_A, ReductionDirection::ALONG_COL);

            // copy B to device

            // since B contains the checksum we have to copy it row by row, we divide the load into multiple streams from [1] to [numStreams-1] (diffent from the A stream[0])

            cudaEvent_t b_memcpyEvent;
            cudaEventCreate(&b_memcpyEvent);

            for (int r = 0; r < ROWS_B; ++r)
            {
                int streamIdx = r % (globals::numStreams - 1) + 1;
                cudaMemcpyAsync(dB + r * (cols_B + 1), B + r * cols_B, cols_B * sizeof(float), cudaMemcpyHostToDevice, streams[streamIdx]);
            }

            cudaEventRecord(b_memcpyEvent, streams[1]);

            // calculate row checksums for B

            cudaStreamWaitEvent(streams[1], b_memcpyEvent);
            cudaEventDestroy(b_memcpyEvent);

            gridDim = dim3(1, ROWS_B);
            blockDim = dim3(tileDim.x, 1);
            sharedMemSize = linearDimToBytes(tileDim.x);
            kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, streams[1]>>>(dB, ROWS_B, cols_B, ReductionDirection::ALONG_ROW);

            cudaDeviceSynchronize();
            CUDA_CHECK
        }

        // print dA and dB (with checksums)
        if (globals::debugPrint)
        {
            float* Aec = matrix::alloc(rows_A + 1, cols_A, false);
            float* Bec = matrix::alloc(ROWS_B, cols_B + 1, false);
            cudaMemcpy(Aec, dA, size_A_ec, cudaMemcpyDeviceToHost);
            cudaMemcpy(Bec, dB, size_B_ec, cudaMemcpyDeviceToHost);
            matrix::print(Aec, rows_A + 1, cols_A, "A (w/ column checksum)", HIGHLIGHT_LAST_ROW);
            matrix::print(Bec, ROWS_B, cols_B + 1, "B (w/ row checksum)", HIGHLIGHT_LAST_COL);
            free(Aec);
            free(Bec);
        }

        // compute the actual matrix multiplication as usual

        {
            ScopedTimer timer("tiled matmul", POST);

            gridDim = dim3(CEIL_DIV(COLS_C + 1, tileDim.x), CEIL_DIV(ROWS_C + 1, tileDim.y));
            blockDim = tileDim;
            sharedMemSize = 2 * dim2ToBytes(tileDim);
            kernels::tiled_matmul<<<gridDim, tileDim, sharedMemSize>>>(dA, dB, dC, rows_A + 1, cols_A, cols_B + 1);

            cudaDeviceSynchronize();
            CUDA_CHECK
        }

        // introduce errors in dC
        {
            ScopedTimer timer("introduce error(s)", POST);

            for (int i = 0; i < errors_count; i++)
                cudaMemcpyAsync(dC + error_ys[i] * (COLS_C + 1) + error_xs[i], &(error_values[i]), sizeof(float), cudaMemcpyHostToDevice, streams[i % globals::numStreams]);

            cudaDeviceSynchronize();
            CUDA_CHECK
        }

        // print dC (with mul checksums)
        if (globals::debugPrint)
        {
            float* Cec = matrix::alloc(ROWS_C + 1, COLS_C + 1, false);
            cudaMemcpy(Cec, dC, size_C_ec, cudaMemcpyDeviceToHost);
            matrix::print(Cec, ROWS_C + 1, COLS_C + 1, "C (w/ mul checksums)", HIGHLIGHT_LAST_ROW_AND_COL, error_xs, error_ys, errors_count);
            free(Cec);
        }

        // compute control checksums after mul
        {
            ScopedTimer timer("calc control checksums", POST);

            // compute col control checksum

            gridDim = dim3(COLS_C + 1);
            blockDim = dim3(1, tileDim.y);
            sharedMemSize = linearDimToBytes(tileDim.y);
            kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, streams[0]>>>(dC, ROWS_C, (COLS_C + 1), ReductionDirection::ALONG_COL, d_cc_control);

            // compute row control checksum

            gridDim = dim3(1, ROWS_C + 1);
            blockDim = dim3(tileDim.x, 1);
            sharedMemSize = linearDimToBytes(tileDim.x);
            kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, streams[1]>>>(dC, (ROWS_C + 1), COLS_C, ReductionDirection::ALONG_ROW, d_rc_control);

            cudaDeviceSynchronize();
            CUDA_CHECK
        }

        // print control checksums
        if (globals::debugPrint)
        {
            float *h_rc_control, *h_cc_control;
            cudaMallocHost(&h_rc_control, (ROWS_C + 1) * sizeof(float));
            cudaMallocHost(&h_cc_control, (COLS_C + 1) * sizeof(float));
            cudaMemcpyAsync(h_rc_control, d_rc_control, (ROWS_C + 1) * sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
            cudaMemcpyAsync(h_cc_control, d_cc_control, (COLS_C + 1) * sizeof(float), cudaMemcpyDeviceToHost, streams[1]);
            cudaDeviceSynchronize();
            CUDA_CHECK
            std::vector<int> zeros(errors_count, 0);
            matrix::print(h_rc_control, ROWS_C + 1, 1, "C control row checksum", HIGHLIGHT_LAST_COL, zeros.data(), error_ys, errors_count);
            matrix::print(h_cc_control, 1, COLS_C + 1, "C control column checksum", HIGHLIGHT_LAST_ROW, error_xs, zeros.data(), errors_count);
            cudaFreeHost(h_rc_control);
            cudaFreeHost(h_cc_control);
            CUDA_CHECK
        }

        // edc

        EDCResult edc_res;

        {
            ScopedTimer timer("EDC", POST);

            edc_res = errors_detect_correct(dC, ROWS_C, COLS_C, d_cc_control, d_rc_control, streams);

            // choice: don't send back the result if it's wrong
            // if (edc_res == UNCORRECTABLE_ERROR)
            //    goto cleanup;
        }

        // send back result (without checksums)

        {
            ScopedTimer timer("C to host", POST);

            for (int r = 0; r < ROWS_C; ++r) // copy row without last column
                cudaMemcpyAsync(C + r * COLS_C, dC + r * (COLS_C + 1), COLS_C * sizeof(float), cudaMemcpyDeviceToHost, streams[r % globals::numStreams]);

            cudaDeviceSynchronize();

            CUDA_CHECK
        }

        // cleanup:

        for (int i = 0; i < globals::numStreams; ++i)
        {
            cudaStreamSynchronize(streams[i]); // we have to synch the device but do the loop to destroy anyway, so we synch here the single streams
            cudaStreamDestroy(streams[i]);
        }

        delete[] streams;

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        cudaHostUnregister(A);
        cudaHostUnregister(B);
        cudaHostUnregister(C);

        CUDA_CHECK

        return edc_res;
    }
}
