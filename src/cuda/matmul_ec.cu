#include "globals.h"
#include "inferred_matrix_sizes.h"
#include "kernels.cuh"
#include "matrix.h"
#include "timer.h"

namespace cuda
{
    void matmul_ec(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B)
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

        // allocate control checksums

        float *ccC, *rcC;

        cudaMalloc(&ccC, (COLS_C + 1) * sizeof(float));
        cudaMalloc(&rcC, (ROWS_C + 1) * sizeof(float));

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
            ScopedTimer timer("A,B to device + checksums", POST);

            // copy A to device

            cudaMemcpyAsync(dA, A, SIZE_A_BYTES, cudaMemcpyHostToDevice, streams[0]);

            // calculate col checksums for A

            gridDim = dim3(cols_A);
            blockDim = dim3(1, tileDim.y);
            sharedMemSize = linearDimToBytes(tileDim.y);
            kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, streams[0]>>>(dA, rows_A, cols_A, CHECKSUM_COMPUTE_COL);

            // copy B to device

            // since B contains the checksum we have to copy it row by row, we divide the load into multiple streams from [1] to [numStreams-1] (diffent from the A stream[0])

            std::vector<cudaEvent_t> memcpyEvents(globals::numStreams - 1);

            for (int s = 0; s < globals::numStreams - 1; ++s)
            {
                cudaEventCreate(&memcpyEvents[s]);
            }

            for (int r = 0; r < ROWS_B; ++r)
            {
                int streamIdx = r % (globals::numStreams - 1) + 1;
                cudaMemcpyAsync(dB + r * (cols_B + 1), B + r * cols_B, cols_B * sizeof(float), cudaMemcpyHostToDevice, streams[streamIdx]);

                // record an event in the stream (only the last copy operation in each stream needs to record)
                if (r >= ROWS_B - (globals::numStreams - 1))
                {
                    cudaEventRecord(memcpyEvents[streamIdx - 1], streams[streamIdx]);
                }
            }

            // calculate row checksums for B

            for (int s = 0; s < globals::numStreams - 1; ++s) // wait for all the copy streams to finish
            {
                cudaStreamWaitEvent(streams[1], memcpyEvents[s]);
                cudaEventDestroy(memcpyEvents[s]);
            }

            gridDim = dim3(1, ROWS_B);
            blockDim = dim3(tileDim.x, 1);
            sharedMemSize = linearDimToBytes(tileDim.x);
            kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, streams[1]>>>(dB, ROWS_B, cols_B, CHECKSUM_COMPUTE_ROW);

            cudaDeviceSynchronize();
            CUDA_CHECK
        }

        if (globals::printMatrices)
        {
            // print dA and dB (with checksums)
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
            ScopedTimer timer("matmul kernel", POST);

            gridDim = dim3(CEIL_DIV(COLS_C + 1, tileDim.x), CEIL_DIV(ROWS_C + 1, tileDim.y));
            blockDim = tileDim;
            sharedMemSize = 2 * dim2ToBytes(tileDim);
            kernels::tiled_matmul<<<gridDim, tileDim, sharedMemSize>>>(dA, dB, dC, rows_A + 1, cols_A, cols_B + 1);

            cudaDeviceSynchronize();
            CUDA_CHECK
        }

        if (globals::printMatrices)
        {
            // print dC (with checksums)
            float* Cec = matrix::alloc(ROWS_C + 1, COLS_C + 1, false);
            cudaMemcpy(Cec, dC, size_C_ec, cudaMemcpyDeviceToHost);
            matrix::print(Cec, ROWS_C + 1, COLS_C + 1, "C (w/ checksums)", HIGHLIGHT_LAST_ROW_AND_COL);
            free(Cec);
        }

        // compute control checksums after mul

        {
            ScopedTimer timer("C checksums", POST);

            gridDim = dim3(COLS_C + 1);
            blockDim = dim3(1, tileDim.y);
            sharedMemSize = linearDimToBytes(tileDim.y);
            kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, streams[0]>>>(dC, ROWS_C, (COLS_C + 1), CHECKSUM_COMPUTE_COL, ccC);

            gridDim = dim3(1, ROWS_C + 1);
            blockDim = dim3(tileDim.x, 1);
            sharedMemSize = linearDimToBytes(tileDim.x);
            kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, streams[1]>>>(dC, (ROWS_C + 1), COLS_C, CHECKSUM_COMPUTE_ROW, rcC);

            cudaDeviceSynchronize();
            CUDA_CHECK
        }

        if (globals::printMatrices)
        {
            // print control checksums
            float* hrcC = matrix::alloc(ROWS_C + 1, 1, false);
            float* hccC = matrix::alloc(1, COLS_C + 1, false);
            cudaMemcpy(hrcC, rcC, (ROWS_C + 1) * sizeof(float), cudaMemcpyDeviceToHost);
            CUDA_CHECK
            cudaMemcpy(hccC, ccC, (COLS_C + 1) * sizeof(float), cudaMemcpyDeviceToHost);
            CUDA_CHECK
            matrix::print(hrcC, ROWS_C + 1, 1, "C control row checksum", HIGHLIGHT_LAST_COL);
            matrix::print(hccC, 1, COLS_C + 1, "C control column checksum", HIGHLIGHT_LAST_ROW);
            free(hrcC);
            free(hccC);
        }

        // TODO: perform here error detection, correction

        // send back result without checksums

        {
            ScopedTimer timer("C to host", POST);

            for (int r = 0; r < ROWS_C; ++r)
            {
                cudaMemcpyAsync(C + r * COLS_C, dC + r * (COLS_C + 1), COLS_C * sizeof(float), cudaMemcpyDeviceToHost, streams[r % globals::numStreams]);
            }

            // destroy all streams

            for (int i = 0; i < globals::numStreams; ++i)
            {
                cudaStreamSynchronize(streams[i]); // we have to synch the device but do the loop to destroy anyway, so we synch here the single streams
                cudaStreamDestroy(streams[i]);
            }

            CUDA_CHECK
        }

        // cleanup

        delete[] streams;

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        cudaHostUnregister(A);
        cudaHostUnregister(B);
        cudaHostUnregister(C);

        CUDA_CHECK
    }
}
