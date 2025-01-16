#include "globals.h"
#include "inferred_matrix_sizes.h"
#include "kernels.cuh"
#include "matrix.h"
#include "timer.h"

namespace cuda
{
    void matmul_ec(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B)
    {
        // allocate device matrices with extra space for checksums A(m+1xn)B(nxp+1) = C(m+1xp+1)

        int size_A_ec = SIZE_A_BYTES + cols_A * sizeof(float);
        int size_B_ec = SIZE_B_BYTES + ROWS_B * sizeof(float);
        int size_C_ec = SIZE_C_BYTES + (COLS_C + ROWS_C + 1) * sizeof(float);

        float *dA, *dB, *dC;

        cudaMalloc(&dA, size_A_ec);
        cudaMalloc(&dB, size_B_ec);
        cudaMalloc(&dC, size_C_ec);

        // create streams for parallel executions

        int numStreams = 4; // TODO: adjust num of streams depending on the architecture?

        cudaStream_t* streams = new cudaStream_t[numStreams];
        for (int i = 0; i < numStreams; ++i)
        {
            cudaStreamCreate(&streams[i]);
        }

        {
            ScopedTimer timer("A,B to device", POST);

            // copy A to device

            cudaMemcpyAsync(dA, A, SIZE_A_BYTES, cudaMemcpyHostToDevice, streams[0]);

            // copy B to device

            for (int r = 0; r < ROWS_B; ++r)
            {
                cudaMemcpyAsync(dB + r * (cols_B + 1), B + r * cols_B, cols_B * sizeof(float), cudaMemcpyHostToDevice, streams[r % numStreams]);
            }

            cudaDeviceSynchronize();
        }

        // define threads organization
        dim3 gridSize;
        dim3 blockSize;
        int sharedMemSize;

        // calculate checksums in parallel on different streams

        {
            ScopedTimer timer("checksums", POST);

            // calculate col checksums for A

            gridSize = dim3(cols_A);
            blockSize = dim3(1, tileDim.y);
            sharedMemSize = linearDimToBytes(tileDim.y);
            kernels::compute_checksums<<<gridSize, blockSize, sharedMemSize, streams[0]>>>(dA, rows_A, cols_A, CHECKSUM_COMPUTE_COL);

            // calculate row checksums for B

            gridSize = dim3(1, ROWS_B);
            blockSize = dim3(tileDim.x, 1);
            sharedMemSize = linearDimToBytes(tileDim.x);
            kernels::compute_checksums<<<gridSize, blockSize, sharedMemSize, streams[1]>>>(dB, ROWS_B, cols_B, CHECKSUM_COMPUTE_ROW);

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
        }

        // compute the actual matrix multiplication as usual

        {
            ScopedTimer timer("matmul kernel", POST);

            gridSize = dim3(CEIL_DIV(COLS_C + 1, tileDim.x), CEIL_DIV(ROWS_C + 1, tileDim.y));
            blockSize = tileDim;
            sharedMemSize = 2 * dim2ToBytes(tileDim);
            kernels::tiled_matmul<<<gridSize, tileDim, sharedMemSize>>>(dA, dB, dC, rows_A + 1, cols_A, cols_B + 1);

            cudaDeviceSynchronize();
            CUDA_CHECK
        }

        if (globals::printMatrices)
        {
            // print dC (with checksums)
            float* Cec = matrix::alloc(ROWS_C + 1, COLS_C + 1, false);
            cudaMemcpy(Cec, dC, size_C_ec, cudaMemcpyDeviceToHost);
            matrix::print(Cec, ROWS_C + 1, COLS_C + 1, "C (w/ checksums)", HIGHLIGHT_LAST_ROW_AND_COL);
        }

        // send back result without checksums

        {
            ScopedTimer timer("C to host", POST);

            for (int r = 0; r < ROWS_C; ++r)
            {
                cudaMemcpyAsync(C + r * COLS_C, dC + r * (COLS_C + 1), COLS_C * sizeof(float), cudaMemcpyDeviceToHost, streams[r % numStreams]);
            }

            // destroy all streams

            for (int i = 0; i < numStreams; ++i)
            {
                cudaStreamSynchronize(streams[i]); // we have to synch the device but do the loop to destroy anyway, so we synch here the single streams
                cudaStreamDestroy(streams[i]);
            }
        }

        // cleanup

        delete[] streams;

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }
}
