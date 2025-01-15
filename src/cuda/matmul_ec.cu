#include "inferred_matrix_sizes.h"
#include "kernels.cuh"
#include "matrix.h"

namespace cuda
{
    void matmul_ec(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B)
    {
        // alloc the result matrix

        C = matrix::alloc(ROWS_C, COLS_C, false);

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

        // copy matrices to device

        cudaMemcpy(dA, A, size_A_ec, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, size_B_ec, cudaMemcpyHostToDevice);

        // define threads organization

        dim3 gridSize(CEIL_DIV(COLS_C + 1, tileDim.x), CEIL_DIV(ROWS_C + 1, tileDim.y)); // we actually don't need the +1 for the checksums but it's no big deal in exchange for cleaner code
        dim3 blockSize = tileDim;
        int sharedMemSize = dim3ToBytes(blockSize);

        // calculate checksums in parallel on different streams

        // calculate col checksums for A
        kernels::compute_checksums<<<gridSize, blockSize, sharedMemSize>>>(dA, rows_A, cols_A, CHECKSUM_COMPUTE_COL);
        // calculate row checksums for B
        kernels::compute_checksums<<<gridSize, blockSize, sharedMemSize, streams[0]>>>(dB, ROWS_B, cols_B, CHECKSUM_COMPUTE_ROW);

        cudaDeviceSynchronize();
        CUDA_CHECK

        // compute the actual matrix multiplication as usual
        kernels::tiled_matmul<<<gridSize, blockSize, 2 * sharedMemSize>>>(dA, dB, dC, rows_A + 1, cols_A, cols_B + 1);
        cudaDeviceSynchronize();
        CUDA_CHECK

        // send back result without checksums

        int rowsPerStream = ROWS_C / numStreams; // how many rows each stream should process
        int remainingRows = ROWS_C % numStreams; // remainder of rows to distribute among streams

        int startRow = 0;
        for (int s = 0; s < numStreams; ++s)
        {
            int rowsToProcess = rowsPerStream + (s < remainingRows ? 1 : 0); // Distribute remaining rows among streams

            for (int i = 0; i < rowsToProcess; ++i) // issue copy commands for each row
            {
                int rowIdx = startRow + i;
                cudaMemcpyAsync(C + rowIdx * COLS_C, dC + rowIdx * (COLS_C + 1), COLS_C * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
            }

            // update startRow for the next stream
            startRow += rowsToProcess;
        }

        // synchronize all streams
        for (int i = 0; i < numStreams; ++i)
        {
            cudaStreamSynchronize(streams[i]);
            cudaStreamDestroy(streams[i]);
        }

        // cleanup

        delete[] streams;

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }
}