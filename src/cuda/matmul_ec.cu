#include "cuda.cuh"
#include "edc.cuh"
#include "globals.h"
#include "inferred_matrix_sizes.h"
#include "kernels.cuh"
#include "matrix.h"
#include "timer.h"

/**
 * @brief Copies a matrix (or a portion of it) to CUDA memory
 *
 * It is able to handle also the copy of a block of a matrix.
 *
 * Assume we have to copy block (i, j) of a matrix HxW, divided into NxM blocks.
 *
 * The block (therefore also the GPU matrix) will have H/N rows and W/M cols.
 *
 * To copy the correct block, we have to start at an initial offset that includes:
 * - a delta of i * W * H/N, to select the correct row of blocks
 * - an extra delta of j * W/M to select the correct block within the row
 *
 * Moreover, if  M != 1 the next row of the matrix is not immediately after the previous one,
 * but it starts W cells after the previous one starts (next_row_offset)
 *
 * @param[in]   matrix                The original, host matrix
 * @param[out]  dst                   The GPU allocated memory where to copy
 * @param[in]   rows                  The number of rows that should be copied
 * @param[in]   cols                  The number of columns that should be copied
 * @param[in]   initial_offset        How much of the original matrix must be skipped at the beginning
 * @param[in]   next_row_offset       How much of the original matrix must be skipped to transition to the next row
 * @param[in]   leave_cell_after_row  If a cell should be left empty after each row (== copying B or a block of B)
 * @param[in]   stream                The stream on which to work
 */
void cp_matrix_to_CUDA(float* matrix, float* dst, int rows, int cols, int initial_offset, int next_row_offset, bool leave_cell_after_row, cudaStream_t stream)
{
    matrix += initial_offset;
    for (int i = 0; i < rows; i++)
    {
        cudaMemcpyAsync(dst, matrix, cols * sizeof(float), cudaMemcpyHostToDevice, stream);

        matrix += next_row_offset;
        dst += cols + (leave_cell_after_row ? 1 : 0);
    }
}

void print_CUDA_matrix(float* mat, int rows, int cols, const char* name, int flags, int* highlight_xs, int* highlight_ys, int highlight_count)
{
    float* mat_host = matrix::alloc(rows, cols, false);
    cudaMemcpy(mat_host, mat, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    matrix::print(mat_host, rows, cols, name, flags, highlight_xs, highlight_ys, highlight_count);
    free(mat_host);
    CUDA_CHECK
}

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
        cudaStream_t stream_A;
        cudaStream_t stream_B;
        // cudaStream_t stream_Abis;
        // cudaStream_t stream_Bbis;

        cudaStreamCreate(&stream_A);
        cudaStreamCreate(&stream_B);
        // cudaStreamCreate(&stream_Abis);
        // cudaStreamCreate(&stream_Bbis);

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
            ScopedTimer timer("A,B to device + compute input checksums", POST);

            // copy A to device
            cp_matrix_to_CUDA(A, dA, rows_A, cols_A, 0, cols_A, false, stream_A);

            // calculate col checksums for A
            gridDim = dim3(cols_A);
            blockDim = dim3(1, tileDim.y);
            sharedMemSize = linearDimToBytes(tileDim.y);
            kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, stream_A>>>(dA, rows_A, cols_A, ReductionDirection::ALONG_COL);

            // copy B to device
            cp_matrix_to_CUDA(B, dB, ROWS_B, cols_B, 0, cols_B, true, stream_B);

            // calculate row checksums for B
            gridDim = dim3(1, ROWS_B);
            blockDim = dim3(tileDim.x, 1);
            sharedMemSize = linearDimToBytes(tileDim.x);
            kernels::compute_checksums<<<gridDim, blockDim, sharedMemSize, stream_B>>>(dB, ROWS_B, cols_B, ReductionDirection::ALONG_ROW);

            cudaDeviceSynchronize();
            CUDA_CHECK
        }

        // print dA and dB (with checksums)
        if (globals::debugPrint)
        {
            print_CUDA_matrix(dA, rows_A + 1, cols_A, "A (w/ column checksum)", HIGHLIGHT_LAST_ROW, NULL, NULL, 0);
            print_CUDA_matrix(dB, ROWS_B, cols_B + 1, "B (w/ row checksum)", HIGHLIGHT_LAST_COL, NULL, NULL, 0);
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
            print_CUDA_matrix(dC, ROWS_C + 1, COLS_C + 1, "C (w/ column checksum)", HIGHLIGHT_LAST_ROW_AND_COL, NULL, NULL, 0);

        // compute control checksums after mul
        {
            ScopedTimer timer("compute control checksums", POST);

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
            std::vector<int> zeros(errors_count, 0);
            print_CUDA_matrix(d_rc_control, ROWS_C + 1, 1, "C control row checksum", HIGHLIGHT_LAST_COL, zeros.data(), error_ys, errors_count);
            print_CUDA_matrix(d_cc_control, 1, COLS_C + 1, "C control column checksum", HIGHLIGHT_LAST_ROW, error_xs, zeros.data(), errors_count);
        }

        // edc

        EDCResult edc_res;

        {
            ScopedTimer timer("error detection (+ correction)", POST);

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

        cudaStreamCreate(&stream_A);
        cudaStreamCreate(&stream_B);

        cudaStreamSynchronize(stream_A);
        cudaStreamSynchronize(stream_B);

        cudaStreamDestroy(stream_A);
        cudaStreamDestroy(stream_B);

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
