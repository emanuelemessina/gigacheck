#include "kernels.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define M1A (write_buf_local + 0 * block_s_size)[row * block_l_size + col]
#define M1B (write_buf_local + 1 * block_s_size)[row * block_l_size + col]
#define M2A (write_buf_local + 2 * block_s_size)[row * block_l_size + col]
#define M2B (write_buf_local + 3 * block_s_size)[row * block_l_size + col]
#define M3A (write_buf_local + 4 * block_s_size)[row * block_l_size + col]
#define M3B (write_buf_local + 5 * block_s_size)[row * block_l_size + col]
#define M4A (write_buf_local + 6 * block_s_size)[row * block_l_size + col]
#define M4B (write_buf_local + 7 * block_s_size)[row * block_l_size + col]
#define M5A (write_buf_local + 8 * block_s_size)[row * block_l_size + col]
#define M5B (write_buf_local + 9 * block_s_size)[row * block_l_size + col]
#define M6A (write_buf_local + 10 * block_s_size)[row * block_l_size + col]
#define M6B (write_buf_local + 11 * block_s_size)[row * block_l_size + col]
#define M7A (write_buf_local + 12 * block_s_size)[row * block_l_size + col]
#define M7B (write_buf_local + 13 * block_s_size)[row * block_l_size + col]

#define M1 (read_buf_local + 0 * block_s_size)[row * block_l_size + col]
#define M2 (read_buf_local + 1 * block_s_size)[row * block_l_size + col]
#define M3 (read_buf_local + 2 * block_s_size)[row * block_l_size + col]
#define M4 (read_buf_local + 3 * block_s_size)[row * block_l_size + col]
#define M5 (read_buf_local + 4 * block_s_size)[row * block_l_size + col]
#define M6 (read_buf_local + 5 * block_s_size)[row * block_l_size + col]
#define M7 (read_buf_local + 6 * block_s_size)[row * block_l_size + col]

#define A11 (read_buf_local)[row * next_row_off + col]
#define A12 (read_buf_local + block_l_size)[row * next_row_off + col]
#define A21 (read_buf_local + 2 * block_s_size)[row * next_row_off + col]
#define A22 (read_buf_local + 2 * block_s_size + block_l_size)[row * next_row_off + col]

#define B11 (read_buf_local + 4 * block_s_size)[row * next_row_off + col]
#define B12 (read_buf_local + block_l_size + 4 * block_s_size)[row * next_row_off + col]
#define B21 (read_buf_local + 2 * block_s_size + 4 * block_s_size)[row * next_row_off + col]
#define B22 (read_buf_local + 2 * block_s_size + block_l_size + 4 * block_s_size)[row * next_row_off + col]

#define C11 (write_buf_local)[row * next_row_off + col]
#define C12 (write_buf_local + block_l_size)[row * next_row_off + col]
#define C21 (write_buf_local + 2 * block_s_size)[row * next_row_off + col]
#define C22 (write_buf_local + 2 * block_s_size + block_l_size)[row * next_row_off + col]

#define switch_buffers()   \
    float* tmp = read_buf; \
    read_buf = write_buf;  \
    write_buf = tmp

unsigned uint_pow(unsigned base, unsigned exp)
{
    unsigned result = 1;
    while (exp)
    {
        if (exp % 2)
            result *= base;
        exp /= 2;
        base *= base;
    }
    return result;
}

__global__ void strassen(float* buf1, float* buf2, int l_size)
{
    int thread = blockIdx.x * blockDim.x + threadIdx.x;

    float* read_buf = buf1;
    float* write_buf = buf2;

    int num_threads = 1;

    // FORWARD PASS: compute at each step the operands for M1, M2, ...
    for (int cur_l_size = l_size; cur_l_size != 1; cur_l_size /= 2)
    {
        int cur_s_size = cur_l_size * cur_l_size;

        int block_l_size = cur_l_size / 2;
        int block_s_size = block_l_size * block_l_size;

        int next_row_off = 2 * block_l_size;

        float* read_buf_local = read_buf + thread * 2 * cur_s_size;
        float* write_buf_local = write_buf + thread * 14 * cur_s_size / 4;

        if (thread < num_threads)
        {
            for (int row = 0; row < block_l_size; row++)
            {
                for (int col = 0; col < block_l_size; col++)
                {
                    // Axx, Bxx and MxA/B are macros that access the right element
                    // of the buffer in order to access Axx[row][col], ...
                    M1A = A11 + A22;
                    M1B = B11 + B22;

                    M2A = A21 + A22;
                    M2B = B11;

                    M3A = A11;
                    M3B = B12 - B22;

                    M4A = A22;
                    M4B = B21 - B11;

                    M5A = A11 + A12;
                    M5B = B22;

                    M6A = A21 - A11;
                    M6B = B11 + B12;

                    M7A = A12 - A22;
                    M7B = B21 + B22;
                }
            }
        }

        num_threads *= 7;

        __syncthreads();
        switch_buffers();
    }

    // MULTIPLICATION PASS: when M1, M2, ... are 1x1, multiply them
    if (thread < num_threads)
        write_buf[thread] = read_buf[2 * thread] * read_buf[2 * thread + 1];
    __syncthreads();
    switch_buffers();

    // BACKWARD PASS: use the C11, ... values to compute higher level products
    for (int cur_l_size = 2; cur_l_size <= l_size; cur_l_size *= 2)
    {
        num_threads /= 7;

        int cur_s_size = cur_l_size * cur_l_size;

        int block_l_size = cur_l_size / 2;
        int block_s_size = block_l_size * block_l_size;

        int next_row_off = 2 * block_l_size;

        float* read_buf_local = read_buf + thread * 7 * cur_s_size / 4;
        float* write_buf_local = write_buf + thread * cur_s_size;

        if (thread < num_threads)
        {
            for (int row = 0; row < block_l_size; row++)
            {
                for (int col = 0; col < block_l_size; col++)
                {
                    C11 = M1 + M4 - M5 + M7;
                    C12 = M3 + M5;
                    C21 = M2 + M4;
                    C22 = M1 - M2 + M3 + M6;
                }
            }
        }

        __syncthreads();
        switch_buffers();
    }
}

namespace cuda
{
    float** strassen_parallelizing_recursion(float** A, float** B, int rowsA, int colsA, int colsB)
    {
        const int rowsB = colsA;
        const int rowsC = rowsA;
        const int colsC = colsB;

        // Compute dimension of square matrix that contains both A and B (separately)
        int N = 1;
        int log = 0;

        while (N < rowsA || N < colsA || N < colsB)
        {
            N *= 2;
            log += 1;
        }

        // Compute buffer size
        const int BUF_SIZE = 2 * uint_pow(7, log);

        // Allocate buffers
        float *buf1, *buf2;
        cudaMalloc(&buf1, BUF_SIZE * sizeof(float));
        cudaMalloc(&buf2, BUF_SIZE * sizeof(float));

        // Fill with zeros the part of the buffer where the matrices should be written
        // This ensures that the cells not overwritten by matrix values are 0
        cudaMemset(&buf1, 0, 2 * N * N);

        // Copy matrices to buffer
        for (int row = 0; row < rowsA; row++)
            cudaMemcpy((buf1 + N * row), A[row], colsA * sizeof(float), cudaMemcpyHostToDevice);
        for (int row = 0; row < rowsB; row++)
            cudaMemcpy((buf1 + N * (row + N)), B[row], colsB * sizeof(float), cudaMemcpyHostToDevice);

        const int num_threads = uint_pow(7, log);

        const int TPB = MIN(num_threads, 1024);
        const int BPG = (num_threads + TPB - 1) / TPB;

        strassen<<<BPG, TPB>>>(buf1, buf2, N);

        float** C = (float**)malloc(rowsC * sizeof(float*));
        for (int row = 0; row < rowsC; row++)
        {
            C[row] = (float*)malloc(colsC * sizeof(float));
            cudaMemcpy(C[row], (buf2 + N * row), colsC * sizeof(float), cudaMemcpyDeviceToHost);
        }

        cudaFree(buf1);
        cudaFree(buf2);

        return C;
    }
}
