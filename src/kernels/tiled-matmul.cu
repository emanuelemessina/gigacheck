#include "kernels.cuh"
#include <math.h>
#include <stdio.h>

#define CEIL_DIV(numerator, denominator) (int)((numerator + denominator - 1) / denominator)

// size_X is total number of elements of matrix X
#define size_A (rows_A * cols_A)
#define size_B (rows_B * cols_B)
#define size_C (rows_C * cols_C)

// size_X_bytes is the dimension in bytes of matrix X
#define size_A_bytes (size_A * sizeof(float))
#define size_B_bytes (size_B * sizeof(float))
#define size_C_bytes (size_C * sizeof(float))

#define rows_B cols_A
#define rows_C rows_A
#define cols_C cols_B

// The #elements to skip to go to the next line
#define nextRow_A cols_A
#define nextRow_B cols_B
#define nextRow_C cols_B
#define nextRow_tile colsPerTile_shared

// Row, col index of the thread within a block
// Correspond to row, col of the matrix cell within the tile
#define row threadIdx.y
#define col threadIdx.x

// Row, col of the block within the grid
// Correspond to row, col of the tile cell
#define tileRow blockIdx.y
#define tileCol blockIdx.x

// Dimension of the side of the square tile
#define tileSize blockDim.x // = blockDim.y

/**
 * @brief Get the index of a matrix cell, given the tile + (local) element coordinates
 *
 * @param[in]  tile_row   Row of the tile within the matrix
 * @param[in]  tile_col   Column of the tile within the matrix
 * @param[in]  el_row     Row of the element within the tile
 * @param[in]  el_col     Column of the element within the tile
 * @param[in]  tile_size  Dimension of the side of the square tile
 * @param[in]  mat_rows   Number of rows in the original matrix
 * @param[in]  mat_cols   Number of columns in the original matrix
 *
 * @return The index within the matrix array that corresponds to the element
 *	(el_row, el_col) within the tile (tile_row, tile_col). If the element is outside
 *	the matrix, it returns -1
 */
__device__ int getMatrixIdx(int tile_row, int tile_col, int el_row, int el_col, int tile_size, size_t mat_rows, size_t mat_cols)
{
    if (tile_row * tile_size + el_row >= mat_rows)
        return -1;

    if (tile_col * tile_size + el_col >= mat_cols)
        return -1;

    return tile_row * tile_size * mat_cols // select the correct row where the tile starts
           + tile_col * tile_size          // select the column of the tile start
           + el_row * mat_cols             // select the right row within the tile
           + el_col;                       // select the right cell within the tile row
}

/**
 * @brief Get the value of a matrix in a specific cell, given the tile + (local)
 *	element coordinates
 *
 * @param[in]   matrix     Matrix from which to read the value
 * @param[in]   tile_row   Row of the tile within the matrix
 * @param[in]   tile_col   Column of the tile within the matrix
 * @param[in]   el_row     Row of the element within the tile
 * @param[in]   el_col     Column of the element within the tile
 * @param[in]   tile_size  Dimension of the side of the square tile
 * @param[in]   mat_rows   Number of rows in the original matrix
 * @param[in]   mat_cols   Number of columns in the original matrix
 *
 * @return The element (el_row, el_col) within the tile (tile_row, tile_col), if exists.
 *	If the element is outside the matrix, it returns 0
 */
__device__ float getMatrixElement(float* matrix, int tile_row, int tile_col, int el_row, int el_col, int tile_size, size_t mat_rows, size_t mat_cols)
{
    int idx = getMatrixIdx(tile_row, tile_col, el_row, el_col, tile_size, mat_rows, mat_cols);
    return idx == -1 ? 0.0 : matrix[idx];
}

/**
 * @brief
 *
 * @param[in,out]  matrix     Matrix from which to read the value
 * @param[in]      val        The value to be inserted in the matrix cell
 * @param[in]      tile_row   Row of the tile within the matrix
 * @param[in]      tile_col   Column of the tile within the matrix
 * @param[in]      el_row     Row of the element within the tile
 * @param[in]      el_col     Column of the element within the tile
 * @param[in]      tile_size  Dimension of the side of the square tile
 * @param[in]      mat_rows   Number of rows in the original matrix
 * @param[in]      mat_cols   Number of columns in the original matrix
 */
__device__ void setMatrixElement(float* matrix, float val, int tile_row, int tile_col, int el_row, int el_col, int tile_size, size_t mat_rows, size_t mat_cols)
{
    int idx = getMatrixIdx(tile_row, tile_col, el_row, el_col, tile_size, mat_rows, mat_cols);
    if (idx != -1)
        matrix[idx] = val;
}

/**
 * @brief CUDA kernel to compute the product between two matrices, using the "tiled" shared
 * memory approach
 *
 * @param[in]   A       The first matrix to multiply
 * @param[in]   B       The second matrix to multiply
 * @param[out]  C       The matrix where to write the result
 * @param[in]   rows_A  #rows in A
 * @param[in]   cols_A  #cols in A
 * @param[in]   cols_B  #cols in B
 */
__global__ void tiled_matmul_kernel(float* A, float* B, float* C, size_t rows_A, size_t cols_A, size_t cols_B)
{
    // Each thread computes one value of C
    // Blocks organize themselves to load and share values in shared memory

    // colsPerTile_shared potentially updates tileSize to avoid it generating bank conflicts
    int colsPerTile_shared = tileSize + (tileSize % 2 == 0 ? 1 : 0);

    int numTiles = CEIL_DIV(cols_A, tileSize);

    float res = 0;

    extern __shared__ float shared_mem[];

    float* shared_A = shared_mem;
    float* shared_B = shared_mem + colsPerTile_shared * tileSize;

    for (int tile = 0; tile < numTiles; tile++)
    {

        // load your elements of the tile:
        //     - element (row, col) of tile (tileRow, tile) of matrix A
        //     - element (row, col) of tile (tile, tileCol) of matrix B
        shared_A[row * nextRow_tile + col] = getMatrixElement(A, tileRow, tile, row, col, tileSize, rows_A, cols_A);
        shared_B[row * nextRow_tile + col] = getMatrixElement(B, tile, tileCol, row, col, tileSize, rows_B, cols_B);

        __syncthreads();

        // compute product on the shared memory tile
        for (int el = 0; el < tileSize; el++)
            res += shared_A[row * nextRow_tile + el] * shared_B[col + el * nextRow_tile];

        __syncthreads();
    }

    // write final cell result
    setMatrixElement(C, res, tileRow, tileCol, row, col, tileSize, rows_C, cols_C);
}

namespace cuda
{

    /**
     * @brief Computes the product between two matrices A and B, exploiting the GPU
     * with the tiled multiplication algorithm
     *
     * @param[in]   A       The first matrix to mutiply
     * @param[in]   B       The second matrix to multiply
     * @param[out]  C       The result matrix
     * @param[in]   rows_A  #rows in A
     * @param[in]   cols_A  #cols in A
     * @param[in]   cols_B  #cols in B
     *
     */
    void tiled_matmul(float* A, float* B, float* C, size_t rows_A, size_t cols_A, size_t cols_B)
    {
        float *dA, *dB, *dC;

        cudaMalloc(&dA, size_A_bytes);
        cudaMalloc(&dB, size_B_bytes);
        cudaMalloc(&dC, size_C_bytes);

        cudaMemcpy(dA, A, size_A_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, size_B_bytes, cudaMemcpyHostToDevice);

        dim3 tiles(CEIL_DIV(cols_C, tileDim.x), CEIL_DIV(rows_C, tileDim.y));

        tiled_matmul_kernel<<<tiles, tileDim, 2 * (tileDim.x + 1) * tileDim.y * sizeof(float)>>>(dA, dB, dC, rows_A, cols_A, cols_B);
        cudaDeviceSynchronize();
        CUDA_CHECK

        cudaMemcpy(C, dC, size_C_bytes, cudaMemcpyDeviceToHost);

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }
}
