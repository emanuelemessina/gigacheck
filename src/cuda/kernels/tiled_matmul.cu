#include "kernels.cuh"

#include "inferred_matrix_sizes.h"

// Dimension of the side of the square tile
#define TILESIDE blockDim.x // = blockDim.y
// sidelength of the tile corrected to avoid bank conflicts
#define TILESIDE_BNK (TILESIDE + (TILESIDE % 2 == 0 ? 1 : 0))

// The #elements to skip to go to the next line
#define SKIPROWS_A cols_A
#define SKIPROWS_B cols_B
#define SKIPROWS_C cols_B
#define SKIPROWS_TILE TILESIDE_BNK

// Row, col index of the thread within a block
// Correspond to row, col of the matrix cell within the tile
#define TH_ROW threadIdx.y
#define TH_COL threadIdx.x

// Row, col of the block within the grid
// Correspond to row, col of the tile cell
#define TILE_IDY blockIdx.y
#define TILE_IDX blockIdx.x
// number of tiles on the x direction (horizontal)
#define NUM_TILES_X CEIL_DIV(cols_A, TILESIDE)

/**
 * @brief Get the index of a matrix cell, given the tile + (local) element coordinates
 *
 * @param[in]  tile_idy   Row of the tile within the matrix
 * @param[in]  tile_idx   Column of the tile within the matrix
 * @param[in]  el_row     Row of the element within the tile
 * @param[in]  el_col     Column of the element within the tile
 * @param[in]  tile_size  Dimension of the side of the square tile
 * @param[in]  mat_rows   Number of rows in the original matrix
 * @param[in]  mat_cols   Number of columns in the original matrix
 *
 * @return The index within the matrix array that corresponds to the element
 *	(el_row, el_col) within the tile (tile_idy, tile_idx). If the element is outside
 *	the matrix, it returns -1
 */
__device__ int getMatrixLinearIndex(size_t mat_rows, size_t mat_cols, int tile_size, int tile_idy, int tile_idx, int el_row, int el_col)
{
    if (tile_idy * tile_size + el_row >= mat_rows)
        return -1;

    if (tile_idx * tile_size + el_col >= mat_cols)
        return -1;

    return tile_idy * tile_size * mat_cols // select the correct row where the tile starts
           + tile_idx * tile_size          // select the column of the tile start
           + el_row * mat_cols             // select the right row within the tile
           + el_col;                       // select the right cell within the tile row
}

/**
 * @brief Get the value of a matrix in a specific cell, given the tile + (local)
 *	element coordinates
 *
 * @param[in]   matrix     Matrix from which to read the value
 * @param[in]   tile_idy   Row of the tile within the matrix
 * @param[in]   tile_idx   Column of the tile within the matrix
 * @param[in]   el_row     Row of the element within the tile
 * @param[in]   el_col     Column of the element within the tile
 * @param[in]   tile_size  Dimension of the side of the square tile
 * @param[in]   mat_rows   Number of rows in the original matrix
 * @param[in]   mat_cols   Number of columns in the original matrix
 *
 * @return The element (el_row, el_col) within the tile (tile_idy, tile_idx), if exists.
 *	If the element is outside the matrix, it returns 0
 */
__device__ float getMatrixElement(float* matrix, int mat_rows, int mat_cols, int tile_size, int tile_idy, int tile_idx, int el_row, int el_col)
{
    int idx = getMatrixLinearIndex(mat_rows, mat_cols, tile_size, tile_idy, tile_idx, el_row, el_col);
    return idx == -1 ? 0.0 : matrix[idx];
}

/**
 * @brief
 *
 * @param[in,out]  matrix     Matrix from which to read the value
 * @param[in]      val        The value to be inserted in the matrix cell
 * @param[in]      tile_idy   Row of the tile within the matrix
 * @param[in]      tile_idx   Column of the tile within the matrix
 * @param[in]      el_row     Row of the element within the tile
 * @param[in]      el_col     Column of the element within the tile
 * @param[in]      tile_size  Dimension of the side of the square tile
 * @param[in]      mat_rows   Number of rows in the original matrix
 * @param[in]      mat_cols   Number of columns in the original matrix
 *
 * @return true if the element was set, false if the element is outside the matrix
 */
__device__ bool setMatrixElement(float* matrix, int mat_rows, size_t mat_cols, int tile_size, int tile_idy, int tile_idx, int el_row, int el_col, float val)
{
    int idx = getMatrixLinearIndex(mat_rows, mat_cols, tile_size, tile_idy, tile_idx, el_row, el_col);
    if (idx == -1)
        return false;
    matrix[idx] = val;
    return true;
}

__global__ void kernels::tiled_matmul(float* A, float* B, float* C, int rows_A, int cols_A, int cols_B)
{
    // Each thread loads operand matrices values as tiles into shared memory
    // then computes the value of C corresponding to its position

    extern __shared__ float shared_mem[];

    float* shared_A = shared_mem;
    float* shared_B = shared_mem + TILESIDE_BNK * TILESIDE;

    float res = 0;

    // load tiles into shared memory

    for (int tile_idx = 0; tile_idx < NUM_TILES_X; tile_idx++) // this thread looks through the tiles on his same row horizontally
    {
        // load matrices elements into the shared tiles memory:
        //     - element (TH_ROW, TH_COL) of tile (TILE_IDY, tile_idx) of matrix A (one of A's tiles on the same grid row as this thread)
        //     - element (TH_ROW, TH_COL) of tile (tile_idx, TILE_IDX) of matrix B (the corresponding B tile, on the same grid column as the current row, and on the same grid row as the current column, because of how matrix multiplication works)
        shared_A[TH_ROW * SKIPROWS_TILE + TH_COL] = getMatrixElement(A, rows_A, cols_A, TILESIDE, TILE_IDY, tile_idx, TH_ROW, TH_COL);
        shared_B[TH_ROW * SKIPROWS_TILE + TH_COL] = getMatrixElement(B, ROWS_B, cols_B, TILESIDE, tile_idx, TILE_IDX, TH_ROW, TH_COL);

        // another thread on the same row as this one will look at the same tiles but load elements from his corresponding column in each tile chunk that he looks at
        // theads from other rows will do the same
        // we wait for all threads in the grid to finish loading their elements

        __syncthreads();

        // at this point the tile for this block is loaded and the current thread can compute the scalar product for its tile

        // compute product on the shared memory tile

        for (int el = 0; el < TILESIDE; el++)                                                     // this thread looks through the elements of the tile horizontally
            res += shared_A[TH_ROW * SKIPROWS_TILE + el] * shared_B[TH_COL + el * SKIPROWS_TILE]; // tileA[TH_ROW, el] * tileB[el, TH_COL] , with the tile this thread belongs to
        // this thread will sum all the products in his tile into the result
        // at the next iteration, this thread will update the result with the product of the next tile

        // wait for all the threads to compute their scalar products

        __syncthreads();

        // at the next iteration, this thread will load an element from an adjacent tile, all the thread combined will thus load the entire tile, and we can compute the scalar product for the next tile and add it to the result for this thread, and so on
    }

    // this thread will write his result into C, other threads will do the same for their positions
    setMatrixElement(C, ROWS_C, COLS_C, TILESIDE, TILE_IDY, TILE_IDX, TH_ROW, TH_COL, res);
}
