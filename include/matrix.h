// common includes
#include "globals.h"
#include "inferred_matrix_sizes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>

namespace matrix
{
    /**
     * @brief Allocates a new matrix
     *
     * @param[in]   rows        #rows
     * @param[in]   cols        #cols
     * @param[in]   initialize  If set to true, the matrix will be initialized with random values, else it will be zeroed
     *
     * @return The newly allocated matrix
     */
    float* alloc(int rows, int cols, bool initialize);

#define HIGHLIGHT_LAST_ROW 1
#define HIGHLIGHT_LAST_COL 2
#define HIGHLIGHT_LAST_ROW_AND_COL 3

    /**
     * @brief Prints a matrix
     *
     * @param[out]  mat   The matrix
     * @param[in]   rows  #rows
     * @param[in]   cols  #cols
     * @param[in]   name  The name of the matrix (optional)
     * @param highlight_xs x coords of the elements to highlight
     * @param highlight_xs y coords of the elements to highlight
     * @param highlight_count number of elements to highlight -> length of xs (and ys)
     */
    void print(float* mat, int rows, int cols, std::string&& name = "", int flags = 0, int* highlight_xs = nullptr, int* highlight_ys = nullptr, int highlight_count = 0);

    /**
     * @brief Checks that the matrix product C = A*B is correct. The function will
     * print the first error (if any), and then will stop
     *
     * @param[out]  A   The first matrix to be multiplied
     * @param[out]  B   The second matrix to be multiplied
     * @param[out]  C   The product to be checked
     * @param[in]   ra  #rows in A
     * @param[in]   ca  #cols in A
     * @param[in]   cb  #cols in B
     *
     * @return True if the product is correct, false otherwise
     */
    bool check_product(float* A, float* B, float* C, int ra, int ca, int cb);

    /**
     * @brief Decideds if the matrices should be split in blocks to fit the global memory, and if so how many
     *
     * @param[in]   rows_A                #rows of matrix A
     * @param[in]   cols_A                #cols of matrix A
     * @param[in]   cols_B                #cols of matrix B
     * @param[out]  num_split_common_dim  The number of blocks for dividing A's columns and B's rows into (= the dimensions multiplied together)
     * @param[out]  num_split_other_dim   The number of blocks for dividing A's rows and B's columns into (= the "other" dimensions)
     * @param[in]   strategy              Which strategy to use when matrices do not fit the GPU memory
     *
     */
    void choose_division(int rows_A, int cols_A, int cols_B, int* num_split_common_dim, int* num_split_other_dim, Strategy strategy);
}
