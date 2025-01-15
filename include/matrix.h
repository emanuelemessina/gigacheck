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

    /**
     * @brief Prints a matrix
     *
     * @param[out]  mat   The matrix
     * @param[in]   rows  #rows
     * @param[in]   cols  #cols
     *
     */
    void print(float* mat, int rows, int cols, std::string&& name);

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
}