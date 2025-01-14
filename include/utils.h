#include "parameters.h"

namespace utils
{
    namespace matrix
    {
        /**
         * @brief Allocates a new matrix, filling it with random numbers if needed
         *
         * @param[in]   rows        #rows
         * @param[in]   cols        #cols
         * @param[in]   initialize  Whether it should be filled with random numbers or not
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
        void print(float* mat, int rows, int cols);
    }

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
     */
    void check_product(float* A, float* B, float* C, int ra, int ca, int cb);
}