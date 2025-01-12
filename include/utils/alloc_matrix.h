#ifndef __ALLOC_MATRIX_H_

#define __ALLOC_MATRIX_H_

#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Allocates a new matrix, filling it with random numbers if needed
 *
 * @param[in]   rows        #rows
 * @param[in]   cols        #cols
 * @param[in]   initialize  Whether it should be filled with random numbers or not
 *
 * @return The newly allocated matrix
 */
float* alloc(size_t rows, size_t cols, bool initialize);

#endif
