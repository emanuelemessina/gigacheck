#ifndef __PRINT_MATRIX_H_

#define __PRINT_MATRIX_H_

#include "parameters.h"

#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Prints a matrix
 *
 * @param[out]  mat   The matrix
 * @param[in]   rows  #rows
 * @param[in]   cols  #cols
 *
 */
void print(float* mat, size_t rows, size_t cols);

#endif
