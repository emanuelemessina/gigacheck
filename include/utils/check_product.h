#ifndef __CHECK_PRODUCT_H_

#define __CHECK_PRODUCT_H_

#include "parameters.h"

#include <stdio.h>
#include <stdlib.h>

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
void check(float* A, float* B, float* C, int ra, int ca, int cb);

#endif
