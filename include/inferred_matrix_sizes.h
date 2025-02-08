#pragma once

// inferred dimensions for matrix multiplication (forced mathematically, defined for simplicity)

// to use in c++
#define rb ca
#define rc ra
#define cc cb
// to use in cuda
#define ROWS_B cols_A
#define ROWS_C rows_A
#define COLS_C cols_B

// inferred sizes for matrices A, B and C, to use in cuda for simplicity

// SIZE_X is the total number of elements of matrix X
#define SIZE_A (rows_A * cols_A)
#define SIZE_B (ROWS_B * cols_B)
#define SIZE_C (ROWS_C * COLS_C)

// SIZE_X_BYTES is the dimension in bytes of matrix X
#define SIZE_A_BYTES (SIZE_A * sizeof(float))
#define SIZE_B_BYTES (SIZE_B * sizeof(float))
#define SIZE_C_BYTES (SIZE_C * sizeof(float))

#define MAX_BLOCK_ROWS_B max_block_cols_A
#define MAX_BLOCK_ROWS_C max_block_rows_A
#define MAX_BLOCK_COLS_C max_block_cols_B
