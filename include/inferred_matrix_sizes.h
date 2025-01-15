#ifndef __INFERRED_MATRIX_SIZES_H_

#define __INFERRED_MATRIX_SIZES_H_

#define rb ca
#define rc ra
#define cc cb

#define ROWS_B cols_A
#define ROWS_C rows_A
#define COLS_C cols_B

// size_X is total number of elements of matrix X
#define SIZE_A (rows_A * cols_A)
#define SIZE_B (ROWS_B * cols_B)
#define SIZE_C (ROWS_C * COLS_C)

// size_X_bytes is the dimension in bytes of matrix X
#define SIZE_A_BYTES (SIZE_A * sizeof(float))
#define SIZE_B_BYTES (SIZE_B * sizeof(float))
#define SIZE_C_BYTES (SIZE_C * sizeof(float))

#endif
