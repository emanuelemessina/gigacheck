#include "inferred_matrix_sizes.h"
#include "matrix.h"

void matrix::print(float* mat, int rows, int cols)
{
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
#ifdef TEST_intValues
            printf("%2.0f ", mat[r * cols + c]);
#else
            printf("%f ", mat[r * cols + c]);
#endif
        printf("\n");
    }
    printf("\n");
}
