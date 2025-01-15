#include "matrix.h"

void matrix::print(float* mat, int rows, int cols, std::string&& name = "")
{
    if (!name.empty())
        printf("%s:\n", name.c_str());

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
            printf(globals::useIntValues ? "%2.0f " : "%f ", mat[r * cols + c]);
        printf("\n");
    }
    printf("\n");
}
