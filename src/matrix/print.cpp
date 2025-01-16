#include "iomod.h"
#include "matrix.h"

void matrix::print(float* mat, int rows, int cols, std::string&& name, int flags)
{
    if (!name.empty())
        printf("%s:\n", name.c_str());

    for (int r = 0; r < rows; r++)
    {
        if (flags & HIGHLIGHT_LAST_ROW && r == rows - 1)
            std::cout << MAGENTA;

        for (int c = 0; c < cols; c++)
        {
            if (flags & HIGHLIGHT_LAST_COL && c == cols - 1)
                std::cout << MAGENTA;
            printf(globals::useIntValues ? "%2.0f " : "%f ", mat[r * cols + c]);
        }

        std::cout << RESET;
        printf("\n");
    }
    printf("\n");
}
