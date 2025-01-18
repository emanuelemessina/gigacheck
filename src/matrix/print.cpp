#include "iomod.h"
#include "matrix.h"

void matrix::print(float* mat, int rows, int cols, std::string&& name, int flags, int* highlight_xs, int* highlight_ys, int highlight_count)
{
    if (!name.empty())
        printf("%s:\n", name.c_str());

    int highlighted_count = 0;

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            if (flags & HIGHLIGHT_LAST_ROW && r == rows - 1)
                std::cout << MAGENTA;

            if (flags & HIGHLIGHT_LAST_COL && c == cols - 1)
                std::cout << MAGENTA;

            if (highlight_xs != nullptr && highlight_ys != nullptr && highlighted_count < highlight_count)
            {
                if (c == highlight_xs[highlighted_count] && r == highlight_ys[highlighted_count])
                {
                    highlighted_count++;
                    std::cout << RED;
                }
            }

            printf(globals::useIntValues ? "%2.0f " : "%f ", mat[r * cols + c]);

            std::cout << RESET;
        }

        printf("\n");
    }

    printf("\n");
}
