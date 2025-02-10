#include "cuda.cuh"
#include "iomod.h"
#include "matrix.h"
#include <algorithm>
#include <vector>

void matrix::print(float* mat, int rows, int cols, std::string&& name, int flags, int* highlight_xs, int* highlight_ys, int highlight_count)
{
    if (flags & IS_DEVICE_MAT)
    {
        float* mat_host = matrix::alloc(rows, cols, false);
        cudaMemcpy(mat_host, mat, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
        matrix::print(mat_host, rows, cols, name.c_str(), flags & ~IS_DEVICE_MAT, highlight_xs, highlight_ys, highlight_count);
        free(mat_host);
        CUDA_CHECK
        return;
    }

    if (!name.empty())
        printf("%s:\n", name.c_str());

    std::vector<std::pair<int, int>> highlights(highlight_count);
    auto highlight_cursor = highlights.cbegin();

    if (highlight_xs != nullptr && highlight_ys != nullptr && highlight_count > 0) // sort the highlights
    {
        for (int i = 0; i < highlight_count; i++)
        {
            highlights[i] = std::pair<int, int>(highlight_ys[i], highlight_xs[i]);
        }
        std::sort(highlights.begin(), highlights.end()); // sort by row, then column
    }

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            if (flags & HIGHLIGHT_LAST_ROW && r == rows - 1)
                COUT << MAGENTA;

            if (flags & HIGHLIGHT_LAST_COL && c == cols - 1)
                COUT << MAGENTA;

            while (highlight_cursor != highlights.cend() && r > (*highlight_cursor).first)
            {
                highlight_cursor++; // advance cursor for row-collinear errors
            }

            bool highlight_row_match = highlight_cursor != highlights.cend() && r == (*highlight_cursor).first;

            if (highlight_row_match && c == (*highlight_cursor).second)
            {
                highlight_cursor++;
                COUT << RED;
            }

            COUT << FMT_FLOAT(mat[r * cols + c]) << " " << RESET;
        }

        printf("\n");
    }

    printf("\n");
}
