#include "cli.h"
#include "kernels.cuh"
#include <iostream>
#include <map>
#include <ranges>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#define rb ca
#define rc ra
#define cc cb

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
void check(float* A, float* B, float* C, int ra, int ca, int cb)
{
    for (int r = 0; r < rc; r++)
    {
        for (int c = 0; c < cc; c++)
        {
            float val = 0;
            for (int k = 0; k < ca; k++)
                val += A[r * ca + k] * B[c + k * cb];

            if (C[c + cb * r] != val)
            {
#ifdef TEST_intValues
                printf("Wrong product (first error at (%d, %d): %2.0f instead of %2.0f)\n", r, c, C[c + cb * r], val);
#else
                printf("Wrong product (first error at (%d, %d): %f instead of %f)\n", r, c, C[c + cb * r], val);
#endif
                return;
            }
        }
    }
}

/**
 * @brief Allocates a new matrix, filling it with random numbers if needed
 *
 * @param[in]   rows        #rows
 * @param[in]   cols        #cols
 * @param[in]   initialize  Whether it should be filled with random numbers or not
 *
 * @return The newly allocated matrix
 */
float* alloc(size_t rows, size_t cols, bool initialize)
{
    float* tmp = (float*)malloc(rows * cols * sizeof(float));

    if (initialize)
    {
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
#ifdef TEST_intValues
                tmp[r * cols + c] = (float)(rand() % 5);
#else
                tmp[r * cols + c] = (float)rand() / RAND_MAX;
#endif
    }

    return tmp;
}

/**
 * @brief Prints a matrix
 *
 * @param[out]  mat   The matrix
 * @param[in]   rows  #rows
 * @param[in]   cols  #cols
 *
 */
void print(float* mat, size_t rows, size_t cols)
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

int main(int argc, char* argv[])
{
    // cli definition

    CLI cli = CLI{"GIGACHECK"};

    cli
        .option({"h", "help", OPTION_INT_UNSET, "Help"})
        .option({"ra", "rows-a", 1000, "A rows"})
        .option({"ca", "cols-a", 1000, "A cols"})
        .option({"cb", "cols-b", 1000, "B cols"})
        .option({"r", "redundancy", 0, "Redundancy Level"})
        .option({"e", "errors", 0, "Introduced errors amount"});

    cli.parse(argc, argv);

    auto help = cli.get("help");

    if (help.isSet())
    {
        cli.help();
        return 0;
    }

    int result = 0;

    auto redundancy = cli.get("redundancy").getValue<int>();
    auto errors = cli.get("errors").getValue<int>();
    auto ra = cli.get("rows-a").getValue<int>();
    auto ca = cli.get("cols-a").getValue<int>();
    auto cb = cli.get("cols-b").getValue<int>();

    float* A = alloc(ra, ca, true);
    float* B = alloc(rb, cb, true);
    float* C = alloc(rc, cc, false);

    cuda::tiled_matmul(A, B, C, ra, ca, cb);
    printf("Computation finished\n");

    check(A, B, C, ra, ca, cb);
    printf("Check finished\n");

    print(A, ra, ca);
    print(B, rb, cb);
    print(C, rc, cc);

    // result = launch
    return result;
}
