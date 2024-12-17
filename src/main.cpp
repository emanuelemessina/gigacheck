#include "cli.h"
#include <iostream>
#include <kernels.cuh>
#include <map>
#include <ranges>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// #define TEST_INT_MATRICES
// #define TEST_PRINT_MATRICES
// #define TEST_AVOID_RANDOM

void print_mat(float** mat, int rows, int cols)
{
    for (int row = 0; row < rows; row++)
    {
        printf("|");
        for (int col = 0; col < cols; col++)
            printf(" %5.02f", mat[row][col]);
        printf(" |\n");
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
    // cli definition

#ifndef TEST_AVOID_RANDOM
    srand(time(NULL));
#endif

    CLI cli = CLI{"GIGACHECK"};

    cli
        .option({"h", "help", OPTION_INT_UNSET, "Help"})
        .option({"ra", "rows-a", 1000, "A rows"})
        .option({"ca", "cols-a", 1000, "A cols"})
        .option({"rb", "cols-b", 1000, "B cols"})
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

    auto rb = ca;
    auto rc = ra;
    auto cc = cb;

    float** A_mat = (float**)malloc(ra * sizeof(float*));
    float** B_mat = (float**)malloc(rb * sizeof(float*));

    for (int row = 0; row < ra; row++)
    {
        A_mat[row] = (float*)malloc(ca * sizeof(float));
        for (int col = 0; col < ca; col++)
        {
#ifdef TEST_INT_MATRICES
            A_mat[row][col] = (float)(rand() % 5);
#else
            A_mat[row][col] = (float)rand() / RAND_MAX;
#endif
        }
    }

    for (int row = 0; row < rb; row++)
    {
        B_mat[row] = (float*)malloc(cb * sizeof(float));
        for (int col = 0; col < cb; col++)
        {
#ifdef TEST_INT_MATRICES
            B_mat[row][col] = (float)(rand() % 5);
#else
            B_mat[row][col] = (float)rand() / RAND_MAX;
#endif
        }
    }

    float** C_mat = cuda::strassen_parallelizing_recursion(A_mat, B_mat, ra, ca, cb);

#ifdef TEST_PRINT_MATRICES
    print_mat(A_mat, ra, ca);
    print_mat(B_mat, rb, cb);
    print_mat(C_mat, rc, cc);
#endif

    for (int row = 0; row < ra; row++)
        free(A_mat[row]);
    for (int row = 0; row < rb; row++)
        free(B_mat[row]);
    for (int row = 0; row < rc; row++)
        free(C_mat[row]);

    free(A_mat);
    free(B_mat);
    free(C_mat);

    // result = launch
    return result;
}
