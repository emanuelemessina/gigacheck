#include "main.h"

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
    auto rb = ca;
    auto rc = ra;
    auto cc = cb;

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
