#include "cli.h"
#include "cuda.cuh"
#include "globals.h"
#include "iomod.h"
#include "memsize_string.h"
#include "programs.h"
#include <algorithm>

int main(int argc, char* argv[])
{
    // cli definition

    CLI cli = CLI{"GIGACHECK"};

    cli
        .option({"h", "help", OPTION_INT_UNSET, "Help"})
        .option({"n", "no-edc", OPTION_BOOL_UNSET, "Use vanilla matrix multiplication, disable fault tolerance (no error detection and correction)"})
        .option({"p", "print", OPTION_BOOL_UNSET, "Print debug info (matrices, checksums, calculations), do not use with big matrices"})
        .option({"v", "cpu-verify", OPTION_BOOL_UNSET, "Verify the GPU product correctness with the CPU (for debugging, do not use with big matrices)"})
        .option({"i", "ints", OPTION_BOOL_UNSET, "Use int values instead of floats (for visualization, still uses floats as underlying type)"})
        .option({"t", "tileside", 32, "Side of square tile blocks / Length of vector blocks"})
        .option({"m", "memory", OPTION_STRING_UNSET, "Max allowed GPU global memory (default: device max) as a human readable memsize string"})
        .option({"ra", "rows-a", 100, "A rows"})
        .option({"ca", "cols-a", 100, "A cols"})
        .option({"cb", "cols-b", 100, "B cols"})
        .option({"e", "errors", 0, "Introduced errors count, must be < max(ra,cb)"})
        .option({"ce", "collinear-errors", false, "If errors_count > 1, whether they should be arranged on the same axis (correctable)"})
        .option({"s", "strategy", 1, "Which strategy to use: \n                               - 1 (default): 3 matrices (A, B, C), with no buffering\n                               - 2: 5 matrices (A, A', B, B', C) to pre-load the next A, B while multiplying\n                               - 3: 6 matrices (A, A', B, B', C, C') to save C while doing the next multiplication\n                               - 4: 6 matrices (A, A', B, B', C, C') to compute two multiplications in parallel (C = AB; C' = A'B')"});

    // cli parse

    if (!cli.parse(argc, argv))
        return 1;

    auto help = cli.get("help");

    if (help.isSet())
    {
        cli.help();
        return 0;
    }

    // launch program(s)

    int result = 0;

    // gigacheck

    auto print = cli.get("print").getValue<bool>();
    auto ints = cli.get("ints").getValue<bool>();
    auto tileside = cli.get("tileside").getValue<int>();
    cuda::MulStrategy strategy = (cuda::MulStrategy)(cli.get("strategy").getValue<int>() - 1);
    globals::debugPrint = print;
    globals::useIntValues = ints;
    globals::tileSide = tileside;

    auto memory = cli.get("memory").getValue<std::string>();
    cuda::Info info = cuda::getInfo();
    try
    {
        globals::maxGlobalMem = memory.empty() ? info.totalGlobalMem : parseMemSizeString(memory);
    }
    catch (const std::invalid_argument& e)
    {
        CERR << RED << "Error: " << e.what() << ENDL;
        return 1;
    }

    auto noEDC = cli.get("no-edc").getValue<bool>();
    globals::noEDC = noEDC;

    auto verify = cli.get("cpu-verify").getValue<bool>();

    auto errors = cli.get("errors").getValue<int>();
    auto collinear_errors = cli.get("collinear-errors").getValue<bool>();

    auto ra = cli.get("rows-a").getValue<int>();
    auto ca = cli.get("cols-a").getValue<int>();
    auto cb = cli.get("cols-b").getValue<int>();

    result = programs::gigacheck(ra, ca, cb, verify, errors, collinear_errors, strategy);

    // end

    return result;
}
