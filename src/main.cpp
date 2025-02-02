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
        .option({"v", "vanilla", OPTION_BOOL_UNSET, "Use vanilla matrix multiplication (no error checking)"})
        .option({"p", "print", OPTION_BOOL_UNSET, "Print debug info (matrices, checksums, calculations), do not use with big matrices"})
        .option({"c", "check", OPTION_BOOL_UNSET, "Check the GPU product correctness with the CPU (for debugging, do not use with big matrices)"})
        .option({"i", "ints", OPTION_BOOL_UNSET, "Use int values instead of floats (for visualization, still uses floats as underlying type)"})
        .option({"t", "tileside", 32, "Side of square tile blocks / Length of vector blocks"})
        .option({"m", "memory", OPTION_STRING_UNSET, "Max allowed GPU global memory (default: device max) as a human readable memsize string"})
        .option({"ra", "rows-a", 100, "A rows"})
        .option({"ca", "cols-a", 100, "A cols"})
        .option({"cb", "cols-b", 100, "B cols"})
        .option({"e", "errors", 0, "Introduced errors count, must be < max(ra,cb)"})
        .option({"ce", "collinear-errors", false, "If errors_count > 1, whether they should be arranged on the same axis (correctable)"});

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

    auto vanilla = cli.get("vanilla").getValue<bool>();
    auto check = cli.get("check").getValue<bool>();

    auto errors = cli.get("errors").getValue<int>();
    auto collinear_errors = cli.get("collinear-errors").getValue<bool>();

    auto ra = cli.get("rows-a").getValue<int>();
    auto ca = cli.get("cols-a").getValue<int>();
    auto cb = cli.get("cols-b").getValue<int>();

    result = programs::gigacheck(ra, ca, cb, vanilla, check, errors, collinear_errors);

    // end

    return result;
}
