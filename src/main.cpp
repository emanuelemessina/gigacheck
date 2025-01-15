#include "cli.h"
#include "globals.h"
#include "programs.h"

int main(int argc, char* argv[])
{
    // cli definition

    CLI cli = CLI{"GIGACHECK"};

    cli
        .option({"h", "help", OPTION_INT_UNSET, "Help"})
        .option({"v", "vanilla", OPTION_BOOL_UNSET, "Use vanilla matrix multiplication (no error checking)"})
        .option({"p", "print", OPTION_BOOL_UNSET, "Print the matrices (for debugging, do not use with big matrices)"})
        .option({"c", "check", OPTION_BOOL_UNSET, "Check the GPU product correctness with the CPU (for debugging, do not use with big matrices)"})
        .option({"i", "ints", OPTION_BOOL_UNSET, "Use int values instead of floats (for visualization, still uses floats as underlying type)"})
        .option({"ra", "rows-a", 100, "A rows"})
        .option({"ca", "cols-a", 100, "A cols"})
        .option({"cb", "cols-b", 100, "B cols"})
        .option({"r", "redundancy", 0, "Redundancy Level"})
        .option({"e", "errors", 0, "Introduced errors amount"});

    // cli parse

    cli.parse(argc, argv);

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
    globals::printMatrices = print;
    globals::useIntValues = ints;

    auto vanilla = cli.get("vanilla").getValue<bool>();
    auto check = cli.get("check").getValue<bool>();

    auto redundancy = cli.get("redundancy").getValue<int>();
    auto errors = cli.get("errors").getValue<int>();

    auto ra = cli.get("rows-a").getValue<int>();
    auto ca = cli.get("cols-a").getValue<int>();
    auto cb = cli.get("cols-b").getValue<int>();

    result = programs::gigacheck(ra, ca, cb, vanilla, check);

    // end

    return result;
}
