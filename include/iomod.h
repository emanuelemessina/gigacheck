#pragma once

#include "globals.h"
#include <iomanip>
#include <iostream>

// colors

#define RED "\033[31m"
#define GREEN "\033[32m"
#define GRAY "\033[90m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN "\033[36m"
#define WHITE "\033[37m"

// modifiers

#define BOLD "\033[1m"
#define UNDERLINE "\033[4m"
#define REVERSED "\033[7m"
#define HIDDEN "\033[8m"

// control

#define CLEAR "\033[2J"
#define CLEARLINE "\033[K"
#define PREVIOUSLINE "\033[F"
#define NEXTLINE "\033[E"

#define RESET "\033[0m"

// floats

#define PRINTF_DEFAULT_PRECISION 6

#define FMT_FLOAT(n) std::fixed << std::setprecision(globals::useIntValues ? 0 : PRINTF_DEFAULT_PRECISION) << std::setw(globals::useIntValues ? 2 : 0) << (n) << std::defaultfloat << std::setprecision(PRINTF_DEFAULT_PRECISION)

// shorts

#define COUT std::cout
#define CERR std::cerr
#define ENDL std::endl
