#include "timer.h"
#include "iomod.h"

inline void print_time(long int nanoseconds)
{
    if (nanoseconds >= 1'000'000'000)
    {
        double seconds = nanoseconds / 1'000'000'000.0;
        COUT << seconds << " s";
    }
    else if (nanoseconds >= 1'000'000)
    {
        double milliseconds = nanoseconds / 1'000'000.0;
        COUT << milliseconds << " ms";
    }
    else if (nanoseconds >= 1'000)
    {
        double microseconds = nanoseconds / 1'000.0;
        COUT << microseconds << " us";
    }
    else
    {
        COUT << nanoseconds << " ns";
    }
}

ScopedTimer::ScopedTimer(const std::string&& blockName, TimerPrintMode mode, long* nanoseconds_out) : name(blockName), mode(mode), nanoseconds_out(nanoseconds_out)
{
    start();

    if (mode == ENCLOSED)
        COUT << YELLOW << "⏱️  [" << name << "]" << RESET << ENDL;
}
ScopedTimer::~ScopedTimer()
{
    auto endpoint = std::chrono::high_resolution_clock::now();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(endpoint - startpoint).count();

    if (mode == ENCLOSED)
        COUT << YELLOW << "🏁 [" << name << "] " << RESET;
    else
        COUT << "\t➡️  " << CYAN << name << RESET << ": ";

    print_time(nanoseconds);

    if (mode == ENCLOSED)
        COUT << "\n";

    COUT << ENDL;

    *nanoseconds_out = nanoseconds;
}

void ScopedTimer::start()
{
    startpoint = std::chrono::high_resolution_clock::now();
}

signed long ScopedTimer::stop()
{
    auto endpoint = std::chrono::high_resolution_clock::now();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(endpoint - startpoint).count();
    return nanoseconds;
}
