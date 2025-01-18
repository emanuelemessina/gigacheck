#pragma once

#ifndef TIMER_H
#define TIMER_H

#include "iomod.h"
#include <chrono>
#include <functional>
#include <optional>

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

enum TimerPrintMode
{
    PRE,
    POST
};

class ScopedTimer
{
  public:
    ScopedTimer(const std::string&& blockName, TimerPrintMode mode)
        : name(blockName), mode(mode), start(std::chrono::high_resolution_clock::now())
    {
        if (mode == PRE)
            COUT << YELLOW << "⏱️  [" << name << "]" << RESET << ENDL;
    }

    ~ScopedTimer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        if (mode == PRE)
            COUT << YELLOW << "🏁 [" << name << "] " << RESET;
        else
            COUT << "\t➡️  " << CYAN << name << RESET << ": ";

        print_time(nanoseconds);

        if (mode == PRE)
            COUT << "\n";

        COUT << ENDL;
    }

  private:
    std::string name;
    TimerPrintMode mode;
    std::chrono::high_resolution_clock::time_point start;
};

#endif
