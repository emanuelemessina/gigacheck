#pragma once

#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <functional>
#include <iostream>
#include <optional>

inline void print_time(long int nanoseconds)
{
    if (nanoseconds >= 1'000'000'000)
    {
        double seconds = nanoseconds / 1'000'000'000.0;
        std::cout << seconds << " s";
    }
    else if (nanoseconds >= 1'000'000)
    {
        double milliseconds = nanoseconds / 1'000'000.0;
        std::cout << milliseconds << " ms";
    }
    else if (nanoseconds >= 1'000)
    {
        double microseconds = nanoseconds / 1'000.0;
        std::cout << microseconds << " us";
    }
    else
    {
        std::cout << nanoseconds << " ns";
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
            std::cout << name << " ..." << std::endl;
    }

    ~ScopedTimer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        if (mode == PRE)
            std::cout << "--> ";
        else
            std::cout << "\t" << name << ": ";

        print_time(nanoseconds);

        if (mode == PRE)
            std::cout << "\n";

        std::cout << std::endl;
    }

  private:
    std::string name;
    TimerPrintMode mode;
    std::chrono::high_resolution_clock::time_point start;
};

#endif