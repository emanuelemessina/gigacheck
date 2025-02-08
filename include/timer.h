#pragma once

#include "cuda.cuh"
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

class AggregatedTimer
{
  public:
    AggregatedTimer(const std::string&& blockName) : name(blockName)
    {
    }

    void pushDuration(signed long duration)
    {
        durations.push_back(duration);
    }

    void aggregate()
    {
        auto total = 0;

        for (auto duration : durations)
        {
            total += duration;
        }

        COUT << "\t🔄️  " << CYAN << name << RESET << ": ";

        print_time(total);

        COUT << ENDL;
    }

  private:
    std::string name;
    std::vector<signed long> durations;
};

enum TimerPrintMode
{
    ENCLOSED,
    POST
};

class ScopedTimer
{
  public:
    ScopedTimer(const std::string&& blockName, TimerPrintMode mode) : name(blockName), mode(mode)
    {
        start();

        if (mode == ENCLOSED)
            COUT << YELLOW << "⏱️  [" << name << "]" << RESET << ENDL;
    }

    ScopedTimer(AggregatedTimer* aggregatedTimer, cudaStream_t* stream_ptr = nullptr) : aggregatedTimer(aggregatedTimer), stream_ptr(stream_ptr)
    {
        start();
    }

    ~ScopedTimer()
    {
        auto endpoint = std::chrono::high_resolution_clock::now();
        auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(endpoint - startpoint).count();

        if (aggregatedTimer)
        {
            aggregatedTimer->pushDuration(nanoseconds);
            return;
        }

        if (mode == ENCLOSED)
            COUT << YELLOW << "🏁 [" << name << "] " << RESET;
        else
            COUT << "\t➡️  " << CYAN << name << RESET << ": ";

        print_time(nanoseconds);

        if (mode == ENCLOSED)
            COUT << "\n";

        COUT << ENDL;
    }

  private:
    void start()
    {
        if (stream_ptr)
        {
            cudaEventCreate(&e_stop);
            CUDA_CREATE_RECORD_EVENT(e_start, *stream_ptr);
            CUDA_CHECK
        }
        else
            startpoint = std::chrono::high_resolution_clock::now();
    }

    signed long stop()
    {
        if (stream_ptr)
        {
            float time = 0;
            cudaEventRecord(e_stop, 0);
            cudaEventSynchronize(e_stop);
            cudaEventElapsedTime(&time, e_start, e_stop);
            cudaEventDestroy(e_start);
            cudaEventDestroy(e_stop);
            CUDA_CHECK
            return static_cast<signed long>(time * 1'000'000);
        }
        else
        {
            auto endpoint = std::chrono::high_resolution_clock::now();
            auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(endpoint - startpoint).count();
            return nanoseconds;
        }
    }

    std::string name;
    TimerPrintMode mode;
    std::chrono::high_resolution_clock::time_point startpoint;
    cudaEvent_t e_start, e_stop;
    cudaStream_t* stream_ptr = nullptr;
    AggregatedTimer* aggregatedTimer = nullptr;
};
