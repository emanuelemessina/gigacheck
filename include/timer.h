#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include <functional>
#include <optional>
#include <stdint.h>
#include <string>
#include <vector>


enum TimerPrintMode
{
    ENCLOSED,
    POST
};

class ScopedTimer
{
  public:
    ScopedTimer(const std::string&& blockName, TimerPrintMode mode, long* nanoseconds_out = nullptr);
    ~ScopedTimer();

  private:
    void start();

    signed long stop();

    std::string name;
    TimerPrintMode mode;
    std::chrono::high_resolution_clock::time_point startpoint;
    long* nanoseconds_out = nullptr;
};

class CudaAggregateTimer
{
  public:
    CudaAggregateTimer()
    {
    }
    ~CudaAggregateTimer();

    void start();

    /**
     * @brief No other calls to start must be issued before a stop!
     *
     */
    void stop();
    /**
     * @brief This synchronizes the device. Call this only at the end of the program!
     *
     */
    uint64_t aggregate();

  private:
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> events;
};
