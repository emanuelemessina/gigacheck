#include "cuda.cuh"
#include "timer.h"

CudaAggregateTimer::~CudaAggregateTimer()
{
    for (auto pair : events)
    {
        cudaEventDestroy(pair.first);
        cudaEventDestroy(pair.second);
    }
}
void CudaAggregateTimer::start()
{
    cudaEvent_t e_start, e_stop;
    CUDA_CREATE_RECORD_EVENT(e_start, 0)
    cudaEventCreate(&e_stop);
    events.push_back(std::make_pair(e_start, e_stop));
}
/**
 * @brief No other calls to start must be issued before a stop!
 *
 */
void CudaAggregateTimer::stop()
{
    cudaEvent_t e_stop = events.back().second;
    cudaEventRecord(e_stop, 0);
}
/**
 * @brief This synchronizes the device. Call this only at the end of the program!
 *
 */
uint64_t CudaAggregateTimer::aggregate()
{
    cudaStreamSynchronize(0);
    uint64_t milliseconds = 0;
    for (auto pair : events)
    {
        cudaEvent_t e_start, e_stop;
        e_start = pair.first;
        e_stop = pair.second;
        float ms = 0;
        cudaEventElapsedTime(&ms, e_start, e_stop);
        milliseconds += ms;
    }
    return milliseconds;
}
