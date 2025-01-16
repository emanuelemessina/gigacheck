#include "cuda.cuh"

namespace cuda
{
    Info getInfo()
    {
        cudaDeviceProp deviceProp;
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&deviceProp, device);
        Info info{
            device,
            deviceProp.name,
            deviceProp.multiProcessorCount,
            deviceProp.totalGlobalMem,
            deviceProp.sharedMemPerBlock,
            deviceProp.sharedMemPerMultiprocessor};
        return info;
    }
}
