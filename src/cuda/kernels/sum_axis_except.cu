#include "kernels.cuh"

__global__ void kernels::sum_axis_except(const float* ec_matrix_start, int rows, int cols, ReductionDirection reduction_direction, int reduction_axis_index, int exclude_index, float* result)
{
    extern __shared__ float shared[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x;

    int limit_reduction = reduction_direction == ReductionDirection::ALONG_ROW ? cols : rows;

    if (tid >= limit_reduction)
        return;

    float local_sum = 0.0f;
    for (int i = lane; i < limit_reduction; i += blockDim.x)
        if (i != exclude_index)
            local_sum += reduction_direction == ReductionDirection::ALONG_ROW ? ec_matrix_start[reduction_axis_index * (cols + 1) + i] : ec_matrix_start[i * (cols + 1) + reduction_axis_index];

    shared[lane] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (lane < stride && lane + stride < limit_reduction)
            shared[lane] += shared[lane + stride];

        __syncthreads();
    }

    if (lane == 0)
        *result = shared[0];
}
