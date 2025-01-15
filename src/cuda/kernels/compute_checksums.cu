#include "kernels.cuh"

__global__ void kernels::compute_checksums(float* matrix, int rows, int cols, bool checksum_compute_mode)
{
    extern __shared__ float shared_data[]; // Shared memory for intermediate sums

    int index_orthogonal = checksum_compute_mode == CHECKSUM_COMPUTE_COL ? (blockIdx.x * blockDim.x + threadIdx.x) : (blockIdx.y * blockDim.y + threadIdx.y);
    int limit_orthogonal = checksum_compute_mode == CHECKSUM_COMPUTE_COL ? cols : rows;

    if (index_orthogonal >= limit_orthogonal)
        return;

    float sum = 0.0f;

    int index_reduction = checksum_compute_mode == CHECKSUM_COMPUTE_COL ? threadIdx.y : threadIdx.x;
    int blockDim_reduction = checksum_compute_mode == CHECKSUM_COMPUTE_COL ? blockDim.y : blockDim.x;
    int limit_reduction = checksum_compute_mode == CHECKSUM_COMPUTE_COL ? rows : cols;

    // Accumulate values in chunks to utilize shared memory and threads
    for (int i = index_reduction; i < limit_reduction; i += blockDim_reduction)
    {
        sum += checksum_compute_mode == CHECKSUM_COMPUTE_COL ? matrix[i * cols + index_orthogonal] : matrix[index_orthogonal * cols + i];
    }

    // Store partial results into shared memory
    shared_data[index_reduction] = sum;
    __syncthreads();

    // Perform reduction to compute the checksum
    for (int stride = blockDim_reduction / 2; stride > 0; stride /= 2)
    {
        if (index_reduction < stride)
        {
            shared_data[index_reduction] += shared_data[index_reduction + stride];
        }
        __syncthreads();
    }

    // first thread writes final result to the matrix directly
    if (index_reduction == 0)
    {
        if (checksum_compute_mode == CHECKSUM_COMPUTE_COL)
        {
            matrix[rows * cols + index_orthogonal] = shared_data[0]; // Last row for column checksums
        }
        else
        {
            matrix[index_orthogonal * cols + cols] = shared_data[0]; // Last column for row checksums
        }
    }
}
