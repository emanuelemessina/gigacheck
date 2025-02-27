#include "kernels.cuh"

__global__ void kernels::compute_checksums(float* matrix, int rows, int cols, ReductionDirection reduction_direction, float* checksum)
{
    extern __shared__ float shared_data[]; // shared memory for intermediate sums, as big as the blockdim vector

    int index_orthogonal = reduction_direction == ReductionDirection::ALONG_COL ? blockIdx.x : blockIdx.y; // 1 block per row/col checksum, the ortho direction is just the block index in that direction

    float sum = 0.0f;

    // info along the reduction direction
    int index_reduction = reduction_direction == ReductionDirection::ALONG_COL ? threadIdx.y : threadIdx.x;
    int blockDim_reduction = reduction_direction == ReductionDirection::ALONG_COL ? blockDim.y : blockDim.x;
    int limit_reduction = reduction_direction == ReductionDirection::ALONG_COL ? rows : cols;

    // this thread accumulates values in blockdim offsets along the reduction direction
    for (int i = index_reduction; i < limit_reduction; i += blockDim_reduction) // (rows : cols)/blockdim iterations
    {
        sum += reduction_direction == ReductionDirection::ALONG_COL ? matrix[i * cols + index_orthogonal] : matrix[index_orthogonal * (cols + 1) + i]; // 1 transf, 1 op
    }

    // this thread stores his partial result into shared memory at his relative offset
    shared_data[index_reduction] = sum;
    __syncthreads(); // other threads do the same for their relative offset, we wait for all threads to finish

    // this threads performs reduction over the shared memory vector
    for (int stride = blockDim_reduction / 2; stride > 0; stride /= 2) // log2(blockdim) iterations
    {
        if (index_reduction < stride && index_reduction + stride < limit_reduction)
        {
            shared_data[index_reduction] += shared_data[index_reduction + stride]; // 1 op
        }
        __syncthreads(); // other threads do the same
    }

    // finally, the first thread writes final reduction result to the matrix directly, or to the checksum array
    if (index_reduction == 0)
    {
        if (checksum != nullptr)
        {
            checksum[index_orthogonal] = shared_data[0]; // in the control checksum case: 1 transf
            return;
        } // in the load input checksums case

        if (reduction_direction == ReductionDirection::ALONG_COL)
        {
            matrix[rows * cols + index_orthogonal] = shared_data[0]; // last row for column checksums // 1 transf
        }
        else
        {
            matrix[index_orthogonal * (cols + 1) + cols] = shared_data[0]; // last column for row checksums // 1 transf
        }
    }
}

// geometric series ar^k, k:0->n
// Sn = a(1-r^(n+1))/(1-r)
// r = 2, a = 1, n = log2(blockdim)-1 --->  Sn = 2^log2(blockdim) - 1 = blockdim - 1
// total ops for reduction in one row/col

// transfers: (ReductionDirection::ALONG_COL ? cols : row)*( ReductionDirection::ALONG_COL ? rows : cols)/blockdim +  (ReductionDirection::ALONG_COL ? cols : row)*1   =  N + (ReductionDirection::ALONG_COL ? cols : row)*1
// ops: (ReductionDirection::ALONG_COL ? cols : row)*(ReductionDirection::ALONG_COL ? rows : cols)/blockdim + (ReductionDirection::ALONG_COL ? cols : row)(blockdim-1)   = N + (ReductionDirection::ALONG_COL ? cols : row)(blockdim-1)
