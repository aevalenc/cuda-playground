/*
 * Jacobi Solver in CUDA
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

#include "examples/jacobi_method/1d_grid/jacobi.h"
#include <cstdint>

__global__ void JacobiSolveGPU1D(const double* A, const double* b, double* x0, double* x, std::int32_t N)
{
    // Get Global thread index
    std::int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
    {
        return;  // Out of bounds check
    }

    // Calculate sum
    double sum = 0;
    for (std::int32_t k = 0; k < N; ++k)
    {
        if (k != idx)
        {
            sum += A[k + N * idx] * x0[k];
        }
    }

    // Calculate next iteration
    x[idx] = (b[idx] - sum) / A[idx + N * idx];
}

__global__ void JacobiSolveWithSharedMemoryGPU(const double* A, const double* b, double* x0, double* x, std::int32_t N)
{
    // Calculate the global thread index
    std::int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::int32_t column = blockIdx.x * blockDim.x + threadIdx.x;

    double final_sum = 0.0;
    for (std::int32_t current_block = 0; current_block < (N + kBlockSize - 1) / kBlockSize; ++current_block)
    {
        double inner_sum = 0.0;

        // Shared memory for A and x0
        __shared__ double shared_A[kBlockSize];
        __shared__ double shared_x0[kBlockSize];

        // Load A and B into shared memory
        if (row < N && (current_block * blockDim.x + threadIdx.x < N) && column < kBlockSize)
        {
            shared_A[threadIdx.x] = A[row * N + current_block * blockDim.x + threadIdx.x];
        }
        else
        {
            shared_A[threadIdx.x] = 0;
        }
        if (column < kBlockSize && current_block * blockDim.y + threadIdx.y < N)
        {
            shared_x0[threadIdx.x] = x0[(current_block * kBlockSize + threadIdx.x)];
        }
        else
        {
            shared_x0[threadIdx.x] = 0;
        }

        __syncthreads();

        // Perform the multiplication
        if (row < N && column < 1)
        {
            // double sum = 0.0;
            for (std::int32_t j = 0; j < kBlockSize; ++j)
            {
                std::int32_t current_index = current_block * kBlockSize + j;
                if (row != current_index && current_index < N)  // Avoid self-multiplication
                {
                    inner_sum += shared_A[j] * shared_x0[j];
                }
            }
            final_sum += inner_sum;
        }

        __syncthreads();
    }

    // Calculate next iteration
    if (row < N && column < 1)
    {
        x[row] = (b[row] - final_sum) / A[row + N * row];
    }
}
