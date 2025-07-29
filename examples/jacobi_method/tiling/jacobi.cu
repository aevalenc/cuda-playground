/*
 * Jacobi Solver in CUDA
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

#include "examples/jacobi_method/tiling/jacobi.h"
#include <cstdint>

__global__ void JacobiSolveWithTilingGPU(const double* A, const double* b, double* x0, double* x, std::int32_t N)
{
    // Calculate the global thread index
    std::int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::int32_t column = blockIdx.x * blockDim.x + threadIdx.x;

    double final_sum = 0.0;
    for (std::int32_t current_tile = 0; current_tile < (N + kTiledBlockSize - 1) / kTiledBlockSize; ++current_tile)
    {
        double inner_sum = 0.0;

        // Shared memory for A and x0
        __shared__ double shared_A[kTiledBlockSize][kTiledBlockSize];
        __shared__ double shared_x0[kTiledBlockSize];

        // Load A and B into shared memory
        if (row < N && (current_tile * blockDim.x + threadIdx.x < N))
        {
            shared_A[threadIdx.y][threadIdx.x] = A[row * N + current_tile * blockDim.x + threadIdx.x];
        }
        else
        {
            shared_A[threadIdx.y][threadIdx.x] = 0;
        }

        if (column < kTiledBlockSize && (current_tile * blockDim.y + threadIdx.y < N))
        {
            shared_x0[threadIdx.y] = x0[(current_tile * blockDim.y + threadIdx.y)];
        }
        else
        {
            shared_x0[threadIdx.y] = 0;
        }

        __syncthreads();

        // Perform the multiplication
        if (row < N && column < 1)
        {
            // double sum = 0.0;
            for (std::int32_t j = 0; j < kTiledBlockSize; ++j)
            {
                std::int32_t current_index = current_tile * kTiledBlockSize + j;
                if (row != current_index && current_index < N)  // Avoid self-multiplication
                {
                    inner_sum += shared_A[threadIdx.y][j] * shared_x0[j];
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
