/*
 * Jacobi Solver in CUDA
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

#include "examples/jacobi_method/jacobi.h"
#include <cstdint>

__global__ void JacobiSolveGPU(const double* A, const double* b, double* x0, double* x, std::int32_t N)
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
