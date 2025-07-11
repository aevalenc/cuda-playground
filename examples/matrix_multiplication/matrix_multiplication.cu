/*
 * add_numbers.cu
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

#include "examples/matrix_multiplication/matrix_multiplication.h"
#include <cstdint>

__global__ void MatVectorMultGPU(std::int32_t* A, std::int32_t* b, std::int32_t* C, std::int32_t N)
{

    // Calculate the global thread index
    std::int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        // Iterate through idx row and column
        std::int32_t sum{0};
        for (std::int32_t j = 0; j < N; ++j)
        {
            sum += A[j + N * idx] * b[j];
        }
        C[idx] = sum;
        // printf("Thread %d: C[%d] = %d\n", idx, idx, C[idx]);
    }
}

__global__ void MatMultGPU(std::int32_t* A, std::int32_t* B, std::int32_t* C, std::int32_t N)
{

    // Calculate the global thread index
    std::int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        // Iterate through idx row and column
        std::int32_t sum{0};
        for (std::int32_t j = 0; j < N; ++j)
        {
            sum += A[j + N * idx] * B[idx + N * j];
        }
        C[idx] = sum;
        // printf("Thread %d: C[%d] = %d\n", idx, idx, C[idx]);
    }
}
