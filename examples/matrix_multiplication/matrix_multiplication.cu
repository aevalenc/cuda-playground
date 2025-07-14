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

__global__ void MatMultGPU(std::int32_t* A,
                           std::int32_t* B,
                           std::int32_t* C,
                           std::int32_t M,
                           std::int32_t N,
                           std::int32_t P)
{

    // Calculate the global thread index
    std::int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < M)
    {
        for (std::int32_t column_B = 0; column_B < P; ++column_B)
        {
            // Iterate through idx row and column
            std::int32_t sum{0};
            for (std::int32_t j = 0; j < N; ++j)
            {
                sum += A[j + N * idx] * B[column_B + P * j];
            }
            C[column_B + idx * P] = sum;
        }
    }
}

__global__ void AccelMatMultGPU(std::int32_t* A,
                                std::int32_t* B,
                                std::int32_t* C,
                                std::int32_t M,
                                std::int32_t N,
                                std::int32_t P)
{

    // Calculate the global thread index
    std::int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    std::int32_t column = blockIdx.y * blockDim.y + threadIdx.y;

    if ((row < M) || column < P)
    {
        // Iterate through idx row and column
        std::int32_t sum{0};
        for (std::int32_t j = 0; j < N; ++j)
        {
            sum += A[j + N * row] * B[column + P * j];
        }
        C[column + row * P] = sum;
        printf("Thread (%d, %d): row = %d, column= %d, C[%d] = %d\n",
               blockIdx.x,
               blockIdx.y,
               row,
               column,
               column + row * P,
               C[column + row * P]);
    }
}
