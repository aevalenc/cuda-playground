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
    std::int32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    std::int32_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || column >= P)
        return;  // Out of bounds check

    // Iterate through idx row and column
    std::int32_t sum{0};
    for (std::int32_t j = 0; j < N; ++j)
    {
        sum += A[j + N * row] * B[column + P * j];
    }
    C[column + row * P] = sum;
}

__global__ void MatMultWithSharedMemoryGPU(std::int32_t* A,
                                           std::int32_t* B,
                                           std::int32_t* C,
                                           std::int32_t M,
                                           std::int32_t N,
                                           std::int32_t P)
{
    // Calculate the global thread index
    std::int32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    std::int32_t row = blockIdx.y * blockDim.y + threadIdx.y;

    for (std::int32_t current_block = 0; current_block < (N + blockDim.x - 1) / blockDim.x; ++current_block)
    {
        // Shared memory for A and B
        __shared__ std::int32_t shared_A[32][32];
        __shared__ std::int32_t shared_B[32][32];

        // Load A and B into shared memory
        if (row < M && current_block * blockDim.x + threadIdx.x < N)
        {
            shared_A[threadIdx.y][threadIdx.x] = A[row * N + current_block * blockDim.x + threadIdx.x];
        }
        else
        {
            shared_A[threadIdx.y][threadIdx.x] = 0;
        }

        if (column < P && current_block * blockDim.y + threadIdx.y < N)
        {
            shared_B[threadIdx.y][threadIdx.x] = B[(current_block * blockDim.y + threadIdx.y) * P + column];
        }
        else
        {
            shared_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Perform the multiplication
        if (row < M && column < P)
        {
            std::int32_t sum{0};
            for (std::int32_t j = 0; j < blockDim.x; ++j)
            {
                sum += shared_A[threadIdx.y][j] * shared_B[j][threadIdx.x];
            }
            C[column + row * P] += sum;
        }

        __syncthreads();
    }
}
