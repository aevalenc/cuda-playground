/*
 * utils.cpp
 */

#include "examples/matrix_multiplication/utils.h"
#include <cuda_runtime.h>
#include <iostream>

namespace utils
{

void PrintVector(const std::int32_t* A, const std::int32_t N)
{
    std::cout << "[" << A[0];
    for (std::int32_t i = 1; i < N; ++i)
    {
        std::cout << "\n " << A[i];
    }
    std::cout << "]\n";
}

void PrintArray(const std::int32_t* A, const std::int32_t N)
{
    std::cout << "[" << A[0];
    for (std::int32_t i = 1; i < N; ++i)
    {
        std::cout << ", " << A[i];
    }
    std::cout << "]\n";
}

void PrintMatrix(const std::int32_t* A, const std::int32_t M, const std::int32_t N)
{
    std::cout << "[";
    for (std::int32_t i = 0; i < M; ++i)
    {

        if (i != (M - 1))
        {
            PrintArray(&A[i * N], N);
        }
        else
        {
            std::cout << "[" << A[N * i];
            for (std::int32_t j = 1; j < N; ++j)
            {
                std::cout << ", " << A[j + N * i];
            }
        }
    }
    std::cout << "]]\n";
}

std::int32_t* InitializeTestMatrix(const std::int32_t M, const std::int32_t N)
{
    std::int32_t* host_A = new std::int32_t[M * N];

    for (std::int32_t i = 0; i < M; ++i)
    {
        for (std::int32_t j = 0; j < N; ++j)
        {
            host_A[j + N * i] = 2 + (i + j);  // Example initialization, can be random or specific values
        }
    }
    return host_A;
}

std::int32_t AllocateAndCopyToDevice(std::int32_t*& device_A,
                                     std::int32_t*& device_B,
                                     std::int32_t*& device_C,
                                     const std::int32_t* host_A,
                                     const std::int32_t* host_B,
                                     const std::int32_t M,
                                     const std::int32_t N,
                                     const std::int32_t P)
{
    if (cudaMalloc((void**)&device_A, sizeof(std::int32_t) * M * N) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for device matrix A\n";
        return -1;
    }
    if (cudaMalloc((void**)&device_B, sizeof(std::int32_t) * N * P) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for device matrix B;\n";
        cudaFree(device_A);
        return -1;
    }
    if (cudaMalloc((void**)&device_C, sizeof(std::int32_t) * M * P) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for device matrix C;\n";
        cudaFree(device_A);
        cudaFree(device_B);
        return -1;
    }

    if (cudaMemcpy(device_A, host_A, sizeof(std::int32_t) * M * N, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from host to device for Matrix A\n";
        cudaFree(device_A);
        cudaFree(device_B);
        return -1;
    }

    if (cudaMemcpy(device_B, host_B, sizeof(std::int32_t) * N * P, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from host to device for Matrix B\n";
        cudaFree(device_A);
        cudaFree(device_B);
        return -1;
    }
    return 0;
}

}  // namespace utils
