/*
 * Copyright (C) 2025 Alejandro Valencia
 */

#include "examples/matrix_multiplication/launch.h"
#include "examples/matrix_multiplication/matrix_multiplication.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <time.h>

namespace
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

}  // namespace

double LaunchGPU(std::int32_t M, std::int32_t N, std::int32_t P)
{
    // Allocate device memory for two arrays of double2 points
    std::int32_t* host_A = new std::int32_t[M * N];
    std::int32_t* host_B = new std::int32_t[N * P];
    std::int32_t* host_C = new std::int32_t[M * P];

    // Initialize the arrays
    for (std::int32_t i = 0; i < M; ++i)
    {
        for (std::int32_t j = 0; j < N; ++j)
        {
            host_A[j + N * i] = 2 * (i + j);  // Example initialization, can be random or specific values
        }
    }

    for (std::int32_t i = 0; i < N; ++i)
    {
        for (std::int32_t j = 0; j < P; ++j)
        {
            host_B[j + P * i] = 2 * (i + j);  // Example initialization, can be random or specific values
        }
    }

    std::int32_t* device_A;
    std::int32_t* device_B;
    std::int32_t* device_C;

    const auto start = clock();
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

    MatMultGPU<<<(N + kBlockSize - 1) / kBlockSize, kBlockSize>>>(device_A, device_B, device_C, M, N, P);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (cudaMemcpy(host_C, device_C, sizeof(std::int32_t) * M * P, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from device to host\n";
        delete[] host_A;
        delete[] host_B;
        delete[] host_C;
        cudaFree(device_A);
        cudaFree(device_B);
        cudaFree(device_C);
        return -1;
    }

    PrintMatrix(host_C, M, P);

    const auto end = clock();
    const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed GPU time: " << elapsed_time << " seconds\n";

    delete[] host_A;
    delete[] host_B;
    delete[] host_C;

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    cudaDeviceReset();

    return elapsed_time;
}

double LaunchCPU(std::int32_t N, std::int32_t M, std::int32_t P)
{

    // Allocate device memory for two arrays of double2 points
    std::int32_t* host_A = new std::int32_t[N * M];
    std::int32_t* host_B = new std::int32_t[M * P];
    std::int32_t* host_C = new std::int32_t[N * P];

    // Initialize the arrays
    for (std::int32_t i = 0; i < N; ++i)
    {
        for (std::int32_t j = 0; j < M; ++j)
        {
            host_A[j + N * i] = 2 * (i + j);  // Example initialization, can be random or specific values
        }
    }

    for (std::int32_t i = 0; i < M; ++i)
    {
        for (std::int32_t j = 0; j < P; ++j)
        {
            host_B[j + M * i] = 2 * (i + j);  // Example initialization, can be random or specific values
        }
    }

    // PrintMatrix(host_A, N);
    // PrintVector(host_B, N);

    const auto start = clock();
    for (std::int32_t i = 0; i < N; ++i)
    {
        // Initialize sum
        for (std::int32_t j = 0; j < P; ++j)
        {
            std::int32_t sum{0};
            for (std::int32_t k = 0; k < M; ++k)
            {
                sum += host_A[k + N * i] * host_B[j + M * k];
            }
            host_C[j + P * i] = sum;
        }
    }

    const auto end = clock();
    const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed CPU time: " << elapsed_time << " seconds\n";

    // PrintVector(host_C, N);

    delete[] host_A;
    delete[] host_B;
    delete[] host_C;

    return elapsed_time;
}
