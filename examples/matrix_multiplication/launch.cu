/*
 * Copyright (C) 2025 Alejandro Valencia
 */

#include "examples/matrix_multiplication/launch.h"
#include "examples/matrix_multiplication/matrix_multiplication.h"
#include "examples/matrix_multiplication/utils.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <time.h>

double LaunchCPU(const std::int32_t M, const std::int32_t N, const std::int32_t P)
{

    // Initialize the arrays
    const auto host_A = utils::InitializeTestMatrix(M, N);
    const auto host_B = utils::InitializeTestMatrix(N, P);
    std::int32_t* host_C = new std::int32_t[M * P];

    const auto start = clock();
    for (std::int32_t i = 0; i < M; ++i)
    {
        // Initialize sum
        for (std::int32_t j = 0; j < P; ++j)
        {
            std::int32_t sum{0};
            for (std::int32_t k = 0; k < N; ++k)
            {
                sum += host_A[k + N * i] * host_B[j + P * k];
            }
            host_C[j + P * i] = sum;
        }
    }

    const auto end = clock();
    const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed CPU time: " << elapsed_time << " seconds\n";

    delete[] host_A;
    delete[] host_B;
    delete[] host_C;

    return elapsed_time;
}

double LaunchGPU(const std::int32_t M, const std::int32_t N, const std::int32_t P)
{
    // Initialize the arrays
    const auto host_A = utils::InitializeTestMatrix(M, N);
    const auto host_B = utils::InitializeTestMatrix(N, P);
    std::int32_t* host_C = new std::int32_t[M * P];

    // Allocate device memory
    std::int32_t* device_A;
    std::int32_t* device_B;
    std::int32_t* device_C;

    const auto start = clock();
    if (utils::AllocateAndCopyToDevice(device_A, device_B, device_C, host_A, host_B, M, N, P) != 0)
    {
        return -1;
    }

    // Launch kernel
    const auto num_blocks = (M + kBlockSize - 1) / kBlockSize;
    std::cout << "Launching kernel with " << num_blocks << " blocks of " << kBlockSize << " threads each\n";
    MatMultGPU<<<num_blocks, kBlockSize>>>(device_A, device_B, device_C, M, N, P);

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

double LaunchGPUAccelerated(const std::int32_t M, const std::int32_t N, const std::int32_t P)
{

    // Initialize the arrays
    const auto host_A = utils::InitializeTestMatrix(M, N);
    const auto host_B = utils::InitializeTestMatrix(N, P);
    std::int32_t* host_C = new std::int32_t[M * P];

    // Allocate device memory
    std::int32_t* device_A;
    std::int32_t* device_B;
    std::int32_t* device_C;

    const auto start = clock();
    if (utils::AllocateAndCopyToDevice(device_A, device_B, device_C, host_A, host_B, M, N, P) != 0)
    {
        return -1;
    }

    const auto num_blocks_x = (P + kBlockSize - 1) / kBlockSize;
    const auto num_blocks_y = (M + kBlockSize - 1) / kBlockSize;

    std::cout << "Launching kernel with configuration: (" << num_blocks_x << ", " << num_blocks_y << ") x ("
              << kBlockSize << ", " << kBlockSize << ") \n";
    AccelMatMultGPU<<<dim3(num_blocks_x, num_blocks_y, 1), dim3(kBlockSize, kBlockSize)>>>(
        device_A, device_B, device_C, M, N, P);

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

double LaunchGPUWithSharedMemory(const std::int32_t M, const std::int32_t N, const std::int32_t P)
{

    // Initialize the arrays
    const auto host_A = utils::InitializeTestMatrix(M, N);
    const auto host_B = utils::InitializeTestMatrix(N, P);
    std::int32_t* host_C = new std::int32_t[M * P];

    // Allocate device memory
    std::int32_t* device_A;
    std::int32_t* device_B;
    std::int32_t* device_C;

    const auto start = clock();
    if (utils::AllocateAndCopyToDevice(device_A, device_B, device_C, host_A, host_B, M, N, P) != 0)
    {
        return -1;
    }

    const auto num_blocks_x = (P + kBlockSize - 1) / kBlockSize;
    const auto num_blocks_y = (M + kBlockSize - 1) / kBlockSize;

    const auto grid_dim = dim3(num_blocks_x, num_blocks_y, 1);
    const auto block_dim = dim3(kBlockSize, kBlockSize);

    std::cout << "Launching kernel with configuration: (" << num_blocks_x << ", " << num_blocks_y << ") x ("
              << kBlockSize << ", " << kBlockSize << ") \n";
    MatMultWithSharedMemoryGPU<<<grid_dim, block_dim>>>(device_A, device_B, device_C, M, N, P);

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
