/*
 * Copyright (C) 2025 Alejandro Valencia
 */

#include "examples/threads_blocks_and_grids/launch.h"
#include "examples/threads_blocks_and_grids/add_arrays.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>

void Launch()
{
    std::int32_t N = 100;  // Number of elements in the arrays

    // Allocate device memory for two arrays of integers
    std::int64_t* h_a = new std::int64_t[N];
    std::int64_t* h_b = new std::int64_t[N];

    // Initialize the arrays
    for (std::int32_t i = 0; i < N; ++i)
    {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    std::int64_t* d_a;
    std::int64_t* d_b;

    if (cudaMalloc((void**)&d_a, sizeof(int) * N) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for d_a\n";
        return;
    }
    if (cudaMalloc((void**)&d_b, sizeof(int) * N) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for d_b\n";
        cudaFree(d_a);
        return;
    }

    if (cudaMemcpy(d_a, h_a, sizeof(int) * N, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from host to device for d_a\n";
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }
    if (cudaMemcpy(d_b, h_b, sizeof(int) * N, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from host to device for d_b\n";
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }

    AddIntsCUDA<<<N / 256 + 1, 256>>>(d_a, d_b, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (cudaMemcpy(h_a, d_a, sizeof(int) * N, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from device to host\n";
        delete[] h_a;
        delete[] h_b;
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }

    for (std::int32_t i = 0; i < 5; ++i)
    {
        std::cout << "h_a[" << i << "] = " << h_a[i] << ", h_b[" << i << "] = " << h_b[i] << "\n";
    }

    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_b;
    std::cout << "CUDA example completed successfully.\n";
    cudaDeviceReset();
}
