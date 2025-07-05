/*
 * add_numbers.cu
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

#include <examples/add_numbers/test/add_numbers.cuh>
#include <iostream>

__global__ void AddIntsCUDA(std::int32_t* a, std::int32_t* b, const std::int32_t number_of_points)
{
    // Thread index
    const std::int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Thread %d: a[%d] = %d, b[%d] = %d\n", idx, idx, a[idx], idx, b[idx]);
    a[idx] += b[idx];
    printf("Thread %d: a[%d] = %d, b[%d] = %d\n", idx, idx, a[idx], idx, b[idx]);
}

void Launch(const std::int32_t number_of_elements, std::int32_t* a, std::int32_t* b)
{

    for (std::int32_t i = 0; i < number_of_elements; ++i)
    {
        std::cout << "a[" << i << "] = " << a[i] << ", b[" << i << "] = " << b[i] << "\n";
    }

    std::int32_t* d_a = nullptr;
    std::int32_t* d_b = nullptr;

    cudaMalloc(&d_a, sizeof(std::int32_t) * number_of_elements);
    cudaMalloc(&d_b, sizeof(std::int32_t) * number_of_elements);

    if (cudaMemcpy(d_a, a, sizeof(std::int32_t) * number_of_elements, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Error copying data to device for a\n";
        return;
    }

    if (cudaMemcpy(d_b, b, sizeof(std::int32_t) * number_of_elements, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Error copying data to device for b\n";
        return;
    }

    AddIntsCUDA<<<1, 1>>>(d_a, d_b, number_of_elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (cudaMemcpy(a, d_a, sizeof(std::int32_t) * number_of_elements, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "Error copying data from device for a\n";
        return;
    }

    for (std::int32_t i = 0; i < 1; ++i)
    {
        std::cout << "a[" << i << "] = " << a[i] << ", b[" << i << "] = " << b[i] << "\n";
    }

    cudaFree(d_a);
    cudaFree(d_b);
}
