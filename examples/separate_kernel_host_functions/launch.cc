/*
 *
 */

#include "examples/separate_kernel_host_functions/launch.h"
#include "examples/separate_kernel_host_functions/add_arrays.h"
#include <cstdint>
#include <iostream>

void Launch()
{
    std::int64_t a = 5;
    std::int64_t b = 10;
    std::int64_t* d_a;
    std::int64_t* d_b;

    if (cudaMalloc((void**)&d_a, sizeof(int)) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for d_a\n";
        return;
    }
    if (cudaMalloc((void**)&d_b, sizeof(int)) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for d_b\n";
        cudaFree(d_a);
        return;
    }

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    AddIntsCUDA<<<1, 1>>>(d_a, d_b);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from device to host\n";
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }

    std::cout << "The answer is " << a << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaDeviceReset();
}
