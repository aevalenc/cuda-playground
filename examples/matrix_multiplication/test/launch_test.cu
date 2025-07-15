/*
 * Closest Neighbor Search using Shared Memory in CUDA Test
 */

#include "examples/matrix_multiplication/matrix_multiplication.h"
#include "examples/matrix_multiplication/test/launch_test.cuh"
#include "examples/matrix_multiplication/utils.h"
#include <cstdint>
#include <gtest/gtest.h>

void Launch(const int32_t* A,
            const int32_t* B,
            int32_t* C,
            const std::int32_t M,
            const std::int32_t N,
            const std::int32_t P)
{

    // Allocate device memory
    std::int32_t* d_A;
    std::int32_t* d_B;
    std::int32_t* d_C;
    cudaMalloc(&d_A, sizeof(std::int32_t) * M * N);
    cudaMalloc(&d_B, sizeof(std::int32_t) * N * P);
    cudaMalloc(&d_C, sizeof(std::int32_t) * M * P);

    // Copy points to device
    cudaMemcpy(d_A, A, sizeof(std::int32_t) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(std::int32_t) * N * P, cudaMemcpyHostToDevice);

    // Launch kernel (4 threads per block)
    const auto num_blocks = (M + kTestBlockSize - 1) / kTestBlockSize;
    printf("Launching kernel with %d blocks of %d threads each\n", num_blocks, kTestBlockSize);
    MatMultGPU<<<num_blocks, kTestBlockSize>>>(d_A, d_B, d_C, M, N, P);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(C, d_C, sizeof(std::int32_t) * M * P, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void LaunchAccelerated(const int32_t* A,
                       const int32_t* B,
                       int32_t* C,
                       const std::int32_t M,
                       const std::int32_t N,
                       const std::int32_t P)
{

    // Allocate device memory
    std::int32_t* d_A;
    std::int32_t* d_B;
    std::int32_t* d_C;
    cudaMalloc(&d_A, sizeof(std::int32_t) * M * N);
    cudaMalloc(&d_B, sizeof(std::int32_t) * N * P);
    cudaMalloc(&d_C, sizeof(std::int32_t) * M * P);

    // Copy points to device
    if (cudaMemcpy(d_A, A, sizeof(std::int32_t) * M * N, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from host to device for Matrix A\n";
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }
    if (cudaMemcpy(d_B, B, sizeof(std::int32_t) * N * P, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from host to device for Matrix A\n";
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    // Launch kernel
    const auto num_blocks_x = (P + kTestBlockSize - 1) / kTestBlockSize;
    const auto num_blocks_y = (M + kTestBlockSize - 1) / kTestBlockSize;

    printf("Launching kernel with %d blocks in x, %d blocks in y, and %d threads per block\n",
           num_blocks_x,
           num_blocks_y,
           kTestBlockSize);

    AccelMatMultGPU<<<dim3(num_blocks_x, num_blocks_y), dim3(kTestBlockSize, kTestBlockSize)>>>(d_A, d_B, d_C, M, N, P);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back

    if (cudaMemcpy(C, d_C, sizeof(std::int32_t) * M * P, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from device to host\n";
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
