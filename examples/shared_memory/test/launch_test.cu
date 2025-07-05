/*
 * Closest Neighbor Search using Shared Memory in CUDA Test
 */

#include "examples/shared_memory/closest_neighbor.h"
#include "examples/shared_memory/test/launch_test.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

void Launch(const std::int32_t number_of_points, const double2* h_points, std::int32_t* h_neighbors)
{

    // Allocate device memory
    double2* d_points = nullptr;
    std::int32_t* d_neighbors = nullptr;
    cudaMalloc(&d_points, sizeof(double2) * number_of_points);
    cudaMalloc(&d_neighbors, sizeof(std::int32_t) * number_of_points);

    // Copy points to device
    cudaMemcpy(d_points, h_points, sizeof(double2) * number_of_points, cudaMemcpyHostToDevice);

    // Launch kernel (4 threads per block)
    FindClosestNeighborWithSharedBlocks<<<(number_of_points + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
        d_points, d_neighbors, number_of_points);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_neighbors, d_neighbors, sizeof(std::int32_t) * number_of_points, cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_neighbors);
}
