/*
 * Copyright (C) 2025 Alejandro Valencia
 */

#include "examples/closest_neighbor/closest_neighbor.h"
#include "examples/closest_neighbor/launch.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <time.h>

double LaunchGPU(std::int32_t number_of_points,
                 std::int32_t N,
                 std::int32_t threads_per_block,
                 std::int32_t blocks_per_grid)
{

    // Allocate device memory for two arrays of double2 points
    double2* h_points = new double2[number_of_points];
    std::int32_t* h_neighbors = new std::int32_t[number_of_points];

    // Initialize the arrays
    for (std::int32_t i = 0; i < number_of_points; ++i)
    {
        h_points[i] = make_double2(static_cast<double>(rand()) / RAND_MAX, static_cast<double>(rand()) / RAND_MAX);
    }
    for (std::int32_t i = 0; i < N; ++i)
    {
        std::cout << "h_point[" << i << "] = (" << h_points[i].x << ", " << h_points[i].y << ")\n";
    }

    double2* device_points;
    std::int32_t* device_neighbors;

    const auto start = clock();
    if (cudaMalloc((void**)&device_points, sizeof(double2) * number_of_points) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for device_points\n";
        return -1;
    }
    if (cudaMalloc((void**)&device_neighbors, sizeof(std::int32_t) * number_of_points) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for device_neighbors;\n";
        cudaFree(device_points);
        return -1;
    }

    if (cudaMemcpy(device_points, h_points, sizeof(double2) * number_of_points, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from host to device for device_points\n";
        cudaFree(device_points);
        cudaFree(device_neighbors);
        return -1;
    }

    FindClosestNeighborGPU<<<blocks_per_grid, threads_per_block>>>(device_points, device_neighbors, number_of_points);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (cudaMemcpy(h_neighbors, device_neighbors, sizeof(std::int32_t) * number_of_points, cudaMemcpyDeviceToHost) !=
        cudaSuccess)
    {
        std::cerr << "Failed to copy data from device to host\n";
        delete[] h_points;
        delete[] h_neighbors;
        cudaFree(device_points);
        cudaFree(device_neighbors);
        return -1;
    }

    const auto end = clock();
    const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed GPU time: " << elapsed_time << " seconds\n";

    for (std::int32_t i = 0; i < N; ++i)
    {
        std::cout << "h_neighbors[" << i << "] = " << h_neighbors[i] << "\n";
    }

    delete[] h_points;
    delete[] h_neighbors;
    cudaFree(device_points);
    cudaFree(device_neighbors);

    cudaDeviceReset();

    return elapsed_time;
}

double LaunchCPU(std::int32_t number_of_points, std::int32_t N)
{

    // Allocate device memory for two arrays of double2 points
    double2* h_points = new double2[number_of_points];
    std::int32_t* h_neighbors = new std::int32_t[number_of_points];

    // Initialize the arrays
    for (std::int32_t i = 0; i < number_of_points; ++i)
    {
        h_points[i] = make_double2(static_cast<double>(rand()) / RAND_MAX, static_cast<double>(rand()) / RAND_MAX);
    }

    for (std::int32_t i = 0; i < N; ++i)
    {
        std::cout << "h_point[" << i << "] = (" << h_points[i].x << ", " << h_points[i].y << ")\n";
    }

    const auto start = clock();
    for (std::int32_t idx = 0; idx < number_of_points; ++idx)
    {
        // Initialize the closest neighbor and minimum distance
        double min_distance = std::numeric_limits<double>::max();  // A large initial value

        // Iterate through all points to find the closest neighbor
        for (std::int32_t current_idx = 0; current_idx < number_of_points; ++current_idx)
        {
            if (current_idx != idx)  // Skip self-comparison
            {
                const double distance = pow(h_points[idx].x - h_points[current_idx].x, 2) +
                                        pow(h_points[idx].y - h_points[current_idx].y, 2);
                if (distance < min_distance)
                {
                    min_distance = distance;
                    h_neighbors[idx] = current_idx;  // Store the closest neighbor
                }
            }
        }
    }
    const auto end = clock();
    const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed CPU time: " << elapsed_time << " seconds\n";

    for (std::int32_t i = 0; i < N; ++i)
    {
        std::cout << "h_neighbor[" << i << "] = " << h_neighbors[i] << "\n";
    }

    delete[] h_points;
    delete[] h_neighbors;

    return elapsed_time;
}
