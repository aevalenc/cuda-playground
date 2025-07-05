/*
 * add_numbers.cu
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

#include "examples/shared_memory/closest_neighbor.h"
#include <cmath>
#include <cstdint>
#include <limits>

__global__ void FindClosestNeighborWithSharedBlocks(double2* points, std::int32_t* neighbors, std::int32_t num_points)
{

    // Calculate the global thread index
    std::int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for points in the block
    __shared__ double2 shared_points[kBlockSize];

    // Initialize the closest neighbor and minimum distance
    std::int32_t closest_idx = -1;                             // To store the index of the closest neighbor
    double min_distance = std::numeric_limits<double>::max();  // A large initial value

    // Loop through the grid of blocks
    for (std::int32_t current_block = 0; current_block < gridDim.x; ++current_block)
    {
        // Copy a block of points into shared memory
        if (current_block * kBlockSize + threadIdx.x < num_points)
        {
            shared_points[threadIdx.x] = points[current_block * kBlockSize + threadIdx.x];
        }

        __syncthreads();  // Ensure all threads have copied their points before proceeding

        // Iterate through all points to find the closest neighbor
        for (std::int32_t current_block_idx = 0; current_block_idx < kBlockSize; ++current_block_idx)
        {
            const std::int32_t compare_idx = current_block * kBlockSize + current_block_idx;
            if ((compare_idx == idx) || compare_idx >= num_points || idx >= num_points)
            {
                continue;  // Skip if the index is out of bounds
            }

            const double distance = pow(points[idx].x - shared_points[current_block_idx].x, 2) +
                                    pow(points[idx].y - shared_points[current_block_idx].y, 2);

            if (distance < min_distance)
            {
                min_distance = distance;
                closest_idx = current_block_idx + current_block * kBlockSize;  // Update the closest neighbor index
            }
        }
        __syncthreads();  // Ensure all threads have completed their calculations before moving to the next block
    }
    neighbors[idx] = closest_idx;  // Store the closest neighbor
}
