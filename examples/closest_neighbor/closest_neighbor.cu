/*
 * add_numbers.cu
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

#include "examples/closest_neighbor/closest_neighbor.h"
#include <cmath>
#include <cstdint>
#include <limits>

__global__ void FindClosestNeighborGPU(double2* points, std::int32_t* neighbors, std::int32_t num_points)
{
    // Calculate the global thread index
    std::int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the index is within bounds
    if (idx < num_points)
    {
        // Initialize the closest neighbor and minimum distance
        double min_distance = std::numeric_limits<double>::max();  // A large initial value

        // Iterate through all points to find the closest neighbor
        for (std::int32_t current_idx = 0; current_idx < num_points; ++current_idx)
        {
            if (current_idx != idx)  // Skip self-comparison
            {
                const double distance =
                    pow(points[idx].x - points[current_idx].x, 2) + pow(points[idx].y - points[current_idx].y, 2);
                if (distance < min_distance)
                {
                    min_distance = distance;
                    neighbors[idx] = current_idx;  // Store the closest neighbor
                }
            }
        }
    }
}
