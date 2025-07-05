/*
 * add_numbers.cu
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

#ifndef EXAMPLES_SHARED_MEMORY_CLOSEST_NEIGHBOR_H
#define EXAMPLES_SHARED_MEMORY_CLOSEST_NEIGHBOR_H

#include <cstdint>

extern "C" {
__global__ void FindClosestNeighborWithSharedBlocks(double2* points, std::int32_t* neighbors, std::int32_t num_points);
__device__ constexpr std::int32_t kBlockSize = 32;  // Define the block size
}

#endif  // EXAMPLES_SHARED_MEMORY_CLOSEST_NEIGHBOR_H
