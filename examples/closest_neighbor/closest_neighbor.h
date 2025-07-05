/*
 * add_numbers.cu
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

#ifndef EXAMPLES_CLOSEST_NEIGHBOR_CLOSEST_NEIGHBOR_H
#define EXAMPLES_CLOSEST_NEIGHBOR_CLOSEST_NEIGHBOR_H

#include <cstdint>

extern "C" {
__global__ void FindClosestNeighborGPU(double2* points, std::int32_t* neighbors, std::int32_t num_points);
}

#endif  // EXAMPLES_CLOSEST_NEIGHBOR_CLOSEST_NEIGHBOR_H
