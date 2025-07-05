/*
 * add_numbers.cu
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

#include "examples/threads_blocks_and_grids/add_arrays.h"
#include <cstdint>

__global__ void AddIntsCUDA(std::int64_t* a, std::int64_t* b, std::int32_t n)
{
    // Calculate the global thread index
    std::int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the index is within bounds
    if (idx < n)
    {
        a[idx] += b[idx];
    }
}
