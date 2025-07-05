/*
 * add_numbers.cu
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

#include "examples/separate_kernel_host_functions/add_arrays.h"
#include <cstdint>

__global__ void AddIntsCUDA(std::int64_t* a, std::int64_t* b)
{
    for (std::int64_t i = 0; i < 10000000; ++i)
    {
        // Simulate some work to trigger a timeout
        a[0] += b[0];
    }
}
