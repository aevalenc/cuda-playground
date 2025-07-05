/*
 * add_numbers.cu
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

#ifndef CUDA_EXAMPLES_THREADS_BLOCKS_AND_GRIDS_ADD_ARRAYS_H
#define CUDA_EXAMPLES_THREADS_BLOCKS_AND_GRIDS_ADD_ARRAYS_H

#include <cstdint>

__global__ void AddIntsCUDA(std::int64_t* a, std::int64_t* b, std::int32_t n);

#endif  // CUDA_EXAMPLES_THREADS_BLOCKS_AND_GRIDS_ADD_ARRAYS_H
