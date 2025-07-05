/*
 * add_numbers.cu
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

#ifndef CUDA_EXAMPLES_SEPARATE_KERNEL_HOST_FUNCTIONS_ADD_ARRAYS_H
#define CUDA_EXAMPLES_SEPARATE_KERNEL_HOST_FUNCTIONS_ADD_ARRAYS_H

#include <cstdint>

__global__ void AddIntsCUDA(std::int64_t* a, std::int64_t* b);

#endif  // CUDA_EXAMPLES_SEPARATE_KERNEL_HOST_FUNCTIONS_ADD_ARRAYS_H
