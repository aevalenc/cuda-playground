/*
 * add_numbers.cu
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

#ifndef EXAMPLES_MATRIX_MULTIPLICATION_MATRIX_MULTIPLICATION_H
#define EXAMPLES_MATRIX_MULTIPLICATION_MATRIX_MULTIPLICATION_H

#include <cstdint>

extern "C" {
__global__ void MatVectorMultGPU(std::int32_t* A, std::int32_t* b, std::int32_t* C, std::int32_t N);
__device__ constexpr std::int32_t kBlockSize = 128;  // Define the block size
}

#endif  // EXAMPLES_MATRIX_MULTIPLICATION_MATRIX_MULTIPLICATION_H
