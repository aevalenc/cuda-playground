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
__global__ void MatMultGPU(std::int32_t* A,
                           std::int32_t* B,
                           std::int32_t* C,
                           std::int32_t N,
                           std::int32_t M,
                           std::int32_t P);
__global__ void AccelMatMultGPU(std::int32_t* A,
                                std::int32_t* B,
                                std::int32_t* C,
                                std::int32_t M,
                                std::int32_t N,
                                std::int32_t P);
}

#endif  // EXAMPLES_MATRIX_MULTIPLICATION_MATRIX_MULTIPLICATION_H
