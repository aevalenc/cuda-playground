/*
 * Jacobi Method CUDA Example
 *
 * Author: Alejandro Valencia
 */

#ifndef EXAMPLES_JACOBI_METHOD_TILING_JACOBI_H
#define EXAMPLES_JACOBI_METHOD_TILING_JACOBI_H

#include <cstdint>

__device__ constexpr std::int32_t kBlockSize = 32;

extern "C" {
__global__ void JacobiSolveWithTilingGPU(const double* A, const double* b, double* x0, double* x, std::int32_t N);
}

#endif  // EXAMPLES_JACOBI_METHOD_TILING_JACOBI_H
