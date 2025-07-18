/*
 * Jacobi Method CUDA Example
 *
 * Author: Alejandro Valencia
 */

#ifndef EXAMPLES_JACOBI_METHOD_JACOBI_H
#define EXAMPLES_JACOBI_METHOD_JACOBI_H

#include <cstdint>

__device__ constexpr std::int32_t kBlockSize = 2;

extern "C" {
__global__ void JacobiSolveGPU(double* A, double* b, double* xn, double* x, std::int32_t N);
}

#endif  // EXAMPLES_JACOBI_METHOD_JACOBI_H
