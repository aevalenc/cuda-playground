/*
 * Jacobi Method CUDA Example
 *
 * Author: Alejandro Valencia
 */

#ifndef EXAMPLES_JACOBI_METHOD_1D_GRID_JACOBI_H
#define EXAMPLES_JACOBI_METHOD_1D_GRID_JACOBI_H

#include <cstdint>

__device__ constexpr std::int32_t kBlockSize = 256;

extern "C" {
__global__ void JacobiSolveGPU1D(const double* A, const double* b, double* x0, double* x, std::int32_t N);
__global__ void JacobiSolveWithSharedMemoryGPU(const double* A, const double* b, double* x0, double* x, std::int32_t N);
}

#endif  // EXAMPLES_JACOBI_METHOD_1D_GRID_JACOBI_H
