/*
 * File launch.h
 */

#ifndef EXAMPLES_JACOBI_METHOD_1D_GRID_LAUNCH_H
#define EXAMPLES_JACOBI_METHOD_1D_GRID_LAUNCH_H

#include <cstdint>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                                                     \
    do                                                                                                       \
    {                                                                                                        \
        cudaError_t err = (expr);                                                                            \
        if (err != cudaSuccess)                                                                              \
        {                                                                                                    \
            fprintf(stderr, "CUDA Error Code  : %d\n     Error String: %s\n", err, cudaGetErrorString(err)); \
            exit(err);                                                                                       \
        }                                                                                                    \
    } while (0)

const std::int32_t kNumberOfElements = 11;
const double kTolerance = 1e-3;
const std::int32_t kMaxIterations = 1e3;

double LaunchGPU();
double LaunchJacobiWithSharedMemoryGPU();

#endif  // EXAMPLES_JACOBI_METHOD_1D_GRID_LAUNCH_H
