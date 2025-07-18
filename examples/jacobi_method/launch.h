/*
 * File launch.h
 */

#ifndef EXAMPLES_JACOBI_METHOD_LAUNCH_H
#define EXAMPLES_JACOBI_METHOD_LAUNCH_H

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

double LaunchCPU(const std::int32_t N);
// double LaunchGPUWithSharedMemory(const std::int32_t N);

#endif  // EXAMPLES_JACOBI_METHOD_LAUNCH_H
