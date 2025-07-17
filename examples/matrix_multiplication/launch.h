/*
 * File launch.h
 */

#ifndef EXAMPLES_MATRIX_MULTIPLICATION_LAUNCH_H
#define EXAMPLES_MATRIX_MULTIPLICATION_LAUNCH_H

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

__device__ constexpr std::int32_t kBlockSizeX = 32;
__device__ constexpr std::int32_t kBlockSizeY = 32;

double LaunchCPU(const std::int32_t M, const std::int32_t N, const std::int32_t P);
double LaunchGPU(const std::int32_t M, const std::int32_t N, const std::int32_t P);
double LaunchGPUAccelerated(const std::int32_t M, const std::int32_t N, const std::int32_t P);
double LaunchGPUWithSharedMemory(const std::int32_t M, const std::int32_t N, const std::int32_t P);

#endif  // EXAMPLES_MATRIX_MULTIPLICATION_LAUNCH_H
