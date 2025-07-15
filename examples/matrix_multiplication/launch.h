/*
 * File launch.h
 */

#ifndef CUDA_EXAMPLES_THREADS_BLOCKS_AND_GRIDS_LAUNCH_H
#define CUDA_EXAMPLES_THREADS_BLOCKS_AND_GRIDS_LAUNCH_H

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

__device__ constexpr std::int32_t kBlockSize = 32;  // Define the block size

double LaunchCPU(std::int32_t M, std::int32_t N, std::int32_t P);
double LaunchGPU(std::int32_t M, std::int32_t N, std::int32_t P);
double LaunchGPUAccelerated(std::int32_t M, std::int32_t N, std::int32_t P);

#endif  // CUDA_EXAMPLES_THREADS_BLOCKS_AND_GRIDS_LAUNCH_H
