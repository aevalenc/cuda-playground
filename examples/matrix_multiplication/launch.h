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

double LaunchCPU(std::int32_t N);
double LaunchGPU(std::int32_t N);

#endif  // CUDA_EXAMPLES_THREADS_BLOCKS_AND_GRIDS_LAUNCH_H
