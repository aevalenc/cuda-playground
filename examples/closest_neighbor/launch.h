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

double LaunchCPU(std::int32_t number_of_points, std::int32_t N = 5);
double LaunchGPU(std::int32_t number_of_points,
                 std::int32_t N = 5,
                 std::int32_t threads_per_block = 256,
                 std::int32_t blocks_per_grid = 16);

#endif  // CUDA_EXAMPLES_THREADS_BLOCKS_AND_GRIDS_LAUNCH_H
