/*
 * File launch.h
 */

#ifndef CUDA_EXAMPLES_SEPARATE_KERNEL_HOST_FUNCTIONS_LAUNCH_H
#define CUDA_EXAMPLES_SEPARATE_KERNEL_HOST_FUNCTIONS_LAUNCH_H

#include <cuda.h>

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

void Launch();

#endif  // CUDA_EXAMPLES_SEPARATE_KERNEL_HOST_FUNCTIONS_LAUNCH_H
