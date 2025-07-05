/*
 * add_numbers.cuh
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

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
extern "C" {
__global__ void AddIntsCUDA(std::int32_t* a, std::int32_t* b, const std::int32_t number_of_points);
}
void Launch(const std::int32_t number_of_elements, std::int32_t* a, std::int32_t* b);
