/*
 * Closest Neighbor Search using Shared Memory in CUDA Test
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

__device__ constexpr int32_t kTestBlockSize = 32;

void Launch(const int32_t* A,
            const int32_t* B,
            int32_t* C,
            const std::int32_t M,
            const std::int32_t N,
            const std::int32_t P);

void LaunchAccelerated(const int32_t* A,
                       const int32_t* B,
                       int32_t* C,
                       const std::int32_t M,
                       const std::int32_t N,
                       const std::int32_t P);

void LaunchWithSharedMemory(const int32_t* A,
                            const int32_t* B,
                            int32_t* C,
                            const std::int32_t M,
                            const std::int32_t N,
                            const std::int32_t P);
