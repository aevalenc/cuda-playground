/*
 * Closest Neighbor Search using Shared Memory in CUDA Test
 */

#include <cstdint>
#include <cuda_runtime.h>

__device__ constexpr int32_t kTestBlockSize = 256;

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
