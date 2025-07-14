/*
 * Closest Neighbor Search using Shared Memory in CUDA Test
 */

#include "examples/matrix_multiplication/test/launch_test.cuh"
#include <cstdint>
#include <gtest/gtest.h>

TEST(ClosestNeighborSharedMemoryTest, SmallPointSet)
{
    const std::int32_t M = 3;
    const std::int32_t N = 2;
    const std::int32_t P = 1;

    const std::int32_t h_A[M * N] = {0, 2, 2, 4, 4, 6};
    const std::int32_t h_B[N * P] = {0, 2};
    std::int32_t h_C[M * P] = {0};

    Launch(h_A, h_B, h_C, M, N, P);

    // Check results
    EXPECT_EQ(h_C[0], 4);
    EXPECT_EQ(h_C[1], 8);
    EXPECT_EQ(h_C[2], 12);
}
