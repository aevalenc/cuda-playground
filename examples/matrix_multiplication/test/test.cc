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

TEST(MultiDimensionalGridTest, BasicFunctionality)
{
    const std::int32_t M = 4;
    const std::int32_t N = 4;
    const std::int32_t P = 4;

    const std::int32_t h_A[M * N] = {0, 2, 4, 6, 2, 4, 6, 8, 4, 6, 8, 10, 6, 8, 10, 12};
    const std::int32_t h_B[M * N] = {0, 2, 4, 6, 2, 4, 6, 8, 4, 6, 8, 10, 6, 8, 10, 12};
    std::int32_t h_C[M * P] = {0};

    LaunchAccelerated(h_A, h_B, h_C, M, N, P);

    // Check results
    EXPECT_EQ(h_C[0], 56);
    EXPECT_EQ(h_C[1], 80);
    EXPECT_EQ(h_C[2], 104);
    EXPECT_EQ(h_C[3], 128);
    EXPECT_EQ(h_C[4], 80);
    EXPECT_EQ(h_C[5], 120);
    EXPECT_EQ(h_C[6], 160);
    EXPECT_EQ(h_C[7], 200);
    EXPECT_EQ(h_C[8], 104);
    EXPECT_EQ(h_C[9], 160);
    EXPECT_EQ(h_C[10], 216);
    EXPECT_EQ(h_C[11], 272);
    EXPECT_EQ(h_C[12], 128);
    EXPECT_EQ(h_C[13], 200);
    EXPECT_EQ(h_C[14], 272);
    EXPECT_EQ(h_C[15], 344);
}

TEST(MultiDimensionalGridTest, GivenSameInputExpectEqualOutput)
{
    // Given
    const std::int32_t M = 4;
    const std::int32_t N = 4;
    const std::int32_t P = 4;

    const std::int32_t h_A[M * N] = {0, 2, 4, 6, 2, 4, 6, 8, 4, 6, 8, 10, 6, 8, 10, 12};
    const std::int32_t h_B[M * N] = {0, 2, 4, 6, 2, 4, 6, 8, 4, 6, 8, 10, 6, 8, 10, 12};
    std::int32_t h_C_gpu[M * P] = {0};
    std::int32_t h_C_accelerated_gpu[M * P] = {0};

    // Call
    Launch(h_A, h_B, h_C_gpu, M, N, P);
    Launch(h_A, h_B, h_C_accelerated_gpu, M, N, P);

    // Check results
    for (std::int32_t i = 0; i < M * P; ++i)
    {
        EXPECT_EQ(h_C_gpu[i], h_C_accelerated_gpu[i]) << "Mismatch at index " << i;
    }
}
