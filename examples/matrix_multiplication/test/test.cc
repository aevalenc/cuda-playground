/*
 * Closest Neighbor Search using Shared Memory in CUDA Test
 */

#include "examples/matrix_multiplication/test/launch_test.cuh"
#include "examples/matrix_multiplication/utils.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

TEST(MatrixMultiplicationTest, GivenSmallDifferentSizedMatricesExpectCorrectOutput)
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

class MultiDimensionalGridTestFixture : public ::testing::Test
{
  public:
    MultiDimensionalGridTestFixture() = default;
    ~MultiDimensionalGridTestFixture() override = default;

    void CheckResults(std::int32_t* result)
    {
        EXPECT_EQ(result[0], 56);
        EXPECT_EQ(result[1], 80);
        EXPECT_EQ(result[2], 104);
        EXPECT_EQ(result[3], 128);
        EXPECT_EQ(result[4], 80);
        EXPECT_EQ(result[5], 120);
        EXPECT_EQ(result[6], 160);
        EXPECT_EQ(result[7], 200);
        EXPECT_EQ(result[8], 104);
        EXPECT_EQ(result[9], 160);
        EXPECT_EQ(result[10], 216);
        EXPECT_EQ(result[11], 272);
        EXPECT_EQ(result[12], 128);
        EXPECT_EQ(result[13], 200);
        EXPECT_EQ(result[14], 272);
        EXPECT_EQ(result[15], 344);
    }

  public:
    std::int32_t M{4};
    std::int32_t N{4};
    std::int32_t P{4};

    std::int32_t h_A[16] = {0, 2, 4, 6, 2, 4, 6, 8, 4, 6, 8, 10, 6, 8, 10, 12};
    std::int32_t h_B[16] = {0, 2, 4, 6, 2, 4, 6, 8, 4, 6, 8, 10, 6, 8, 10, 12};
};

TEST_F(MultiDimensionalGridTestFixture, TestWithAcceleratedKernel)
{
    // Given
    std::vector<std::int32_t> h_C(M * P, 0);

    // Call
    LaunchAccelerated(h_A, h_B, h_C.data(), M, N, P);

    // Expect
    CheckResults(h_C.data());
}

TEST_F(MultiDimensionalGridTestFixture, TestWithSharedMemoryKernel)
{
    // Given
    std::vector<std::int32_t> h_C(M * P, 0);

    // Call
    LaunchWithSharedMemory(h_A, h_B, h_C.data(), M, N, P);

    // Expect
    CheckResults(h_C.data());
}

TEST_F(MultiDimensionalGridTestFixture, GivenSameInputExpectEqualOutput)
{
    // Given
    std::vector<std::int32_t> h_C_gpu(M * P, 0);
    std::vector<std::int32_t> h_C_accelerated_gpu(M * P, 0);
    std::vector<std::int32_t> h_C_shared_gpu(M * P, 0);

    // Call
    Launch(h_A, h_B, h_C_gpu.data(), M, N, P);
    LaunchAccelerated(h_A, h_B, h_C_accelerated_gpu.data(), M, N, P);
    LaunchWithSharedMemory(h_A, h_B, h_C_shared_gpu.data(), M, N, P);

    // Check results
    for (std::int32_t i = 0; i < M * P; ++i)
    {
        EXPECT_EQ(h_C_gpu[i], h_C_accelerated_gpu[i]) << "Mismatch w/ accelerated matmult at index " << i;
        EXPECT_EQ(h_C_gpu[i], h_C_shared_gpu[i]) << "Mismatch w/ shared memory matmult at index " << i;
    }
}
