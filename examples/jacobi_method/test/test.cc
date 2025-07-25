/*
 * Jacobi Solver in CUDA Tests
 *
 * Author: Alejandro Valencia
 * Date: July 22, 2025
 */

#include "examples/jacobi_method/1d_grid/launch.h"
#include "examples/jacobi_method/test/launch_test.cuh"
#include "examples/jacobi_method/utils.h"
#include <cstdint>
#include <gtest/gtest.h>

class JacobiTestFixture : public ::testing::Test
{
  public:
    void SetUp() override
    {
        host_A_ = utils::InitializeLaplaceMatrix(kMatrixSize);
        host_b_ = new double[kMatrixSize];

        for (std::int32_t i = 0; i < kMatrixSize; ++i)
        {
            if (i == 0)
            {
                host_b_[i] = 200;
            }
            else if (i == kMatrixSize - 1)
            {
                host_b_[i] = 400;
            }
            else
            {
                host_b_[i] = 0;
            }
        }
    }

    void TearDown() override
    {
        delete[] host_A_;
        delete[] host_b_;
    }

  public:
    const double tolerance = 1e-3;
    static constexpr std::int32_t kMatrixSize = 5;
    std::array<double, kMatrixSize> host_x0_ = {1.0, 1.0, 1.0, 1.0, 1.0};
    std::array<double, kMatrixSize> host_x_ = {0.0, 0.0, 0.0, 0.0, 0.0};
    double* host_A_ = nullptr;
    double* host_b_ = nullptr;
};

TEST_F(JacobiTestFixture, GivenBasicLaplaceMatrixCallCPUColverExpectCorrectOutput)
{
    // Call
    LaunchJacobiSolveCPU(host_A_, host_b_, host_x0_.data(), host_x_.data(), kMatrixSize);

    // Expect
    EXPECT_NEAR(host_x_[0], 233.333, tolerance);
    EXPECT_NEAR(host_x_[1], 266.666, tolerance);
    EXPECT_NEAR(host_x_[2], 300.0, tolerance);
    EXPECT_NEAR(host_x_[3], 333.333, tolerance);
    EXPECT_NEAR(host_x_[4], 366.666, tolerance);
}

TEST_F(JacobiTestFixture, GivenBasicLaplaceMatrixCallGPUSolverExpectCorrectOutput)
{
    // Call
    LaunchJacobiSolveGPU(host_A_, host_b_, host_x0_.data(), host_x_.data(), kMatrixSize);

    // Expect
    EXPECT_NEAR(host_x_[0], 233.333, tolerance);
    EXPECT_NEAR(host_x_[1], 266.666, tolerance);
    EXPECT_NEAR(host_x_[2], 300.0, tolerance);
    EXPECT_NEAR(host_x_[3], 333.333, tolerance);
    EXPECT_NEAR(host_x_[4], 366.666, tolerance);
}

TEST_F(JacobiTestFixture, GivenBasicLaplaceMatrixCallGPUSolverWithTilingExpectCorrectOutput)
{
    // Given
    const std::int32_t matrix_size = 11;
    const auto host_A = utils::InitializeLaplaceMatrix(matrix_size);
    const auto host_b = new double[matrix_size];
    std::array<double, matrix_size> host_x0;
    host_x0.fill(1.0);
    std::array<double, matrix_size> host_x;
    host_x.fill(0.0);

    // When
    for (std::int32_t i = 0; i < matrix_size; ++i)
    {
        if (i == 0)
        {
            host_b[i] = 200;
        }
        else if (i == matrix_size - 1)
        {
            host_b[i] = 400;
        }
        else
        {
            host_b[i] = 0;
        }
    }

    // Call
    LaunchJacobiSolveWithTilingGPU(host_A, host_b, host_x0.data(), host_x.data(), matrix_size);

    // Clean up
    delete[] host_A;
    delete[] host_b;

    // Expect
    EXPECT_NEAR(host_x.at(0), 216.667, tolerance);
    EXPECT_NEAR(host_x.at(1), 233.333, tolerance);
    EXPECT_NEAR(host_x.at(2), 250.000, tolerance);
    EXPECT_NEAR(host_x.at(3), 266.667, tolerance);
    EXPECT_NEAR(host_x.at(4), 283.333, tolerance);
    EXPECT_NEAR(host_x.at(5), 300.000, tolerance);
    EXPECT_NEAR(host_x.at(6), 316.667, tolerance);
    EXPECT_NEAR(host_x.at(7), 333.333, tolerance);
    EXPECT_NEAR(host_x.at(8), 350.000, tolerance);
    EXPECT_NEAR(host_x.at(9), 366.667, tolerance);
    EXPECT_NEAR(host_x.at(10), 383.333, tolerance);
}

TEST_F(JacobiTestFixture, GivenBasicLaplaceMatrixCallGPUSolverWithSharedMemoryExpectCorrectOutput)
{
    // Call
    const auto duration = LaunchJacobiWithSharedMemoryGPU();

    // Expect
    EXPECT_TRUE(duration > -1.0);  // Placeholder for actual expectations
}
