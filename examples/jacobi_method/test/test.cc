/*
 * Jacobi Solver in CUDA Tests
 *
 * Author: Alejandro Valencia
 * Date: July 22, 2025
 */

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
    const double kTestTolerance = 1e-3;
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
    EXPECT_NEAR(host_x_[0], 233.333, kTestTolerance);
    EXPECT_NEAR(host_x_[1], 266.666, kTestTolerance);
    EXPECT_NEAR(host_x_[2], 300.0, kTestTolerance);
    EXPECT_NEAR(host_x_[3], 333.333, kTestTolerance);
    EXPECT_NEAR(host_x_[4], 366.666, kTestTolerance);
}

TEST_F(JacobiTestFixture, GivenBasicLaplaceMatrixCallGPUSolverExpectCorrectOutput)
{
    // Call
    LaunchJacobiSolveGPU(host_A_, host_b_, host_x0_.data(), host_x_.data(), kMatrixSize);

    // Expect
    EXPECT_NEAR(host_x_[0], 233.333, kTestTolerance);
    EXPECT_NEAR(host_x_[1], 266.666, kTestTolerance);
    EXPECT_NEAR(host_x_[2], 300.0, kTestTolerance);
    EXPECT_NEAR(host_x_[3], 333.333, kTestTolerance);
    EXPECT_NEAR(host_x_[4], 366.666, kTestTolerance);
}
