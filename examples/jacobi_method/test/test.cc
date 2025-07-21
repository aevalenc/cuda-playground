/*
 * Closest Neighbor Search using Shared Memory in CUDA Test
 */

#include "examples/jacobi_method/launch.h"
#include "examples/jacobi_method/test/launch_test.cuh"
#include "examples/jacobi_method/utils.h"
#include <cstdint>
#include <gtest/gtest.h>
// #include <vector>

TEST(JacobiSolverCPU, GivenBasicLaplaceMatrixExpectCorrectOutput)
{
    const std::int32_t N = 5;

    LaunchCPU(N);

    // Check results
    // EXPECT_EQ(h_C[0], 4);
    // EXPECT_EQ(h_C[1], 8);
    // EXPECT_EQ(h_C[2], 12);
    EXPECT_TRUE(true);  // Placeholder for actual checks
}

TEST(JacobiSolverGPU, GivenBasicLaplaceMatrixExpectCorrectOutput)
{
    const std::int32_t N = 5;
    double* host_A = utils::InitializeLaplaceMatrix(N);
    double* host_x0 = utils::InitializeTestMatrix(N, 1);
    double* host_b = new double[N];
    for (std::int32_t i = 0; i < N; ++i)
    {
        // Boundary conditions
        if (i == 0)
        {
            host_b[i] = 200.0;
        }
        else if (i == N - 1)
        {
            host_b[i] = 400.0;
        }
        else
        {
            host_b[i] = 0.0;
        }
    }

    double* host_x = new double[N];

    utils::PrintMatrix(host_A, N, N);
    utils::PrintVector(host_b, N);

    LaunchJacobiSolveGPU(host_A, host_b, host_x0, host_x, N);

    utils::PrintVector(host_x, N);

    // Check results
    // EXPECT_EQ(h_C[0], 4);
    // EXPECT_EQ(h_C[1], 8);
    // EXPECT_EQ(h_C[2], 12);
    EXPECT_TRUE(true);  // Placeholder for actual checks
}
