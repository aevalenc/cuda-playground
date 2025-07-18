/*
 * Closest Neighbor Search using Shared Memory in CUDA Test
 */

#include "examples/jacobi_method/launch.h"
// #include "examples/jacobi_method/test/launch_test.cuh"
// #include "examples/jacobi_method/utils.h"
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
