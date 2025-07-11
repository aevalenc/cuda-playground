/*
 * Closest Neighbor Search using Shared Memory in CUDA Test
 */

#include "examples/shared_memory/test/launch_test.cuh"
#include <cstdint>
#include <gtest/gtest.h>

TEST(ClosestNeighborSharedMemoryTest, SmallPointSet)
{
    constexpr int num_points = 6;
    const double2 h_points[num_points] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {5.0, 5.0}, {6.0, 6.0}, {10.0, 10.0}};
    std::int32_t h_neighbors[num_points] = {0};

    Launch(num_points, h_points, h_neighbors);

    // Check results
    EXPECT_TRUE(h_neighbors[0] == 1 || h_neighbors[0] == 2);
    EXPECT_EQ(h_neighbors[1], 0);
    EXPECT_EQ(h_neighbors[2], 0);
    EXPECT_EQ(h_neighbors[3], 4);
    EXPECT_EQ(h_neighbors[4], 3);
    EXPECT_EQ(h_neighbors[5], 4);
}
