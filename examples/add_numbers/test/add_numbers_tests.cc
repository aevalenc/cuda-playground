/*
 * Add 2 arrays using CUDA
 */

#include "examples/add_numbers/test/add_numbers.cuh"
#include <cstdint>
#include <gtest/gtest.h>

TEST(ClosestNeighborSharedMemoryTest, SmallPointSet)
{
    constexpr int num_points = 4;
    std::int32_t h_a[num_points] = {1, 2, 3, 4};
    std::int32_t h_b[num_points] = {5, 6, 7, 8};

    Launch(num_points, h_a, h_b);

    // Check results
    EXPECT_EQ(h_a[0], 6);
    EXPECT_EQ(h_a[1], 8);
    EXPECT_EQ(h_a[2], 10);
    EXPECT_EQ(h_a[3], 12);
}
