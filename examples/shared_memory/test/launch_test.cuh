/*
 * Closest Neighbor Search using Shared Memory in CUDA Test
 */

#include <cstdint>
#include <cuda_runtime.h>

void Launch(const std::int32_t number_of_points, const double2* points, std::int32_t* neighbors);
