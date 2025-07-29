/*
 * Jacobi Method CUDA Example
 *
 * Author: Alejandro Valencia
 */

#define USE_CPU_CONSTANTS 1

#include "examples/jacobi_method/cpu/launch.h"
#include <cstdint>
#include <tuple>

std::int32_t main()
{
    std::ignore = LaunchCPU();
    return 0;
}
