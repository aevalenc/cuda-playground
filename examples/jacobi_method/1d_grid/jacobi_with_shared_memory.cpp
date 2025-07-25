/*
 * main.cpp
 */

#include "examples/jacobi_method/1d_grid/launch.h"
#include <iostream>

int main()
{
    std::int32_t number_of_runs = 10;
    double min_elapsed_time = 1e9;

    for (std::int32_t i = 0; i < number_of_runs; ++i)
    {
        double elapsed_time = LaunchJacobiWithSharedMemoryGPU();
        min_elapsed_time = std::min(min_elapsed_time, elapsed_time);
    }

    std::cout << "Min Elapsed time: " << min_elapsed_time << " seconds\n";
    return 0;
}
