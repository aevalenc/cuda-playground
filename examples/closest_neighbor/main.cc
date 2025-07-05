/*
 * This file is part of the GPU Timeout Test project.
 */

#include "examples/closest_neighbor/launch.h"
#include <iostream>

int main()
{
    // Set the number of points to be processed
    std::int64_t number_of_points = 10000;  // Adjust as needed

    const std::int32_t max_runs = 10;
    auto min_time_cpu = 1000;
    for (std::int32_t run = 0; run < max_runs; ++run)
    {
        std::cout << "Run " << run + 1 << " of 5" << "\n";
        const auto cpu_time = LaunchCPU(number_of_points, 0);
        if (cpu_time < min_time_cpu)
        {
            min_time_cpu = cpu_time;
        }
    }
    std::cout << "Minimum CPU time: " << min_time_cpu << " ms" << "\n";

    // Try different GPU configurations
    auto min_time_gpu = 1000;
    for (std::int32_t run = 0; run < max_runs; ++run)
    {
        std::cout << "Run " << run + 1 << " of " << max_runs << "\n";
        const auto gpu_time = LaunchGPU(number_of_points, 0, 256, 16);
        if (gpu_time < min_time_gpu)
        {
            min_time_gpu = gpu_time;
        }
    }
    std::cout << "GPU time: " << min_time_gpu << " ms" << "\n";

    std::cout << "-----Increased threads per block" << "\n";
    LaunchGPU(number_of_points, 0, 512, 32);

    std::cout << "-----Further increased threads per block" << "\n";
    LaunchGPU(number_of_points, 0, 1024, 64);

    std::cout << "-----Increased blocks per grid" << "\n";
    LaunchGPU(number_of_points, 0, 256, 32);

    std::cout << "-----Increased both threads per block and blocks per grid" << "\n";
    LaunchGPU(number_of_points, 0, 512, 64);

    std::cout << "-----Maximum threads per block and blocks per grid" << "\n";
    LaunchGPU(number_of_points, 0, 1024, 128);  // Maximum

    std::cout << "-----Maximum threads per block and blocks per grid" << "\n";
    LaunchGPU(number_of_points, 0, 1024, 1024);  // Maximum

    return 0;
}
