/*
 * This file is part of the GPU Timeout Test project.
 */

#include "examples/matrix_multiplication/launch.h"
#include <iostream>

int main()
{
    // Set the number of points to be processed
    std::int32_t N = 3;  // Adjust as needed
    const std::int32_t max_runs = 1;

    // auto min_time_cpu = 1000.0;
    // for (std::int32_t run = 0; run < max_runs; ++run)
    // {
    //     std::cout << "Run " << run + 1 << " of " << max_runs << "\n";
    //     const auto cpu_time = LaunchCPU(N);
    //     if (cpu_time < min_time_cpu)
    //     {
    //         min_time_cpu = cpu_time;
    //     }
    // }
    // std::cout << "Minimum CPU time: " << min_time_cpu << " s" << "\n";

    // Try different GPU configurations
    auto min_time_gpu = 1000.0;
    for (std::int32_t run = 0; run < max_runs; ++run)
    {
        std::cout << "Run " << run + 1 << " of " << max_runs << "\n";
        const auto gpu_time = LaunchGPU(N);

        if (gpu_time < min_time_gpu)
        {
            min_time_gpu = gpu_time;
        }
    }
    std::cout << "GPU time: " << min_time_gpu << " s" << "\n";

    return 0;
}
