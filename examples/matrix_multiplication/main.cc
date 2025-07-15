/*
 * This file is part of the GPU Timeout Test project.
 */

#include "examples/matrix_multiplication/launch.h"
#include <iostream>

int main()
{
    // Set the number of points to be processed
    const std::int32_t M = 3000;
    const std::int32_t N = 200;
    const std::int32_t P = 2000;
    const std::int32_t max_runs = 3;

    auto min_time_cpu = 1000.0;
    for (std::int32_t run = 0; run < max_runs; ++run)
    {
        std::cout << "Run " << run + 1 << " of " << max_runs << "\n";
        const auto cpu_time = LaunchCPU(M, N, P);
        if (cpu_time < min_time_cpu)
        {
            min_time_cpu = cpu_time;
        }
    }

    // Try different GPU configurations
    auto min_time_gpu = 1000.0;
    for (std::int32_t run = 0; run < max_runs; ++run)
    {
        std::cout << "Run " << run + 1 << " of " << max_runs << "\n";
        const auto gpu_time = LaunchGPU(M, N, P);

        if (gpu_time < min_time_gpu)
        {
            min_time_gpu = gpu_time;
        }
    }

    auto min_gpu_accelerated_time = 1000.0;
    for (std::int32_t run = 0; run < max_runs; ++run)
    {
        std::cout << "Run " << run + 1 << " of " << max_runs << "\n";
        const auto gpu_accelerated_time = LaunchGPUAccelerated(M, N, P);

        if (gpu_accelerated_time < min_gpu_accelerated_time)
        {
            min_gpu_accelerated_time = gpu_accelerated_time;
        }
    }

    std::cout << "\nMinimum CPU time: " << min_time_cpu << " s" << "\n";
    std::cout << "Minimum GPU time: " << min_time_gpu << " s" << "\n";
    std::cout << "Minimum Accelerated GPU time: " << min_gpu_accelerated_time << " s" << "\n";

    return 0;
}
