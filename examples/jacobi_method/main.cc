/*
 * This file is part of the GPU Timeout Test project.
 */

#include "examples/jacobi_method/launch.h"
#include <iostream>

int main()
{
    // Set the number of points to be processed
    const std::int32_t N = 20001;
    const std::int32_t max_runs = 3;

    auto min_time_cpu = 1000.0;
    for (std::int32_t run = 0; run < max_runs; ++run)
    {
        std::cout << "Run " << run + 1 << " of " << max_runs << "\n";
        const auto cpu_time = LaunchCPU(N);
        if (cpu_time < min_time_cpu)
        {
            min_time_cpu = cpu_time;
        }
    }

    // Try different GPU configurations
    auto min_time_gpu = 1000.0;
    for (std::int32_t run = 0; run < max_runs; ++run)
    {
        std::cout << "GPU Run " << run + 1 << " of " << max_runs << "\n";
        const auto gpu_time = LaunchJacobiSolveGPU(N);

        if (gpu_time < min_time_gpu)
        {
            min_time_gpu = gpu_time;
        }
    }

    auto min_gpu_shared_memory_time = 1000.0;
    // for (std::int32_t run = 0; run < max_runs; ++run)
    // {
    //     std::cout << "Shared Memory Run " << run + 1 << " of " << max_runs << "\n";
    //     const auto gpu_shared_memory_time = LaunchGPUWithSharedMemory(M, N, P);

    //     if (gpu_shared_memory_time < min_gpu_shared_memory_time)
    //     {
    //         min_gpu_shared_memory_time = gpu_shared_memory_time;
    //     }
    // }

    std::cout << "\nMinimum CPU time: " << min_time_cpu << " s" << "\n";
    std::cout << "Minimum GPU time: " << min_time_gpu << " s" << "\n";
    std::cout << "Minimum Shared Memory GPU time: " << min_gpu_shared_memory_time << " s" << "\n";

    return 0;
}
