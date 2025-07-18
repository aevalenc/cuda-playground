/*
 * Jacobi Method CUDA Example
 *
 * Author: Alejandro Valencia
 */

#include "examples/jacobi_method/jacobi.h"
#include "examples/jacobi_method/launch.h"
#include "examples/jacobi_method/utils.h"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <time.h>

namespace
{

const double kTolerance = 1e-5;
const std::int32_t kMaxIterations = 1000;

double residual(const std::int32_t* x, const std::int32_t* xn, const std::int32_t N)
{
    double res = 0.0;
    for (std::int32_t i = 0; i < N; ++i)
    {
        res += (x[i] - xn[i]) * (x[i] - xn[i]);
    }
    return sqrt(res);
}

}  // namespace

double LaunchCPU(const std::int32_t N)
{

    // Initialize the arrays
    const auto host_A = utils::InitializeLaplaceMatrix(N);
    std::int32_t* host_b = new std::int32_t[N];
    for (std::int32_t i = 0; i < N; ++i)
    {
        if (i == 0)
        {
            host_b[i] = 200;
        }
        else if (i == N - 1)
        {
            host_b[i] = 400;
        }
        else
        {
            host_b[i] = 0;
        }
    }

    const auto host_xn = utils::InitializeTestMatrix(N, 1);
    std::int32_t* host_x = new std::int32_t[N];

    utils::PrintMatrix(host_A, N, N);
    utils::PrintMatrix(host_b, N, 1);

    std::int32_t iteration{0};
    const auto start = clock();
    while (residual(host_xn, host_x, N) > kTolerance)
    {
        ++iteration;
        std::cout << "Iteration: " << iteration << "\n";

        // Copy current x to xn
        std::copy(host_x, host_x + N, host_xn);

        // Jacobi iteration
        for (std::int32_t i = 0; i < N; ++i)
        {
            // Initialize sum
            double sum = 0;
            for (std::int32_t k = 0; k < N; ++k)
            {
                if (k != i)
                {
                    sum += host_A[k + N * i] * host_xn[k];
                }
            }

            // Calculate next iteration
            host_x[i] = (host_b[i] - sum) / host_A[i + N * i];
        }

        if (iteration == kMaxIterations)
        {
            std::cout << "Maximum iterations reached: " << kMaxIterations << "\n";
            break;
        }
    }

    const auto end = clock();
    const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed CPU time: " << elapsed_time << " seconds\n";

    utils::PrintVector(host_x, N);

    delete[] host_A;
    delete[] host_b;
    delete[] host_x;
    delete[] host_xn;

    return elapsed_time;
}

// double LaunchGPUWithSharedMemory(const std::int32_t N)
// {

//     // Initialize the arrays
//     const auto host_A = utils::InitializeLaplaceMatrix(N);
//     const auto host_B = utils::InitializeTestMatrix(N, 1);
//     const auto host_xn = utils::InitializeTestMatrix(N, 1);
//     std::int32_t* host_x = new std::int32_t[N];

//     // Allocate device memory
//     std::int32_t* device_A;
//     std::int32_t* device_B;
//     std::int32_t* device_C;

//     const auto start = clock();
//     if (utils::AllocateAndCopyToDevice(device_A, device_B, device_C, host_A, host_B, M, N, P) != 0)
//     {
//         return -1;
//     }

//     const auto num_blocks_x = (P + kBlockSize - 1) / kBlockSize;
//     const auto num_blocks_y = (M + kBlockSize - 1) / kBlockSize;

//     const auto grid_dim = dim3(num_blocks_x, num_blocks_y, 1);
//     const auto block_dim = dim3(kBlockSize, kBlockSize);

//     std::cout << "Launching kernel with configuration: (" << num_blocks_x << ", " << num_blocks_y << ") x ("
//               << kBlockSize << ", " << kBlockSize << ") \n";
//     MatMultWithSharedMemoryGPU<<<grid_dim, block_dim>>>(device_A, device_B, device_C, M, N, P);

//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     if (cudaMemcpy(host_C, device_C, sizeof(std::int32_t) * M * P, cudaMemcpyDeviceToHost) != cudaSuccess)
//     {
//         std::cerr << "Failed to copy data from device to host\n";
//         delete[] host_A;
//         delete[] host_B;
//         delete[] host_C;
//         cudaFree(device_A);
//         cudaFree(device_B);
//         cudaFree(device_C);
//         return -1;
//     }

//     const auto end = clock();
//     const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
//     std::cout << "Elapsed GPU time: " << elapsed_time << " seconds\n";

//     delete[] host_A;
//     delete[] host_B;
//     delete[] host_C;

//     cudaFree(device_A);
//     cudaFree(device_B);
//     cudaFree(device_C);

//     cudaDeviceReset();

//     return elapsed_time;
// }
