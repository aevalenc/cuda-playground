/*
 * Jacobi Method CUDA Example
 *
 * Author: Alejandro Valencia
 */

#include "examples/jacobi_method/jacobi.h"
#include "examples/jacobi_method/launch.h"
#include "examples/jacobi_method/utils.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <time.h>

double LaunchCPU(const std::int32_t N)
{

    // Initialize the arrays
    double* host_A = utils::InitializeLaplaceMatrix(N);
    std::vector<double> host_x0(N, 1.0);
    std::vector<double> host_b(N);
    std::vector<double> host_x(N, 0.0);

    for (std::int32_t i = 0; i < N; ++i)
    {
        if (i == 0)
        {
            host_b.at(i) = 200.0;
        }
        else if (i == N - 1)
        {
            host_b.at(i) = 400.0;
        }
        else
        {
            host_b.at(i) = 0.0;
        }
    }

    std::int32_t iteration{0};
    auto residual = utils::L2Norm(host_x0.data(), host_x.data(), N);
    const auto start = clock();
    while (residual > kTolerance)
    {
        ++iteration;

        // Jacobi iteration
        for (std::int32_t i = 0; i < N; ++i)
        {
            // Initialize sum
            double sum = 0;
            for (std::int32_t k = 0; k < N; ++k)
            {
                if (k != i)
                {
                    sum += host_A[k + N * i] * host_x0[k];
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
        else
        {
            residual = utils::L2Norm(host_x.data(), host_x0.data(), N);
            std::copy(host_x.data(), host_x.data() + N, host_x0.data());
        }
    }

    const auto end = clock();
    const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed CPU time: " << elapsed_time << " seconds\n";

    delete[] host_A;

    return elapsed_time;
}

double LaunchJacobiSolveGPU(const std::int32_t N)
{
    double* host_A = utils::InitializeLaplaceMatrix(N);
    std::vector<double> host_x0(N, 1.0);
    std::vector<double> host_b(N);
    std::vector<double> host_x(N, 0.0);

    // Initialize b with boundary conditions
    for (std::int32_t i = 0; i < N; ++i)
    {
        // Boundary conditions
        if (i == 0)
        {
            host_b.at(i) = 200.0;
        }
        else if (i == N - 1)
        {
            host_b.at(i) = 400.0;
        }
        else
        {
            host_b.at(i) = 0.0;
        }
    }

    // Allocate device memory
    double* device_A;
    double* device_b;
    double* device_x0;
    double* device_x;

    const auto start = clock();
    if (utils::AllocateAndCopyToDevice(
            device_A, device_b, device_x0, device_x, host_A, host_b.data(), host_x0.data(), host_x.data(), N) != 0)
    {
        return -1;
    }

    // Launch kernel
    const auto num_blocks = (N + kBlockSize - 1) / kBlockSize;

    const auto grid_dim = dim3(num_blocks, 1, 1);
    const auto block_dim = dim3(kBlockSize);

    printf("Launching kernel with configuration: (%d, 1, 1) x (%d, 1, 1) threads per block\n", num_blocks, kBlockSize);

    double residual = 0.0;
    for (std::int32_t i = 0; i < kMaxIterations; ++i)
    {
        JacobiSolveGPU<<<grid_dim, block_dim>>>(device_A, device_b, device_x0, device_x, N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back
        if (cudaMemcpy(host_x.data(), device_x, sizeof(double) * N, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::cerr << "Failed to copy x data from device to host\n";
            cudaFree(device_A);
            cudaFree(device_b);
            cudaFree(device_x0);
            cudaFree(device_x);
            return -1;
        }

        if (cudaMemcpy(host_x0.data(), device_x0, sizeof(double) * N, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::cerr << "Failed to copy x0 data from device to host\n";
            cudaFree(device_A);
            cudaFree(device_b);
            cudaFree(device_x0);
            cudaFree(device_x);
            return -1;
        }

        // Check for convergence
        residual = utils::L2Norm(host_x0.data(), host_x.data(), N);
        if (residual < kTolerance)
        {
            std::cout << "Converged after " << i + 1 << " iterations with residual: " << residual << "\n";
            break;
        }

        // Copy previous result to x0
        if (cudaMemcpy(device_x0, device_x, sizeof(double) * N, cudaMemcpyDeviceToDevice) != cudaSuccess)
        {
            std::cerr << "Failed to copy data from device to device\n";
            cudaFree(device_A);
            cudaFree(device_b);
            cudaFree(device_x0);
            cudaFree(device_x);
            return -1;
        }
    }

    const auto end = clock();
    const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed GPU time: " << elapsed_time << " seconds\n";

    cudaFree(device_A);
    cudaFree(device_b);
    cudaFree(device_x0);
    cudaFree(device_x);
    cudaDeviceReset();

    return elapsed_time;
}

// double LaunchGPUWithSharedMemory(const std::int32_t N)
// {

//     // Initialize the arrays
//     const auto host_A = utils::InitializeLaplaceMatrix(N);
//     const auto host_B = utils::InitializeTestMatrix(N, 1);
//     const auto host_x0 = utils::InitializeTestMatrix(N, 1);
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
