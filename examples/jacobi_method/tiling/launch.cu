
/*
 * Jacobi Method CUDA Example
 *
 * Author: Alejandro Valencia
 */

#include "examples/jacobi_method/tiling/jacobi.h"
#include "examples/jacobi_method/tiling/launch.h"
#include "examples/jacobi_method/utils.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <time.h>

double LaunchJacobiMethodWithTilingGPU(std::int32_t N)
{
    double* host_A = utils::InitializeLaplaceMatrix(N);
    std::vector<double> host_x0(N);
    std::vector<double> host_b(N);
    std::vector<double> host_x(N);
    std::vector<double> residuals(N);
    for (std::int32_t i = 0; i < N; ++i)
    {
        host_x0.at(i) = 1.0;
        host_x.at(i) = 0.0;
    }

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
    const auto num_blocks = (N + kTiledBlockSize - 1) / kTiledBlockSize;
    const auto grid_dim = dim3(num_blocks, num_blocks, 1);
    const auto block_dim = dim3(kTiledBlockSize, kTiledBlockSize, 1);
    printf("Launching kernel with configuration: (%d, %d, 1) x (%d, %d, 1) threads per block\n",
           num_blocks,
           num_blocks,
           kTiledBlockSize,
           kTiledBlockSize);

    double residual = 0.0;
    for (std::int32_t i = 0; i < kMaxIterations; ++i)
    {
        JacobiSolveWithTilingGPU<<<grid_dim, block_dim>>>(device_A, device_b, device_x0, device_x, N);

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
        utils::MatrixMultiply(host_A, host_x0.data(), residuals.data(), N, N, 1);
        std::ignore =
            std::transform(residuals.begin(), residuals.end(), host_b.begin(), residuals.begin(), std::minus<double>());
        residual = utils::L2Norm(residuals.data(), N);

        if (residual < kTolerance)
        {
            std::cout << "Converged after " << i + 1 << " iterations with residual: " << residual << "\n";
            break;
        }

        if (i == kMaxIterations - 1)
        {
            std::cout << "Maximum iterations reached: " << kMaxIterations << "\n";
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

    cudaFree(device_A);
    cudaFree(device_b);
    cudaFree(device_x0);
    cudaFree(device_x);
    cudaDeviceReset();

    return (static_cast<double>(end - start) / CLOCKS_PER_SEC);
}
