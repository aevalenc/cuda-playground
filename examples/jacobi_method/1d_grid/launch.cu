/*
 * Jacobi Method CUDA Example
 *
 * Author: Alejandro Valencia
 */

#include "examples/jacobi_method/1d_grid/jacobi.h"
#include "examples/jacobi_method/1d_grid/launch.h"
#include "examples/jacobi_method/utils.h"
#include <algorithm>
#include <array>
#include <cuda_runtime.h>
#include <iostream>
#include <time.h>

double LaunchGPU()
{
    double* host_A = utils::InitializeLaplaceMatrix(kNumberOfElements);
    std::array<double, kNumberOfElements> host_x0;
    std::array<double, kNumberOfElements> host_b;
    std::array<double, kNumberOfElements> host_x;
    std::array<double, kNumberOfElements> residuals;
    for (std::int32_t i = 0; i < kNumberOfElements; ++i)
    {
        host_x0.at(i) = 1.0;
        host_x.at(i) = 0.0;
    }

    for (std::int32_t i = 0; i < kNumberOfElements; ++i)
    {
        if (i == 0)
        {
            host_b.at(i) = 200.0;
        }
        else if (i == kNumberOfElements - 1)
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
    if (utils::AllocateAndCopyToDevice(device_A,
                                       device_b,
                                       device_x0,
                                       device_x,
                                       host_A,
                                       host_b.data(),
                                       host_x0.data(),
                                       host_x.data(),
                                       kNumberOfElements) != 0)
    {
        return -1;
    }

    // Launch kernel
    const auto num_blocks = (kNumberOfElements + kBlockSize - 1) / kBlockSize;
    const auto grid_dim = dim3(num_blocks, 1, 1);
    const auto block_dim = dim3(kBlockSize);
    printf("Launching kernel with configuration: (%d, 1, 1) x (%d, 1, 1) threads per block\n", num_blocks, kBlockSize);

    double residual = 0.0;
    for (std::int32_t i = 0; i < kMaxIterations; ++i)
    {
        JacobiSolveGPU<<<grid_dim, block_dim>>>(device_A, device_b, device_x0, device_x, kNumberOfElements);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back
        if (cudaMemcpy(host_x.data(), device_x, sizeof(double) * kNumberOfElements, cudaMemcpyDeviceToHost) !=
            cudaSuccess)
        {
            std::cerr << "Failed to copy x data from device to host\n";
            cudaFree(device_A);
            cudaFree(device_b);
            cudaFree(device_x0);
            cudaFree(device_x);
            return -1;
        }

        if (cudaMemcpy(host_x0.data(), device_x0, sizeof(double) * kNumberOfElements, cudaMemcpyDeviceToHost) !=
            cudaSuccess)
        {
            std::cerr << "Failed to copy x0 data from device to host\n";
            cudaFree(device_A);
            cudaFree(device_b);
            cudaFree(device_x0);
            cudaFree(device_x);
            return -1;
        }

        // Check for convergence
        utils::MatrixMultiply(host_A, host_x0.data(), residuals.data(), kNumberOfElements, kNumberOfElements, 1);
        std::ignore =
            std::transform(residuals.begin(), residuals.end(), host_b.begin(), residuals.begin(), std::minus<double>());
        residual = utils::L2Norm(residuals.data(), kNumberOfElements);

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
        if (cudaMemcpy(device_x0, device_x, sizeof(double) * kNumberOfElements, cudaMemcpyDeviceToDevice) !=
            cudaSuccess)
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
