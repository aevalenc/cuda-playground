/*
 * Jacobi Solver with Shared Memory in CUDA
 *
 */

#include "examples/jacobi_method/jacobi.h"
#include "examples/jacobi_method/test/launch_test.cuh"
#include "examples/jacobi_method/utils.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <time.h>

void LaunchJacobiSolveGPU(const double* A, const double* b, double* x0, double* x, const std::int32_t N)

{
    // Allocate device memory
    double* device_A;
    double* device_b;
    double* device_x0;
    double* device_x;

    const auto start = clock();
    if (utils::AllocateAndCopyToDevice(device_A, device_b, device_x0, device_x, A, b, x0, x, N) != 0)
    {
        return;  // Error already handled in AllocateAndCopyToDevice
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
        if (cudaMemcpy(x, device_x, sizeof(double) * N, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::cerr << "Failed to copy data from device to host\n";
            cudaFree(device_A);
            cudaFree(device_b);
            cudaFree(device_x0);
            cudaFree(device_x);
            return;
        }

        // Check for convergence
        residual = utils::L2Norm(x, x0, N);
        if (residual < kTolerance)
        {
            std::cout << "Converged after " << i + 1 << " iterations with residual: " << residual << "\n";
            break;
        }
        else
        {
            std::cout << "Iteration: " << i + 1 << " | Residual: " << residual << "\n";
        }

        // Copy previous result to x0
        if (cudaMemcpy(device_x0, device_x, sizeof(double) * N, cudaMemcpyDeviceToDevice) != cudaSuccess)
        {
            std::cerr << "Failed to copy data from device to device\n";
            cudaFree(device_A);
            cudaFree(device_b);
            cudaFree(device_x0);
            cudaFree(device_x);
            return;
        }
    }

    const auto end = clock();
    const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed GPU time: " << elapsed_time << " seconds\n";

    cudaFree(device_A);
    cudaFree(device_b);
    cudaFree(device_x0);
    cudaFree(device_x);
}
