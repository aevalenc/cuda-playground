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

void LaunchJacobiSolveCPU(const double* A,
                          const double* b,
                          double* x0,
                          double* x,
                          const std::int32_t N,
                          const double tolerance,
                          std::int32_t max_iterations)
{

    std::int32_t iteration{0};
    std::vector<double> residuals(N, 0.0);

    utils::MatrixMultiply(A, x0, residuals.data(), N, N, 1);
    std::ignore = std::transform(residuals.begin(), residuals.end(), b, residuals.begin(), std::minus<double>());
    auto residual = utils::L2Norm(residuals.data(), N);

    const auto start = clock();
    while (residual > tolerance)
    {
        ++iteration;

        // Copy current x to x0
        std::copy(x, x + N, x0);

        // Jacobi iteration
        for (std::int32_t i = 0; i < N; ++i)
        {
            // Initialize sum
            double sum = 0;
            for (std::int32_t k = 0; k < N; ++k)
            {
                if (k != i)
                {
                    sum += A[k + N * i] * x0[k];
                }
            }

            // Calculate next iteration
            x[i] = (b[i] - sum) / A[i + N * i];
        }

        if (iteration == max_iterations)
        {
            std::cout << "Maximum iterations reached: " << max_iterations << "\n";
            break;
        }
        else
        {
            utils::MatrixMultiply(A, x0, residuals.data(), N, N, 1);
            std::ignore =
                std::transform(residuals.begin(), residuals.end(), b, residuals.begin(), std::minus<double>());
            residual = utils::L2Norm(residuals.data(), N);
            std::cout << "Iteration: " << iteration << "| Residual: " << residual << "\n";
        }
    }

    const auto end = clock();
    const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed CPU time: " << elapsed_time << " seconds\n";

    utils::PrintVector(x, N);
}

void LaunchJacobiSolveGPU(const double* A,
                          const double* b,
                          double* x0,
                          double* x,
                          const std::int32_t N,
                          const double tolerance,
                          std::int32_t max_iterations)
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

    std::vector<double> residuals(N, 0.0);
    double residual = 0.0;
    for (std::int32_t i = 0; i < max_iterations; ++i)
    {
        JacobiSolveGPU<<<grid_dim, block_dim>>>(device_A, device_b, device_x0, device_x, N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back
        if (cudaMemcpy(x, device_x, sizeof(double) * N, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::cerr << "Failed to copy x data from device to host\n";
            cudaFree(device_A);
            cudaFree(device_b);
            cudaFree(device_x0);
            cudaFree(device_x);
            return;
        }

        if (cudaMemcpy(x0, device_x0, sizeof(double) * N, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::cerr << "Failed to copy x0 data from device to host\n";
            cudaFree(device_A);
            cudaFree(device_b);
            cudaFree(device_x0);
            cudaFree(device_x);
            return;
        }

        // Check for convergence
        utils::MatrixMultiply(A, x0, residuals.data(), N, N, 1);
        std::ignore = std::transform(residuals.begin(), residuals.end(), b, residuals.begin(), std::minus<double>());
        residual = utils::L2Norm(residuals.data(), N);
        if (residual < tolerance)
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

        if (i == max_iterations - 1)
        {
            std::cout << "Maximum iterations reached: " << max_iterations << "\n";
        }
    }

    const auto end = clock();
    const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed GPU time: " << elapsed_time << " seconds\n";

    utils::PrintVector(x, N);

    cudaFree(device_A);
    cudaFree(device_b);
    cudaFree(device_x0);
    cudaFree(device_x);
}

void LaunchJacobiSolveWithTilingGPU(const double* A,
                                    const double* b,
                                    double* x0,
                                    double* x,
                                    const std::int32_t N,
                                    const double tolerance,
                                    std::int32_t max_iterations)
{
    // Allocate device memory
    double* device_A;
    double* device_b;
    double* device_x0;
    double* device_x;

    const auto start = clock();
    if (utils::AllocateAndCopyToDevice(device_A, device_b, device_x0, device_x, A, b, x0, x, N) != 0)
    {
        return;
    }

    // Launch kernel
    const auto num_blocks = (N + kBlockSize - 1) / kBlockSize;

    const auto grid_dim = dim3(num_blocks, num_blocks, 1);
    const auto block_dim = dim3(kBlockSize, kBlockSize, 1);

    printf("Launching kernel with configuration: (%d, %d, 1) x (%d, %d, 1) threads per block\n",
           num_blocks,
           num_blocks,
           kBlockSize,
           kBlockSize);

    std::vector<double> residuals(N, 0.0);
    double residual = 0.0;
    for (std::int32_t i = 0; i < max_iterations; ++i)
    {
        JacobiSolveWithTilingGPU<<<grid_dim, block_dim>>>(device_A, device_b, device_x0, device_x, N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back
        if (cudaMemcpy(x, device_x, sizeof(double) * N, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::cerr << "Failed to copy x data from device to host\n";
            cudaFree(device_A);
            cudaFree(device_b);
            cudaFree(device_x0);
            cudaFree(device_x);
            return;
        }

        if (cudaMemcpy(x0, device_x0, sizeof(double) * N, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::cerr << "Failed to copy x0 data from device to host\n";
            cudaFree(device_A);
            cudaFree(device_b);
            cudaFree(device_x0);
            cudaFree(device_x);
            return;
        }

        // Check for convergence
        utils::MatrixMultiply(A, x0, residuals.data(), N, N, 1);
        std::ignore = std::transform(residuals.begin(), residuals.end(), b, residuals.begin(), std::minus<double>());
        residual = utils::L2Norm(residuals.data(), N);

        if (residual < tolerance)
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

        if (i == max_iterations - 1)
        {
            std::cout << "Maximum iterations reached: " << max_iterations << "\n";
        }
    }

    const auto end = clock();
    const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed GPU time: " << elapsed_time << " seconds\n";

    utils::PrintVector(x, N);

    cudaFree(device_A);
    cudaFree(device_b);
    cudaFree(device_x0);
    cudaFree(device_x);
}
