/*
 * Jacobi Method CUDA Example
 *
 * Author: Alejandro Valencia
 */

#include "examples/jacobi_method/utils.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <time.h>

const std::int32_t N = 10001;
const double kTolerance = 1e-3;
const std::int32_t kMaxIterations = 1;

std::int32_t main()
{

    // Initialize the arrays
    double* host_A = utils::InitializeLaplaceMatrix(N);
    std::array<double, N> host_x0;
    std::array<double, N> host_b;
    std::array<double, N> host_x;
    std::array<double, N> residuals;
    for (std::int32_t i = 0; i < N; ++i)
    {
        host_x0[i] = 1.0;
        host_x[i] = 0.0;
    }

    for (std::int32_t i = 0; i < N; ++i)
    {
        if (i == 0)
        {
            host_b[i] = 200.0;
        }
        else if (i == N - 1)
        {
            host_b[i] = 400.0;
        }
        else
        {
            host_b[i] = 0.0;
        }
    }

    std::int32_t iteration{0};
    utils::MatrixMultiply(host_A, host_x0.data(), residuals.data(), N, N, 1);
    std::ignore =
        std::transform(residuals.begin(), residuals.end(), host_b.begin(), residuals.begin(), std::minus<double>());
    auto residual = utils::L2Norm(residuals.data(), N);

    const auto start = clock();
    while (residual > kTolerance)
    {
        ++iteration;

        // Jacobi iteration

        double sum = 0.0;
        for (std::int32_t i = 0; i < N; ++i)
        {
            sum = 0.0;
            for (std::int32_t k = 0; k < N; ++k)
            {
                if (k != i)
                {
                    sum += host_A[k + N * i] * host_x0[k];
                }
            }

            host_x[i] = (host_b[i] - sum) / host_A[i + N * i];
        }

        if (iteration == kMaxIterations)
        {
            std::cout << "Maximum iterations reached: " << kMaxIterations << "\n";
            break;
        }
        else
        {
            utils::MatrixMultiply(host_A, host_x0.data(), residuals.data(), N, N, 1);
            std::ignore = std::transform(
                residuals.begin(), residuals.end(), host_b.begin(), residuals.begin(), std::minus<double>());
            residual = utils::L2Norm(residuals.data(), N);
            std::cout << "Iteration: " << iteration << ", Residual: " << residual << "\n";
            std::copy(host_x.begin(), host_x.end(), host_x0.begin());
        }
    }

    const auto end = clock();
    const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed CPU time: " << elapsed_time << " seconds\n";

    delete[] host_A;

    return 0;
}
