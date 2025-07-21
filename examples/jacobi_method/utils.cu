/*
 * utils.cpp
 */

#include "examples/jacobi_method/utils.h"
#include <cuda_runtime.h>
#include <iostream>

namespace utils
{

void PrintVector(const double* A, const std::int32_t N)
{
    std::cout << "[" << A[0];
    for (std::int32_t i = 1; i < N; ++i)
    {
        std::cout << "\n " << A[i];
    }
    std::cout << "]\n";
}

void PrintArray(const double* A, const std::int32_t N)
{
    std::cout << "[" << A[0];
    for (std::int32_t i = 1; i < N; ++i)
    {
        std::cout << ", " << A[i];
    }
    std::cout << "]\n";
}

void PrintMatrix(const double* A, const std::int32_t M, const std::int32_t N)
{
    std::cout << "[";
    for (std::int32_t i = 0; i < M; ++i)
    {

        if (i != (M - 1))
        {
            PrintArray(&A[i * N], N);
        }
        else
        {
            std::cout << "[" << A[N * i];
            for (std::int32_t j = 1; j < N; ++j)
            {
                std::cout << ", " << A[j + N * i];
            }
        }
    }
    std::cout << "]]\n";
}

double* InitializeTestMatrix(const std::int32_t M, const std::int32_t N)
{
    double* host_A = new double[M * N];

    for (std::int32_t i = 0; i < M; ++i)
    {
        for (std::int32_t j = 0; j < N; ++j)
        {
            host_A[j + N * i] = 2 + (i + j);  // Example initialization, can be random or specific values
        }
    }
    return host_A;
}

double* InitializeLaplaceMatrix(const std::int32_t N)
{
    double* host_A = new double[N * N];

    for (std::int32_t i = 0; i < N; ++i)
    {
        for (std::int32_t j = 0; j < N; ++j)
        {
            if (i == j)
            {
                host_A[j + N * i] = 2;  // Diagonal elements
            }
            else if (abs(i - j) == 1 || abs(i - j) == N)
            {
                host_A[j + N * i] = -1;  // Adjacent elements
            }
            else
            {
                host_A[j + N * i] = 0;  // Other elements
            }
        }
    }
    return host_A;
}

double L2Norm(const double* x, const double* xn, const std::int32_t N)
{
    double res = 0.0;
    for (std::int32_t i = 0; i < N; ++i)
    {
        res += (x[i] - xn[i]) * (x[i] - xn[i]);
    }
    return sqrt(res);
}

std::int32_t AllocateAndCopyToDevice(double*& device_A,
                                     double*& device_b,
                                     double*& device_x0,
                                     double*& device_x,
                                     const double* host_A,
                                     const double* host_b,
                                     double* host_x0,
                                     double* host_x,
                                     const std::int32_t N)
{
    // Allocate device memory
    if (cudaMalloc((void**)&device_A, sizeof(double) * N * N) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for matrix A\n";
        return -1;
    }
    if (cudaMalloc((void**)&device_b, sizeof(double) * N) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for matrix b;\n";
        cudaFree(device_A);
        return -1;
    }
    if (cudaMalloc((void**)&device_x0, sizeof(double) * N) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for initial guess x0;\n";
        cudaFree(device_A);
        cudaFree(device_b);
        return -1;
    }
    if (cudaMalloc((void**)&device_x, sizeof(double) * N) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for solution x;\n";
        cudaFree(device_A);
        cudaFree(device_b);
        cudaFree(device_x0);
        return -1;
    }

    // Copy data from host to device
    if (cudaMemcpy(device_A, host_A, sizeof(double) * N * N, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from host to device for Matrix A\n";
        cudaFree(device_A);
        cudaFree(device_b);
        cudaFree(device_x0);
        cudaFree(device_x);
        return -1;
    }
    if (cudaMemcpy(device_b, host_b, sizeof(double) * N, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from host to device for Matrix B\n";
        cudaFree(device_A);
        cudaFree(device_b);
        cudaFree(device_x0);
        cudaFree(device_x);
        return -1;
    }
    if (cudaMemcpy(device_x0, host_x0, sizeof(double) * N, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from host to device for Matrix B\n";
        cudaFree(device_A);
        cudaFree(device_b);
        cudaFree(device_x0);
        cudaFree(device_x);
        return -1;
    }

    return 0;
}

}  // namespace utils
