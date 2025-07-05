/*
 * add_numbers.cu
 *
 * Copyright (C) 2025 Name Alejandro Valencia
 */

#include <cuda.h>
#include <iostream>

#define CUDA_CHECK(expr)                                                                                     \
    do                                                                                                       \
    {                                                                                                        \
        cudaError_t err = (expr);                                                                            \
        if (err != cudaSuccess)                                                                              \
        {                                                                                                    \
            fprintf(stderr, "CUDA Error Code  : %d\n     Error String: %s\n", err, cudaGetErrorString(err)); \
            exit(err);                                                                                       \
        }                                                                                                    \
    } while (0)

__global__ void AddIntsCUDA(int* a, int* b)
{
    a[0] += b[0];
}

void Launch()
{
    int a = 5;
    int b = 10;
    int* d_a;
    int* d_b;

    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    AddIntsCUDA<<<1, 1>>>(d_a, d_b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "The answer is " << a << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
}
