/*
 * Jacobi Solver with Shared Memory in CUDA
 *
 */

#include "examples/matrix_multiplication/matrix_multiplication.h"
#include "examples/matrix_multiplication/test/launch_test.cuh"
#include "examples/matrix_multiplication/utils.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <time.h>

// void LaunchWithSharedMemory(const int32_t* A,
//                             const int32_t* B,
//                             int32_t* C,
//                             const std::int32_t M,
//                             const std::int32_t N,
//                             const std::int32_t P)
// {  // Allocate device memory
//     std::int32_t* d_A;
//     std::int32_t* d_B;
//     std::int32_t* d_C;

//     const auto start = clock();
//     if (utils::AllocateAndCopyToDevice(d_A, d_B, d_C, A, B, M, N, P) != 0)
//     {
//         return;  // Error already handled in AllocateAndCopyToDevice
//     }

//     // Launch kernel
//     const auto num_blocks_x = (P + kBlockSize - 1) / kBlockSize;
//     const auto num_blocks_y = (M + kBlockSize - 1) / kBlockSize;

//     const auto grid_dim = dim3(num_blocks_x, num_blocks_y, 1);
//     const auto block_dim = dim3(kBlockSize, kBlockSize);

//     printf("Launching kernel with configuration: (%d, %d) x (%d, %d) threads per block\n",
//            num_blocks_x,
//            num_blocks_y,
//            kBlockSize,
//            kBlockSize);

//     MatMultWithSharedMemoryGPU<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, P);

//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // Copy results back

//     if (cudaMemcpy(C, d_C, sizeof(std::int32_t) * M * P, cudaMemcpyDeviceToHost) != cudaSuccess)
//     {
//         std::cerr << "Failed to copy data from device to host\n";
//         cudaFree(d_A);
//         cudaFree(d_B);
//         cudaFree(d_C);
//         return;
//     }
//     const auto end = clock();
//     const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
//     std::cout << "Elapsed GPU time: " << elapsed_time << " seconds\n";

//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
// }
