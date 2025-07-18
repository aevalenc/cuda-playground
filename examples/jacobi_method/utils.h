/*
 * utils.cpp
 */

#ifndef EXAMPLES_JACOBI_METHOD_UTILS_H
#define EXAMPLES_JACOBI_METHOD_UTILS_H

#include <cstdint>

namespace utils
{

void PrintVector(const std::int32_t* A, const std::int32_t N);
void PrintArray(const std::int32_t* A, const std::int32_t N);
void PrintMatrix(const std::int32_t* A, const std::int32_t M, const std::int32_t N);
std::int32_t* InitializeTestMatrix(const std::int32_t M, const std::int32_t N);
std::int32_t* InitializeLaplaceMatrix(const std::int32_t N);
std::int32_t AllocateAndCopyToDevice(std::int32_t*& device_A,
                                     std::int32_t*& device_B,
                                     std::int32_t*& device_C,
                                     const std::int32_t* host_A,
                                     const std::int32_t* host_B,
                                     const std::int32_t M,
                                     const std::int32_t N,
                                     const std::int32_t P);

}  // namespace utils
#endif  // EXAMPLES_JACOBI_METHOD_UTILS_H
