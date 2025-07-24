/*
 * utils.cpp
 */

#ifndef EXAMPLES_JACOBI_METHOD_UTILS_H
#define EXAMPLES_JACOBI_METHOD_UTILS_H

#include <cstdint>

namespace utils
{

void PrintVector(const double* A, const std::int32_t N);
void PrintArray(const double* A, const std::int32_t N);
void PrintMatrix(const double* A, const std::int32_t M, const std::int32_t N);

double* InitializeTestMatrix(const std::int32_t M, const std::int32_t N);
double* InitializeLaplaceMatrix(const std::int32_t N);
double L2Norm(const double* x, const std::int32_t N);
void MatrixMultiply(const double* A,
                    const double* B,
                    double* C,
                    const std::int32_t M,
                    const std::int32_t N,
                    const std::int32_t P);

std::int32_t AllocateAndCopyToDevice(double*& device_A,
                                     double*& device_b,
                                     double*& device_x0,
                                     double*& device_x,
                                     const double* A,
                                     const double* b,
                                     double* x0,
                                     double* x,
                                     const std::int32_t N);

}  // namespace utils
#endif  // EXAMPLES_JACOBI_METHOD_UTILS_H
