/*
 * utils.cpp
 */

#include "examples/matrix_multiplication/utils.h"
#include <iostream>

namespace utils
{

void PrintVector(const std::int32_t* A, const std::int32_t N)
{
    std::cout << "[" << A[0];
    for (std::int32_t i = 1; i < N; ++i)
    {
        std::cout << "\n " << A[i];
    }
    std::cout << "]\n";
}

void PrintArray(const std::int32_t* A, const std::int32_t N)
{
    std::cout << "[" << A[0];
    for (std::int32_t i = 1; i < N; ++i)
    {
        std::cout << ", " << A[i];
    }
    std::cout << "]\n";
}

void PrintMatrix(const std::int32_t* A, const std::int32_t M, const std::int32_t N)
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

}  // namespace utils
