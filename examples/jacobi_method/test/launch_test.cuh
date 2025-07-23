/*
 * Closest Neighbor Search using Shared Memory in CUDA Test
 */

#include <cstdint>
#include <cuda_runtime.h>

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

constexpr std::int32_t kMaxIterations = 1000;
constexpr double kTolerance = 1e-5;

void LaunchJacobiSolveCPU(const double* A,
                          const double* b,
                          double* x0,
                          double* x,
                          const std::int32_t N,
                          const double tolerance = kTolerance,
                          std::int32_t max_iterations = kMaxIterations);
void LaunchJacobiSolveGPU(const double* A,
                          const double* b,
                          double* x0,
                          double* x,
                          const std::int32_t N,
                          const double tolerance = kTolerance,
                          std::int32_t max_iterations = kMaxIterations);
void LaunchJacobiSolveWithTilingGPU(const double* A,
                                    const double* b,
                                    double* x0,
                                    double* x,
                                    const std::int32_t N,
                                    const double tolerance = kTolerance,
                                    std::int32_t max_iterations = kMaxIterations);
