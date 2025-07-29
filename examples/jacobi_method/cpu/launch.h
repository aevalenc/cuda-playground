/*
 * File launch.h
 */

#ifndef EXAMPLES_JACOBI_METHOD_CPU_LAUNCH_H
#define EXAMPLES_JACOBI_METHOD_CPU_LAUNCH_H

#include <cstdint>

const std::int32_t kCPUNumberOfElements = 10001;
const double kCPUTolerance = 1e-3;
const std::int32_t kCPUMaxIterations = 1;

double LaunchCPU(const std::int32_t N = kCPUNumberOfElements);

#endif  // EXAMPLES_JACOBI_METHOD_CPU_LAUNCH_H
