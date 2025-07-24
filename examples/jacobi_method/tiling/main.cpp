/*
 * main.cpp
 */

#include "examples/jacobi_method/tiling/launch.h"
#include <iostream>

int main()
{
    double elapsed_time = LaunchGPU();
    std::cout << "Elapsed time: " << elapsed_time << " seconds\n";
    return 0;
}
