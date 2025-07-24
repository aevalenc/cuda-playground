/*
 * main.cpp
 */

#include "examples/jacobi_method/1d_grid/launch.h"
#include <iostream>

int main()
{
    double elapsed_time = LaunchGPU();
    std::cout << "Elapsed time: " << elapsed_time << " seconds\n";
    return 0;
}
