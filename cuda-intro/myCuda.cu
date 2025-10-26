#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceId;
    cudaGetDevice(&deviceId); // Get the ID of the current device

    cudaDeviceProp properties; // struct reference to store device properties  
    cudaGetDeviceProperties(&properties, deviceId); // Get properties of the device 

    std::cout << "--- " << properties.name << " ---" << std::endl;
    std::cout << "Compute Capability: " << properties.major << "." << properties.minor << std::endl;
    std::cout << std::endl;

    std::cout << "--- Grid Dimensions ---" << std::endl;
    std::cout << "Max Grid Size (x, y, z): ("
              << properties.maxGridSize[0] << ", "
              << properties.maxGridSize[1] << ", "
              << properties.maxGridSize[2] << ")" << std::endl;
    std::cout << std::endl;

    std::cout << "--- Block Dimensions ---" << std::endl;
    std::cout << "Max Block Size (x, y, z): ("
              << properties.maxThreadsDim[0] << ", "
              << properties.maxThreadsDim[1] << ", "
              << properties.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max Threads per Block: " << properties.maxThreadsPerBlock << std::endl;
    std::cout << std::endl;

    std::cout << "--- Warp and Multiprocessor Info ---" << std::endl;
    std::cout << "Warp Size: " << properties.warpSize << " threads" << std::endl;
    std::cout << "Multiprocessor Count: " << properties.multiProcessorCount << std::endl;
    std::cout << "Max Threads per Multiprocessor: " << properties.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Total Concurrent Threads: " << properties.multiProcessorCount * properties.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  (This represents the theoretical maximum number of threads that can execute simultaneously)" << std::endl;
    std::cout << "  (Calculated as: " << properties.multiProcessorCount << " SMs × " << properties.maxThreadsPerMultiProcessor << " threads/SM)" << std::endl;
    std::cout << std::endl;

    return 0;
}