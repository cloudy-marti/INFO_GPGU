#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

// kernels are C++ functions defined with CUDA
// They will be called with << >>()

// cudaGetDeviceCount (int* count)
// 		Returns the number of compute-capable devices

// cudaGetDeviceProperties (cudaDeviceProp* prop, int device)
//		Returns information about the compute-device.


// Program that gives the information of the GPUs on the boards
int main() {
    int devices;
    cudaDeviceProp prop;
    
    try {
        cudaGetDeviceCount(&devices); 
        // Get information of all the Nvidia devices on the computer
        for(int device = 0; device < devices; device++) {
            cudaGetDeviceProperties(&prop, device);
            // using std::cout as a display function
            // using std::endl as a end of line character
            std::cout << "Device Number                : " << device << std::endl;
            std::cout << "Device name                  : " << prop.name << std::endl;
            std::cout << "Memory Clock Rate (KHz)      : " << prop.memoryClockRate << std::endl;
            std::cout << "Global Memory size (bits)    : " << prop.memoryBusWidth << std::endl;
            // get the warp size, i.e. the number of threads in a warp
            std::cout << "Warp Size                    : " << prop.warpSize << std::endl;
            std::cout << "Peak Memory Bandwidth (GB/s) : " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl;
        }
    }
    catch (const cudaError_t & e) {
        std::cerr << e;
    }

    return 0;
}

/*
    Device Number                : 0
    Device name                  : GeForce RTX 2060 SUPER
    Memory Clock Rate (KHz)      : 7001000
    Global Memory size (bits)    : 256
    Warp Size                    : 32
    Peak Memory Bandwidth (GB/s) : 448.064
*/