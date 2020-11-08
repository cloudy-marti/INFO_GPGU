#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

// kernels are C++ functions defined with CUDA
// They will be called with << >>()

// cudaGetDeviceCount (int* count)
// 		Returns the number of compute-capable devices

// cudaGetDeviceProperties (cudaDeviceProp* prop, int device)
//		Returns information about the compute-device.

// the name
// the clock frequency
// the global memory size
// the warp size of all the GPUs on board
/*
int main() {
    int count;
    cudaGetDeviceCount(&count);

    cudaDeviceProp stats;
    for(int index = 0; index < count; index++) {
    	cudaGetDeviceProperties(&stats, index);
    	printf("Device Number\t\t\t: %d\nDevice Name\t\t\t: %s\nClock Frequency (kHz)\t\t: %d\nGlobal Memory Size (bytes)\t: %zd\nWarp size (threads)\t\t: %d\n",
    		index, stats.name, stats.clockRate, stats.totalGlobalMem, stats.warpSize);
    }
}
*/

int main() {
    int devices;
    cudaDeviceProp prop;
    
    try {
        cudaGetDeviceCount(&devices);
        for(int device = 0; device < devices; device++) {
            cudaGetDeviceProperties(&prop, device);
            std::cout << "Device Number                : " << device << std::endl;
            std::cout << "Device name                  : " << prop.name << std::endl;
            std::cout << "Memory Clock Rate (KHz)      : " << prop.memoryClockRate << std::endl;
            std::cout << "Global Memory size (bits)    : " << prop.memoryBusWidth << std::endl;
            std::cout << "Warp Size                    : " << prop.warpSize << std::endl;
            std::cout << "Peak Memory Bandwidth (GB/s) : " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl;
        }
    }
    catch (const cudaError_t & e) {
        std::cerr << e;
    }

    return 0;
}