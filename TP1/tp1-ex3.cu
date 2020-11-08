#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void kernel() {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t n = tid;
	uint32_t sum = 0;
    uint32_t prod = 1;
    while(n != 0){
        uint32_t digit = n % 10;
        n /= 10;
        sum += digit;
        prod *= digit;
    }
    if(sum*prod == tid) printf("%u\n", tid);
	return;
}

void checkrange(uint32_t range){
    double dim = sqrt(range);
    printf("Checking %u for sum-product numbers\n", range);

    uint32_t nthreads = (uint32_t)ceil(range/(dim));
    nthreads = nthreads <= 1024 ? nthreads : 1024;
    kernel<<<(uint32_t)dim, nthreads, 0>>>();
    cudaDeviceSynchronize();
}

int main() {
	// main iteration
	checkrange(1024);
    checkrange(2048);
    checkrange(262144);

    checkrange(524288);
    checkrange(1048576); // sqrt 1024

    checkrange(2097152);
    checkrange(16777216);
	return 0;
}


/*
 * checkrange(16777216) doesn't run the kernel function
 * A block contains at most 1024 threads. 
 */