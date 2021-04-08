
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <algorithm>
#include <bitset>

#include <inttypes.h>

#include <cuda_runtime.h>
#include <thrust/extrema.h>

#include "cuStopwatch.cu"

#define MAXCOUNT        (1<<32)
#define BASE			2
#define DIGITS			64

#define MAXDISPLAYCOUNT 16

#define THREADS			(1<<10)

__device__
void split_kernel(uint64_t* input, uint32_t size, uint64_t* output, size_t iteration) {
	// Divide the huge array into arrays of 1024
	// Share a chunk of the array within the block
	// they are seen by all threads of one block
	__shared__ uint32_t tmp_buffer[THREADS];
	__shared__ uint32_t scan_output[THREADS];
	__shared__ uint32_t totalFalses;
	__shared__ uint32_t dest_address[THREADS];

	uint32_t t_idx = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t t_idx_block = threadIdx.x;

	// Set a 1 for all bits at 0 and a 0 for the others
	// on a tmp buffer shared by all threads of a block
	tmp_buffer[t_idx_block] = ((input[t_idx] >> iteration) & 1) == 1 ? 0 : 1;
	scan_output[t_idx_block] = 0;
	cudaDeviceSynchronize();

	// This is the scan of the temp buffer step
	// those will be the new indices for the false sort keys
	if(t_idx_block != 0) {
		scan_output[t_idx_block] = tmp_buffer[t_idx_block-1] + scan_output[t_idx_block-1];
	}
	cudaDeviceSynchronize();

	// The last element in the scan's output now contains the total number of false sort keys.
	// We write this value to a shared variable, totalFalses.
	if(t_idx_block == 0) {
		totalFalses = scan_output[THREADS-1] + tmp_buffer[THREADS-1];
	}
	cudaDeviceSynchronize();
	
	// compute the destination address for the sort keys (whose bit of interest is 1)
	dest_address[t_idx_block] = t_idx_block - scan_output[t_idx_block] + totalFalses;
	dest_address[t_idx_block] = ((input[t_idx] >> iteration) & 1) == 1 ? dest_address[t_idx_block] : scan_output[t_idx_block];
	cudaDeviceSynchronize();

	// Finally, we scatter the original sort keys to destination address d.
	// shared_output[dest_address[t_idx_block]] = input[t_idx];
	// cudaDeviceSynchronize();

	// output[t_idx_block] = shared_output[t_idx_block];
	output[dest_address[t_idx_block]+blockIdx.x] = input[t_idx];
	cudaDeviceSynchronize();
}

__global__
void radix_sort(uint64_t* arr, uint64_t* output_arr, uint32_t size) {
	for(int iteration = 0; iteration < DIGITS; iteration++) {
		split_kernel(arr, size, output_arr, iteration);
		cudaDeviceSynchronize();
		memcpy(arr, output_arr, size*sizeof(uint64_t));
	}
}

// __global__
// void computeScatter(uint64_t* input, uint64_t* output, uint32_t* dest_address) {
// 	uint32_t t_idx = threadIdx.x + blockIdx.x * blockDim.x;

// 	output[dest_address[t_idx]+blockIdx.x] = input[t_idx];
// }

// __global__
// void computeAddress(uint64_t* input, uint32_t* dest_address, uint32_t* scan_output, uint32_t totalFalses, int iteration) {
// 	uint32_t t_idx = threadIdx.x + blockIdx.x * blockDim.x;

// 	dest_address[t_idx] = t_idx - scan_output[t_idx] + totalFalses;
// 	dest_address[t_idx] = ((input[t_idx] >> iteration) & 1) == 1 ? dest_address[t_idx] : scan_output[t_idx];
// }

// __global__
// void computeHistogram(uint64_t* input, uint32_t* scan_output, uint32_t* tmp_buffer, uint32_t size, int iteration) {

// 	uint32_t t_idx = threadIdx.x + blockIdx.x * blockDim.x;

// 	tmp_buffer[t_idx] = ((input[t_idx] >> iteration) & 1) == 1 ? 0 : 1;
// 	if(t_idx == 0) {
// 		scan_output[t_idx] = 0;
// 	}
// 	__syncthreads();

// 	if(t_idx != 0) {
// 		scan_output[t_idx] = tmp_buffer[t_idx-1] + scan_output[t_idx-1];
// 	}
// }

float sort_gpu(uint64_t* arr, size_t size) {
    cuStopwatch timer;
	float elapsed_time;
	
	uint64_t *arr_dev, *output_arr_dev;
	// uint32_t *scan_output, *tmp_buffer, *dest_address;

	// uint32_t totalFalses;

	// initialize array on the device and transfer data from host to device
	cudaMalloc((void**)&arr_dev, sizeof(uint64_t)*size);
	cudaMalloc((void**)&output_arr_dev, sizeof(uint64_t)*size);

	// cudaMalloc((void**)&scan_output, sizeof(uint32_t)*size);
	// cudaMalloc((void**)&tmp_buffer, sizeof(uint32_t)*size);
	// cudaMalloc((void**)&dest_address, sizeof(uint32_t)*size);

	cudaMemcpy(arr_dev, arr, size*sizeof(uint64_t), cudaMemcpyHostToDevice);

	timer.start();

	int block_size = size/THREADS < 1 ? 1 : size/THREADS;
	radix_sort<<<block_size, THREADS>>>(arr_dev, output_arr_dev, size);
	// for(int iteration = 0; iteration < DIGITS; iteration++) {
		// computeHistogram<<<block_size, THREADS>>>(arr_dev, scan_output, tmp_buffer, size, iteration);
		// cudaDeviceSynchronize();

		// totalFalses = scan_output[size-1] + tmp_buffer[size-1];
		
		// computeAddress<<<block_size, THREADS>>>(arr_dev, dest_address, scan_output, totalFalses, iteration);
		// cudaDeviceSynchronize();

		// computeScatter<<<block_size, THREADS>>>(arr_dev, output_arr_dev, dest_address);
		// cudaDeviceSynchronize();
		// memcpy(arr_dev, output_arr_dev, size*sizeof(uint64_t));
	// }

    elapsed_time = timer.stop()/1000;

    cudaMemcpy(arr, arr_dev, size*sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaFree(arr_dev);
	cudaFree(output_arr_dev);

    return elapsed_time;
}

// Trying to develop the SCAN algorithm only on the number of digit at 1 for each element.
// static const int grid_size = 24;

// __global__ void counting_sort_gpu(uint64_t* arr, size_t size, uint64_t* output, int iteration) {
// 	uint32_t t_idx = threadIdx.x;
// 	uint32_t g_idx = threadIdx.x + blockIdx.x*THREADS;
// 	uint32_t grid_size = THREADS*blockDim.x;

// 	size_t number_of_ones = 0;

// 	for(int i = g_idx; i < size; i += grid_size) {
	// maybe use atomic add
// 		number_of_ones += ((arr[i] >> iteration) & 1);
// 	}
// 	__shared__ size_t shared_arr[THREADS];
// 	shared_arr[t_idx] = number_of_ones;
// 	__syncthreads();
// 	for(int n = THREADS/2; n > 0; n /= 2) {
// 		if(t_idx < n) {
// 			shared_arr[t_idx] += shared_arr[t_idx+n];
// 		}
// 		__syncthreads();
// 	}
// 	if(t_idx == 0) {
// 		output[blockIdx.x] = shared_arr[0];
// 	}
// }

// Radix sort CPU
void counting_sort_cpu(uint64_t* arr, size_t size, int iteration, size_t* buckets, uint64_t* output_arr) {
	for(int i = 0; i < size; i++) {
		buckets[(arr[i] >> iteration) & 1] += 1;
	}

	buckets[1] += buckets[0];

	for(int i = size-1; i >= 0; i--) {
		int index = (arr[i] >> iteration) & 1;
		buckets[index] -= 1;
		output_arr[buckets[index]] = arr[i];
	}

	std::memcpy(arr, output_arr, size*sizeof(uint64_t));
}

float rsort_cpu(uint64_t* arr, size_t size) {
	time_t		start_time, end_time;
	
	size_t* 	buckets;
	uint64_t* 	output_arr;

	output_arr 	= (uint64_t*)malloc(sizeof(uint64_t)*size);
	buckets		= (size_t*)malloc(sizeof(size_t)*BASE);

	start_time 	= clock();

	for(int iteration = 0; iteration < DIGITS; iteration++) {
		buckets[0] = 0;
		buckets[1] = 0;
		counting_sort_cpu(arr, size, iteration, buckets, output_arr);
	}

	end_time = (clock() - start_time) / CLOCKS_PER_SEC;
	
	free(output_arr);
	free(buckets);
	return end_time;
}

// Sort CPU function from STD library
float sort_cpu(uint64_t* arr, size_t size) {
    time_t start_time = clock();
    std::sort(arr, arr+size);
    return (clock() - start_time) / CLOCKS_PER_SEC;
}

void randgen(uint64_t* arr, size_t count){
    uint64_t state = time(NULL);
    state ^= state << 12;
    state += state >> 7;
    state ^= state << 23;
    state += state >> 6;
    state ^= state << 45;
    state -= state >> 4;
    state++;
    for(uint64_t i = 0; i < count; i++){
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        arr[i] = state;
    }
    return;
}

void usage() {
	std::cout << "usage: sort-gpu.exe size";
	exit(0);
}

int main(int argc, char** argv){

    size_t 		size;
    uint64_t 	*cpu_array, *r_cpu_array, *gpu_array;
	float 		elapsed;

	if(argc < 2) {
		usage();
	}
	
	std::string arg = argv[1];
	try {
		size = 1<<std::stoi(arg, nullptr);
	} catch (std::invalid_argument const &e) {
		std::cerr << "Invalid number: " << arg << std::endl;
		usage();
	} catch (std::out_of_range const &e) {
		std::cerr << "Number out of range: " << arg << std::endl;
		usage();
	}

	std::cout << "Randomizing array of size " << size << std::endl;

    /*---------------------- Initialize arrays ----------------------*/
    cudaMallocHost((void**)&cpu_array, sizeof(uint64_t)*size);
	cudaMallocHost((void**)&r_cpu_array, sizeof(uint64_t)*size);
	cudaMallocHost((void**)&gpu_array, sizeof(uint64_t)*size);

	randgen(cpu_array, size);

	memcpy(r_cpu_array, cpu_array, size*sizeof(uint64_t));
    memcpy(gpu_array, cpu_array, size*sizeof(uint64_t));

    // Display unsorted array
    std::cout << "Unsorted array:\n";
    std::cout << "[\n";
    for(int i = 0; i < MAXDISPLAYCOUNT; i++) {
        std::cout << "\t" << cpu_array[i] << ",\n";
    }
	std::cout << "...]\n";
    
	/*------------------ Apply Sorting algorithms ------------------*/
	// Processing by GPU using RADIX SORT
	elapsed = sort_gpu(gpu_array, size);
	std::cout << "\n[GPU version] Using radix sort algorithm, runtime " << elapsed << "s\n";
	std::cout << "[\n";
	for(int i = 0; i < MAXDISPLAYCOUNT; i++) {
		std::cout << "\t" << gpu_array[i] << ",\n";
	}
	std::cout << "..." /*] => max: " << gpu_array[size-1] */<< std::endl;
    
	// Processing by CPU using built-in SORT function
	elapsed = sort_cpu(cpu_array, size);
    std::cout << "\n[CPU version] Using quick sort algorithm, runtime " << elapsed << "s\n";
    std::cout << "[\n";
    for(int i = 0; i < MAXDISPLAYCOUNT; i++) {
		std::cout << "\t" << cpu_array[i] << ",\n";
    }
	std::cout << "...] "/*=> max: " << cpu_array[size-1]*/ << std::endl;
	
	// Processing by CPU using RADIX SORT
	elapsed = rsort_cpu(r_cpu_array, size);
	std::cout << "\n[CPU version] Using radix sort algorithm, runtime " << elapsed << "s\n";
    std::cout << "[\n";
    for(int i = 0; i < MAXDISPLAYCOUNT; i++) {
		std::cout << "\t" << r_cpu_array[i] << ",\n";
    }
	std::cout << "...] "/*=> max: " << r_cpu_array[size-1]*/ << std::endl;

	cudaFree(cpu_array);
	cudaFree(r_cpu_array);
    cudaFree(gpu_array);

    exit(0);
}