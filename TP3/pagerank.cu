#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <utility>
#include <time.h>
#include "cuStopwatch.cu"

#define COUNT (1<<23)
#define LINK_PER_PAGE 4
#define ERMIX 0.25f
#define MAXINT (4294967295.0f)
#define DAMPING 0.9f
#define EPSILON 0.00000001f
#define MAXPRCOUNT 16
#define INITPROJ 1024

/* ------------ Pagerank computation, GPU part ------------ */

__global__ void pr_init_gpu(float* pr){
    int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
    // each thread will initialize one element of the array
    if(t_idx < COUNT) {
        pr[t_idx] = 1/(float)COUNT;
    }
}

__global__ void pr_damping_gpu(float* pr){
    int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
    // each thread will initialize one element of the array
    if(t_idx < COUNT) {
        pr[t_idx] = (1-DAMPING)/(float)COUNT;
    }
}

__global__ void pr_iter_gpu(const uint2* links, const float* oldp, float* newp){
    int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(t_idx < COUNT*LINK_PER_PAGE) {
        float value = DAMPING * oldp[links[t_idx].x]/LINK_PER_PAGE;
        // using atomicAdd since multiple threads may try to access the same variable (= data-race)
        // an atomicAdd cannot be interrupted
        atomicAdd(&newp[links[t_idx].y], value);
    }
}

__global__ void pr_conv_check_gpu(const float* oldp, const float* newp, uint32_t* conv){
    int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
    *conv = 0;
    // if one pair does not converge, change *conv to 1
    if(t_idx < COUNT && fabs(oldp[t_idx] - newp[t_idx]) > EPSILON) {
        *conv = 1;
    }

    // int hasNotConverged = fabs(oldp[t_idx] - newp[t_idx]) > EPSILON ? 1 : 0;
    // if(__any_sync(0xffffffff, hasNotConverged)) {
    //     return;
    // }
    // *conv = 0;

    // int hasConverged = fabs(oldp[t_idx] - newp[t_idx]) <= EPSILON ? 1 : 0;
    // if(__all_sync(0xffffffff, hasConverged)) {
    //     *conv = 0;
    // }
}

// control GPU computation, returns computation time (in seconds, not counting memory transfer time)
float pr_compute_gpu(const uint2* links, float* pr){
    cuStopwatch timer;
    float* oldpr;
    float* newpr;
    uint32_t conv;
    uint32_t* conv_device;
    
    // Allocate arrays and convergence variable in the device memory
    cudaMalloc((void**)&oldpr, sizeof(float)*COUNT);
    cudaMalloc((void**)&newpr, sizeof(float)*COUNT);
    cudaMalloc((void**)&conv_device, sizeof(uint32_t));

    // get the number of blocks necessary to have one thread per element on the pagerank arrays
    // considering that each block will be called with exactly 1024 (1<<10) threads
    int blocks_by_count = ceil(COUNT/(1<<10));
    
    // start timer to calculate execution time
    timer.start();

    // initialize the values of the oldpr array
    pr_init_gpu<<<blocks_by_count, (1<<10)>>>(oldpr);
        
    while(true) {
        pr_damping_gpu<<<blocks_by_count, (1<<10)>>>(newpr);
        // calculate the contribution for each webpage
        // give blocks_by_count*LINK_PER_PAGE so that each thread will check one element of the links array
        pr_iter_gpu<<<blocks_by_count*LINK_PER_PAGE, (1<<10)>>>(links, oldpr, newpr);
        
        cudaMemcpyAsync(conv_device, &conv, sizeof(uint32_t), cudaMemcpyHostToDevice);
        // check if the pairs oldpr[i] and newpr[i] converge
        pr_conv_check_gpu<<<blocks_by_count, (1<<10)>>>(oldpr, newpr, conv_device);
        cudaMemcpyAsync(&conv, conv_device, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if(conv == 0) {
            break;
        }
        // put values of newpr into oldpr
        cudaMemcpy(oldpr, newpr, sizeof(float)*COUNT, cudaMemcpyDeviceToDevice);
    }

    float exec_time = timer.stop()/1000;

    // transfer the data calculated from the device memory to the host memory
    cudaMemcpy(pr, newpr, sizeof(float)*COUNT, cudaMemcpyDeviceToHost);

    // Free the device memory for all structures allocated in this scope
    cudaFree(oldpr);
    cudaFree(newpr);
    cudaFree(conv_device);

    return exec_time;
}

/* ------------ Pagerank computation, CPU part ------------ */
// fill the array with the value 1/(float)COUNT
// this is the first probability to get on one webpage or another
// since webpages are represented in a graph, each one can be the first to be visited with the same probability as the others
void pr_init_cpu(float* pr){
    for(int i = 0; i < COUNT; i++) {
        // cast to float to get a float result (the number with its integer and decimal parts)
        // or else it will be seen as an integer division (and only the integer part will be given as a result)
        pr[i] = 1/(float)COUNT; 
    }
}

// fill the array with the value (1-DAMPING)/(float)COUNT
// this will be the base value of each element of the array in each iteration
void pr_damping_cpu(float* pr){
    for(int i = 0; i < COUNT; i++) {
        // cast to float to get a float result
        pr[i] = (1-DAMPING)/(float)COUNT;
    }
}

// perform the calculation of each probability to be visited on the newp array of webpages
// using the links array to get each hyperlink source and destination
void pr_iter_cpu(const uint2* links, const float* oldp, float* newp){
    // add contributions for each link for pagerank
    // the source of the link (links[i].x) value on oldp will contribute to the new value of the destination of the link (links[i].y)
    for(int i = 0; i < COUNT*LINK_PER_PAGE; i++) {
        newp[links[i].y] += DAMPING * oldp[links[i].x]/LINK_PER_PAGE;
    }
}

// check if each pair oldp[i]/newp[i] reached the convergence constant (EPSILON)
void pr_conv_check_cpu(const float* oldp, const float* newp, uint32_t* conv){
    // iterate through the arrays to check each pair
    for(int i = 0; i < COUNT; i++) {
        if(abs(oldp[i] - newp[i]) > EPSILON) {  // at least one pair did not converge
            return;
        }
    }
    *conv = 0;  // all pairs converged
}

float pr_compute_cpu(const uint2* links, float* pr){
    float* oldpr;
    cudaMallocHost((void**)&oldpr, sizeof(float)*COUNT);

    uint32_t conv = 1;
    
    // start the clock to calculate execution time
    clock_t start_time = clock();
    
    // initialize all values of the oldpr and newpr arrays
    pr_init_cpu(oldpr);
    pr_damping_cpu(pr);

    while(true) {
        // calculate the contribution for each webpage
        pr_iter_cpu(links, oldpr, pr);
        // check if the pairs oldpr[i] and newpr[i] converge
        pr_conv_check_cpu(oldpr, pr, &conv);
        if(conv == 0) {
            break; // finish the loop if convergence is achieved for each pair
        }   
        // newpr becomes the oldpr for the new iteration of the calculation
        // values on oldpr are no longer used so they can be overwritten
        memcpy(oldpr, pr, sizeof(float)*COUNT);
        // place base value for each calculation on the newpr array
        pr_damping_cpu(pr);
    }

    // free the array allocated within this scope
    cudaFreeHost(oldpr);
    // return execution time
    return (clock() - start_time)/CLOCKS_PER_SEC;
}

/* ------------ Random graph generation ------------ */

uint32_t randstate;

uint32_t myrand(){
    randstate ^= randstate << 13;
    randstate ^= randstate >> 17;
    randstate ^= randstate << 5;
    return randstate;
}

void seed(){
    randstate = time(NULL);
    for(int i = 0; i < 16; i++) myrand();
    return;
}

void randgen(uint2* links){
    uint32_t state = time(NULL);
    uint32_t *weight = (uint32_t*)malloc(sizeof(uint32_t) * COUNT);
    memset((void*)weight, 0, sizeof(uint32_t) * COUNT);
    uint32_t totalweight = 0;
    uint32_t lcnt = 0;
    
    // Initial five
    for(int i = 0; i < INITPROJ; i++){
        weight[i] = 1;
        for(int j = 0; j < 4; j++){
            links[lcnt].x = i;
            links[lcnt].y = (uint32_t)(myrand()*(COUNT/MAXINT));
            lcnt++;
        }
    }
    totalweight = INITPROJ;
    
    // Barabasi-Albert with Erdos-Renyi mix-in
    for(uint32_t i = INITPROJ; i < COUNT; i++){
        for(int k = 0; k < LINK_PER_PAGE; k++){
             if(myrand()/MAXINT < ERMIX){
                links[lcnt].x = i;
                links[lcnt].y = (uint32_t)(myrand()*(COUNT/MAXINT));
                lcnt++;
            }else{
                uint32_t randweight = (uint32_t)(myrand()/MAXINT*totalweight);
                uint32_t idx = 0;
                while(randweight > weight[idx]){
                    randweight -= weight[idx];
                    idx++;
                }
                links[lcnt].x = i;
                links[lcnt].y = idx;
                lcnt++;
                weight[idx]++;
                totalweight++;
            }
        }
    }
    return;
}

/* ------------ Main control ------------ */

void pr_extract_max(const float* pr, float* prmax, uint32_t* prmaxidx){
    for(int i = 0; i < MAXPRCOUNT; i++) prmax[i] = -1.0f;
    for(uint32_t i = 0; i < COUNT; i++){
        if(pr[i] > prmax[MAXPRCOUNT-1]){
            int ptr = 0;
            while(pr[i] <= prmax[ptr]) ptr++;
            float oldval, newval;
            uint32_t oldidx, newidx;
            newval = pr[i];
            newidx = i;
            for(int j = ptr; j < MAXPRCOUNT; j++){
                oldval = prmax[j];
                oldidx = prmaxidx[j];
                prmax[j] = newval;
                prmaxidx[j] = newidx;
                newval = oldval;
                newidx = oldidx;
            }
        }
    }
    return;
}

int main(){
    // Generating random network
    uint2* randlinks;
    cudaHostAlloc((void**)&randlinks, sizeof(uint2)*COUNT*LINK_PER_PAGE, cudaHostAllocDefault);
    seed();
    randgen(randlinks);
    printf("Finished generating graph\n\n");
    
    // Declaration of needed variables and arrays
    float prmax[MAXPRCOUNT];
    uint32_t prmaxidx[MAXPRCOUNT];
    float elapsed;
    float *pagerank;
    float check;
    cudaHostAlloc((void**)&pagerank, sizeof(float)*COUNT, cudaHostAllocDefault);
    
    // Processing by GPU
    elapsed = pr_compute_gpu(randlinks, pagerank);
    printf("GPU version, runtime %7.4fs\n", elapsed);
    check = 0.0f;
    for(uint32_t i = 0; i <COUNT; i++) check+=pagerank[i];
    printf("Deviation: %.6f\n", check);
    pr_extract_max(pagerank, prmax, prmaxidx);
    for(int i = 0; i < MAXPRCOUNT; i++){
        printf("Rank %d, index %u, normalized pagerank %8.7f\n", i, prmaxidx[i], prmax[i] / check);
    }
    printf("\n");
    
    // Processing by CPU
    elapsed = pr_compute_cpu(randlinks, pagerank);
    printf("CPU version, runtime %7.4fs\n", elapsed);
    check = 0.0f;
    for(uint32_t i = 0; i <COUNT; i++) check+=pagerank[i];
    printf("Deviation: %.6f\n", check);
    pr_extract_max(pagerank, prmax, prmaxidx);
    for(int i = 0; i < MAXPRCOUNT; i++){
        printf("Rank %d, index %u, normalized pagerank %8.7f\n", i, prmaxidx[i], prmax[i] / check);
    }
    
    // Free memory
    cudaFreeHost(randlinks);
    cudaFreeHost(pagerank);
	return 0;
}