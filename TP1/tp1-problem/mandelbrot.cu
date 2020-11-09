#include "SDL.h"
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <complex>

#include <thrust/complex.h>

#define LEN 1024
#define LENSHIFT 10
#define ITERMAX 1024
#define getindex(i, j) (((i)<<LENSHIFT)+(j))
#define NCOLOR 64
#define NCOLORMASK 63

SDL_Window *screen;
SDL_Renderer *ren;
SDL_Texture *tex;
SDL_Surface *mysurf;

uint32_t iterscpu[LEN*LEN];
uint32_t colors[NCOLOR+1];
uint32_t* iters;

// Do the mandelbrot computation
uint32_t compute_iteration(double i, double j, uint32_t itermax)
{
    // use complex type from thrust to represent the complex number
    std::complex<double> z(0);
    std::complex<double> c(i, j);

    uint32_t iter = 0;
    for (iter = 0; iter < itermax; iter++) {
        // mandelbrot operation
        z = (z * z) + c;
        // if the result is not within [-2, 2], break
        if(abs(z) >= 2) {
            break;
        }
    }
    // if iter == ITERMAX-1, the pixel is within the mandelbrot suite
    return iter;
}

// Iterate through the 2D array to calculate mandelbrot for each pixel
void iterate_cpu(uint32_t *arr, double x, double y, double delta, uint32_t itermax)
{
    for (int i = 0; i < LEN; i++) {
        for (int j = 0; j < LEN; j++) {
            double ci = x + (j * delta);
            double cj = y - (i * delta);
            arr[getindex(i, j)] = compute_iteration(ci, cj, itermax);
        }
    }
    return;
}

// Do the mandelbrot computation
// Only a kernel function can be called within a kernel function
__device__ uint32_t compute_iteration_gpu(double ci, double cj, int itermax) {
    // Cannot use std so doing it manually
    // simulate imaginary number
    double zreal = 0;           // real part
    double zi = 0;              // imaginary part

    double zreal_result = 0;    // real part
    double zi_result = 0;       // imaginary part

    uint32_t iter = 0;
    for (iter = 0; iter < itermax; iter++){
        // mandelbrot operation
        // do the complex operation manually
        zi = zreal * zi;
        zi += zi;
        zi += cj;
        zreal = zreal_result - zi_result + ci;
        zreal_result = zreal * zreal;
        zi_result = zi * zi;
        // Only results bounded within [-2, 2] are taken in consideration
        if (zreal_result + zi_result > 4.0) {
            break;  
        } 
    }
    // if iter == ITERMAX-1, the pixel is within the mandelbrot suite
    return iter;
}

// std cannot be used with CUDA
// we can use insted thrust/complex.h from CUDA libraries
// #include <thrust/complex.h>
// thrust::complex<double> z(0);
// thrust::complex<double> c(ci, cj);
// for (iter = 0; iter < itermax; iter++){
//     z = (z * z) + c;
//     if (abs(z) >= 2) break;
// }

// Calculate mandelbrot using the GPU
// Each thread will calculate mandelbrot for one pixel of the window
__global__ void iterate_gpu(uint32_t* arr, double x, double y, double delta, uint32_t itermax){
    int t_id = blockDim.x * blockIdx.x + threadIdx.x;
    // Get the corresponding pixel
    int xi = t_id % LEN;
    int yj = t_id / LEN;
    double ci = x + (yj * delta);
    double cj = y - (xi * delta);
    arr[getindex(xi, yj)] = compute_iteration_gpu(ci, cj, itermax);
    return;
}

void kernel_call(uint32_t* arr, double x, double y, double delta, uint32_t itermax){
    // Using 1024 blocks and 1024 threads for each block
    iterate_gpu<<<ITERMAX, ITERMAX, 0>>>(arr, x, y, delta, itermax);
    cudaDeviceSynchronize();
    return;
}

void generate_colors(const SDL_PixelFormat* format){
    double h = 0.0;
    for(int i=0; i<NCOLOR; i++){
        int ph = h / 60;
        float f = (h/60.0 - ph);
        int v = 255;
        int p = 64;
        int q = (int)(255*(1 - f*0.75f));
        int t = (int)(255*(0.25f + f*0.75f));
        switch(ph){
            case 0:
                colors[i] = SDL_MapRGB(format, v, t, p);
                break;
            case 1:
                colors[i] = SDL_MapRGB(format, q, v, p);
                break;
            case 2:
                colors[i] = SDL_MapRGB(format, p, v, t);
                break;
            case 3:
                colors[i] = SDL_MapRGB(format, p, q, v);
                break;
            case 4:
                colors[i] = SDL_MapRGB(format, t, p, v);
                break;
            case 5:
                colors[i] = SDL_MapRGB(format, v, p, q);
                break;
            default:
                break;
        }
        h += 360.0/NCOLOR;
    }
    colors[NCOLOR] = SDL_MapRGB(format, 0, 0, 0);
    return;
}

int main(int argc, char** argv){
    SDL_Event e;
    bool usegpu = false;
    if(argc > 1){
        usegpu = (strcmp(argv[1], "gpu") == 0);
    }
    uint32_t* gpuarray;
    uint32_t* hostarray;
    
    // Initialize SDL
    if( SDL_Init(SDL_INIT_VIDEO) < 0 ) {
        fprintf(stderr, "Couldn't initialize SDL: %s\n", SDL_GetError());
        exit(1);
    }
	atexit(SDL_Quit);
    // Create window
	screen = SDL_CreateWindow("Mandelbrot", 
                        SDL_WINDOWPOS_UNDEFINED,
                        SDL_WINDOWPOS_UNDEFINED,
                        LEN, LEN, SDL_WINDOW_SHOWN);
    if ( screen == NULL ) {
        fprintf(stderr, "Couldn't set up window: %s\n", SDL_GetError());
        exit(1);
    }
    
    // Initialize CUDA
    if(usegpu){
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        cudaMalloc((void**)&gpuarray, LEN*LEN*sizeof(uint32_t));
        cudaHostAlloc((void**)&hostarray, LEN*LEN*sizeof(uint32_t), cudaHostAllocDefault);
    }
    
    // Create renderer and texture
    SDL_PixelFormat* fmt = SDL_AllocFormat(SDL_PIXELFORMAT_RGBA32);
    generate_colors(fmt);
    ren = SDL_CreateRenderer(screen, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    tex = SDL_CreateTexture(ren, fmt->format, SDL_TEXTUREACCESS_STREAMING, LEN, LEN);
    
    // Timing
    float totaltime = 0.0f;
    uint32_t frames = 0;
    
    // Window for Mandelbrot
    double targetx = -0.743643887037158704752191506114774;
    double targety = 0.131825904205311970493132056385139;
    double centerx = 0.0;
    double centery = 0.0;
    double delta = 4.0/LEN;
    const double scale = 0.94;
    uint32_t itermax = 32;
    const uint32_t iterstep = 8;
    
    while(true){
        bool flag = false;
        while(SDL_PollEvent(&e)){
            if(e.type==SDL_QUIT){
                flag = true;
            }
        }
        if(flag) break;
        clock_t t;
        float tsec;
        t = clock();
        // renderer
        if(!usegpu){
            iterate_cpu(iterscpu, centerx - delta*LEN/2, centery + delta*LEN/2, delta, itermax);
            iters = iterscpu;
        }else{
            kernel_call(gpuarray, centerx - delta*LEN/2, centery + delta*LEN/2, delta, itermax);
            cudaMemcpyAsync(hostarray, gpuarray, LEN * LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            iters = hostarray;
        }
        
        int len = LEN;
        uint32_t* surf = NULL;
        SDL_LockTexture(tex, NULL, (void**)(&surf), &len);
        for(uint32_t i=0; i<LEN*LEN; i++){
                if (iters[i] < itermax){
                    surf[i] = colors[iters[i]&NCOLORMASK];
                }else{
                    surf[i] = colors[NCOLOR];
                }
        }
        SDL_UnlockTexture(tex);
        SDL_RenderClear(ren);
        SDL_RenderCopy(ren, tex, NULL, NULL);
        SDL_RenderPresent(ren);
        centerx = targetx + (centerx - targetx)*scale;
        centery = targety + (centery - targety)*scale;
        delta *= scale;
        itermax += iterstep;
        t = clock() - t;
        tsec = ((float)t)/CLOCKS_PER_SEC;
        totaltime += tsec;
        tsec = 1.0f/60 - tsec;
        if(tsec > 0) SDL_Delay((uint32_t)(tsec*1000));
        frames++;
        if(frames>=530) break;
    }
    
    char s[100];
    sprintf(s, "Average FPS: %.1f\nFrame count: %u", frames/totaltime, frames);
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "Benchmark", s, screen);
    SDL_FreeFormat(fmt);
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(screen);
    if(usegpu){
        cudaFree(gpuarray);
        cudaFreeHost(hostarray);
    }
    exit(0);
}