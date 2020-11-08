#include "SDL_image.h"
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuStopwatch.cu"

enum conv_t{
    CONV_IDENTITY,
    CONV_EDGE,
    CONV_SHARP,
    CONV_GAUSS
};

SDL_Window *screen;
SDL_Window *screen_res;
SDL_Renderer *ren;
SDL_Renderer *ren_res;
SDL_Texture *tex;
SDL_Texture *tex_res;
SDL_Surface *surf;

int32_t width, height;
float filter[9];

float __constant__ filter_device[9];

__global__ void conv_global(const unsigned char* src, unsigned char* dest, int32_t w, int32_t h){
    // todo: write a kernel to apply the given filter on the given image stored in the global memory
}

float conv_global_gpu(unsigned char* pixels, int32_t w, int32_t h){
    // todo: write the code that manages memory (global memory) and invokes the kernel conv_global, it should return the running time
}

__global__ void conv_texture(cudaTextureObject_t src, unsigned char* dest, int32_t w, int32_t h){
    // todo: write a kernel to apply the given filter on the given image stored as a texture
}

float conv_texture_gpu(unsigned char* pixels, int32_t w, int32_t h){
    // todo: write the code that manages memory (texture memory) and invokes the kernel conv_global, it should return the running time
}

int main(int argc, char** argv){
    SDL_Event event;
    bool withtex = false;
    
    // Initialize SDL
    if( SDL_Init(SDL_INIT_VIDEO) < 0 ) {
        fprintf(stderr, "Couldn't initialize SDL: %s\n", SDL_GetError());
        exit(1);
    }
	atexit(SDL_Quit);
    
    if(argc == 1){
        exit(1);
    }
    
    // Read image and option
    IMG_Init(IMG_INIT_PNG);
    surf = IMG_Load(argv[1]);
    if(surf == NULL){
        fprintf(stderr, "Error loading image.\n");
        exit(1);
    }
    width = surf->w;
    height = surf->h;
    SDL_SetSurfaceRLE(surf, 1);
    
    // Initialize involution kernel
    conv_t conv_type;
    if(argc >= 3){
        if (strcmp(argv[2], "identity") == 0) conv_type = CONV_IDENTITY;
        else if (strcmp(argv[2], "edge") == 0) conv_type= CONV_EDGE;
        else if (strcmp(argv[2], "sharp") == 0) conv_type= CONV_SHARP;
        else if (strcmp(argv[2], "gauss") == 0) conv_type = CONV_GAUSS;
        else conv_type = CONV_IDENTITY;
    }
    switch(conv_type){
        case CONV_EDGE:
            filter[0] = -1; filter[1] = -1; filter[2] = -1; 
            filter[3] = -1; filter[4] = 8; filter[5] = -1; 
            filter[6] = -1; filter[7] = -1; filter[8] = -1; 
            break;
        case CONV_SHARP:
            filter[0] = 0; filter[1] = -1; filter[2] = 0; 
            filter[3] = -1; filter[4] = 5; filter[5] = -1; 
            filter[6] = 0; filter[7] = -1; filter[8] = 0; 
            break;
        case CONV_GAUSS:
            filter[0] = 1.0f/16; filter[1] = 1.0f/8; filter[2] = 1.0f/16; 
            filter[3] = 1.0f/8; filter[4] = 1.0f/4; filter[5] = 1.0f/8; 
            filter[6] = 1.0f/16; filter[7] = 1.0f/8; filter[8] = 1.0f/8; 
            break;
        default:
            filter[0] = 0; filter[1] = 0; filter[2] = 0; 
            filter[3] = 0; filter[4] = 1; filter[5] = 0; 
            filter[6] = 0; filter[7] = 0; filter[8] = 0; 
            break;
    }
    cudaMemcpyToSymbolAsync(filter_device, filter, sizeof(float)*9, 0, cudaMemcpyHostToDevice);
    
    if(argc >= 4){
        if(strcmp(argv[3], "texture") == 0) withtex = true;
    }
    
    // Create window
	screen = SDL_CreateWindow("Original", 
                        100,
                        100,
                        width, height, SDL_WINDOW_SHOWN);
    if ( screen == NULL ) {
        fprintf(stderr, "Couldn't set up window: %s\n", SDL_GetError());
        exit(1);
    }
    screen_res = SDL_CreateWindow("Filtered", 
                        300,
                        300,
                        width, height, SDL_WINDOW_SHOWN);
    if ( screen_res == NULL ) {
        fprintf(stderr, "Couldn't set up window: %s\n", SDL_GetError());
        exit(1);
    }
    
    // Initialize CUDA
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    
    // Create renderer and texture
    ren = SDL_CreateRenderer(screen, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    tex = SDL_CreateTextureFromSurface(ren, surf);
    
    // Show image
    SDL_RenderCopy(ren, tex, NULL, NULL);
    SDL_RenderPresent(ren);
    
    // Compute
    SDL_LockSurface(surf);
    float elapsed;
    if(withtex){
        elapsed = conv_texture_gpu((unsigned char*)surf->pixels, width, height);
    }else{
        elapsed = conv_global_gpu((unsigned char*)surf->pixels, width, height);
    }
    SDL_UnlockSurface(surf);
    
    // Show computed image
    ren_res = SDL_CreateRenderer(screen_res, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    tex_res = SDL_CreateTextureFromSurface(ren_res, surf);
    SDL_RenderCopy(ren_res, tex_res, NULL, NULL);
    SDL_RenderPresent(ren_res);
    SDL_FreeSurface(surf);
    
    while (1) {
        SDL_WaitEvent(&event);
        if ((event.type == SDL_QUIT) || ((event.type == SDL_WINDOWEVENT) && (event.window.event == SDL_WINDOWEVENT_CLOSE))) break;
    }
    
    char s[100];
    sprintf(s, "Kernel execution time: %.4fms", elapsed);
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "Timing", s, screen);
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(screen);
    SDL_DestroyWindow(screen_res);
    exit(0);
}