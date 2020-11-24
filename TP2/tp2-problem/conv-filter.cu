#include "SDL_image.h"
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <assert.h>
#include "cuStopwatch.cu"

enum conv_t{
    CONV_IDENTITY,
    CONV_EDGE,
    CONV_SHARP,
    CONV_GAUSS
};

#define FILTER_SIZE 9

SDL_Window *screen;
SDL_Window *screen_res;
SDL_Renderer *ren;
SDL_Renderer *ren_res;
SDL_Texture *tex;
SDL_Texture *tex_res;
SDL_Surface *surf;

int32_t width, height;
float filter[FILTER_SIZE];

float __constant__ filter_device[FILTER_SIZE];

// gives the char representing the color from the pixel on row-y and column-x
__device__ int compute_filter(const unsigned char *src, int t_idx, int t_idy, int width, int height, int color)
{
    // We multiply by 3 to take in consideration the three colors represented 
    int tid_index = t_idx * 3;

    // get the 2D coordinates
    int row    = t_idx/width;   // == t_idy%height
    int column = t_idy/height;  // == t_idx%width

    int result = 0;
    
    for (int i = -1; i <= 1; i++) {
        /*  Check the rows
            > [ , , ]
            > [ ,X, ]
            > [ , , ]
        */
        int check_row = row + i;
        if(check_row < 0 || check_row == height) { // if out of bound (up or down), don't move the index
            check_row = 0;
        } else {
            check_row = i * width * 3;
        }
        for (int j = -1; j <= 1; j++) {
            /*  Check the columns
                [ , , ]
                [ ,X, ]
                [ , , ]
                 ^ ^ ^
            */
            int check_column = column + j;
            if(check_column < 0 || check_column == width) { // if out of bound (left or right), don't move the index
                check_column = 0;
            } else {
                check_column = j * 3;
            }
            // on filter_device index: add 1 to i and j because we began with -1
            // on src index: search for the adjacent elements (seen on a 2D array) on the 1D array
            result += filter_device[(i+1)*3+(j+1)] * src[tid_index+check_row+check_row+color];
        }
    }

    // colors are represented on chars (1 byte) from 0 to 255, so clip it if necessary
    if(result < 0) {
        return 0;
    } else if(result > 255){
        return 255;
    } else {
        return result;
    }
}

// 2D array to 1D array : index = idx_x * width + idx_y
__global__ void conv_global(const unsigned char* src, unsigned char* dest, int32_t w, int32_t h){
    // write a kernel to apply the given filter on the given image stored in the global memory
    int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int t_idy = blockDim.y * blockIdx.y + threadIdx.y;

    // check if the thread is within the array
    if(t_idx < w*h*3) {
        // for each color (r=0, g=1, b=2)
        for(int color = 0; color < 3; color++) {
            //int idx = get_idx(t_idx, t_idy, w, h, i);
            dest[t_idx*3 + color] = compute_filter(src, t_idx, t_idy, w, h, color);
        }
    }
}

// cudaAlloc/cudaMemoryCopy from the host and back to the host
float conv_global_gpu(unsigned char* pixels, int32_t w, int32_t h){
    // write the code that manages memory (global memory) and invokes the kernel conv_global, it should return the running time
    
    // total size of the 1D array that will represent each color layer of each pixel
    int size = w*h*3;
    unsigned char *input_picture, *output_picture, *host_picture;
    cuStopwatch timer;

    // start the cuWatch before allocating
    // malloc operations take time and need to be taken in consideration when counting the time elapsed
    timer.start();

    // Allocate host memory that is accessible to the device
    // This will allow us to transfer the picture information from the GPU to the host
    cudaMallocHost((void **)&host_picture, size * sizeof(unsigned char));
    // Allocate linear memory of the device for the internal structures (the 1D arrays representing the picture)
    cudaMalloc((void **)&input_picture, size * sizeof(unsigned char));
    cudaMalloc((void **)&output_picture, size * sizeof(unsigned char));

    // copy the content of the BMP picture to the host_picture array on the host
    memcpy(host_picture, pixels, size * sizeof(unsigned char));
    // copy the content of the host_picture array on the allocated array
    // to transfer the picture from the host memory to the device memory
    cudaMemcpy(input_picture, host_picture, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // This allows us to get as much total threads as pixels of the picture
    int block_number = ceil((h * w) / 1024);
    // Apply the filter
    conv_global<<<block_number, 1024>>>(input_picture, output_picture, w, h);

    // Copy the output pixels (after applying the filter) to the host_picture array
    // to transfer the new picture from the device to the host memory
    cudaMemcpy(host_picture, output_picture, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    memcpy(pixels, host_picture, size);

    // Free the allocated memory for the arrays
    cudaFreeHost(host_picture);
    cudaFree(input_picture);
    cudaFree(output_picture);
    
    // Return the total execution time
	return timer.stop();
}

__global__ void conv_texture(cudaTextureObject_t src, unsigned char* dest, int32_t w, int32_t h){
    // todo: write a kernel to apply the given filter on the given image stored as a texture
}

float conv_texture_gpu(unsigned char* pixels, int32_t w, int32_t h){
    // todo: write the code that manages memory (texture memory) and invokes the kernel conv_global, it should return the running time
    
    return 0.0f;
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
        printf("usage: conv-filter.exe [picture.png]");
        exit(1);
    }
    
    // Read image and option
    IMG_Init(IMG_INIT_PNG);
    surf = IMG_Load(argv[1]);
    // surf = IMG_Load("cat.bmp");
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