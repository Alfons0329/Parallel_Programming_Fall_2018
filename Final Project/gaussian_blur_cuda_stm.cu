#include "bmpReader.h"
#include "bmpReader.cpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <bits/stdc++.h>
#include <sys/time.h>

using namespace std;

// CUDA Stream
#define N_STREAMS 8

// Gaussian filter
int filter_size;
unsigned int filter_scale, filter_row;
unsigned int *filter;

// Image IO
int img_col, img_row;
unsigned char *img_input, *img_output;

// CUDA error checker
void cuda_err_chk(const cudaError_t& e, const int& cudaError_cnt){
    if(e != cudaSuccess){
        fprintf(stderr, "cudaError in no. %d: %s\n", cudaError_cnt, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}


// Kernel
__global__ void cuda_gaussian_filter(unsigned char* img_input_cuda, unsigned char* img_output_cuda,int img_col, int img_row, int shift, unsigned int* filter_cuda, int filter_row, unsigned int filter_scale, int img_border_upper){
    int cuda_col = blockIdx.x * blockDim.x + threadIdx.x;
    int cuda_row = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int tmp = 0;
    int target = 0;
    int a, b;
    
    if (3 * (cuda_row * img_col + cuda_col) + shift >= img_border_upper){
        return;
    }
    
    for(int j = 0; j < filter_row; j++){
        for(int i = 0; i < filter_row; i++){
            a = cuda_col + i - (filter_row / 2);
            b = cuda_row + j - (filter_row / 2);

            target = 3 * (b * img_col + a) + shift;
            if (target >= img_border_upper || target < 0){
                continue;
            }

			tmp += filter_cuda[j * filter_row + i] * img_input_cuda[target];  
        }
    }
    tmp /= filter_scale;

    if(tmp > 255){
        tmp = 255;
    }
    
    img_output_cuda[3 * (cuda_row * img_col + cuda_col) + shift] = tmp;
}

int main(int argc, char* argv[])
{
    /*--------------- Init -------------------*/
    int thread_cnt, block_row;
    int cudaError_cnt = 0;
    string fname_input;
    string fname_output;

    if(argc < 2){
        fprintf(stderr, "%s", "Please provide filename for Gaussian Blur. usage ./gb_std.o <BMP image file> \n");
        return -1;
    }
    else if(argc == 3){
        sscanf(argv[2], "%d", &thread_cnt);
        printf("Testing with %d threads in each CUDA block\n", thread_cnt);
    }
    else{
        // Set default thread count to 1024
        thread_cnt = 1024;
    }
    block_row = (int)sqrt(thread_cnt);

    /*---------------- Image and mask IO ----*/
    FILE* mask;
    mask = fopen("mask_Gaussian.txt", "r");
    fscanf(mask, "%d", &filter_size);
    filter_row = (int)sqrt(filter_size);
    filter = new unsigned int [filter_size];

    for(int i = 0; i < filter_size; i++){
        fscanf(mask, "%u", &filter[i]);
    }

    filter_scale = 0;
    for(int i = 0; i < filter_size; i++){
        filter_scale += filter[i];	
    }
    fclose(mask);

    
    /*-------------- CUDA init ------------*/
    int num = 0;
    cudaGetDeviceCount(&num);
    cudaDeviceProp prop;
    if(num > 0){
        cudaGetDeviceProperties(&prop, 0);
        cout << "Device: " << prop.name << '\n';
    }
    else{
        fprintf(stderr, "%s", "No NVIDIA GPU detected!\n");
        return 1;
    }
    
    fname_input = argv[1];
    BmpReader* bmp_io = new BmpReader();
    img_input = bmp_io->ReadBMP(fname_input.c_str(), &img_col, &img_row);
    printf("Filter scale = %u, filter size %d x %d and image size W = %d, H = %d\n", filter_scale, filter_row, filter_row, img_col, img_row);

    int resolution = 3 * (img_col * img_row);
    img_output = (unsigned char*)malloc(resolution * sizeof(unsigned char));
    memset(img_output, 0, sizeof(img_output));
    // Apply the Gaussian filter to the image, RGB respectively
    string tmp(fname_input);

    // Init CUDA streams
    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    int offset = 0;
    int chunk_size = resolution / N_STREAMS;

    unsigned char* img_input_cuda;
    unsigned char* img_output_cuda;
    unsigned int* filter_cuda;
    cuda_err_chk(cudaMalloc((void**) &img_input_cuda, resolution * sizeof(unsigned char)), cudaError_cnt++);
    cuda_err_chk(cudaMalloc((void**) &img_output_cuda, resolution * sizeof(unsigned char)), cudaError_cnt++);
    cuda_err_chk(cudaMalloc((void**) &filter_cuda, filter_size * sizeof(unsigned int)), cudaError_cnt++);
    
    // Copy memory from host to GPU
    cuda_err_chk(cudaMemcpy(filter_cuda, filter, filter_size * sizeof(unsigned int), cudaMemcpyHostToDevice), cudaError_cnt++); // The filter matrix with constant memory
    
    // Grid and block, divide the image into 1024 per block
    /* 
    Split into 8 parts
    [
        _
        _
        _
        _
        _
        _
        _
        _
    ]   
    */
    const dim3 block_size(block_row, block_row);
    // const dim3 grid_size((img_col + block_row - 1) / block_row, (img_row + block_row - 1) / block_row / N_STREAMS); // BUG BAD VERSION, CAUSE ZEBRA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    const dim3 grid_size((img_col + block_row - 1) / block_row, (img_row / N_STREAMS + block_row - 1) / (block_row));
    
    /*-------------- CUDA run ------------*/
    // R G B channel respectively
    struct timeval start, end; 
    gettimeofday(&start, 0);
    for(int j = 0; j < N_STREAMS; j++){
        offset = chunk_size * j;
        
        cuda_err_chk(cudaMemcpyAsync(img_input_cuda + offset, img_input + offset, chunk_size * sizeof(unsigned char), cudaMemcpyHostToDevice, streams[j]), cudaError_cnt++);
        cuda_err_chk(cudaMemcpyAsync(img_output_cuda + offset, img_output + offset, chunk_size * sizeof(unsigned char), cudaMemcpyHostToDevice, streams[j]), cudaError_cnt++);

        for(int i = 0; i < 3; i++) {
            cuda_gaussian_filter<<<grid_size, block_size, 0, streams[j]>>>(img_input_cuda + offset, img_output_cuda + offset, img_col, img_row, i, filter_cuda, filter_row, filter_scale, chunk_size); 
        }
        
        cuda_err_chk(cudaMemcpyAsync(img_output + offset, img_output_cuda + offset, chunk_size * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[j]), cudaError_cnt++);
    }
    cuda_err_chk(cudaDeviceSynchronize(), cudaError_cnt++);
    gettimeofday(&end, 0);
    int sec = end.tv_sec - start.tv_sec;
    int usec = end.tv_usec - start.tv_usec;
    int t_gpu = sec * 1000 + (usec / 1000);
    printf("GPU time (ms): %d\n", t_gpu);
    
    /*-------------- Finalize ------------*/
    // Copy memory from GPU to host

    // Write output BMP file
    fname_output = fname_input.substr(0, fname_input.size() - 4)+ "_blur_cuda_stm.bmp";
    bmp_io->WriteBMP(fname_output.c_str(), img_col, img_row, img_output);
    
    // Free memory space
    free(img_input);
    free(img_output);
    free(filter);
    cuda_err_chk(cudaFree(img_input_cuda), cudaError_cnt++);
    cuda_err_chk(cudaFree(img_output_cuda), cudaError_cnt++);
    cuda_err_chk(cudaFree(filter_cuda), cudaError_cnt++);

    printf("Finished \n");
    return 0;
}