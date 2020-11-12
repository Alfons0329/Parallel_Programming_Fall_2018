#include "bmpReader.h"
#include "bmpReader.cpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <pthread.h>
#include <string>

// openCV libraries for showing the images dont change
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define uint32 unsigned int

#define MYRED	2
#define MYGREEN 1
#define MYBLUE	0
int img_width, img_height;

int FILTER_SIZE;
uint32 FILTER_SCALE;
uint32 *filter_G;

unsigned char *input_image, *pic_blur, *output_image;

// variables for cuda parallel processing
#define SHM_FILTER_WITDH 101
__constant__ unsigned int const_filter_G[SHM_FILTER_WITDH * SHM_FILTER_WITDH]; // for constant memory in the GPU processing, used to cache and accelerate

__global__ void cuda_gaussian_filter(unsigned char* cuda_input_image, unsigned char* cuda_output_image,int img_width, int img_height, int shift, int ws, uint32 FILTER_SCALE, int img_border)
{
    // for CUDA parallelization
    int cuda_width = blockIdx.x * blockDim.x + threadIdx.x;
    int cuda_height = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int tmp = 0;
    int a, b;

    if (cuda_width * img_height + cuda_height >= img_border)
    {
        return;
    }

    for (int j = 0; j  <  ws; j++)
    {
        for (int i = 0; i  <  ws; i++)
        {
            a = cuda_width + i - (ws / 2);
            b = cuda_height + j - (ws / 2);

            tmp += const_filter_G[j * ws + i] * cuda_input_image[3 * (b * img_width + a) + shift]; 
        }
    }
    tmp /= FILTER_SCALE;

    if (tmp > 255)
    {
        tmp = 255;
    }
    cuda_output_image[3 * (cuda_height * img_width + cuda_width) + shift] = tmp;

}
// show the progress of gaussian segment by segment
// const float segment[] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };
void write_and_show(BmpReader* bmpReader, string outputblur_name, int k)
{
    bmpReader->WriteBMP(outputblur_name.c_str(), img_width, img_height, output_image);

    // show the output file
    Mat img = imread(outputblur_name);
    while(1)
    {
        imshow("Current progress", img);
        if (waitKey(0) % 256 == 27)
        {
            break;
        }
    }
}

int main(int argc, char* argv[])
{
    int THREAD_CNT = 1024;

    // read input filename
    string inputfile_name;
    string outputblur_name;

    if (argc < 2)
    {
        printf("Please provide filename for Gaussian Blur. usage ./gb_std.o <BMP image file>");
        return 1;
    }
    else if (argc == 3)
    {
        sscanf(argv[2], "%d", &THREAD_CNT);
        printf("Testing with %d threads in each CUDA block\n", THREAD_CNT);
    }

    // read Gaussian mask file from system
    FILE* mask;
    mask = fopen("mask_Gaussian.txt", "r");
    fscanf(mask, "%d", &FILTER_SIZE);
    filter_G = new uint32 [FILTER_SIZE];

    for (int i = 0; i < FILTER_SIZE; i++)
    {
        fscanf(mask, "%u", &filter_G[i]);
    }

    FILTER_SCALE = 0; //recalculate
    for (int i = 0; i < FILTER_SIZE; i++)
    {
        FILTER_SCALE += filter_G[i];	
    }
    fclose(mask);

    // main part of Gaussian blur
    BmpReader* bmpReader = new BmpReader();

    // platform information
    int num = 0;
    cudaGetDeviceCount(&num);

    // get gpu properties
    cudaDeviceProp prop;
    if (num > 0)
    {
        cudaGetDeviceProperties(&prop, 0);
        // get device name
        cout << "Device: " <<prop.name << endl;
        if (!prop.deviceOverlap)
        {
            printf("Device does not support overlap, so stream is invalid!");
            return -1;
        }
    }
    else
    {
        printf("No NVIDIA GPU detected! \n");
        return 2;
    }

    // read input BMP file
    inputfile_name = argv[1];
    input_image = bmpReader -> ReadBMP(inputfile_name.c_str(), &img_width, &img_height);
    printf("Filter scale = %u, filter size %d x %d and image size W = %d, H = %d\n", FILTER_SCALE, (int)sqrt(FILTER_SIZE), (int)sqrt(FILTER_SIZE), img_width, img_height);

    // allocate space for output image
    int resolution = 3 * (img_width * img_height); //padding
    output_image = (unsigned char*)malloc(resolution * sizeof(unsigned char));
    memset(output_image, 0, sizeof(output_image));
    // apply the Gaussian filter to the image, RGB respectively
    string tmp(inputfile_name);

    //---------------------CUDA main part-------------------------//
    // allocate space
    cudaError_t cuda_err, cuda_err2, cuda_err3, cuda_err4;

    //create cuda pipeline stream
    cudaStream_t stm0, stm1;
    cuda_err = cudaStreamCreate(&stm0);
    cuda_err2 = cudaStreamCreate(&stm1);

    if(cuda_err != cudaSuccess || cuda_err2 != cudaSuccess) // || cuda_err3 != cudaSuccess)
    {
        printf("Failed with error part stream %s \n", cudaGetErrorString(cuda_err));
        printf("Failed with error part0 stream err2: %s \n", cudaGetErrorString(cuda_err2));
        // printf("Failed with error part1 err3: %s \n", cudaGetErrorString(cuda_err3));
        return 3;

    }

    unsigned char* cuda_input_image_s0;
    unsigned char* cuda_output_image_s0;
    unsigned char* cuda_input_image_s1;
    unsigned char* cuda_output_image_s1;
    // uint32* cuda_filter_G;
    int half_res = resolution / 2;
    cuda_err = cudaMalloc((void**) &cuda_input_image_s0, half_res * sizeof(unsigned char));
    cuda_err2 = cudaMalloc((void**) &cuda_output_image_s0, resolution * sizeof(unsigned char));
    cuda_err3 = cudaMalloc((void**) &cuda_input_image_s1, half_res * sizeof(unsigned char));
    cuda_err4 = cudaMalloc((void**) &cuda_output_image_s1, resolution * sizeof(unsigned char));
    // cuda_err3 = cudaMalloc((void**) &cuda_filter_G, FILTER_SIZE * sizeof(uint32)); //dont forget to allocate space for it
    if(cuda_err != cudaSuccess || cuda_err2 != cudaSuccess || cuda_err3 != cudaSuccess || cuda_err4 != cudaSuccess)
    {
        printf("Failed with error part2 %s \n", cudaGetErrorString(cuda_err));
        printf("Failed with error part2 err2: %s \n", cudaGetErrorString(cuda_err2));
        printf("Failed with error part2 err3: %s \n", cudaGetErrorString(cuda_err3));
        printf("Failed with error part2 err4: %s \n", cudaGetErrorString(cuda_err4));
        return 4;

    }

    cuda_err4 = cudaMemcpyToSymbol(const_filter_G, filter_G, FILTER_SIZE * sizeof(uint32)); // the filter matrix with constant memory
    if(cuda_err4 != cudaSuccess)
    {
        printf("Constant memory allocation failed! %s \n", cudaGetErrorString(cuda_err4));
        return 5;
    }

    // grid and block, divide the image into 1024 per block
    const dim3 block_size((int)sqrt(THREAD_CNT), (int) sqrt(THREAD_CNT));
    // const dim3 grid_size((img_height + THREAD_CNT - 1) / THREAD_CNT, (img_width + THREAD_CNT - 1) / THREAD_CNT);
    const dim3 grid_size((img_width + THREAD_CNT - 1) / THREAD_CNT, (img_height + THREAD_CNT - 1) / THREAD_CNT);

    // divide the pics into two for pipeline
    for (int shift_cnt = 2; shift_cnt >= 0; shift_cnt--)
    {
        for (int i = 0; i < resolution; i += half_res)
        {
            cuda_err = cudaMemcpyAsync(cuda_input_image_s0, input_image, half_res * sizeof(unsigned char), cudaMemcpyHostToDevice);
            cuda_err2 = cudaMemcpyAsync(cuda_input_image_s1, input_image + i, half_res * sizeof(unsigned char), cudaMemcpyHostToDevice);
            if(cuda_err != cudaSuccess || cuda_err2 != cudaSuccess)// || cuda_err3 != cudaSuccess || cuda_err4 != cudaSuccess)
            {
                printf("Failed with error part3 %s \n", cudaGetErrorString(cuda_err));
                printf("Failed with error part3 err2: %s \n", cudaGetErrorString(cuda_err2));
                printf("Failed with error part3 err3: %s \n", cudaGetErrorString(cuda_err3));
                printf("Failed with error part3 err4: %s \n", cudaGetErrorString(cuda_err4));
                return 6;
            }
            cuda_gaussian_filter<<<grid_size, block_size, 0, stm0>>>(cuda_input_image_s0, cuda_output_image_s0, img_width, img_height, shift_cnt, (int)sqrt((int)FILTER_SIZE), FILTER_SCALE, half_res);
            cuda_gaussian_filter<<<grid_size, block_size, 0, stm1>>>(cuda_input_image_s1, cuda_output_image_s1, img_width, img_height, shift_cnt, (int)sqrt((int)FILTER_SIZE), FILTER_SCALE, resolution);
            
            cuda_err = cudaDeviceSynchronize();

            if(cuda_err != cudaSuccess)
            {
                printf("Failed with threadsync error: %s \n", cudaGetErrorString(cuda_err));
                return 127;
            }

            cuda_err = cudaMemcpyAsync(output_image, cuda_output_image_s0, half_res * sizeof(unsigned char), cudaMemcpyDeviceToHost);
            cuda_err2 = cudaMemcpyAsync(output_image + i, cuda_output_image_s1, half_res * sizeof(unsigned char), cudaMemcpyDeviceToHost);
            
            if(cuda_err != cudaSuccess || cuda_err2 != cudaSuccess)// || cuda_err3 != cudaSuccess || cuda_err4 != cudaSuccess)
            {
                printf("Failed with error part4 %s \n", cudaGetErrorString(cuda_err));
                printf("Failed with error part4 err2: %s \n", cudaGetErrorString(cuda_err2));
                // printf("Failed with error part3 err3: %s \n", cudaGetErrorString(cuda_err3));
                // printf("Failed with error part3 err4: %s \n", cudaGetErrorString(cuda_err4));
                return 7;
            }
        }
    }
    

    cuda_err = cudaStreamSynchronize(stm0);
    cuda_err2 = cudaStreamSynchronize(stm1);
    if(cuda_err != cudaSuccess || cuda_err2 != cudaSuccess)// || cuda_err3 != cudaSuccess || cuda_err4 != cudaSuccess)
    {
        printf("Failed with error part5 %s \n", cudaGetErrorString(cuda_err));
        printf("Failed with error part5 err2: %s \n", cudaGetErrorString(cuda_err2));
        // printf("Failed with error part3 err3: %s \n", cudaGetErrorString(cuda_err3));
        // printf("Failed with error part3 err4: %s \n", cudaGetErrorString(cuda_err4));
        return 8;
    }
    // write output BMP file
    outputblur_name = inputfile_name.substr(0, inputfile_name.size() - 4)+ "_blur_cuda_stream.bmp";
    bmpReader->WriteBMP(outputblur_name.c_str(), img_width, img_height, output_image);
    // if demo, decomment this to show
    // write_and_show(bmpReader, outputblur_name, 0);
    // free memory space
    free(input_image);
    free(output_image);
    cudaFree(cuda_input_image_s0);
    cudaFree(cuda_input_image_s0);
    cudaFree(cuda_output_image_s1);
    cudaFree(cuda_output_image_s1);
    cudaFree(const_filter_G);

    return 0;
}
