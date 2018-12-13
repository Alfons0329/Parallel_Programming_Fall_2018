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

#define MYRED	2
#define MYGREEN 1
#define MYBLUE	0
#define RATE 1000000
int img_width, img_height;

int FILTER_SIZE;
float FILTER_SCALE;
float *filter_G;

unsigned char *input_image, *pic_blur, *output_image;

// variables for cuda parallel processing
int TILE_WIDTH;

__global__ void cuda_gaussian_filter(unsigned char* input_image, unsigned char* output_image,int img_width, int img_height, int shift, float* filter_G, int ws, float FILTER_SCALE)
{
    // for CUDA parallelization
    int cuda_width = blockIdx.x * blockDim.x + threadIdx.x;
    int cuda_height = blockIdx.y * blockDim.y + threadIdx.y;
    if (cuda_width >= img_width || cuda_height >= img_height)
    {
        return;
    }
    // printf("cuda_width = %d, cuda_height = %d ", cuda_width, cuda_height);
    int tmp = 0;
    int a, b;
    for (int j = 0; j < ws; j++)
    {
        for (int i = 0; i < ws; i++)
        {
            a = cuda_width + i - (ws / 2);
            b = cuda_height + j - (ws / 2);
            // detect for borders of the image
            if (a < 0 || b < 0 || a >= img_width || b >= img_height)
            {
                continue;
            }
            //printf("Location = %d \n", 3 * (b * img_width + a) + shift);
            printf(" j * ws + i = %d j = %d i = %d\n", j * ws + i, j, i);
            printf("process w = %d h = %d  ws = %d\n", a, b, ws);
            tmp += filter_G[j * ws + i] * input_image[3 * (b * img_width + a) + shift];
            // printf(" , Value = %d ", input_image[3 * (b * img_width + a) + shift]);
            // printf("\n");
            // printf(" i = %d j = %d \n", i, j);
        }
    }

    tmp /= FILTER_SCALE;

    if (tmp < 0)
    {
        tmp = 0;
    } 
    if (tmp > 255)
    {
        tmp = 255;
    }

    output_image[3 * (cuda_height * img_width + cuda_width) + shift] = tmp;
}
// show the progress of gaussian segment by segment
const float segment[] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };
void write_and_show(BmpReader* bmpReader, string outputblur_name, int k)
{

    // write output BMP file
    outputblur_name = "input" + to_string(k) + "_blur.bmp";
    bmpReader->WriteBMP(outputblur_name.c_str(), img_width, img_height, output_image);

    // show the output file
    Mat img = imread(outputblur_name);
    imshow("Current progress", img);
    waitKey(20);
}

int main(int argc, char* argv[])
{
    TILE_WIDTH = 1024;

    // read input filename
    string inputfile_name;
    string outputblur_name;

    if (argc < 2)
    {
        printf("Please provide filename for Gaussian Blur. usage ./gb_std.o <BMP image file>");
        return 1;
    }

    // read Gaussian mask file from system
    FILE* mask;
    mask = fopen("mask_Gaussian.txt", "r");
    fscanf(mask, "%d", &FILTER_SIZE);
    filter_G = new float[FILTER_SIZE];

    for (int i = 0; i < FILTER_SIZE; i++)
    {
        fscanf(mask, "%f", &filter_G[i]);
    }

    FILTER_SCALE = 0.0f; //recalculate
    for (int i = 0; i < FILTER_SIZE; i++)
    {
        filter_G[i] *= RATE;
        FILTER_SCALE += filter_G[i];	
    }
    fclose(mask);

    // main part of Gaussian blur
    BmpReader* bmpReader = new BmpReader();

    for (int k = 1; k < argc; k++)
    {

        // read input BMP file
        inputfile_name = argv[k];
        input_image = bmpReader -> ReadBMP(inputfile_name.c_str(), &img_width, &img_height);
        printf("Filter scale = %f and image size W = %d, H = %d\n", FILTER_SCALE, img_width, img_height);

        // allocate space for output image
        int resolution = 3 * (img_width * img_height); //padding
        output_image = (unsigned char*)malloc(resolution * sizeof(unsigned char));
        memset(output_image, 0, sizeof(output_image));
        // apply the Gaussian filter to the image, RGB respectively
        string tmp(inputfile_name);

        //---------------------CUDA main part-------------------------//
        // allocate space
        cudaError_t cuda_err;

        unsigned char* cuda_input_image;
        unsigned char* cuda_output_image;

        cuda_err = cudaMalloc((void**) &cuda_input_image, resolution * sizeof(unsigned char));
        cuda_err = cudaMalloc((void**) &cuda_output_image, resolution * sizeof(unsigned char));
        if(cuda_err != cudaSuccess)
        {
            printf("Failed with error %s \n", cudaGetErrorString(cuda_err));
        }

        // copy memory from host to GPU
        cudaMemcpy(cuda_input_image, input_image, resolution * sizeof(unsigned char), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_output_image, output_image, resolution * sizeof(unsigned char), cudaMemcpyHostToDevice);
        if(cuda_err != cudaSuccess)
        {
            printf("Failed with error %s \n", cudaGetErrorString(cuda_err));
        }


        for (int i = 2; i >= 0; i++) //R G B channel respectively
        {
            cuda_gaussian_filter<<<(resolution + TILE_WIDTH) / TILE_WIDTH, TILE_WIDTH>>>(cuda_input_image, cuda_output_image, img_width, img_height, i, filter_G, (int)sqrt((int)FILTER_SIZE), FILTER_SCALE);
            cudaError_t cuda_err = cudaDeviceSynchronize();

            if(cuda_err != cudaSuccess)
            {
                printf("Failed with error %s \n", cudaGetErrorString(cuda_err));
                return -1;
            }
        }

        // copy memory from GPU to host
        cudaMemcpy(output_image, cuda_output_image, resolution * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        //---------------------CUDA main part-------------------------//

        // write output BMP file
        outputblur_name = inputfile_name.substr(0, inputfile_name.size() - 4)+ "_blur.bmp";
        bmpReader->WriteBMP(outputblur_name.c_str(), img_width, img_height, output_image);
        /*Mat img = imread(outputblur_name);
          while(1)
          {
          imshow("Current progress", img);
          if(waitKey(10) == 27)
          {
          break;
          }
          }*/

        // free memory space
        free(input_image);
        free(output_image);
        cudaFree(cuda_input_image);
        cudaFree(cuda_output_image);
    }

    return 0;
}
