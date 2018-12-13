#include "bmpReader.h"
#include "bmpReader.cpp"
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

unsigned char *pic_in, *pic_blur, *pic_out;

// variables for cuda parallel processing
int TILE_WIDTH;

__global__ void cuda_gaussian_filter(unsigned char* input_image, unsigned char* output_image,int img_width, int img_height, int shift)
{
	// for CUDA parallelization
    int cuda_width = blockIdx.x * blockDim.x + threadIdx.x;
    int cuda_height = blockIdx.y * blockDim.y + threadIdx.y;
    if (cuda_width >= img_width || cuda_height >= img_height)
    {
        return;
    }

    int tmp = 0;
	int a, b;
	int ws = (int)sqrt((int)FILTER_SIZE);

	for (int j = 0; j  <  ws; j++)
	{
		for (int i = 0; i  <  ws; i++)
		{
			a = cuda_width + i - (ws / 2);
			b = cuda_height + j - (ws / 2);

			// detect for borders of the image
			if (a < 0 || b < 0 || a>=img_width || b>=img_height)
			{
				continue;
			} 
			tmp += filter_G[j * ws + i] * input_image[3 * (b * img_width + a) + shift];
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

    output_image[cuda_height * img_width + cuda_width] = tmp;
}
// show the progress of gaussian segment by segment
const float segment[] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };
void write_and_show(BmpReader* bmpReader, string outputblur_name, int k)
{

	// write output BMP file
    outputblur_name = "input" + to_string(k) + "_blur.bmp";
	bmpReader->WriteBMP(outputblur_name.c_str(), img_width, img_height, pic_out);

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
		pic_in = bmpReader -> ReadBMP(inputfile_name.c_str(), &img_width, &img_height);
	    printf("Filter scale = %f and image size W = %d, H = %d\n", FILTER_SCALE, img_width, img_height);

		// allocate space for output image
        int resolution = 3 * (img_width * img_height  + 1); //padding
		pic_out = (unsigned char*)malloc(resolution * sizeof(unsigned char));

		// apply the Gaussian filter to the image, RGB respectively
        string tmp(inputfile_name);
        
        //---------------------CUDA main part-------------------------//
        // allocate space
        unsigned char* cuda_input_image;
        unsigned char* cuda_output_image;

        cudaMalloc((void**) &cuda_input_image, resolution * sizeof(unsigned char));
        cudaMalloc((void**) &cuda_output_image, resolution * sizeof(unsigned char));

        // copy memory from host to GPU
        cudaMemcpy(cuda_input_image, pic_in, resolution * sizeof(unsigned char), cudaMemcpyHostToDevice);
        //cudaMemcpy(cuda_output_image, pic_out, resolution * sizeof(unsigned char), cudaMemcpyHostToDevice);

        for (int i = 0; i < 3; i++) //R G B channel respectively
        {
            cuda_gaussian_filter<<<(resolution + TILE_WIDTH) / TILE_WIDTH, TILE_WIDTH>>>(cuda_input_image, cuda_output_image, img_width, img_height, i);_
        }

        // copy memory from GPU to host
        cudaMemcpy(pic_out, cuda_output_image, resolution * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        //---------------------CUDA main part-------------------------//
        
        // write output BMP file
        outputblur_name = "input" + to_string(k) + "_blur.bmp";
		bmpReader->WriteBMP(outputblur_name.c_str(), img_width, img_height, pic_out);
        //write_and_show(bmpReader, inputfile_name, k);

		// free memory space
		free(pic_in);
		free(pic_out);
        cudaFree(cuda_input_image);
        cudaFree(cuda_output_image);
	}

	return 0;
}
