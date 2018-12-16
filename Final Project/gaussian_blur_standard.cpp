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
#define RATE 10000
int img_width, img_height;

int FILTER_SIZE;
unsigned long long  FILTER_SCALE;
unsigned long long  *filter_G;

unsigned char *pic_in, *pic_blur, *pic_out;

unsigned char gaussian_filter(int w, int h,int shift, int img_border)
{
	int a, b;
	int ws = (int)sqrt((int)FILTER_SIZE);
    int half = ws >> 1;
    int target = 0;

	/*for (int j = 0; j  <  ws; j++)
	{
		for (int i = 0; i  <  ws; i++)
		{
			a = w + i - (ws / 2);
			b = h + j - (ws / 2);

			// detect for borders of the image
			if (a < 0 || b < 0 || a>=img_width || b>=img_height)
			{
				continue;
			} 
			tmp += filter_G[j * ws + i] * pic_in[3 * (b * img_width + a) + shift];
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
	} */

	/*if (3 * (h * img_width + w) + shift >= img_border)
    {
        return 0;
    }*/

	unsigned long long tmp = 0;
    for (int i = -half; i <= half; i++)
    {
        for (int j = -half; j <= half; j++)
        {
            target = 3 * ((h + i) * img_width + (w + j)) + shift;
            if (target >= img_border || target < 0)
            {
                continue;
            }
            tmp += filter_G[i * ws + j] * pic_in[target];
        }
    }
	printf("tmp before %lld", tmp);
    tmp /= (unsigned long long int)FILTER_SCALE;
	printf("tmp after %lld \n", tmp);
    if (tmp < 0)
    {
        tmp = 0;
    } 
    if (tmp > 255)
    {
        tmp = 255;
    }
	// printf("Input %d Output %d img_birder %d filter_g %f\n", pic_in[3 * (h * img_width + w) + shift], tmp, img_border, filter_G[w % 35]);
	return (unsigned char)tmp;
}
// show the progress of gaussian segment by segment
const float segment[] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };
void write_and_show(BmpReader* bmpReader, string outputblur_name, int k)
{
	bmpReader->WriteBMP(outputblur_name.c_str(), img_width, img_height, pic_out);

    // show the output file
    Mat img = imread(outputblur_name);
    imshow("Current progress", img);
    waitKey(20);
}

int main(int argc, char* argv[])
{
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
	filter_G = new unsigned long long [FILTER_SIZE];

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

		int resolution = 3 * (img_width * img_height);
		// allocate space for output image
		pic_out = (unsigned char*)malloc(3 * img_width * img_height * sizeof(unsigned char));

		//apply the Gaussian filter to the image, RGB respectively
        string tmp(inputfile_name);
        int segment_cnt = 1;
        outputblur_name = inputfile_name.substr(0, inputfile_name.size() - 4)+ "_blur.bmp";

		for (int j = 0; j < img_height; j++) 
		{
			for (int i = 0; i < img_width; i++)
			{
				pic_out[3 * (j * img_width + i) + MYRED] = gaussian_filter(i, j, MYRED, resolution);
				pic_out[3 * (j * img_width + i) + MYGREEN] = gaussian_filter(i, j, MYGREEN, resolution);
				pic_out[3 * (j * img_width + i) + MYBLUE] = gaussian_filter(i, j, MYBLUE, resolution);
			}
            
            // show the progress of image every 10% of work progress
            if (j >= segment[segment_cnt - 1] * img_height && j <= segment[segment_cnt] * img_height)
            {
                printf("Show segment %d with j is now %d \n", segment_cnt, j);
                write_and_show(bmpReader, outputblur_name, k);
                segment_cnt = (segment_cnt >= 10) ? segment_cnt : segment_cnt + 1;   
            }
        }
		// write output BMP file
        bmpReader->WriteBMP(outputblur_name.c_str(), img_width, img_height, pic_out);
        write_and_show(bmpReader, outputblur_name, k);
		// free memory space
		free(pic_in);
		free(pic_out);
	}

	return 0;
}
