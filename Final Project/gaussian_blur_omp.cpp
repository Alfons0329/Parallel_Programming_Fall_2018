#include "bmpReader.h"
#include "bmpReader.cpp"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string>
#include <omp.h>

// openCV libraries for showing the images dont change
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define ull unsigned long long int

// variables for Gaussian filter
int FILTER_SIZE;
ull FILTER_SCALE;
ull *filter_G;

// variables for image processing
#define MYRED	2
#define MYGREEN 1
#define MYBLUE	0
int img_width, img_height;
unsigned char *pic_in, *pic_in_padded, *pic_out;

unsigned char gaussian_filter(int w, int h,int shift, int img_border)
{
	int ws = (int)sqrt((int)FILTER_SIZE);
	int target = 0;
	ull tmp = 0;
	int a, b;

	for (int j = 0; j  <  ws; j++)
	{
		for (int i = 0; i  <  ws; i++)
		{
			a = w + i/* - (ws / 2)*/;
			b = h + j/* - (ws / 2)*/;

			// detect for borders of the image
			/*if (a < 0 || b < 0 || a >= img_width || b >= img_height)
			{
				continue;
			} */
			tmp += filter_G[j * ws + i] * pic_in_padded[3 * (b * (img_width+ws) + a) + shift];
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
	return (unsigned char)tmp;
}

unsigned char *padding_pic_for_clamp(unsigned char *pic_in, int img_width, int img_height, int FILTER_SIZE) {
	int ws = (int)sqrt((int)FILTER_SIZE);
	int half = (ws-1)/2;
	ws = half*2+1;
	unsigned char *pic_out = (unsigned char*) malloc(sizeof(unsigned char[3]) * (img_width + ws) * (img_height + ws));
	for (int i = 0; i < ws+img_height; i++) {
		for (int j = 0; j < ws+img_width; j++) {
			// clamp to edge
			int y = i - half;
			int x = j - half;
			if (y >= img_height) y = img_height - 1;
			if (y < 0) y = 0;
			if (x >= img_width) x = img_width - 1;
			if (x < 0) x = 0;
			int idx = (i * (ws+img_width) + j) * 3;
			int orig_idx = (y * img_width + x) * 3;
			pic_out[idx+0] = pic_in[orig_idx+0];
			pic_out[idx+1] = pic_in[orig_idx+1];
			pic_out[idx+2] = pic_in[orig_idx+2];
		}
	}
	return pic_out;
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
		printf("Please provide filename for Gaussian Blur. usage ./gb_omp.o <BMP image file>\n");
		return 1;
	}
	
	// read Gaussian mask file from system
	FILE* mask;
	mask = fopen("mask_Gaussian.txt", "r");
	fscanf(mask, "%d", &FILTER_SIZE);
	filter_G = new unsigned long long [FILTER_SIZE];

	for (int i = 0; i < FILTER_SIZE; i++)
	{
		fscanf(mask, "%llu", &filter_G[i]);
	}

	FILTER_SCALE = 0; //recalculate
	for (int i = 0; i < FILTER_SIZE; i++)
	{
		FILTER_SCALE += filter_G[i];
		//printf("filter_G %d ", filter_G[i]);
	}
	fclose(mask);

	// main part of Gaussian blur
	BmpReader* bmpReader = new BmpReader();
	for (int k = 1; k < argc; k++)
	{
		// read input BMP file
		inputfile_name = argv[k];
		pic_in = bmpReader -> ReadBMP(inputfile_name.c_str(), &img_width, &img_height);
		printf("Filter scale = %llu and image size W = %d, H = %d\n", FILTER_SCALE, img_width, img_height);
		pic_in_padded = padding_pic_for_clamp(pic_in, img_width, img_height, FILTER_SIZE);

		int resolution = 3 * (img_width * img_height);
		// allocate space for output image
		pic_out = (unsigned char*)malloc(3 * img_width * img_height * sizeof(unsigned char));

		//apply the Gaussian filter to the image, RGB respectively
		string tmp(inputfile_name);
		int segment_cnt = 1;
		outputblur_name = inputfile_name.substr(0, inputfile_name.size() - 4)+ "_blur_omp.bmp";

		for (segment_cnt = 1; segment_cnt <= 10; segment_cnt++)
		{
			int seg_start = segment_cnt == 1 ? 0 : img_height * segment[segment_cnt-2];
			int seg_end = img_height * segment[segment_cnt-1];
			// magic parallel flag
			#pragma omp parallel for
			for (int j = seg_start; j < seg_end; j++)
			{
				for (int i = 0; i < img_width; i++)
				{
					pic_out[3 * (j * img_width + i) + MYRED] = gaussian_filter(i, j, MYRED, resolution);
					pic_out[3 * (j * img_width + i) + MYGREEN] = gaussian_filter(i, j, MYGREEN, resolution);
					pic_out[3 * (j * img_width + i) + MYBLUE] = gaussian_filter(i, j, MYBLUE, resolution);
				}
			}
			// show the progress of image every 10% of work progress
			printf("Show segment %d with j is now %d \n", segment_cnt, seg_end-1);
			write_and_show(bmpReader, outputblur_name, k);
		}
		// write output BMP file
		bmpReader->WriteBMP(outputblur_name.c_str(), img_width, img_height, pic_out);
		write_and_show(bmpReader, outputblur_name, k);
		// free memory space
		free(pic_in);
		free(pic_in_padded);
		free(pic_out);
	}

	return 0;
}
