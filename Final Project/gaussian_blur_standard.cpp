// Student ID:
// Name      :
// Date      : 2017.11.03

#include "bmpReader.h"
#include "bmpReader.cpp"
#include  <stdio.h>
#include  <iostream>
#include  <math.h>
#include  <pthread.h>
#include  <semaphore.h>
using namespace std;

#define MYRED	2
#define MYGREEN 1
#define MYBLUE	0

int img_width, img_height;

int FILTER_SIZE;
float FILTER_SCALE;
float *filter_G;

const char *inputfile_name[5] = 
{
	"input1.bmp",
	"input2.bmp",
	"input3.bmp",
	"input4.bmp",
	"input5.bmp"
};
const char *outputBlur_name[5] = 
{
	"Blur1.bmp",
	"Blur2.bmp",
	"Blur3.bmp",
	"Blur4.bmp",
	"Blur5.bmp"
};
/*
const char *outputSobel_name[5] = {
	"Sobel1.bmp",
	"Sobel2.bmp",
	"Sobel3.bmp",
	"Sobel4.bmp",
	"Sobel5.bmp"
};*/

unsigned char *pic_in, *pic_grey, *pic_blur, *pic_out;

unsigned char gaussian_filter(int w, int h,int shift)
{
	int tmp = 0;
	int a, b;
	int ws = (int)sqrt((int)FILTER_SIZE);
	// process R, G and B respectively, shift 0 is R, 1 is G and 2 is B respectively

	for (int j = 0; j  <  ws; j++)
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
	} 

	return (unsigned char)tmp;
}

int main()
{
	// read mask file
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
		filter_G[i] *= 1000000;
		FILTER_SCALE += filter_G[i];	
	}
	printf("filter scale [%f] \n", FILTER_SCALE);
	fclose(mask);


	BmpReader* bmpReader = new BmpReader();
	for (int k = 0; k  <  5; k++)
	{
		// read input BMP file
		pic_in = bmpReader -> ReadBMP(inputfile_name[k], &img_width, &img_height);
		// allocate space for output image
		pic_out = (unsigned char*)malloc(3 * img_width * img_height * sizeof(unsigned char));

		//apply the Gaussian filter to the image, RGB respectively
		int cnt = 0;
		for (int j = 0; j < img_height; j++) 
		{
			for (int i = 0; i < img_width; i++)
			{
				pic_out[3 * (j * img_width + i) + MYRED] = gaussian_filter(i, j, MYRED);
				pic_out[3 * (j * img_width + i) + MYGREEN] = gaussian_filter(i, j, MYGREEN);
				pic_out[3 * (j * img_width + i) + MYBLUE] = gaussian_filter(i, j, MYBLUE);
			}
		}

		// write output BMP file
		bmpReader->WriteBMP(outputBlur_name[k], img_width, img_height, pic_out);

		//free memory space
		free(pic_in);
		free(pic_out);
	}

	return 0;
}
