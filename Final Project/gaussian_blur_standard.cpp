#include "bmpReader.h"
#include "bmpReader.cpp"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <pthread.h>
#include <string>

// openCV libraries for showing the images
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

#define MYRED	2
#define MYGREEN 1
#define MYBLUE	0
#define RATE 1000000
int img_width, img_height;

int FILTER_SIZE;
float FILTER_SCALE;
float *filter_G;

unsigned char *pic_in, *pic_blur, *pic_out;

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
    else
    {
        inputfile_name = argv[1];
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
	printf("filter scale [%f] \n", FILTER_SCALE);
	fclose(mask);

    // main part of Gaussian blur
	BmpReader* bmpReader = new BmpReader();
	for (int k = 1; k < argc; k++)
	{
		// read input BMP file
		pic_in = bmpReader -> ReadBMP(inputfile_name.c_str(), &img_width, &img_height);

		// allocate space for output image
		pic_out = (unsigned char*)malloc(3 * img_width * img_height * sizeof(unsigned char));

		//apply the Gaussian filter to the image, RGB respectively
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
        string tmp(inputfile_name);
        outputblur_name = "input" + to_string(k) + "_blur.bmp";
		bmpReader->WriteBMP(outputblur_name.c_str(), img_width, img_height, pic_out);

		// free memory space
		free(pic_in);
		free(pic_out);
	}

	return 0;
}
