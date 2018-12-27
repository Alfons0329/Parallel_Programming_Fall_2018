#include "bmpReader.h"
#include "bmpReader.cpp"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string>
#include <pthread.h>

#define THREAD_COUNT 4

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

unsigned char *pic_in, *pic_blur, *pic_out;

int thread_id[THREAD_COUNT];
pthread_t thread_pid[THREAD_COUNT];

// count how many threads finished a specific segment
int finish_count[10+1];

// protect shared variable finishCount
pthread_mutex_t finish_count_mutex;

// notify main thread that a segment is done
pthread_cond_t finish_cond;

unsigned char gaussian_filter(int w, int h,int shift)
{
	int ws = (int)sqrt((int)FILTER_SIZE);
	int target = 0;
	uint32 tmp = 0;
	int a, b;

	for (int j = 0; j  <  ws; j++)
	{
		for (int i = 0; i  <  ws; i++)
		{
			a = w + i - (ws / 2);
			b = h + j - (ws / 2);

			// detect for borders of the image
			if (a < 0 || b < 0 || a >= img_width || b >= img_height)
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

void *worker(void *param) {
	int id = *(int *) param;
	int segment_cnt;
	for (segment_cnt = 1; segment_cnt <= 10; segment_cnt++)
	{
		int seg_start = segment_cnt == 1 ? 0 : img_height * segment[segment_cnt-2];
		int seg_end = img_height * segment[segment_cnt-1];

		// sub segment
		int size = (seg_end - seg_start + THREAD_COUNT-1) / THREAD_COUNT;
		int part_start = seg_start + size * id;
		int part_end = seg_start + size * (id+1);
		if (part_end > seg_end) part_end = seg_end;

		for (int j = part_start; j < part_end; j++)
		{
			for (int i = 0; i < img_width; i++)
			{
				pic_out[3 * (j * img_width + i) + MYRED] = gaussian_filter(i, j, MYRED);
				pic_out[3 * (j * img_width + i) + MYGREEN] = gaussian_filter(i, j, MYGREEN);
				pic_out[3 * (j * img_width + i) + MYBLUE] = gaussian_filter(i, j, MYBLUE);
			}
		}
		// notify main thread that a segment is finished
		pthread_mutex_lock(&finish_count_mutex);
		finish_count[segment_cnt] += 1;
		pthread_cond_signal(&finish_cond);
		pthread_mutex_unlock(&finish_count_mutex);
	}
	return NULL;
}

int main(int argc, char* argv[])
{
	// read input filename
	string inputfile_name;
	string outputblur_name;

	if (argc < 2)
	{
		printf("Please provide filename for Gaussian Blur. usage ./gb_pthread.o <BMP image file>\n");
		return 1;
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
		printf("Filter scale = %u and image size W = %d, H = %d\n", FILTER_SCALE, img_width, img_height);

		int resolution = 3 * (img_width * img_height);
		// allocate space for output image
		pic_out = (unsigned char*)malloc(3 * img_width * img_height * sizeof(unsigned char));

		//apply the Gaussian filter to the image, RGB respectively
		string tmp(inputfile_name);
		int segment_cnt = 1;
		outputblur_name = inputfile_name.substr(0, inputfile_name.size() - 4)+ "_blur_pthread.bmp";

		// reset finish count
		for (segment_cnt = 1; segment_cnt <= 10; segment_cnt++) {
			finish_count[segment_cnt] = 0;
		}
		pthread_mutex_init(&finish_count_mutex, NULL);
		pthread_cond_init(&finish_cond, NULL);

		// create worker
		for (int i = 0; i < THREAD_COUNT; i++)
		{
			thread_id[i] = i;
			pthread_create(&thread_pid[i], NULL, worker, &thread_id[i]);
		}

		for (segment_cnt = 1; segment_cnt <= 10; segment_cnt++)
		{
			pthread_mutex_lock(&finish_count_mutex);
			// wait until every thread finishes a segment
			while (finish_count[segment_cnt] < THREAD_COUNT) {
				pthread_cond_wait(&finish_cond, &finish_count_mutex);
			}
			pthread_mutex_unlock(&finish_count_mutex);

			// show the progress of image every 10% of work progress
			// printf("Show segment %d with j is now %d \n", segment_cnt, seg_end-1);
			write_and_show(bmpReader, outputblur_name, k);
		}
		// write output BMP file
		bmpReader->WriteBMP(outputblur_name.c_str(), img_width, img_height, pic_out);
		write_and_show(bmpReader, outputblur_name, k);

		// destroy worker
		for (int i = 0; i < THREAD_COUNT; i++) {
			pthread_join(thread_pid[i], NULL);
		}
		pthread_mutex_destroy(&finish_count_mutex);
		pthread_cond_destroy(&finish_cond);
        
        // if demo, decomment this to show until ESC is pressed, dont remove the following, just comment or decomment
        /*
        Mat img = imread(outputblur_name);
        while(1)
        {
            imshow("Current progress", img);
            if (waitKey(0) % 256 == 27)
            {
                break;
            }
        }
        */
 
		// free memory space
		free(pic_in);
		free(pic_out);
	}

	return 0;
}
