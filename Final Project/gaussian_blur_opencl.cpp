#include "bmpReader.h"
#include "bmpReader.cpp"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <pthread.h>
#include <string>
#include <fstream>
#include <sstream>
#include <CL/cl.h>

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

unsigned char *pic_in, *pic_out;

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
	// OpenCL initialization
	cl_uint num_platform, num_device;
	cl_platform_id pid;
	cl_device_id did;
	cl_int err_platform, err_device, err_context, err_cmdq;
	err_platform = clGetPlatformIDs(1, &pid, &num_platform);
	err_device = clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 1, &did, &num_device);
	cl_context ctx = clCreateContext(NULL, 1, &did, 0, 0, &err_context);
	cl_command_queue cmd_queue = clCreateCommandQueueWithProperties(ctx, did, 0, &err_cmdq);  
	if(err_platform == CL_SUCCESS &&
			err_device == CL_SUCCESS &&
			err_context == CL_SUCCESS &&
			err_cmdq == CL_SUCCESS){
		std::cout << "Initialize without error!" << std::endl;
	}
	else{
		std::cerr << "[Error] Initialize fail!" << std::endl;
		return 0;
	}

	// read input filename
	string inputfile_name;
	string outputblur_name;

	if (argc < 2)
	{
		printf("Please provide filename for Gaussian Blur. usage ./gb_opencl.o <BMP image file>");
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

    int ws = (int)sqrt((int)FILTER_SIZE);
	// main part of Gaussian blur
	BmpReader* bmpReader = new BmpReader();
	for (int k = 1; k < argc; k++)
	{
		// read input BMP file
		inputfile_name = argv[k];
		pic_in = bmpReader -> ReadBMP(inputfile_name.c_str(), &img_width, &img_height);
		printf("Filter scale = %u, filter size %d x %d and image size W = %d, H = %d\n", FILTER_SCALE, (int)sqrt(FILTER_SIZE), (int)sqrt(FILTER_SIZE), img_width, img_height);

		// allocate space for output image
		int resolution = 3 * (img_width * img_height);
		pic_out = (unsigned char*)malloc(3 * img_width * img_height * sizeof(unsigned char));

        // opencl memory allocation
        cl_int err_mem;
        cl_mem in_img_mem = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, resolution * sizeof(unsigned char), pic_in, &err_mem);
        if(err_mem != CL_SUCCESS) std::cerr << "[Error] clCreateBuffer" << std::endl;
        cl_mem out_img_mem = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, resolution * sizeof(unsigned char), NULL, &err_mem);
        if(err_mem != CL_SUCCESS) std::cerr << "[Error] clCreateBuffer" << std::endl;
        cl_mem filter_mem = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE * sizeof(unsigned int), filter_G, &err_mem);
        if(err_mem != CL_SUCCESS) std::cerr << "[Error] clCreateBuffer" << std::endl;
        cl_mem width_mem = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &img_width, &err_mem);
        if(err_mem != CL_SUCCESS) std::cerr << "[Error] clCreateBuffer" << std::endl;
        cl_mem height_mem = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &img_height, &err_mem);
        if(err_mem != CL_SUCCESS) std::cerr << "[Error] clCreateBuffer" << std::endl;
        cl_mem ws_mem = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &ws, &err_mem);
        if(err_mem != CL_SUCCESS) std::cerr << "[Error] clCreateBuffer" << std::endl;
        cl_mem filter_scale_mem = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int), &FILTER_SCALE, &err_mem);
        if(err_mem != CL_SUCCESS) std::cerr << "[Error] clCreateBuffer" << std::endl;

		// Load cl program from file to string
		fstream fs("./gaussian_filter.cl", std::ios_base::in);
		stringstream ss;
		string gf_str;
		ss << fs.rdbuf();
		gf_str = ss.str();
		const char* gf_cstr = gf_str.c_str();
		size_t gf_len = gf_str.size();

		// opencl create and build program
		cl_int err_program, err_build, err_kernel, err_args;
		cl_program program = clCreateProgramWithSource(ctx, 1, &gf_cstr, &gf_len, &err_program);
		if(err_program != CL_SUCCESS) std::cerr << "[Error] clCreateProgramWithSource" << std::endl;
		err_build = clBuildProgram(program, 1, &did, "-Werror", 0, 0);
		if(err_build != CL_SUCCESS){
			std::cerr << "[Error] clBuildProgram" << std::endl;   
			size_t len = 0;
			cl_int ret = CL_SUCCESS;
			ret = clGetProgramBuildInfo(program, did, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
			char *buffer = new char[len];
			ret = clGetProgramBuildInfo(program, did, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
			std::cerr << buffer << std::endl;
		}

        
        //apply the Gaussian filter to the image, RGB respectively
        string tmp(inputfile_name);
        int segment_cnt = 1;

        cl_mem shift_mem[3];
        cl_kernel kernel[3];

        for (int i = 2; i >= 0; i--)
        {
			cl_int err;
			size_t global_work_size[2] = {(size_t) img_width, (size_t) img_height};
        	cl_int err_kernel;
            kernel[i] = clCreateKernel(program, "cl_gaussian_filter", &err_kernel);
            if(err_kernel != CL_SUCCESS) std::cerr << "[Error] clCreateKernel" << std::endl; 

            shift_mem[i] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &i, &err_mem);
            if(err_mem != CL_SUCCESS) std::cerr << "[Error] clCreateBuffer" << std::endl;
            err_args = clSetKernelArg(kernel[i], 0, sizeof(in_img_mem), &in_img_mem);
            if(err_args != CL_SUCCESS) std::cerr << "[Error] clSetKernelArg" << std::endl;
            err_args = clSetKernelArg(kernel[i], 1, sizeof(out_img_mem), &out_img_mem);
            if(err_args != CL_SUCCESS) std::cerr << "[Error] clSetKernelArg" << std::endl;
            err_args = clSetKernelArg(kernel[i], 2, sizeof(filter_mem), &filter_mem);
            if(err_args != CL_SUCCESS) std::cerr << "[Error] clSetKernelArg" << std::endl;
            err_args = clSetKernelArg(kernel[i], 3, sizeof(width_mem), &width_mem);
            if(err_args != CL_SUCCESS) std::cerr << "[Error] clSetKernelArg" << std::endl;
            err_args = clSetKernelArg(kernel[i], 4, sizeof(height_mem), &height_mem);
            if(err_args != CL_SUCCESS) std::cerr << "[Error] clSetKernelArg" << std::endl;
            err_args = clSetKernelArg(kernel[i], 5, sizeof(shift_mem[i]), &shift_mem[i]);
            if(err_args != CL_SUCCESS) std::cerr << "[Error] clSetKernelArg" << std::endl;
            err_args = clSetKernelArg(kernel[i], 6, sizeof(ws_mem), &ws_mem);
            if(err_args != CL_SUCCESS) std::cerr << "[Error] clSetKernelArg" << std::endl;
            err_args = clSetKernelArg(kernel[i], 7, sizeof(filter_scale_mem), &filter_scale_mem);
            if(err_args != CL_SUCCESS) std::cerr << "[Error] clSetKernelArg" << std::endl;

			err = clEnqueueNDRangeKernel(cmd_queue, kernel[i], 2, 0, global_work_size, 0, 0, 0, 0);

			if(err != CL_SUCCESS)
			{
				cout << "[Error] clEnqueueNDRangeKernel" << endl;
				return 3;
			} 
        }

        clFinish(cmd_queue);

        cl_uint err_readbuffer;
        err_readbuffer = clEnqueueReadBuffer(cmd_queue, out_img_mem, CL_TRUE, 0, 3 * img_width * img_height * sizeof(unsigned char), pic_out, 0, 0, 0);
        if(err_readbuffer != CL_SUCCESS) std::cerr << "[Error] clEnqueueReadBuffer" << std::endl;
        
        outputblur_name = inputfile_name.substr(0, inputfile_name.size() - 4)+ "_blur_opencl.bmp";
        // bmpReader->WriteBMP(outputblur_name.c_str(), img_width, img_height, pic_out);

        // write_and_show(bmpReader, outputblur_name, k);
        // if demo, decomment this to show until ESC is pressed

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

        // free opencl object
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseMemObject(in_img_mem);
        clReleaseMemObject(out_img_mem);
        clReleaseMemObject(filter_mem);
        clReleaseMemObject(width_mem);
        clReleaseMemObject(height_mem);
        clReleaseMemObject(ws_mem);
        clReleaseMemObject(filter_scale_mem);
        clReleaseCommandQueue(cmd_queue);
        clReleaseContext(ctx);

        // free memory space
        free(pic_in);
        free(pic_out);
	}

	return 0;
}
