#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <CL/opencl.h>
#include <fstream>
#include <iostream>
/***************************Image histogram algorithm abstract by myself*************************************
 * Calculate the statistic data of the (R, G, B) of image, 
 * where each color can be describe as
 * 0xRRGGBB and the R_stat[RR]++; G_stat[GG]++; B_stat[BB]++;, after the iterations
 * the result can be shown as an image histogram
 * **********************************************************************************************************/
unsigned int * histogram(unsigned char *image_data, unsigned int _size) {

    unsigned char *img = image_data;
    unsigned int *ref_histogram_results;
    unsigned int *ptr;

    ref_histogram_results = (unsigned int *)malloc(256 * 3 * sizeof(unsigned int));
    ptr = ref_histogram_results;
    memset (ref_histogram_results, 0x0, 256 * 3 * sizeof(unsigned int));
    // Data encoded with R G B R G B R G B...
    // histogram of R
    for (unsigned int i = 0; i < _size; i += 3)
    {
        unsigned int index = img[i];
        ptr[index]++;
    }

    // histogram of G
    ptr += 256;
    for (unsigned int i = 1; i < _size; i += 3)
    {
        unsigned int index = img[i];
        ptr[index]++;
    }

    // histogram of B
    ptr += 256;
    for (unsigned int i = 2; i < _size; i += 3)
    {
        unsigned int index = img[i];
        ptr[index]++;
    }

    return ref_histogram_results;
}

int main(int argc, char const *argv[])
{
    // OpenCL init starts here
    cl_int cl_err_num;
    cl_uint num_device;
    cl_device_id device_id;
    cl_platform_id plat_pd;

    // OpenCL init ends here
    
    // Input size and required variables starts here
    unsigned char* image; // For input image 
    unsigned int* histogram_results; // For output result
    unsigned int i = 0, a, input_size;

    // Input size and required variables ends here
    // Use traditional C FILE IO will be faster for performance
    // std::fstream inFile("input", std::ios_base::in);
    // std::ofstream outFile("xxxxxx.out", std::ios_base::out);
    
    // Traditional C FILE IO starts here
    FILE* inFile = fopen("input", "r");
    FILE* outFile = fopen("0416324.out", "w");
    fscanf(inFile, "%u", &input_size);

    // Allocate memory starts here
    
    // OpenCL memory allocation starts here
    // OpenCL memory allocation ends here
    
    
    image = (unsigned char* ) malloc (sizeof(unsigned char) * input_size);//R, G, B ranging from 0x00 to 0xFF
    histogram_results = (unsigned int* ) malloc (sizeof(unsigned int) * 256 * 3);//R 256, G 256, B 256, total 768 statistical data
    // Allocate memory ends here
    while(fscanf(inFile, "%u", &a) != EOF)
    {
        image[i++] = a;
    }
    // Traditional C FILE IO ends here
    histogram_results = histogram(image, input_size);
    
    for(unsigned int i = 0; i < 256 * 3; ++i) 
    {
        if (i % 256 == 0 && i != 0)
        {
            fprintf(outFile, "\n");
        }
        fprintf(outFile, "%u", histogram_results[i]);
    }

    fclose(inFile);
    fclose(outFile);
    return 0;
}
