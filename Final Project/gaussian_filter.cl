__kernel void cl_gaussian_filter(
        __global unsigned char* cl_input_image,     // data in global memory 
        __global unsigned char* cl_output_image,    // data in global memory 
        __global unsigned int* cl_filter_G, 
        __global int* img_width, 
        __global int* img_height, 
        __global int* shift, 
        __global int* ws,                                     // sqrt(FILTER_SIZE) 
        __global int* FILTER_SCALE) 
{
    // for opencl parallelization
    const int cl_width = get_global_id(0);
    const int cl_height = get_global_id(1);

	unsigned int tmp = 0;
	int a, b;

    if (cl_width >= *img_width || cl_height >= *img_height)
    {
        return; 
    }

	for (int j = 0; j  <  *ws; j++)
	{
		for (int i = 0; i  <  *ws; i++)
		{
			a = cl_width + i - (*ws / 2);
			b = cl_height + j - (*ws / 2);

			tmp += cl_filter_G[j * (*ws) + i] * cl_input_image[3 * (b * *img_width + a) + *shift];
		}
	}
	tmp /= *FILTER_SCALE;

	if (tmp > 255)
	{
		tmp = 255;
	}

    cl_output_image[3 * (cl_height * (*img_width) + cl_width) + (*shift)] = tmp;
}
