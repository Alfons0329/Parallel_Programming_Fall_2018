/*
	You can NOT modify anything in this file.
*/
#include <stdio.h>
#include <stdlib.h>

unsigned char* BmpReader::ReadBMP(char const *f, int *a_width, int *a_height )
{
	fp_s = NULL;    // source file handler
	image_s = NULL; // source image array

	static unsigned char tmp_header[54] = {
		0x42,        // identity : B
		0x4d,        // identity : M
		0, 0, 0, 0,  // file size
		0, 0,        // reserved1
		0, 0,        // reserved2
		54, 0, 0, 0, // RGB data offset
		40, 0, 0, 0, // struct BITMAPINFOHEADER size
		0, 0, 0, 0,  // bmp width
		0, 0, 0, 0,  // bmp height
		1, 0,        // planes
		24, 0,       // bit per pixel
		0, 0, 0, 0,  // compression
		0, 0, 0, 0,  // data size
		0, 0, 0, 0,  // h resolution
		0, 0, 0, 0,  // v resolution
		0, 0, 0, 0,  // used colors
		0, 0, 0, 0   // important colors
	};
	header = tmp_header;

	fp_s = fopen(f, "rb");
	if (fp_s == NULL) {
		printf(" Error: %s not exist.\n",f);
		return 0;
	}

	// move offset to 10 to find rgb raw data offset
	fseek(fp_s, 10, SEEK_SET);
	fread(&rgb_raw_data_offset, sizeof(unsigned int), 1, fp_s);
	// move offset to 18    to get width & height;
	fseek(fp_s, 18, SEEK_SET);
	fread(&width,  sizeof(unsigned int), 1, fp_s);
	fread(&height, sizeof(unsigned int), 1, fp_s);
	// get  bit per pixel
	fseek(fp_s, 28, SEEK_SET);
	fread(&bit_per_pixel, sizeof(unsigned short), 1, fp_s);
	byte_per_pixel = bit_per_pixel / 8;
	// move offset to rgb_raw_data_offset to get RGB raw data
	fseek(fp_s, rgb_raw_data_offset, SEEK_SET);

	image_s = (unsigned char *)malloc((size_t)width * height * byte_per_pixel);
	if (image_s == NULL) {
		printf("malloc images_s error\n");
		return 0;
	}

	fread(image_s, sizeof(unsigned char), (size_t)(long)width * height * byte_per_pixel, fp_s);

	*a_width = width;
	*a_height = height;
	fclose(fp_s);

	return image_s;
}

int BmpReader::WriteBMP(char const *f, int a_width, int a_height, unsigned char *ptr )
{
	fp_t = NULL;    // target file handler

	// write to new bmp
	fp_t = fopen(f, "wb");
	if (fp_t == NULL) {
		printf("fopen fname_t error\n");
		return -1;
	}

	// file size
	file_size = width * height * byte_per_pixel + rgb_raw_data_offset;
	header[2] = (unsigned char)(file_size & 0x000000ff);
	header[3] = (file_size >> 8)  & 0x000000ff;
	header[4] = (file_size >> 16) & 0x000000ff;
	header[5] = (file_size >> 24) & 0x000000ff;

	// width
	header[18] = width & 0x000000ff;
	header[19] = (width >> 8)  & 0x000000ff;
	header[20] = (width >> 16) & 0x000000ff;
	header[21] = (width >> 24) & 0x000000ff;

	// height
	header[22] = height &0x000000ff;
	header[23] = (height >> 8)  & 0x000000ff;
	header[24] = (height >> 16) & 0x000000ff;
	header[25] = (height >> 24) & 0x000000ff;

	// bit per pixel
	header[28] = bit_per_pixel;

	// write header
	fwrite(header, sizeof(unsigned char), rgb_raw_data_offset, fp_t);
	// write image
	fwrite(ptr, sizeof(unsigned char), (size_t)(long)width * height * byte_per_pixel, fp_t);

	fclose(fp_t);

	return 0;
}
