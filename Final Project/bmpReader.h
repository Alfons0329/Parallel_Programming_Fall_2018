/* 
	You can NOT modify anything in this file.
*/
#include <stdio.h>
#include <stdlib.h>

class BmpReader {
public:
	unsigned char *ReadBMP(char const *f, int *a_width, int *a_height );
	int WriteBMP(char const *f, int a_width, int a_height, unsigned char *ptr );
private:
	FILE          *fp_s;    // source file handler
	FILE          *fp_t;    // target file handler 
	unsigned int  x, y;             // for loop counter
	unsigned int  width, height;   // image width, image height
	unsigned char *image_s; // source image array
	unsigned char *header;
	unsigned int file_size;           // file size
	unsigned int rgb_raw_data_offset; // RGB raw data offset
	unsigned short bit_per_pixel;     // bit per pixel
	unsigned short byte_per_pixel;    // byte per pixel
};