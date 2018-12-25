#include <bits/stdc++.h>
#include "bmpReader.h"
#include "bmpReader.cpp"
using namespace std;

int main(int argc, char* argv[])
{
    string unpadded(argv[1]);
    string f_name1 = unpadded.substr(0, unpadded.size() - 4) + "_blur_unpadded.bmp";
    string f_name2 = unpadded.substr(0, unpadded.size() - 4) + "_blur_cuda.bmp";


    ifstream f1;
    ifstream f2;
    
    f1.open(f_name1);
    f2.open(f_name2);

    if (!f1.is_open())
    {
        cout << f_name1 << " open failed! \n";
        return 1;
    }
    if (!f2.is_open())
    {
        cout << f_name2 << " open failed! \n";
        return 1;
    }

    BmpReader* my_reader = new BmpReader();
    unsigned char* serial_img;
    unsigned char* cuda_img;
    int img_width, img_height, diff_cnt = 0;
    serial_img = my_reader -> ReadBMP(f_name1.c_str(), &img_width, &img_height);
    cuda_img = my_reader -> ReadBMP(f_name2.c_str(), &img_width, &img_height);
    
    for (int i = 0; i < img_width * img_height * 3; i += 3)
    {

        if (serial_img[i] != cuda_img[i] || serial_img[i + 1] != cuda_img[i + 1] || serial_img[i + 2] != cuda_img[i + 2])
        {
            diff_cnt++;
        }
    }
    int img_size = img_width * img_height;
    printf("Unpadded image, check difference in %s and %s with size of W: %d. H: %d \
            \nGot %d diffs in %d points rate %3.1f percent \n", f_name1.c_str(), f_name2.c_str(), img_width, img_height, diff_cnt, img_size, (float) diff_cnt * 100.0f / (float) img_size);
    return 0;
}



