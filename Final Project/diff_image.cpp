#include <bits/stdc++.h>
#include "bmpReader.h"
#include "bmpReader.cpp"
using namespace std;

int main(int argc, char* argv[])
{
    string fname_serial(argv[1]);
    string fname_parallel(argv[2]);

    ifstream f1;
    ifstream f2;
    
    f1.open(fname_serial);
    f2.open(fname_parallel);

    if (!f1.is_open())
    {
        cout << fname_serial << " open failed! \n";
        return 1;
    }
    if (!f2.is_open())
    {
        cout << fname_parallel << " open failed! \n";
        return 1;
    }

    BmpReader* my_reader = new BmpReader();
    unsigned char* serial_img;
    unsigned char* cuda_img;
    int img_col, img_row, diff_cnt = 0;
    serial_img = my_reader -> ReadBMP(fname_serial.c_str(), &img_col, &img_row);
    cuda_img = my_reader -> ReadBMP(fname_parallel.c_str(), &img_col, &img_row);
    
    for (int i = 0; i < img_col * img_row * 3; i += 3)
    {

        if (serial_img[i] != cuda_img[i] || serial_img[i + 1] != cuda_img[i + 1] || serial_img[i + 2] != cuda_img[i + 2])
        {
            diff_cnt++;
        }
    }
    int img_size = img_col * img_row;
    printf("Serial image, check difference in %s and %s with size of W: %d. H: %d \
            \nGot %d diffs in %d points rate %3.1f percent \n", fname_serial.c_str(), fname_parallel.c_str(), img_col, img_row, diff_cnt, img_size, (float) diff_cnt * 100.0f / (float) img_size);
    return 0;
}



