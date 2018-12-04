#include <bits/stdc++.h>

using namespace std;


int main(int argc, char* argv[])
{
    ifstream serial;
    ifstream cuda;

    string agv1(argv[1]);
    string agv2(argv[2]);
    string::size_type sz;     // alias of size_t
    
    cout << "agv1 = " << agv1 << " , agv2 = " << agv2 << endl;
    serial.open(agv1.c_str());
    cuda.open(agv2.c_str());

    if(!serial.is_open() || !cuda.is_open())
    {
        printf("fopen failed!!! \n");
        return 1;
    }
    vector<float> serial_point;
    vector<float> cuda_point;

    int cnt = 0;
    string str;
    while(serial >> str)
    {
        ++cnt;
        if(cnt >= 22)
        {
            serial_point.push_back(atof(str.c_str()));
        }
    }
    cnt = 0;
    while(cuda >> str)
    {
        ++cnt;
        if(cnt >= 22)
        {
            cuda_point.push_back(atof(str.c_str()));
        }
    }

    int diff_cnt = 0;
    for (int i = 0; i < cuda_point.size(); ++i)
    {
        if(abs(cuda_point[i] - serial_point[i]) >= 0.01f)
        {
            ++diff_cnt;
            //cout << cuda_point[i] << "  ,  " << serial_point[i] << endl;
        }
    }
    printf("Diff is %d in %d points, rate %f \n", diff_cnt, cuda_point.size(), (float) diff_cnt / cuda_point.size());
    return 0;
}
