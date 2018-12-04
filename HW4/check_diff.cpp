#include <bits/stdc++.h>

using namespace std;


int main()
{
    ifstream serial;
    ifstream cuda;

    serial.open("serial_result.txt");
    cuda.open("cuda_result.txt");

    vector<string> serial_point;
    vector<string> cuda_point;

    int cnt = 0;
    string str;
    while(serial >> str)
    {
        ++cnt;
        if(cnt >= 22)
        {
            serial_point.push_back(str);
        }
    }
    cnt = 0;
    while(cuda >> str)
    {
        ++cnt;
        if(cnt >= 22)
        {
            cuda_point.push_back(str);
        }
    }

    int diff_cnt = 0;
    for (int i = 0; i < cuda_point.size(); ++i)
    {
        if(cuda_point[i] != serial_point[i])
        {
            ++diff_cnt;
            //cout << cuda_point[i] << "  ,  " << serial_point[i] << endl;
        }
    }
    printf("Diff is %d in %d points, rate %f \n", diff_cnt, cuda_point.size(), (float) diff_cnt / cuda_point.size());
    return 0;
}
