#include <bits/stdc++.h>
using namespace std;

int main()
{
    ifstream mask;
    mask.open("mask_Gaussian.txt");
    if (!mask.is_open())
    {
        printf("fopen failed !!! \n");
        return 1;
    }

    vector<int> vals;
    int tmp;
    
    while(mask >> tmp)
    {
        vals.push_back(tmp);
    }
    
    int n = vals.size();
    int nonzero = 0, max_val = 0;
    for (int i = 1; i < n; i++)
    {
        if(vals[i])
        {
            nonzero++;
            max_val = max(max_val, vals[i]);
        }

    }
    float rate = (float) nonzero / (float) n; 
    printf("max %d, nonzero %d in %d rate %f \n", max_val, nonzero, n, rate);
    return 0;
}
