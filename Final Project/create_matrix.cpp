#include <cmath> 
#include <iomanip> 
#include <iostream> 
#include <fstream>
#include <vector>
#define ull unsigned long long int
using namespace std; 
int matrix_size;
const float sigma = 15.0f; 
const ull RATE = 1e7;
void FilterCreation(vector<vector <float> >& GKernel) 
{ 
    float r, s = 2.0 * sigma * sigma; 
    // sum is for normalization 
    float sum = 0.0; 
  
    // generating  kernel 
    for (int x = -matrix_size / 2; x <= matrix_size / 2; x++)
    { 
        for (int y = -matrix_size / 2; y <= matrix_size / 2; y++) 
        { 
            r = sqrt(x * x + y * y); 
            GKernel[x + matrix_size / 2][y + matrix_size / 2] = (exp(-(r * r) / s)) / (M_PI * s); 
            sum += GKernel[x + matrix_size / 2][y + matrix_size / 2];
        } 
    } 
  
    // normalising the Kernel 
    for (int i = 0; i < matrix_size; ++i)
    {
        for (int j = 0; j < matrix_size; ++j)
        {
            GKernel[i][j] /= sum;
        }
    }
        
} 
  
// Driver program to test above function 
int main() 
{ 
    ofstream output_file("mask_Gaussian.txt");
    if(!output_file.is_open())
    {
        printf("Error reading file");
    }

    printf("Input N for N * N Gaissuan square matrix: ");
    scanf("%d", &matrix_size);
    if((matrix_size * matrix_size) % 2 == 0)
    {
        printf("Matrix size MUST be an odd number ! \n");
        return 1;
    }
    printf("matrix = %d , stddev = %lf \n", matrix_size, sigma);

    vector<vector<float> > GKernel(matrix_size, vector<float>(matrix_size, 0.0f));
    FilterCreation(GKernel); 
  
    output_file << matrix_size * matrix_size << endl;

    ull tmp = 0;
    for (int i = 0; i < matrix_size; ++i) 
    { 
        for (int j = 0; j < matrix_size; ++j)
        {
            tmp = (unsigned long long) ( GKernel[i][j] * RATE );
            output_file << tmp << " " ;
        }
        output_file << endl; 
    }
    return 0;
} 
