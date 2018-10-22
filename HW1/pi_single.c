#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

int main(int argc, char* argv[])
{
    float x = 0.0f, y = 0.0f;
    float pi_estimated = 0.0f;
    long long int in_circle=0;

    int cpu_cores=atoi(argv[1]);
    long long int num_tosses=atoll(argv[2]);
    srand(time(NULL));

    for(register long long int i = 0; i < num_tosses; i++)
    {
        x = rand() / ((float)RAND_MAX);
        y = rand() / ((float)RAND_MAX);
        if( x*x + y*y <=1 )
        {
            in_circle++;
        }
    }

    pi_estimated = 4 * in_circle / ((float)num_tosses);
    printf("Pi: %f\n", pi_estimated);
    return 0;
}
