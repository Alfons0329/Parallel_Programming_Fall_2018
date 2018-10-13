#include <bits/stdc++.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

using namespace std;
int cpu_cores=0;
long long int num_tosses=0;

inline void multi_thread_monte_carlo()
{


}

inline void single_thread_monte_carlo()
{
    float x = 0.0f, y = 0.0f;
    float pi_estimated = 0.0f;
    long long int in_circle=0;

    for(register long long int i = 0; i < num_tosses; i++)
    {
        x = rand() / ((float)RAND_MAX);
        y = rand() / ((float)RAND_MAX);
        //printf("x=[%f] y=[%f] \n", x, y);
        if( x*x + y*y <=1 )
        {
            in_circle++;
        }
    }

    pi_estimated = 4 * in_circle / ((float)num_tosses);
    printf("Pi: %f\n", pi_estimated);
}


int main(int argc, char* argv[])
{
    cpu_cores=atoi(argv[1]);
    num_tosses=atoll(argv[2]);
    printf("cpu cores %d, num_tosses %lld \n", cpu_cores, num_tosses);
    srand(time(NULL));

    //--------------------------single thread monte carlo-----------------------//
    struct timeval start, end;

    gettimeofday(&start, 0);
    single_thread_monte_carlo(); //measure the time of single thread process
    gettimeofday(&end, 0);

    int sec = abs(end.tv_sec - start.tv_sec);
    int usec = abs(end.tv_usec - start.tv_usec);
    printf("single thread elapsed time %f ms \n", sec * 1000.0f + usec / 1000.0f);
    //--------------------------multi thread monte carlo-----------------------//
    return 0;
}
