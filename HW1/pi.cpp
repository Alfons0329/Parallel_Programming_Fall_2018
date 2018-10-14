#include <bits/stdc++.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

using namespace std;
int cpu_cores = 0;
long long int num_tosses = 0;

long long int* each_in_circle;

inline int fast_rand()
{
    int g_seed = time(NULL);
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) & 0x7FFF;
}

inline int fast_rand2(int low, int high)
{
    std::default_random_engine rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(low, high);

    return dis(gen);
}

inline void* monte_carlo_runner(void* args)
{
    long cur_thread = (long) args;
    //printf("cur thread %d \n", cur_thread);
   
    float x = 0.0f, y = 0.0f;
    float pi_estimated = 0.0f;
    long long int in_circle = 0;
    unsigned int seed = time(NULL);

    for (register long long int i = num_tosses / cpu_cores; i >= 0; i--)
    // for (register long long int i = 0; i < num_tosses / cpu_cores; i++)
    {
        x = rand_r(&seed) / ((float)RAND_MAX); //thread-safe random number generator
        y = rand_r(&seed) / ((float)RAND_MAX); //thread-safe random number generator

        // x = fast_rand2(-1, 1) / (float) 1.0;
        // y = fast_rand2(-1, 1) / (float) 1.0;

        // printf("x=[%f] y=[%f] \n", x, y);
        if (x * x + y * y <= 1)
        {
            in_circle++;
        }
    }

    each_in_circle[cur_thread] = in_circle;
    pthread_exit(EXIT_SUCCESS);
}

inline void single_thread_monte_carlo()
{
    float x = 0.0f, y = 0.0f;
    float pi_estimated = 0.0f;
    long long int in_circle = 0;

    for (register long long int i = 0; i < num_tosses; i++)
    {
        x = rand() / ((float)RAND_MAX);
        y = rand() / ((float)RAND_MAX);
        //printf("x=[%f] y=[%f] \n", x, y);
        if (x * x + y * y <= 1)
        {
            in_circle++;
        }
    }

    pi_estimated = 4 * in_circle / ((float)num_tosses);
    printf("Pi: %f\n", pi_estimated);
}

int main(int argc, char *argv[])
{
    cpu_cores = atoi(argv[1]);
    num_tosses = atoll(argv[2]);
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
    
    each_in_circle = new long long int[cpu_cores]; //stores the in_circle count of each thread
    pthread_t thread_id[cpu_cores];

    for (long i = 0; i < cpu_cores; i++) //i is the current thread
    {
        pthread_create(&thread_id[i], NULL, monte_carlo_runner, (void *) i);
    }

    gettimeofday(&start, 0);
    for (long i = 0; i < cpu_cores; i++)
    {
        pthread_join(thread_id[i], NULL);
    }
    //calculate pi from # of cpu_cores threads
    long long int sum = 0;
    for (int i = 0; i < cpu_cores; i++)
    {
        sum += each_in_circle[i];
    }
    gettimeofday(&end, 0);
    float pi_estimated = 4 * sum / ((float)num_tosses);
    
    sec = abs(end.tv_sec - start.tv_sec);
    usec = abs(end.tv_usec - start.tv_usec);
    printf("Pi: %f\n", pi_estimated);
    printf("multithread thread ([%d] threads) elapsed time %f ms \n",cpu_cores ,sec * 1000.0f + usec / 1000.0f);

    return 0;
}
