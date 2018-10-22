#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

int cpu_cores=0;
long long int num_tosses=0;

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
long long int total_in_circle; //mutex version

void* monte_carlo_runner(void* args)
{
    long cur_thread = (long) args;
   
    float x = 0.0f, y = 0.0f;
    float pi_estimated = 0.0f;
    long long int in_circle = 0;
    unsigned int seed = time(NULL);

    for (register long long int i = num_tosses / cpu_cores; i >= 0; i--)
    {
        x = rand_r(&seed) / ((float)RAND_MAX); //thread-safe random number generator
        y = rand_r(&seed) / ((float)RAND_MAX); //thread-safe random number generator
        if (x * x + y * y <= 1.0f)
        {
            in_circle++;
        }
    }
    pthread_mutex_lock(&mutex1); 
    //each_in_circle[cur_thread] = in_circle; non-mutex version
    total_in_circle+=in_circle;//mutex version
    pthread_mutex_unlock(&mutex1);
    pthread_exit(EXIT_SUCCESS);
}

int main(int argc, char* argv[])
{
    cpu_cores=atoi(argv[1]);
    num_tosses=atoll(argv[2]);
    pthread_t thread_id[cpu_cores];
    for (long i = 0; i < cpu_cores; i++) //i is the current thread
    {
        pthread_create(&thread_id[i], NULL, monte_carlo_runner, (void *) i);
    }

    for (long i = 0; i < cpu_cores; i++)
    {
        pthread_join(thread_id[i], NULL);
    }

    float pi_estimated = 4 * total_in_circle / ((float)num_tosses);
    printf("Pi: %f\n", pi_estimated);
    return 0;
}
