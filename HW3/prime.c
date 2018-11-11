#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int isprime(int n)
{
    int i, squareroot;
    if (n > 10)
    {
        squareroot = (int)sqrt(n);
        for (i = 3; i <= squareroot; i = i + 2)
            if ((n % i) == 0)
                return 0;
        return 1;
    }
    else
        return 0;
}
/* algorithm abstract
 * distribute the total / N primes to count for each
 * mod the number into range from 0 to N - 1
 * the slave CPU #0 search the number K { K | K % N = 0 } to determine the prime, and the like
 * */
int main(int argc, char *argv[])
{
    int 
        pc,       /* prime counter */
        foundone; /* most recent prime found */

    long long int 
        n, /* number to start count on */ 
        limit /* upper size of count */
        search_for_each_cpu; /* primes to search for eahc CPU */


    int rank, /* CPU rank to identify which CPU is now used */ 
        size; /* task size for each cpu */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0)
    {
        sscanf(argv[1], "%llu", &limit);
        printf("Starting. Numbers to be scanned= %lld\n", limit);

        pc = 4; /* Assume (2,3,5,7) are counted <F4>here */

        for (n = 11; n <= limit; n = n + 2)
        {
            if (isprime(n))
            {
                pc++;
                foundone = n;
            }
        }
    }
    else
    {

    }

    printf("Done. Largest prime is %d Total primes %d\n", foundone, pc);

    MPI_Finalize(); //terminate all distributive computation
    return 0;
}
