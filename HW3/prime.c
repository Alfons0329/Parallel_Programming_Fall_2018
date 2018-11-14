#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#define ll long long
#define LLSPRP_MAX 7

struct prime_st
{
    ll max_prime, cnt;
};

int isprime(int n)
{
    int i, squareroot;
    if (n > 10)
    {
        squareroot = (int)sqrt(n);
        for (i = 3; i <= squareroot; i = i + 2)
        {
            if ((n % i) == 0)
            {
                return 0;
            }
        }
        return 1;
    }
    else
    {
        return 0;
    }
}
/* referenece from https://computing.llnl.gov/tutorials/mpi/ */
int main(int argc, char *argv[])
{

    long long int n, /* number to start count on */ 
    limit; /* upper size of count */

    /*-------------------------------MPI data declaration starts--------------------------------*/
    int rank, /* CPU rank to identify which CPU is now used */ 
    size; /* task size for each cpu */

    struct prime_st sen, rcv; /* prime data struct for MPI computing, send and receive*/
    sen.max_prime = 0;
    sen.cnt = 4;

    MPI_Datatype datatype, oldtype[1]; /* only one datatype, so oldtype length be 1 */
    int blockcount[1];
    MPI_Aint offsets[1];
    MPI_Status status;
    /*-------------------------------MPI data declaration ends--------------------------------*/

    /*-------------------------------MPI init starts--------------------------------*/
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    blockcount[0] = 2; /* long long gots two: max_prime and cnt */
    offset[0] = 0; /* No offsets is needed */
    oldtype[0] = MPI_LONG_LONG_INT;

    /* define structured type and commit it */
    MPI_Type_create_struct(1, blockcount, offsets, oldtype, &datatype);
    MPI_Type_commit(&datatype);
    /*-------------------------------MPI init ends--------------------------------*/

    if(rank == 0) /* master machine in the cluster send data to slave machine */
    {
        sscanf(argv[1], "%llu", &limit);
        printf("Starting. Numbers to be scanned= %lld\n", limit);

        for (n = 11; n <= limit; n = n + 2)
        {
            if (isprime(n))
            {
                sen.max_prime = n;
                sen.cnt++;
            }
        }

        for (int i = 0; i < size; i++)
        {
            MPI_Send(&sen, 1, datatype, i, 1, MPI_COMM_WORLD);
        }
    }
    
    /* collect result from all the machines in the cluster and update */

    for (int i = 0; i < size; i++)
    {
        MPI_Recv(&rcv, 1, datatype, i, 1, MPI_COMM_WORLD, &status);
        sen.cnt += rcv.cnt;
        
        if(sen.max_prime < rcv.max_prime)
        {
            sen.max_prime = rcv.max_prime;
        }
    }

    /* terminate all distributive computation */
    MPI_Type_free(&datatype);
    MPI_Finalize(); 

    if (rank == 0)
    {
        printf("Done. Largest prime is %lld Total primes %lld\n", sen.max_prime, sen.cnt);
    }
    return 0;
}
