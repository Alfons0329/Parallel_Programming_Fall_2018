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

    long long int limit; /* upper size of count */

    /*-------------------------------MPI data declaration starts--------------------------------*/
    int rank, /* CPU rank to identify which CPU is now used */ 
        size; /* task size for each cpu */

    struct prime_st sen, rcv; /* prime data struct for MPI computing, send and receive*/

    MPI_Datatype datatype, oldtype[1]; /* only one datatype, so oldtype length be 1 */
    int blockcount[1];
    MPI_Aint offsets[1];
    MPI_Status status;
    /*-------------------------------MPI data declaration ends--------------------------------*/

    /*-------------------------------MPI init starts--------------------------------*/
    MPI_Init(&argc, &argv);

    sscanf(argv[1], "%llu", &limit); /* all the machines, including master and slave, gets the limit */

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    blockcount[0] = 2; /* long long gots two: max_prime and cnt */
    offsets[0] = 0; /* No offsets is needed */
    oldtype[0] = MPI_LONG_LONG_INT;

    /* define structured type and commit it */
    MPI_Type_create_struct(1, blockcount, offsets, oldtype, &datatype);
    MPI_Type_commit(&datatype);

    sen.max_prime = 0;
    sen.cnt = 0;
    /*-------------------------------MPI init ends--------------------------------*/
    if (rank == 0)
    {
        printf("Starting. Numbers to be scanned= %lld\n",limit);
    }

    for (register long long int n = 11 + rank * 2; n <= limit; n += size * 2) 
    {
        /*if (n & 1)
        {*/
            if (n % 6 == 5 || n % 6 == 1)
            {
                if (isprime(n)) 
                {
                    sen.max_prime = n;
                    sen.cnt++;
                }
            }
            else
            {
                continue;
            }

       /* }
        else
        {
            continue;
        }*/
    }

    if (rank != 0)
    {
        MPI_Send(&sen, 1, datatype, 0, 0, MPI_COMM_WORLD);
    }
    else 
    {
        for (int i = 1 ; i < size ; i++) /* collect slave */
        {
            MPI_Recv(&rcv, 1, datatype, i, 0, MPI_COMM_WORLD, &status);
            sen.cnt += rcv.cnt;
            if (sen.max_prime < rcv.max_prime)
            {
                sen.max_prime = rcv.max_prime;
            }
        }
    }

    MPI_Type_free(&datatype);
    MPI_Finalize();

    if (rank == 0)
    {
        printf("Done. Largest prime is %lld Total primes %lld\n",sen.max_prime,sen.cnt + 4);
    }

    return 0;
}
