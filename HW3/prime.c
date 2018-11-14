#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#define LLSPRP_MAX 7
/*
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
*/
long long mod_mul(long long a, long long b, long long m)
{
    a %= m, b %= m;
    long long y = (long long)((double)a * b / m + 0.5); /* fast for m < 2^58 */
    long long r = (a * b - y * m) % m;
    return r < 0 ? r + m : r;
}

long long powmod(long long a, long long b, long long mod)
{ 
    long long ans = 1;
    for (; b; a = mod_mul(a, a, mod), b >>= 1)
        if (b & 1)
            ans = mod_mul(ans, a, mod);
    return ans;
}

int sprp[3] = {2, 7, 61};                                           //int range
int llsprp[7] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022}; //unsigned long long

bool isprime(long long n, int *llsprp, int num)
{
    int t = 0;
    long long u = n - 1;

    for (; u % 2 == 0; ++t)
    {
        u >>= 1;
    } 

    for (int i = 0; i < num; ++i)
    {
        long long a = llsprp[i] % n;
        if (a == 0 || a == 1 || a == n - 1)
            continue;
        long long x = powmod(a, u, n);
        if (x == 1 || x == n - 1)
            continue;
        for (int j = 1; j < t; ++j)
        {
            x = mod_mul(x, x, n);
            if (x == 1)
                return 0;
            if (x == n - 1)
                break;
        }
        if (x == n - 1)
            continue;
        return 0;
    }
    return 1;
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
        foundone, /* most recent prime found */
        max_prime = 0,
        total_prime = 0;

    long long int 
        n, /* number to start count on */ 
        limit; /* upper size of count */


    int 
        rank, /* CPU rank to identify which CPU is now used */ 
        size; /* task size for each cpu */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0) /* master machine in the cluster */
    {
        sscanf(argv[1], "%llu", &limit);
        printf("Starting. Numbers to be scanned= %lld\n", limit);

        pc = 4; /* Assume (2,3,5,7) are counted <F4>here */

        for (n = 11; n <= limit; n = n + 2)
        {
            if(n & 1 == 0) /* even number will not be prime */
            {
                continue; 
            }
            if (isprime(n, llsprp, LLSPRP_MAX))
            {
                pc++;
                foundone = n;
            }
        }
        MPI_Reduce(&pc, &total_prime, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&foundone, &total_prime, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        printf("Done. Largest prime is %d Total primes %d\n", foundone, pc);

    }
    /* 
     * slave machine example given for the MPI distributed computation
     master search for 11 16
     slave 1 search for 12 17
     slave 2 search for 13 18
     slave 3 search for 14 19
     slave 4 search for 15 20
     */
    else if(rank > 0)
    {
        for (n = 11 + rank; n <= limit; n+=rank)
        {
            if(n & 1 == 0) /* even number will not be prime */
            {
                continue; 
            }
            if (isprime(n, llsprp, LLSPRP_MAX))
            {
                pc++;
                foundone = n;
            }
        }
        MPI_Reduce(&pc, &total_prime, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&foundone, &total_prime, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    /* update the result */
    MPI_Finalize(); /* terminate all distributive computation */
    return 0;
}
