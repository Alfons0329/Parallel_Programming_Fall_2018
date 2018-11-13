#include <stdio.h>
#include <stdlib.h>
#include <math.h>

long long c[100];

void coef(int n)
{
    int i, j;

    if (n < 0 || n > 63) abort(); // gracefully deal with range issue

    for (c[i=0] = 1; i < n; c[0] = -c[0], i++)
        for (c[1 + (j=i)] = 1; j > 0; j--)
            c[j] = c[j-1] - c[j];

}

int is_prime(long long n)
{
    long long i;

    coef(n);
    c[0] += 1, c[i=n] -= 1;
    while (i-- && !(c[i] % n));

    return i < 0;
}

int main(int argc, char *argv[])
{
    int pc,       /* prime counter */
        foundone; /* most recent prime found */
    long long int n, limit;

    sscanf(argv[1],"%llu",&limit);
    printf("Starting. Numbers to be scanned= %lld\n",limit);    

    pc=4;     /* Assume (2,3,5,7) are counted here */
    for (n = 0; n < 100; ++n)
    {
        coef(n);
    }
    for (n = 11; n <= limit; n = n+2) 
    {
        if (isprime(n)) 
        {
            pc++;
            foundone = n;
        }
    }
    printf("Done. Largest prime is %d Total primes %d\n",foundone, pc);

    return 0;

} 
