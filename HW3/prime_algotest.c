#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define ll long long

/*int prime[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};

ll mulmod(ll a, ll b, ll c)
{ 
    //return (a*b)%c   O(log(N))
    ll x = 0, y = a % c;
    while (b > 0)
    {
        if (b % 2 == 1)
            x = (x + y) % c;
        y = (y * 2) % c;
        b /= 2;
    }
    return x % c;
}
ll mulmodfast(ll a, ll b, ll mod)
{ 
    //return (a*b)%mod  O(1)
    a %= mod;
    b %= mod;
    long double res = a;
    res *= b;
    ll c = (ll)(res / mod);
    a *= b;
    a -= (c * mod);
    a %= mod;
    if (a < 0)
        a += mod;
    return a;
}

ll modulo(ll a, ll b, ll c)
{
    ll x = 1, y = a; // long long is taken to avoid overflow of intermediate results
    while (b > 0)
    {
        if (b % 2 == 1)
            x = mulmod(x, y, c); //we can use mulmod or multiply fxn here
        y = mulmod(y, y, c);     // squaring the base
        b /= 2;
    }
    return x % c;
}

int isprime(ll p, int iteration)
{
    if (p < 2)
        return 0;
    if (p != 2 && p % 2 == 0)
        return 0;
    for (int i = 0; i < 25; i++)
    {
        if (p == prime[i])
            return 1;
        else if (p % prime[i] == 0)
            return 0;
    }
    long long s = p - 1;
    while (s % 2 == 0)
        s /= 2;
    for (int i = 0; i < iteration; i++)
    {
        long long a = rand() % (p - 1) + 1, temp = s;
        long long mod = modulo(a, temp, p);
        while (temp != p - 1 && mod != 1 && mod != p - 1)
        {
            mod = mulmodfast(mod, mod, p);
            temp *= 2;
        }
        if (mod != p - 1 && temp % 2 == 0)
            return 0;
    }
    return 1;
}*/
int isprime(long long n)
{
    long long sq = sqrt(n);
    if (n % 6 == 1 || n % 6 == 5)
    {
        for (long long i = 5; i < sqrt(n); i = i + 2)
        {
            if (n % i == 0)
            {
                return 0;
            }
        }
        return 1;
    }
    return 0;
}
int main(int argc, char *argv[])
{
    int pc,       /* prime counter */
        foundone; /* most recent prime found */
    long long int n, limit;

    sscanf(argv[1],"%llu",&limit);
    printf("Starting. Numbers to be scanned= %lld\n",limit);    

    pc=4;     /* Assume (2,3,5,7) are counted here */

    for (n = 11; n <= limit; n = n + 2) 
    {
        if(n & 1 == 0)
        {
            continue;
        }
        else if (isprime(n)) 
        {
            pc++;
            foundone = n;
        }
    }
    printf("Done. Largest prime is %d Total primes %d\n",foundone, pc);

    return 0;

} 
