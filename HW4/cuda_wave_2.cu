/**********************************************************************
 * DESCRIPTION:
 *   Serial Concurrent Wave Equation - C Version
 *   This program implements the concurrent wave equation
 *********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAXPOINTS 1000000
#define MAXSTEPS 1000000
#define MINPOINTS 20
#define PI 3.14159265
#define BLOCKSIZE 128
void check_param(void);
void init_line(void);
void update (void);
void printfinal (void);

int nsteps,                 	/* number of time steps */
    tpoints, 	     		/* total points along string */
    rcode;                  	/* generic return code */
float  values[MAXPOINTS+2], 	/* values at time t */
       oldval[MAXPOINTS+2], 	/* values at time (t-dt) */
       newval[MAXPOINTS+2]; 	/* values at time (t+dt) */


/**********************************************************************
 *	Checks input values from parameters
 *********************************************************************/
void check_param(void)
{
    char tchar[20];

    /* check number of points, number of iterations */
    while ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS)) {
        printf("Enter number of points along vibrating string [%d-%d]: "
                ,MINPOINTS, MAXPOINTS);
        scanf("%s", tchar);
        tpoints = atoi(tchar);
        if ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS))
            printf("Invalid. Please enter value between %d and %d\n", 
                    MINPOINTS, MAXPOINTS);
    }
    while ((nsteps < 1) || (nsteps > MAXSTEPS)) {
        printf("Enter number of time steps [1-%d]: ", MAXSTEPS);
        scanf("%s", tchar);
        nsteps = atoi(tchar);
        if ((nsteps < 1) || (nsteps > MAXSTEPS))
            printf("Invalid. Please enter value between 1 and %d\n", MAXSTEPS);
    }

    printf("Using points = %d, steps = %d\n", tpoints, nsteps);

}




/**********************************************************************
 *     Update all values along line a specified number of times
 *********************************************************************/


__global__ void update (float *values_d, int tpoints,  int nsteps)
{
    int i;
    int k =  1+threadIdx.x + blockIdx.x * blockDim.x;
    if(k <= tpoints){
        float values_t;
        float newval_t;
        float oldval_t;

        float x, fac , tmp ; 
        fac = 2.0 * PI;
        tmp = tpoints - 1;
        x = (float)(k-1)/tmp ;
        values_t = sin(fac * x);
        oldval_t = values_t;

        for (i = 1; i <= nsteps ; i++) {
            if ((k == 1) || (k == tpoints ))
                newval_t = 0.0f;
            else
                newval_t = (2.0f * values_t) - oldval_t + ( 0.09f * ( -2.0f * values_t ));
            oldval_t = values_t;
            values_t = newval_t;
        }

        values_d[k] = values_t;
    }
}
/**********************************************************************
 *     Print final results
 *********************************************************************/
void printfinal()
{
    int i;

    for (i = 1; i <= tpoints; i++) {
        printf("%6.4f ", values[i]);
        if (i%10 == 0)
            printf("\n");
    }
}

/**********************************************************************
 *	Main program
 *********************************************************************/
int main(int argc, char *argv[])
{
    sscanf(argv[1],"%d",&tpoints);
    sscanf(argv[2],"%d",&nsteps);
    check_param();
    printf("Initializing points on the line...\n");
    printf("Updating all points for all time steps...\n");
    float *values_d;
    int size = (1+tpoints)*sizeof(float);
    cudaMalloc((void**)&values_d,size);
    update<<<(tpoints+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(values_d, tpoints, nsteps);
    cudaMemcpy(values, values_d, size, cudaMemcpyDeviceToHost);
    printf("Printing final results...\n");
    printfinal();
    printf("\nDone.\n\n");
    cudaFree(values_d);

    return 0;
}
