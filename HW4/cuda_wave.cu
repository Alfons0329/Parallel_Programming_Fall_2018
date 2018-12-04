/**********************************************************************
 * DESCRIPTION:
 *   Serial Concurrent Wave Equation - C Version
 *   This program implements the concurrent wave equation
 *   TA machine is GTX1060, so the TILE_WIDTH should be set at 
 *   corresponding to the spec (see L6 p155 or NVIDIA site) 
 *********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAXPOINTS 1000000
#define MAXSTEPS 1000000
#define MINPOINTS 20
#define PI 3.14159265

int TILE_WIDTH; /*testing purpose, change it to define after finished testing!!*/
const float FAC = 6.2831925f;/*use for share memory in GPU*/

void check_param(void);
void update (void);
void printfinal (void);
float cuda_sin(float);

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
    while ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS)) 
    {
        printf("Enter number of points along vibrating string [%d-%d]: "
                ,MINPOINTS, MAXPOINTS);
        scanf("%s", tchar);
        tpoints = atoi(tchar);
        if ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS))
            printf("Invalid. Please enter value between %d and %d\n", 
                    MINPOINTS, MAXPOINTS);
    }

    while ((nsteps < 1) || (nsteps > MAXSTEPS)) 
    {
        printf("Enter number of time steps [1-%d]: ", MAXSTEPS);
        scanf("%s", tchar);
        nsteps = atoi(tchar);
        if ((nsteps < 1) || (nsteps > MAXSTEPS))
            printf("Invalid. Please enter value between 1 and %d\n", MAXSTEPS);
    }

    printf("Using points = %d, steps = %d\n", tpoints, nsteps);

}

/**********************************************************************
 * CUDA sine function
 *********************************************************************/
__host__ __device__ float cuda_sin(float input)
{
    return sinf(input);
}
/**********************************************************************
 *  Update all values along line a specified number of times
 *  Use each thread for the parallelization of for loop (parallelize the for i (1 to t) tpoints thread)   
 *********************************************************************/
/* Kernel function is this one */
__global__ void update(float *cuda_value, int tpoints, int nsteps)
{
    int thread_ID = threadIdx.x + blockIdx.x * blockDim.x + 1; /* the thread index of CUDA core for loop from 1 */ 

    int i;
    float x;
    float tmp = tpoints - 1;
    /* float fac, tmp; */
    /* __constant__ float fac = 6.2831925f; */
    
    float old_tmp, new_tmp, val_tmp;
    /* Update values for each time step */
    if(thread_ID > tpoints)
    {
        return;
    }
    else
    {
        /* init line */

        /* Calculate initial values based on sine curve */
        /* fac = 2.0 * PI; */
        /* k = 0.0; */ 
        /* tmp = tpoints - 1; */
        x = (thread_ID - 1)/tmp;
        val_tmp = sin(FAC * x); 
        /* k = k + 1.0; */

        /* Initialize old values array */
        old_tmp = val_tmp;
        for (i = nsteps - 1; i >= 0; i--) 
        {
            /* Update points along line for this time step */
            /* global endpoints */
            if ((thread_ID == 1) || (thread_ID  == tpoints))
            {
                new_tmp = 0.0f;
            }
            else
            {
                new_tmp = (2.0f * val_tmp) - old_tmp + ( (-0.18f) * val_tmp);
                //new_tmp = (1.82f * val_tmp) - old_tmp; lots of errors but dunno why
            }
            /* Update old values with new values */
            old_tmp = val_tmp;
            val_tmp = new_tmp;
        }
        /* write back the result */
        cuda_value[thread_ID] = val_tmp;
    }
}

/**********************************************************************
 *     Print final results
 *********************************************************************/
void printfinal()
{
    int i;

    for (i = 1; i <= tpoints; i++) 
    {
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
    //sscanf(argv[3],"%d",&TILE_WIDTH);
    TILE_WIDTH = 1024;
    check_param();

    printf("Initializing points on the line...\n");    
    printf("Updating all points for all time steps...\n");

    float *cuda_value;
    int size = (tpoints + 1) * sizeof(float); /* padding */

    //cudaMalloc((void**)&cuda_value, size); /* allocate the memory space for cuda computation */
    size_t pitch = 0;
    cudaMallocPitch((void**)&cuda_value, &pitch, size, 1) ; /* alignment the memory allocation use 1 row for 1d array */
    /* dim3 dimGrid(tpoints / TILE_WIDTH, tpoints / TILE_WIDTH); */
    /* dim3 dimBlock(TILE_WIDTH, TILE_WIDTH); */
    update<<<(tpoints + TILE_WIDTH)/ TILE_WIDTH, TILE_WIDTH>>>(cuda_value, pitch / sizeof(float), nsteps);
    cudaMemcpy(values, cuda_value, size, cudaMemcpyDeviceToHost); /* copy the memory from GPU memory to main memory */

    printf("Printing final results...\n");
    printfinal();
    printf("\nDone.\n\n");
    cudaFree(cuda_value);
    return 0;
}
