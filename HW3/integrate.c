#include <stdio.h>
#include <math.h>
#include <mpi.h>
#define PI 3.1415926535
double sen, rcv;
int main(int argc, char **argv) 
{
    long long i, num_intervals;
    double area, rect_width, sum, x_middle; 

    sscanf(argv[1],"%llu",&num_intervals); /* all the machines, including master and slave, gets the limit */
    rect_width = PI / num_intervals;
    /*-------------------------------MPI data declaration starts--------------------------------*/
    int rank, /* CPU rank to identify which CPU is now used */ 
        size; /* task size for each cpu */

    MPI_Datatype datatype, oldtype[1]; /* only one datatype, so oldtype length be 1 */
    int blockcount[1];
    MPI_Aint offsets[1];
    MPI_Status status;
    /*-------------------------------MPI data declaration ends--------------------------------*/

    /*-------------------------------MPI init starts--------------------------------*/
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    blockcount[0] = 2; /* double gots one: sum */
    offsets[0] = 0; /* No offsets is needed */
    oldtype[0] = MPI_LONG_LONG_INT;

    /* define structured type and commit it */
    MPI_Type_create_struct(1, blockcount, offsets, oldtype, &datatype);
    MPI_Type_commit(&datatype);

    /*-------------------------------MPI init ends--------------------------------*/


    sum = 0;
    for(i = 1 + rank; i < num_intervals + 1; i += size) 
    {
        /* find the middle of the interval on the X-axis. */
        x_middle = (i - 0.5) * rect_width;
        area = sin(x_middle) * rect_width; 
        sum = sum + area;
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
            sen += rcv;
        }
    }

    if (rank == 0)
    {
        printf("The total area is: %f\n", (float)sum);
    }

    return 0;

}   
