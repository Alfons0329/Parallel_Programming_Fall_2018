#include <stdio.h>
#include <math.h>

#define PI 3.1415926535

int main(int argc, char **argv) 
{
    long long i, num_intervals;
    double rect_width, area, sum, x_middle; 

    sscanf(argv[1],"%llu",&num_intervals);

    rect_width = PI / num_intervals;

    sum = 0;
    for(i = 1; i < num_intervals + 1; i++) {

        /* find the middle of the interval on the X-axis. */ 

        x_middle = (i - 0.5) * rect_width;
        area = sin(x_middle) * rect_width; 
        sum = sum + area;

    } 

    printf("The total area is: %f\n", (float)sum);
    return 0;

}   
