# Parallel Programming HW2: Accelerating Conjugate Gradient Computation  using OpenMP

## Tips for optimzation with example

* Using omp reduction for parallization of iteration, see [this topic](https://blog.csdn.net/gengshenghong/article/details/7000685)
```c
#pragma omp for reduction(+:norm_temp1, norm_temp2)
    for (j = 0; j < lastcol - firstcol + 1; j++) {
      norm_temp1 = norm_temp1 + x[j]*z[j];
      norm_temp2 = norm_temp2 + z[j]*z[j];
    }
```