# Parallel Programming HW4 - CUDA

## The statistics of error diffs
* 
    ```
    new_tmp = (2.0 * val_tmp) - old_tmp + ( (-0.09 * 2.0) * val_tmp);
 
    ```
    diff is 2034 in 10000 pts
*
    ```
    new_tmp = (2.0 * val_tmp) - old_tmp + ( (-0.18) * val_tmp);
    ```
    diff is 2034 in 10000 pts
*
    ```
    new_tmp = (2.0f * val_tmp) - old_tmp + ( (-0.18f) * val_tmp);
    ```
    diff is 206 in 10000 pts
