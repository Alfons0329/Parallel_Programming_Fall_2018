# Parallel Programming HW4: CUDA

## Build, run, and validate for correctness
* Build the serial(CPU) version of wave euqation
    ```sh
    make normal
    ```
* Build the CUDA(GPU) version of wave euqation
    ```sh
    make cuda
    ``` 
* Build the diff checker (check_diff.cpp) for validate the error rate (abs (serial_point - cuda_point) > 1e-2) is count as error.
    ```sh
    make diff
    ```
* Or use the run.sh to do all the tings
    ```sh
    ./run.sh $tpoints $nsteps 1
    ```

