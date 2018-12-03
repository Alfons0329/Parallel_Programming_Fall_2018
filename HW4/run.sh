#!/bin/bash
set -e
make clean

make normal
./serial_wave $1 $2 > serial_result.txt

make cuda
./cuda_wave $1 $2 > cuda_result.txt

#nvcc cuda_wave_2.cu -o cuda_wave_2
#time ./cuda_wave_2 $1 $2 > cuda_result.txt
