#!/bin/bash
set -e

tile_width_test()
{
    make cuda
    for i in 64 128 256 512 1024 ;
    do
        echo "TILE_WIDTH " $i
        time ./cuda_wave $1 $2 $i > cuda_result.txt
        ./check_diff
    done
}

make normal
time ./serial_wave $1 $2 > serial_result.txt

#2048 wrong answer
make diff
tile_width_test $1 $2

echo "--------------Steven------------------"
nvcc cuda_wave_2.cu -o cuda_wave_2
time ./cuda_wave_2 $1 $2 > cuda_result.txt
./check_diff
