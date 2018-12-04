#!/bin/bash
set -e

tile_width_test()
{
    make cuda
    make diff

    echo "--------------Steven------------------"
    nvcc cuda_wave_2.cu -o cuda_wave_2
    time ./cuda_wave_2 $1 $2 > cuda_result_2.txt

    echo "--------------Mine------------------"
    #2048 wrong answer
    for i in 64 128 256 512 1024;
    do
        echo "TILE_WIDTH " $i
        time ./cuda_wave $1 $2 $i > cuda_result.txt
        ./check_diff cuda_result.txt cuda_result_2.txt
    done
}

correctness_test()
{
    make normal
    make cuda
    make diff
    time ./serial_wave $1 $2 > serial_result.txt

    time ./cuda_wave $1 $2 > cuda_result.txt
    ./check_diff serial_result.txt cuda_result.txt
}

to_do=$3

if [ $to_do -eq 1 ];
then
    correctness_test $1 $2
else
    tile_width_test $1 $2
fi

