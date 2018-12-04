#!/bin/bash
set -e
make clean

#make normal
#time ./serial_wave $1 $2 > serial_result.txt

make cuda
for i in 64 128 256 512 1024 2048;
do
    echo "TILE_WIDTH" $i
    time ./cuda_wave $1 $2 $i > cuda_result.txt
done

#nvcc cuda_wave_2.cu -o cuda_wave_2
#time ./cuda_wave_2 $1 $2 > cuda_result.txt
