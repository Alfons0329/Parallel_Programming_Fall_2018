#!/bin/bash
set -e

if [ $# -eq 0 ]
then
    printf "Usage: . / compare.sh <BMP image file>\n"
    exit
else
    read -p "CUDA vs OpenCL run times: " cnt
    make cuda
    make opencl
    make matrix
    ./cm.o
    ./check.o
    for i in $(seq 1 $cnt)
    do
        printf "CUDA\n"
        time ./gb_cuda.o $1
        printf "OpenCL\n"
        time ./gb_opencl.o $1
    done
fi

