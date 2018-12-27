#!/bin/bash
set -e

if [ $# -eq 0 ]
then
    printf "Usage: . / compare.sh <BMP image file>\n"
    exit
else
    read -p "CUDA vs OpenCL run times: " cnt
    read -p "CUDA const memory? 1 yes 2 no " yn
    if [ $yn -eq 1 ]
    then
        make cuda_shm
    else
        make cuda
    fi
    make opencl
    make matrix
    ./cm.o
    ./check.o
    for i in $(seq 1 $cnt)
    do
        printf "CUDA\n"
        if [ $yn -eq 1 ]
        then
            time ./gb_cuda_shm.o $1
        else
            time ./gb_cuda.o $1
        fi
        printf "OpenCL\n"
        time ./gb_opencl.o $1
    done
fi

