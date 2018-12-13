#!/bin/bash
set -e

read -p "1: Standard, 2: pthread, 3: OpenMP, 4: CUDA, 5:Generate Gaussian Matrix 7: Clean output files " choose

case $choose in
    1)
        make standard
        ;;
    2)
        make pthread
        ;;
    3)
        make omp
        ;;
    4)
        make cuda
        ;;
    5)
        make matrix
        ./cm.o
        ;;
    7)
        make clean
        ;;
esac
python show.py $

