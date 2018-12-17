#!/bin/bash
set -e

if [ $# -eq 0 ]
then
    echo "Please provide filename for Gaussian Blur. usage . / run.sh <BMP image file1>"
else

    read -p "1: Standard, 2: pthread, 3: OpenMP, 4: CUDA, 5:Generate Gaussian Matrix 7: Clean output files " choose

    case $choose in
        1)
            make standard
            time ./gb_std.o $1
            ;;
        2)
            make pthread
            ;;
        3)
            make omp
            time ./gb_omp.o $1
            ;;
        4)
            make cuda
            time ./gb_cuda.o $1
            ;;
        5)
            make matrix
            ./cm.o
            ;;
        7)
            make clean
            ;;
    esac

fi

