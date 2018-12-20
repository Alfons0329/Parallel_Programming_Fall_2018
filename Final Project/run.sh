#!/bin/bash
set -e

if [ $# -eq 0 ]
then
    echo "Please provide filename for Gaussian Blur. usage . / run.sh <BMP image file1>"
else

    read -p "1: Standard, 2: pthread, 3: OpenMP, 4: CUDA, 5: Generate Gaussian Matrix, 6: OpenCL, 7: Diff CUDA output vs serial (standard) output, 8: Clean output files " choose

    case $choose in
        1)
            make standard
            time taskset -c 1 ./gb_std.o $1
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
            ./check.o
            ;;
        6)
            ;;

        7)
            make diff
            time taskset -c 1 ./gb_std_unpadded.o $1
            time ./gb_cuda.o $1
            ./diff.o $1
            ;;
        8)
            make clean
            ;;
        *)
            exit
            ;;
    esac

fi

