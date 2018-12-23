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
            time taskset -c 1 ./gb_std_unpadded.o $1
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
            read -p "Test thread in 4 16 64 256 1024 vs time? 1 no 2 yes " yn
            if [ $yn -eq 1 ];
            then
                time ./gb_cuda.o $1
            else
                for i in 4 16 64 256 1024;
                do
                    time ./gb_cuda.o $1 $i
                done
            fi

            read -p "Test shared memory CUDA Gaussian Blur? 1 no 2 yes " yn
            if [ $yn -eq 2 ];
            then
                make cuda_shm
                echo "Non shared memory Gaussian Blur: "
                time ./gb_cuda.o $1
                echo "Shared memory Gaussian Blur: "
                time ./gb_cuda_shm.o $1
            fi
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

