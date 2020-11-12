#!/bin/bash
# set -e

if [ $# -eq 0 ]
then
    printf "Please provide filename for Gaussian Blur. usage . / run.sh <BMP image file>\n"
else

    read -p "1: Standard, 2: pthread, 3: OpenMP, 4: CUDA, 5: Generate Gaussian Matrix, 6: OpenCL, 7: Diff CUDA output vs serial (standard) output, 8: Run all platform at once, 9: Clean output files " choose

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

            read -p "Test constant memory CUDA Gaussian Blur? 1 no 2 yes " yn
            if [ $yn -eq 2 ];
            then
                read -p "Test thread in 4 16 64 256 1024 vs time? 1 no 2 yes " yn
                if [ $yn -eq 1 ];
                then
                    make cuda_shm
                    time ./gb_cuda_shm.o $1
                else
                    make cuda_shm
                    for i in 4 16 64 256 1024;
                    do
                        printf "\nGaussian Blur without constant memory: "
                        time ./gb_cuda.o $1 $i
                        printf "\nGaussian Blur with constant memory "
                        time ./gb_cuda_shm.o $1 $i
                    done
                fi
            fi

            read -p "Test stream pipeline CUDA Gaussian Blur 1 no 2 yes " yn
            if [ $yn -eq 2 ];
            then
                read -p "Test thread in 4 16 64 256 1024 vs time? 1 no 2 yes " yn
                if [ $yn -eq 1 ];
                then
                    make cuda_stream
                    time ./gb_cuda_stream.o $1
                else
                    make cuda_stream
                    for i in 4 16 64 256 1024;
                    do
                        printf "\nGaussian Blur without stream pipeline , only const memory: "
                        time ./gb_cuda_shm.o $1 $i
                        printf "\nGaussian Blur with stream pipeline and const memory "
                        time ./gb_cuda_stream.o $1 $i
                    done
                fi
            fi

            #./diff.o $1

            ;;
        5)
            make matrix
            ./cm.o
            ./check.o
            ;;
        6)
            make opencl
            time ./gb_opencl.o $1
            ;;

        7)
            make diff
            time taskset -c 1 ./gb_std_unpadded.o $1
            time ./gb_cuda.o $1
            ./diff.o $1
            ;;
        8)
            make all -j8
            printf "\nSerial: "
            time taskset -c 1 ./gb_std_unpadded.o $1
            printf "\nOpenMP: "
            time ./gb_omp.o $1
            printf "\nCUDA without constant memory: "
            time ./gb_cuda.o $1
            printf "\nCUDA with constant memory: "
            time ./gb_cuda_shm.o $1
            # printf "\nOpenCL: "
            # time ./gb_opencl.o $1
            ;;
        9)
            make clean
            ;;
        *)
            exit
            ;;
    esac

fi

