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
            time ./gb_pthread.o $1
            ;;

        3)
            make omp
            time ./gb_omp.o $1
            ;;

        4)
            make cuda cuda_shm cuda_stm -j8
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

            read -p "Test constant, shared memory CUDA Gaussian Blur? 1 no 2 yes " yn
            if [ $yn -eq 2 ];
            then
                read -p "Test thread in 4 16 64 256 1024 vs time? 1 no 2 yes " yn
                if [ $yn -eq 1 ];
                then
                    time ./gb_cuda_shm.o $1
                else
                    for i in 4 16 64 256 1024;
                    do
                        printf "\nGaussian Blur without constant, shared memory "
                        time ./gb_cuda.o $1 $i
                        printf "\nGaussian Blur with constant, shared memory "
                        time ./gb_cuda_shm.o $1 $i
                    done
                fi
            fi

            read -p "Test stream pipeline CUDA Gaussian Blur 1 no 2 yes " yn
            if [ $yn -eq 2 ];
            then
                read -p "Test thread in 4 16 64 256 1024 vs time? 1 no 2 yes " yn
                read -p "Thread configuration in 2D ? 0 no 1 yes" dim
                if [ $yn -eq 1 ];
                then
                    time ./gb_cuda_stm.o $1 1024 $dim
                else
                    for i in 4 16 64 256 1024;
                    do
                        printf "\nGaussian Blur without stream pipline "
                        time ./gb_cuda.o $1 $i $dim
                        printf "\nGaussian Blur with stream pipline "
                        time ./gb_cuda_stm.o $1 $i $dim
                    done
                fi
            fi
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
            f_name="$(basename $1 .bmp)"
            if [ -e $f_name\_blur\_unpadded.bmp ];
            then
                printf "\nFile for diff (Golden) already exists, directly diff! "
            else
                printf "\nFile for diff (Golden) does not exist, make first! "
                make do_diff -j8
                # time taskset -c 1 ./gb_std_unpadded.o $1
            fi
            
            printf "\nCUDA without constant, shared memory: "
            ./gb_cuda.o $1
            ./diff.o $f_name\_blur\_unpadded.bmp $f_name\_blur\_cuda.bmp
            printf "\nCUDA with constant, shared memory: "
            ./gb_cuda_shm.o $1 $f_name_blur_cuda_shm.bmp
            ./diff.o $f_name\_blur\_unpadded.bmp $f_name\_blur\_cuda.bmp
            ;;

        8)
            make all -j8
            printf "\nSerial: "
            time taskset -c 1 ./gb_std_unpadded.o $1
            printf "\nOpenMP: "
            time ./gb_omp.o $1
            printf "\nCUDA without constant, shared memory: "
            time ./gb_cuda.o $1
            printf "\nCUDA with constant, shared memory: "
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

