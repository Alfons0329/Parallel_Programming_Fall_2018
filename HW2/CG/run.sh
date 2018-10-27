#!/bin/bash

#use this to run all test and comparisons

#single thread execution
cd ../original_no_omp/CG/
make clean
make
cd bin
./cg > ../../../CG/result/single_thread_res.txt 

#multi thread with OpenMP execution
cd ../../../CG/
make clean
make
cd bin
./cg > ../result/multi_thread_res.txt

#compare and show
cd ../result
echo "Single thread without OpenMP optimization"
echo $(cat "single_thread_res.txt" | grep VERIFICATION)
echo $(cat "single_thread_res.txt" | grep Initialization)
echo $(cat "single_thread_res.txt" | grep Execution)
echo ""
echo "Multi thread with OpenMP optimization"
echo $(cat "multi_thread_res.txt" | grep VERIFICATION)
echo $(cat "multi_thread_res.txt" | grep Initialization)
echo $(cat "multi_thread_res.txt" | grep Execution)
cd ..

