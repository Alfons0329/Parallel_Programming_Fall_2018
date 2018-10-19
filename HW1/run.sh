#!/bin/bash
cpu_cnt=$1
n_tosses=$2
run_times=$3
echo "cpu_cnt $cpu_cnt , n_tosses $n_tosses , run_times $run_times"
make
echo "---------------c  -----------------------"
for i in $(seq 1 $run_times);
do
    ./pi_out.o $1 $2
done
echo "---------------cpp-----------------------"
for i in $(seq 1 $run_times);
do
    ./pi_out_cpp.o $1 $2
done
echo "-----------run $3 times test end---------"

