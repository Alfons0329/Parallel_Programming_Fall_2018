#!/bin/bash
cpu_cnt=$1
n_tosses=$2
run_times=$3
echo "cpu_cnt $cpu_cnt , n_tosses $n_tosses , run_times $run_times"
make
echo "---------------c  -----------------------"
for i in $(seq 1 $run_times);
do
    start=`date "+%s%N"`
    ./pi_out_single.o $cpu_cnt $n_tosses
    end=`date "+%s%N"`
    t1=$(($end-$start))
    echo "1 thread, elapsed time $t1 ns"

    start=`date "+%s%N"`
    ./pi_out.o $cpu_cnt $n_tosses
    end=`date "+%s%N"`
    t2=$(($end-$start))
    t3=$((t1/t2))
    echo "$cpu_cnt threads, elapsed time $t2 ns, accelerate rate = $t3"
done
# echo "---------------cpp-----------------------"
# for i in $(seq 1 $run_times);
# do
#     ./pi_out_cpp.o $cpu_cnt $n_tosses
# done
echo "-----------run $3 times test end---------"

