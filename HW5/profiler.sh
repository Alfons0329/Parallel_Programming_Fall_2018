#!/bin/bash
if [ $# -eq 0 ]
then
    echo "input Student ID"
else
    zip "HW5_$1.zip" histogram.cpp
    unzip -o "HW5_$1.zip"
    g++ -pg -o histogram histogram.cpp -lOpenCL
    /home/net/test.sh $1
    gprof histogram gmon.out
fi

