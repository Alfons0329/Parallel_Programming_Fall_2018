#!/bin/bash
function Test
{
 if [ $# -eq 0 ]
 then
     echo "At least 1 argument"
 else
     mkfifo input
     cat $1 > input&
     time ./histogram
     diff $2.out "$1.out" > /dev/null
     if [ $? -eq 0 ]
     then
         echo $1 is correct
     fi
     rm input $2.out
 fi
}

if [ $# -eq 0 ]
then
    echo "input Student ID"
else
    zip "HW5_$1.zip" histogram.cpp
    unzip -o "HW5_$1.zip"
    g++ -pg -o histogram histogram.cpp -lOpenCL
    Test "/home/net/input-307M" $1
    gprof histogram gmon.out
fi

