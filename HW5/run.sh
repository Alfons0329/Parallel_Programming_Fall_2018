#!/bin/bash
if [ $# -eq 0 ]
then
    echo "input Student ID"
else
    zip "HW5_$1.zip" histogram.cpp
    /home/net/build.sh $1
    /home/net/test.sh $1
fi

