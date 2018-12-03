#!/bin/bash
set -e

make normal
time ./serial_wave $1 $2 > serial_result.txt

make cuda
time ./cuda_wave $1 $2 > cuda_result.txt

cat serial_result.txt | grep real
cat cuda_result.txt | grep real
