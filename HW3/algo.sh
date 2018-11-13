#!/bin/bash
set -e

make prime_normal
make algo_test

read -p "Enter the prime numbers to be searched" nums

echo "Normal"
time ./prime_normal $nums

echo "Primality checking algo improved"
time ./prime_algotest $nums

