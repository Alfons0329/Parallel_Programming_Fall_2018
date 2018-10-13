#!/bin/bash
make
./pi_out.o 1 12345678
echo "---------------cpp-----------------------"
./pi_out_cpp.o 1 12345678

