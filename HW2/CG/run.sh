#!/bin/sh
make
cd bin
./cg > multi_thread_res.txt
cd ..