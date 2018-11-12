#!/bin/bash
#To make mpiexec work properly, you need to be able to execute jobs on remote nodes without typing a password.
#You will need to generate an ssh key by yourself.
#De comment this line for SSH configuration
#mkdir -p ∼/. ssh
#ssh-keygen -t rsa
#cat ∼/. ssh / id_rsa . pub >> ∼/. ssh / authorized_keys

#Build the source code
echo "Build the normal version prime checker"
make prime_normal
echo "Build the MPI(distributed computing) version prime checker"
make prime

#Executing the program
echo "Normal version, non MPI version"
read -p "Enter the numbers to be searched: " nums
time ./prime_normal $nums

#For the cluster system in the PP-f18 course
echo "MPI version"
read -p "Enter the numbers to be searched: " nums
time /usr/lib64/openmpi/bin/mpiexec -n 3 -hostfile hostfile --map-by node ./prime $nums
#time /usr/lib64/openmpi/bin/mpiexec -n 3 -host pp1.cs.nctu.edu.tw, pp2.cs.nctu.edu.tw, pp3.cs.nctu.edu.tw ./prime $nums
