#!/bin/bash
#To make mpiexec work properly, you need to be able to execute jobs on remote nodes without typing a password.
#You will need to generate an ssh key by yourself.
#De comment this line for SSH configuration
#mkdir -p ∼/.ssh
#ssh-keygen -t rsa
#cat ∼/.ssh/id_rsa.pub >> ∼/.ssh/authorized_keys

set -e
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
echo "1 clusters"
time /usr/lib64/openmpi/bin/mpiexec -n 1 -hostfile hostfile --map-by node ./prime $nums

echo "2 clusters"
time /usr/lib64/openmpi/bin/mpiexec -n 2 -hostfile hostfile --map-by node ./prime $nums

echo "3 clusters"
time /usr/lib64/openmpi/bin/mpiexec -n 3 -hostfile hostfile --map-by node ./prime $nums

