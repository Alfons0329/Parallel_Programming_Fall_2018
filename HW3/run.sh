#!/bin/bash
#To make mpiexec work properly, you need to be able to execute jobs on remote nodes without typing a password.
#You will need to generate an ssh key by yourself.
mkdir -p ∼/. ssh
ssh - keygen -t rsa
cat ∼/. ssh / id_rsa . pub >> ∼/. ssh / authorized_keys

echo "Build the normal version prime checker"
make prime_normal
echo "Build the MPI(distributed computing) version prime checker"
make prime


#for the cluster system in the PP-f18 course
