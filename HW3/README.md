# Parallel Programming HW3: Distributive computing with MPI

## Build and run
* If you need to configure your cluster
    ```sh
    mkdir -p ~/.ssh #make the ssh folder to storage keys in your master machine
    ssh-keygen -t rsa #generate the ssh key, the default settings are OK
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys #store the authorized keys.
    ```
    For more information of configurting the cluster machines, please visit the PDF of this homework [here](HW3.pdf)

* Then run the program on the cluster, the shell IO will read the input for control process
    ```sh
    ./run.sh
    ```
