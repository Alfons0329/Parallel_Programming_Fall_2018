# Parallel Programming HW1: Calculating Pi using Monte Carlo Method

## Build and run

* Accelerte with pthread library, make sure pthread (POSIX thread is runable on your computer, mine is Ubuntu16.04)
* Check if pthread is installed in your UNIX-like system
    ```sh
    $ locate libpthread.so      
    # should return the message like the following
    /lib/i386-linux-gnu/libpthread.so.0
    /lib/x86_64-linux-gnu/libpthread.so.0
    /lib32/libpthread.so.0
    /usr/lib/x86_64-linux-gnu/libpthread.so
    ```
* Run the program(makefile included), see the speed up percentage.
```sh
./run.sh $threads $tossed_times $testing_times
```

## Valid the result correctness and some writeup

* PI should be about 3.14XXXXX (accurate to 2 digits of floating number)

* While PI are right, speed seems not have been accelerated with pthread

  * Single thread = 1161437863ns
  * Rand with normal `rand()`  = 1153890195ns

* See [this topic](https://www.nctusam.com/2017/10/22/rand-%E8%88%87-rand_r%E7%9A%84%E5%B7%AE%E7%95%B0/) for more details
* Rand with thread-safe `rand_r(&seed)` = 263168401ns, did speed up
