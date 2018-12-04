TARGET=serial_wave
SOURCE=serial_wave.c

TARGET2=cuda_wave 
SOURCE2=cuda_wave.cu

TARGET3=check_diff
DIFF=check_diff.cpp

CC=gcc
CPP=g++
NVCC=nvcc
LIBS=-lm

normal: serial_wave.c
	$(CC) $(SOURCE) -o $(TARGET) $(LIBS)

cuda: cuda_wave.cu
	$(NVCC) $(SOURCE2) -o $(TARGET2) $(LIBS)

diff: check_diff.cpp
	$(CPP) $(DIFF) -o $(TARGET3) -std=c++11 
.PHONY: clean

clean:
	rm -f $(TARGET) $(TARGET2) $(TARGET3)
