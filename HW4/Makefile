TARGET=serial_wave
SOURCE=serial_wave.c

TARGET2=cuda_wave 
SOURCE2=cuda_wave.cu

CC=gcc
NVCC=nvcc
LIBS=-lm

normal: serial_wave.c
	$(CC) $(SOURCE) -o $(TARGET) $(LIBS)

cuda: cuda_wave.c
	$(NVCC) $(SOURCE2) -o $(TARGET2) $(LIBS)

.PHONY: clean

clean:
	rm -f $(TARGET) $(TARGET2)
