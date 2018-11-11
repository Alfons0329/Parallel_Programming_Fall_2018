TARGET=integrate
SOURCE=integrate.c

TARGET2=prime
SOURCE2=prime.c

TARGET3=prime_normal
SOURCE3=prime_normal.c

CC=gcc
MPICC=/usr/lib64/openmpi/bin/mpicc
LIBS=-lm

integrate: integrate.c
		$(MPICC) -o $(TARGET) $(SOURCE) $(LIBS) 

prime: prime.c
		$(MPICC) -o $(TARGET2) $(SOURCE2) $(LIBS)

prime_normal: prime_normal.c
		$(CC) -o $(TARGET3) $(SOURCE3) $(LIBS) 

.PHONY: clean
	rm -f $(TARGET) $(TARGET2) $(TARGET3)
