TARGET=prime
SOURCE=prime.c

TARGET2=prime_normal
SOURCE2=prime_normal.c

TARGET3=integrate
SOURCE3=integrate.c

TARGET4=integrate_normal
SOURCE4=integrate_normal.c

CC=gcc
MPICC=/usr/lib64/openmpi/bin/mpicc
LIBS=-lm

prime_mpi: prime.c
	$(MPICC) -o $(TARGET) $(SOURCE) $(LIBS)

prime_normal: prime_normal.c
	$(CC) -o $(TARGET2) $(SOURCE2) $(LIBS) 

integrate_mpi: integrate.c
	$(MPICC) -o $(TARGET3) $(SOURCE3) $(LIBS) 

integrate_normal: integrate_normal.c
	$(CC) -o $(TARGET4) $(SOURCE4) $(LIBS) 

.PHONY: clean

clean:
	rm -f $(TARGET) $(TARGET2) $(TARGET3) $(TARGET4)
