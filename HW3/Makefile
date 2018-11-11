TARGET=integrate
SOURCE=integrate.c


TARGET2=prime
SOURCE2=prime.c

MPICC=/usr/lib64/openmpi/bin/mpicc
LIBS=-lm

integrate: integrate.c
		$(MPICC) -o $(TARGET) $(SOURCE) $(LIBS) 

prime: prime.c
		$(MPICC) -o $(TARGET2) $(SOURCE2) $(LIBS) 
