TARGET=pi_out.o
CC=gcc
C_FLAGS=-std=c11 -O2
SOURCE=pi.c

LIBS=-lm -pthread

TARGET2=pi_out_single.o
CC=gcc
C_FLAGS=-std=c11 -O2
SOURCE2=pi_single.c

# TARGET2=pi_out_cpp.o
# CPP2=g++
# CPP_FLAGS2=-std=c++11 -O2
# SOURCE2=pi.cpp

all: pi.c pi.cpp
	$(CC) $(SOURCE) -o $(TARGET) $(C_FLAGS) $(LIBS)
	$(CC) $(SOURCE2) -o $(TARGET2) $(C_FLAGS) $(LIBS)
	# $(CPP2) $(SOURCE2) -o $(TARGET2) $(CPP_FLAGS2)  $(LIBS)

.PHONY: clean
	rm  *.o
