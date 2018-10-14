TARGET=pi_out.o
CPP=gcc
CPP_FLAGS=-std=c11
SOURCE=pi.c

LIBS=-lm -pthread

TARGET2=pi_out_cpp.o
CPP2=g++
# CPP_FLAGS2=-std=c++11
SOURCE2=pi.cpp

all: pi.c pi.cpp
	$(CPP) $(SOURCE) -o $(TARGET) $(CPP_FLAGS) $(LIBS)
	$(CPP2) $(SOURCE2) -o $(TARGET2) $(LIBS)

.PHONY: clean
	rm  *.o
