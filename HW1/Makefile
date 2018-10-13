TARGET=pi_out.o
CPP=g++
CPP_FLAGS=-std=c++11
SOURCE=pi.cpp
LIBS=-lm -pthread

all: pi.cpp 
	$(CPP) $(SOURCE) -o $(TARGET) $(CPP_FLAGS) $(LIBS)

.PHONY: clean
	rm  *.o
