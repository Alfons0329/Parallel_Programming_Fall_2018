TARGET=gb_pthread.o
SOURCE=gaussian_blur_pthread.cpp

TARGET2=gb_std.o
SOURCE2=gaussian_blur_standard.cpp

TARGET3=cm.o
SOURCE3=create_matrix.cpp

LIBS=-pthread -lm
CPP=g++
CPP_FLAGS=-std=c++11 -O2 -Wno-unused-result

all: gaussian_blur_standard.cpp
	# $(CPP) $(SOURCE) -o $(TARGET) $(CPP_FLAGS) $(LIBS)
	$(CPP) $(SOURCE2) -o $(TARGET2) $(CPP_FLAGS) $(LIBS)

matrix: create_matrix.cpp
	$(CPP) $(SOURCE3) -o $(TARGET3) $(CPP_FLAGS)

.PHONY: clean
	rm -f *.o
