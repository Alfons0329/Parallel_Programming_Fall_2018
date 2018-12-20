#---------------------------Required files-----------------------#
#Standard Gaussian blur (serial version)
TARGET=gb_std.o
SOURCE=gaussian_blur_standard.cpp

#Pthread Gaussian blur
TARGET2=gb_pthread.o
SOURCE2=gaussian_blur_pthread.cpp

#OpenMP Gaussian blur
TARGET3=gb_omp.o
SOURCE3=gaussian_blur_omp.cpp

#CUDA Gaussian blur
TARGET4=gb_cuda.o
SOURCE4=gaussian_blur_cuda.cu

#CUDA Gaussian blur with OpenCV
TARGET5=gb_cuda_cv.o
SOURCE5=gaussian_blur_cuda_cv.cu

#Gaussian matrix generator and check matrix
TARGET_CM=cm.o
SOURCE_CM=create_matrix.cpp
TARGET_CHECK=check.o
SOURCE_CHECK=check_matrix.cpp

#diff the cuda, unpadded pic
TARGET_DIFF=diff.o
SOURCE_DIFF=diff_image.cpp
TARGET_UNPADDED=gb_std_unpadded.o
SOURCE_UNPADDED=gaussian_blur_unpadded.cpp

#------------------------Compiler and flag-----------------------#
#Compilers
CPP=g++
MPICC=mpicc
NVCC=nvcc
#LIBS and FLAGS
PTHREAD_LIBS=-pthread 
OTHER_LINS=-lm
CPP_FLAGS=-std=c++11 
OPENCV=`pkg-config --cflags --libs opencv`

#---------------------------Rules-------------------------------#
#Rules for standard Gaussian blur (serial version)
standard: gaussian_blur_standard.cpp
	$(CPP) $(SOURCE) -o $(TARGET) $(CPP_FLAGS) $(LIBS) $(OPENCV)

#Rules for pthread Gaussian blur
pthread: gaussian_blur_pthread.cpp
	$(CPP) $(SOURCE2) -o $(TARGET2) $(CPP_FLAGS) $(PTHREAD_LIBS) $(LIBS) $(OPENCV)

#Rules for OpenMp Gaussian blur
omp: gaussian_blur_omp.cpp
	$(CPP) $(SOURCE3) -o $(TARGET3) $(CPP_FLAGS) -fopenmp $(LIBS) $(OPENCV)

#Rules for CUDA Gaussian blur
cuda: gaussian_blur_cuda.cu
	$(NVCC) $(SOURCE4) -o $(TARGET4) $(CPP_FLAGS) $(LIBS) $(OPENCV)

#Rules for Gaussian matrix generator
matrix: create_matrix.cpp
	$(CPP) $(SOURCE_CM) -o $(TARGET_CM) $(CPP_FLAGS) 
	$(CPP) $(SOURCE_CHECK) -o $(TARGET_CHECK) $(CPP_FLAGS) 

diff: diff_image.cpp
	$(CPP) $(SOURCE_DIFF) -o $(TARGET_DIFF) $(CPP_FLAGS) 
	$(NVCC) $(SOURCE4) -o $(TARGET4) $(CPP_FLAGS) $(LIBS) $(OPENCV)
	$(CPP) $(SOURCE_UNPADDED) -o $(TARGET_UNPADDED) $(CPP_FLAGS) $(OPENCV)


.PHONY: clean

clean:
	rm -f *.o
