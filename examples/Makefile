#Optionally specify the path to the OpenCL headers
#and uncomment the line below
CXXFLAGS=-fPIC -Wall
CXXFLAGS+= -I${CUDAROOT}/include

CXX ?= g++
LDFLAGS+=-L${CUDAROOT}/lib64
LDLIBS+=-lOpenCL

#On Windows specify the path to the OpenCL lib file
#the first commented line is the typical path for NVIDIA GPUs
#the second is for AMD GPUS.
#LDFLAGS= -L"$(CUDA_PATH)\lib\x64" -lOpenCL
#LDFLAGS= -L"$(AMDAPPSDKROOT)lib\x86_64" -lOpenCL

example1: util.o example1.cpp
	${CXX} ${CXXFLAGS} -c example1.cpp -o example1.o
	${CXX} ${LDFLAGS} util.o example1.o ${LDLIBS} -o example1

util.o: util.h util.c
	${CXX} ${CXXFLAGS} -c util.c -o util.o

all: example1

clean:
	rm -f *.o example1
