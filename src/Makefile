#Optionally specify the path to the OpenCL headers
#and uncomment the line below
CXXFLAGS=-fPIC -Wall
CXXFLAGS+= -I${CUDAROOT}/include

CXX ?= g++
LDFLAGS+=-L${CUDAROOT}/lib/x64
LDLIBS+=-lOpenCL

#On Windows specify the path to the OpenCL lib file
#the first commented line is the typical path for NVIDIA GPUs
#the second is for AMD GPUS.
#LDFLAGS= -L"$(CUDA_PATH)\lib\x64" -lOpenCL
#LDFLAGS= -L"$(AMDAPPSDKROOT)lib\x86_64" -lOpenCL

library: clPRNG.cpp clPRNG.hpp
	${CXX} ${CXXFLAGS} -c clPRNG.cpp -o clPRNG.o
	${AR} rcs libclPRNG.a *.o
	${CXX} ${LDFLAGS} -shared clPRNG.o -o libclPRNG.so.0.0.1 ${LDLIBS}

all: library

clean:
	rm -f *.o *.lib *.so.* *.a
