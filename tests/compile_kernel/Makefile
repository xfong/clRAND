#Optionally specify the path to the OpenCL headers
#and uncomment the line below
CCFLAGS=-fPIC -Wall -m64 -pthread -O3
PREFIX?=/usr/local

GCC ?= g++
INSTALL ?= install

ifeq ($(OS),Windows_NT)
    CCFLAGS+= -I${CUDA_PATH}\\include
    LDFLAGS+=-L${CUDA_PATH}\\lib\\x64 -L../../src
else
    CCFLAGS+= -I${CUDAROOT}/include
    LDFLAGS+=-L${CUDAROOT}/lib64 -L../../src
endif

LDLIBS+=-lOpenCL -lclrand

#On Windows specify the path to the OpenCL lib file
#the first commented line is the typical path for NVIDIA GPUs
#the second is for AMD GPUS.
#LDFLAGS= -L"$(CUDA_PATH)\lib\x64" -lOpenCL
#LDFLAGS= -L"$(AMDAPPSDKROOT)lib\x86_64" -lOpenCL

compile_kernel: utils.o compile_kernel.cpp
	${GCC} ${CCFLAGS} -c compile_kernel.cpp -o compile_kernel.o
	${GCC} ${LDFLAGS} utils.o compile_kernel.o -o compile_kernel ${LDLIBS}

utils.o: utils.h utils.c
	${GCC} ${CCFLAGS} -c utils.c -o utils.o

all: compile_kernel

clean:
	rm -f *.o compile_kernel
