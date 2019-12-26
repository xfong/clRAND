#Optionally specify the path to the OpenCL headers
#and uncomment the line below
CCFLAGS=-fPIC -Wall -m64 -pthread -O3

GCC ?= gcc
ifeq ($(OS),Windows_NT)
    CCFLAGS+= -I${CUDAROOT}\\include
    LDFLAGS+=-L${CUDAROOT}\\lib\\x64 -L../src
else
    CCFLAGS+= -I${CUDAROOT}/include
    LDFLAGS+=-L${CUDAROOT}/lib64 -L../src
endif

LDLIBS+=-lOpenCL -lclPRNG

#On Windows specify the path to the OpenCL lib file
#the first commented line is the typical path for NVIDIA GPUs
#the second is for AMD GPUS.
#LDFLAGS= -L"$(CUDA_PATH)\lib\x64" -lOpenCL
#LDFLAGS= -L"$(AMDAPPSDKROOT)lib\x86_64" -lOpenCL

example1: util.o example1.c
	${GCC} ${CCFLAGS} -c example1.c -o example1.o
	${GCC} ${LDFLAGS} util.o example1.o -o example1 ${LDLIBS}

util.o: util.h util.c
	${GCC} ${CCFLAGS} -c util.c -o util.o

all: example1

clean:
	rm -f *.o example1
