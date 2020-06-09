#Optionally specify the path to the OpenCL headers
#and uncomment the line below
CCFLAGS=-fPIC -Wall -m64 -pthread -O3
PREFIX?=/usr/local

GCC ?= g++
INSTALL ?= install

ifeq ($(OS),Windows_NT)
    CCFLAGS+= -I${CUDA_PATH}\\include -DCLRAND_EXPORT
    LDFLAGS+=-L${CUDA_PATH}\\lib\\x64 -L../../src
else
    CCFLAGS+= -I${CUDAROOT}/include
    LDFLAGS+=-L${CUDAROOT}/lib64 -L../../src -Wl,-rpath=../../src
endif

LDLIBS+=-lOpenCL -Wl,--as-needed -lclrand

#On Windows specify the path to the OpenCL lib file
#the first commented line is the typical path for NVIDIA GPUs
#the second is for AMD GPUS.
#LDFLAGS= -L"$(CUDA_PATH)\lib\x64" -lOpenCL
#LDFLAGS= -L"$(AMDAPPSDKROOT)lib\x86_64" -lOpenCL

seed_test: utils.o seed_test.cpp
	${GCC} ${CCFLAGS} -c seed_test.cpp -o seed_test.o
	${GCC} ${LDFLAGS} utils.o seed_test.o -o seed_test ${LDLIBS}

generator_test: utils.o generator_test.cpp
	${GCC} ${CCFLAGS} -c generator_test.cpp -o generator_test.o
	${GCC} ${LDFLAGS} utils.o generator_test.o -o generator_test ${LDLIBS}

utils.o: utils.h utils.c
	${GCC} ${CCFLAGS} -c utils.c -o utils.o

all: seed_test generator_test

clean:
	rm -f *.o seed_test generator_test
