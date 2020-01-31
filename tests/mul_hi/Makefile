#Optionally specify the path to the OpenCL headers
#and uncomment the line below
CCFLAGS=-fPIC -Wall -m64 -pthread -O3
PREFIX?=/usr/local

GCC ?= g++
INSTALL ?= install

mul_hi: mul_hi.cpp
	${GCC} ${CCFLAGS} -c mul_hi.cpp -o mul_hi.o
	${GCC} ${LDFLAGS} mul_hi.o -o mul_hi ${LDLIBS}

all: mul_hi

clean:
	rm -f *.o mul_hi
