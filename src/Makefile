#Optionally specify the path to the OpenCL headers
#and uncomment the line below
CXXFLAGS=-fPIC -Wall -m64 -pthread -O3
CXX ?= g++
INSTALL?=install
LDLIBS+=-lOpenCL

PREFIX?=/usr/local

MAJOR_VERSION=0
MINOR_VERSION=0
REV_VERSION=1
ifeq ($(OS),Windows_NT)
    CXXFLAGS+=-I${CUDA_PATH}\\include
    CXXFLAGS+=-D__WIN32 -DCLRAND_EXPORT
    LDFLAGS+=-L${CUDA_PATH}\\lib\\x64
    DLLFLAGS=-Wl,--out-implib,libclrand.a -Wl,--major-image-version,0,--minor-image-version,1
    LIBEXT=.dll
else
    CXXFLAGS+= -I${CUDAROOT}/include
    LDFLAGS+=-L${CUDAROOT}/lib64
    LIBEXT=.so.${MAJOR_VERSION}.${MINOR_VERSION}.${REV_VERSION}
    DLLFLAGS=-Wl,-soname,libclrand.so.${MAJOR_VERSION}
endif

INSTALL_INCLUDEDIR=$(PREFIX)/include
INSTALL_LIBDIR=$(PREFIX)/lib

#On Windows specify the path to the OpenCL lib file
#the first commented line is the typical path for NVIDIA GPUs
#the second is for AMD GPUS.
#LDFLAGS= -L"$(CUDA_PATH)\lib\x64" -lOpenCL
#LDFLAGS= -L"$(AMDAPPSDKROOT)lib\x86_64" -lOpenCL

library/shared: clrand.cpp clrand.hpp
	${CXX} ${CXXFLAGS} -c clrand.cpp -o clrand.o
	${CXX} ${LDFLAGS} -shared clrand.o -o libclrand${LIBEXT} ${LDLIBS} ${DLLFLAGS}
	ln -s libclrand${LIBEXT} libclrand.so.${MAJOR_VERSION}
	ln -s libclrand.so.${MAJOR_VERSION} libclrand.so

library/static: clrand.cpp clrand.hpp
	${CXX} ${CXXFLAGS} -c clrand.cpp -o clrand.o
	${AR} rcs libclrand.a *.o

all: library/shared library/static

install: all
	$(INSTALL) -D -m 644 -t $(INSTALL_INCLUDEDIR) ../include/clrand.h
	$(INSTALL) -D -m 755 -t $(INSTALL_LIBDIR) *.so.$(MAJOR_VERSION).$(MINOR_VERSION).$(REV_VERSION)
	ln -Lrs $(INSTALL_LIBDIR)/libclrand.so.$(MAJOR_VERSION).$(MINOR_VERSION).$(REV_VERSION) $(INSTALL_LIBDIR)/libclrand.so.$(MAJOR_VERSION).$(MINOR_VERSION)
	ln -Lrs $(INSTALL_LIBDIR)/libclrand.so.$(MAJOR_VERSION).$(MINOR_VERSION) $(INSTALL_LIBDIR)/libclrand.so.$(MAJOR_VERSION)
	ln -Lrs $(INSTALL_LIBDIR)/libclrand.so.$(MAJOR_VERSION) $(INSTALL_LIBDIR)/libclrand.so
	$(INSTALL) -D -m 644 -t $(INSTALL_LIBDIR) *.a

clean:
	rm -f *.o *.lib lib*.so* *.a *.dll
