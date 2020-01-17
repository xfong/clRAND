#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include "../../src/clrand.hpp"
#include "utils.h"

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

int main(int argc, char **argv) {
    cl_event          event = NULL;
    cl_int            err = -1;

    oclStruct* tmpStructPtr = (oclStruct*) malloc(sizeof(oclStruct));
    int res = makeOclStruct(argc, argv, tmpStructPtr);
    if (res != 0) {
        fprintf(stderr,"Unable to create oclStruct!\n");
        return res;
    }
    res = printOclStructInfo(tmpStructPtr);
    if (res != 0) {
        fprintf(stderr,"Unable to print information about oclStruct!\n");
        return res;
    }

    cl_device_id tmpDev = (*tmpStructPtr).target_device;
    (*tmpStructPtr).ctx = clCreateContext(NULL, 1, &tmpDev, NULL, NULL, &err);
    if (err) {
        std::cout << "ERROR: unable to create context to extract random uint!" << std::endl;
        return -1;
    }

    clRAND test;
    std::cout << "Initializing stream" << std::endl;
    test.Init((*tmpStructPtr).target_device, (*tmpStructPtr).ctx, CLRAND_GENERATOR_TYCHE_I);
    std::cout << "Building kernel source" << std::endl;
    test.BuildSource();
    std::cout << test.GetSource() << std::endl;
    std::cout << "Compiling kernel source" << std::endl;
    err = test.BuildKernelProgram();
    if (err != 0) {
        std::cout << "ERROR: failed to compile kernel program" << std::endl;
    }
    std::cout << "\n\nComplete..." << std::endl;

    return err;
}
