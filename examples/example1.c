// First example of using clPRNG to generate random bitstream
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <getopt.h>
#include "util.h"
#include "../include/clPRNG.h"

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

    ClPRNG* test = create_clPRNG_stream();
    initialize_prng(test, (*tmpStructPtr).target_device, "tinymt32");
    err = set_precision(test, "uint");
    free(tmpStructPtr);
    return res;
}
