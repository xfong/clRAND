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

#define ISAAC_RANDSIZL   (8)
#define ISAAC_RANDSIZ    (1<<ISAAC_RANDSIZL)

void* clRAND::GetLocalStateMem() {
    static isaac_state* outState = new isaac_state[this->GetNumValidEntries()];
    isaac_state* tmpPtr = (isaac_state*)(this->local_state_mem);
    for (int idx = 0; idx < this->GetNumValidEntries(); idx++) {
        outState[idx] = tmpPtr[idx];
    }
    return (void*)(outState);
}

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

    clRAND* test = clrand_create_stream();
    clrand_initialize_prng(test, (*tmpStructPtr).target_device, CLRAND_GENERATOR_ISAAC);
    err = clrand_ready_stream(test);
    if (err) {
        fprintf(stderr,"Unable to ready the bitstream!\n");
        return -1;
    }
    isaac_state* tmpArray = (isaac_state*)(test->GetLocalStateMem());
    free(tmpStructPtr);
    return res;
}
