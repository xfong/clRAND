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

void xorwow_seed(xorwow_state* state, unsigned long j){
        state->x[0] = 123456789U;
        state->x[1] = 362436069U;
        state->x[2] = 521288629U;
        state->x[3] = 88675123U;
        state->x[4] = 5783321U;

        state->d = 6615241U;

        // Constants are arbitrary prime numbers
        const uint s0 = (uint)(j) ^ 0x2c7f967fU;
        const uint s1 = (uint)(j >> 32) ^ 0xa03697cbU;
        const uint t0 = 1228688033U * s0;
        const uint t1 = 2073658381U * s1;
        state->x[0] += t0;
        state->x[1] ^= t0;
        state->x[2] += t1;
        state->x[3] ^= t1;
        state->x[4] += t0;
        state->d += t1 + t0;

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

    cl_device_id tmpDev = (*tmpStructPtr).target_device;
    (*tmpStructPtr).ctx = clCreateContext(NULL, 1, &tmpDev, NULL, NULL, &err);
    if (err) {
        std::cout << "ERROR: unable to create context to extract random uint!" << std::endl;
        return -1;
    }

    clRAND* test = clrand_create_stream();
    clrand_initialize_prng(test, (*tmpStructPtr).target_device, (*tmpStructPtr).ctx, CLRAND_GENERATOR_XORWOW);

    err = test->SetupWorkConfigurations();
    if (err) {
        fprintf(stderr,"Unable to set the execution configuration!\n");
        return -1;
    }

    // Initialize the counters that tracks available random number generators
    size_t numPRNGs = test->GetNumberOfRNGs();
    size_t bufMult = 2;

    err = test->SetupStreamBuffers(bufMult, numPRNGs);
    test->SetReady();

    // Seed the RNGs
    err = test->SeedGenerator();
    if (err != 0) {
        std::cout << "ERROR: failed to seed PRNG" << std::endl;
        return -1;
    }

    size_t stateStructSize = test->GetStateStructSize();
    size_t stateMemSize = test->GetStateBufferSize();
    // Prepare host memory to copy RNG states from device to host
    xorwow_state* state_mem = new xorwow_state[numPRNGs];
    if (stateMemSize == numPRNGs * sizeof(xorwow_state)) {
        err = test->CopyStateToHost((void*)(state_mem));
        if (err) {
            std::cout << "ERROR: unable to copy state buffer to host!" << std::endl;
        }
    } else {
        std::cout << "ERROR: something went wrong setting up memory sizes!" << std::endl;
        std::cout << "State Structure Size (host side): " << sizeof(xorwow_state) << std::endl;
        std::cout << "State Structure Size (obj side): " << stateStructSize << std::endl;
        std::cout << "Number of PRNGs: " << numPRNGs << std::endl;
        std::cout << "Size of state buffer: " << stateMemSize << std::endl;
    }

    // Generate RNG states on host side
    xorwow_state* golden_states = new xorwow_state[numPRNGs];
    ulong init_seedVal = test->GetSeed();
    uint err_counts = 0;
    for (int idx = 0; idx < numPRNGs; idx++) {
        ulong newSeed = (ulong)(idx);
        newSeed <<= 1;
        newSeed += init_seedVal;
        if (newSeed == 0) {
            newSeed += 1;
        }
        xorwow_seed(&golden_states[idx], newSeed);
        if (golden_states[idx].d != state_mem[idx].d) {
            err_counts++;
            std::cout << "Mismatch in d at idx = " << idx << std::endl;
            continue;
        }
        for (uint idx1 = 0; idx1 < 5; idx1++) {
            if (golden_states[idx].x[idx1] != state_mem[idx].x[idx1]) {
                err_counts++;
                std::cout << "Mismatch in x at idx = " << idx << std::endl;
                continue;
            }
        }
    }
    if (err_counts == 0) {
        std::cout << "No errors detected!" << std::endl;
    }

    // Completed checks...
    std::cout << "Checks completed!..." << std::endl;
    delete [] state_mem;
    delete [] golden_states;
    free(tmpStructPtr);
    return res;
}
