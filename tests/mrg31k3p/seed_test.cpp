#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <bitset>
#include <unistd.h>
#include <getopt.h>
#include "../../src/clrand.hpp"
#include "../../generator/mrg31k3p.hpp"
#include "utils.h"

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#define MRG31K3P_M1 2147483647
#define MRG31K3P_M2 2147462579

void mrg31k3p_seed(mrg31k3p_state* state, ulong j){
        uint64_t kk = (uint64_t)(j);
	state->x10 = (uint)j;
	state->x11 = (uint)(j >> 5);
	state->x12 = (uint)(j >> 11);
	state->x20 = (uint)(j >> 22);
	state->x21 = (uint)(j >> 30);
	state->x22 = (uint)(j >> 33);
	if(j == 0){
		state->x10++;
		state->x21++;
	}
	if (state->x10 > MRG31K3P_M1) state->x10 -= MRG31K3P_M1;
	if (state->x11 > MRG31K3P_M1) state->x11 -= MRG31K3P_M1;
	if (state->x12 > MRG31K3P_M1) state->x12 -= MRG31K3P_M1;

	if (state->x20 > MRG31K3P_M2) state->x20 -= MRG31K3P_M2;
	if (state->x21 > MRG31K3P_M2) state->x21 -= MRG31K3P_M2;
	if (state->x22 > MRG31K3P_M2) state->x22 -= MRG31K3P_M2;

	if (state->x10 > MRG31K3P_M1) state->x10 -= MRG31K3P_M1;
	if (state->x11 > MRG31K3P_M1) state->x11 -= MRG31K3P_M1;
	if (state->x12 > MRG31K3P_M1) state->x12 -= MRG31K3P_M1;

	if (state->x20 > MRG31K3P_M2) state->x20 -= MRG31K3P_M2;
	if (state->x21 > MRG31K3P_M2) state->x21 -= MRG31K3P_M2;
	if (state->x22 > MRG31K3P_M2) state->x22 -= MRG31K3P_M2;
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
    clrand_initialize_prng(test, (*tmpStructPtr).target_device, (*tmpStructPtr).ctx, CLRAND_GENERATOR_MRG31K3P);

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
    mrg31k3p_state* state_mem = new mrg31k3p_state[numPRNGs];
    if (stateMemSize == numPRNGs * sizeof(mrg31k3p_state)) {
        err = test->CopyStateToHost((void*)(state_mem));
        if (err) {
            std::cout << "ERROR: unable to copy state buffer to host!" << std::endl;
        }
    } else {
        std::cout << "ERROR: something went wrong setting up memory sizes!" << std::endl;
        std::cout << "State Structure Size (host side): " << sizeof(mrg31k3p_state) << std::endl;
        std::cout << "State Structure Size (obj side): " << stateStructSize << std::endl;
        std::cout << "Number of PRNGs: " << numPRNGs << std::endl;
        std::cout << "Size of state buffer: " << stateMemSize << std::endl;
    }

    // Generate RNG states on host side
    mrg31k3p_state* golden_states = new mrg31k3p_state[numPRNGs];
    ulong init_seedVal = test->GetSeed();
    uint err_counts = 0;
    for (uint idx = 0; idx < numPRNGs; idx++) {
        ulong newSeed = (ulong)(idx);
        newSeed <<= 1;
        newSeed += init_seedVal;
        if (newSeed == 0) {
            newSeed += 1;
        }
        mrg31k3p_seed(&golden_states[idx], newSeed);
        if (golden_states[idx].x10 != state_mem[idx].x10) {
            err_counts++;
            std::cout << "Mismatch in x10 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].x11 != state_mem[idx].x11) {
            err_counts++;
            std::cout << "Mismatch in x11 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].x12 != state_mem[idx].x12) {
            err_counts++;
            std::cout << "Mismatch in x12 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].x20 != state_mem[idx].x20) {
            err_counts++;
            std::cout << "Mismatch in x20 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].x21 != state_mem[idx].x21) {
            err_counts++;
            std::cout << "Mismatch in x21 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].x22 != state_mem[idx].x22) {
            err_counts++;
            std::cout << "Mismatch in x22 at idx = " << idx << std::endl;
            continue;
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
