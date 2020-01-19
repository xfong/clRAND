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

#define MRG63K3A_M1    9223372036854769163
#define MRG63K3A_M2    9223372036854754679
#define MRG63K3A_A12   1754669720
#define MRG63K3A_Q12   5256471877
#define MRG63K3A_R12   251304723
#define MRG63K3A_A13N  3182104042
#define MRG63K3A_Q13   2898513661
#define MRG63K3A_R13   394451401
#define MRG63K3A_A21   31387477935
#define MRG63K3A_Q21   293855150
#define MRG63K3A_R21   143639429
#define MRG63K3A_A32N  6199136374
#define MRG63K3A_Q23   1487847900
#define MRG63K3A_R23   985240079

void mrg63k3a_seed(mrg63k3a_state* state, ulong j){
	state->s10 = j;
	state->s11 = j;
	state->s12 = j;
	state->s20 = j;
	state->s21 = j;
	state->s22 = j;
	if(j == 0){
		state->s10++;
		state->s21++;
	}
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
    clrand_initialize_prng(test, (*tmpStructPtr).target_device, (*tmpStructPtr).ctx, CLRAND_GENERATOR_MRG63K3A);

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
    mrg63k3a_state* state_mem = new mrg63k3a_state[numPRNGs];
    if (stateMemSize == numPRNGs * sizeof(mrg63k3a_state)) {
        err = test->CopyStateToHost((void*)(state_mem));
        if (err) {
            std::cout << "ERROR: unable to copy state buffer to host!" << std::endl;
        }
    } else {
        std::cout << "ERROR: something went wrong setting up memory sizes!" << std::endl;
        std::cout << "State Structure Size (host side): " << sizeof(mrg63k3a_state) << std::endl;
        std::cout << "State Structure Size (obj side): " << stateStructSize << std::endl;
        std::cout << "Number of PRNGs: " << numPRNGs << std::endl;
        std::cout << "Size of state buffer: " << stateMemSize << std::endl;
    }

    // Generate RNG states on host side
    mrg63k3a_state* golden_states = new mrg63k3a_state[numPRNGs];
    ulong init_seedVal = test->GetSeed();
    uint err_counts = 0;
    for (int idx = 0; idx < numPRNGs; idx++) {
        ulong newSeed = (ulong)(idx);
        newSeed <<= 1;
        newSeed += init_seedVal;
        if (newSeed == 0) {
            newSeed += 1;
        }
        mrg63k3a_seed(&golden_states[idx], newSeed);
        if (golden_states[idx].s10 != state_mem[idx].s10) {
            err_counts++;
            std::cout << "Mismatch in s10 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].s11 != state_mem[idx].s11) {
            err_counts++;
            std::cout << "Mismatch in s11 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].s12 != state_mem[idx].s12) {
            err_counts++;
            std::cout << "Mismatch in s12 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].s20 != state_mem[idx].s20) {
            err_counts++;
            std::cout << "Mismatch in s20 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].s21 != state_mem[idx].s21) {
            err_counts++;
            std::cout << "Mismatch in s21 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].s22 != state_mem[idx].s22) {
            err_counts++;
            std::cout << "Mismatch in s22 at idx = " << idx << std::endl;
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
