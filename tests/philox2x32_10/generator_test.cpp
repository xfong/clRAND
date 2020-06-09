#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include "../../src/clrand.hpp"
#include "../../generator/philox2x32_10.hpp"
#include "utils.h"

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#ifdef R
    #undef R
#endif

#define PHILOX2X32_10_MULTIPLIER 0xd256d193
#define PHILOX2X32_10_KEY_INC 0x9E3779B9

/*
  Returns most significant 64-bit of result of
  multiplication between two 64-bit numbers
*/
uint32_t mul_hi32(const uint32_t x, const uint32_t y) {
    uint32_t x_lo = x & 0x00000000ffff;
    uint32_t x_hi = x >> 16;
    uint32_t y_lo = y & 0x00000000ffff;
    uint32_t y_hi = y >> 16;
    uint32_t x_x_y_lo = x_lo * y_lo;
    uint32_t x_x_y_mid = x_hi * y_lo;
    uint32_t y_x_x_mid = y_hi * x_lo;
    uint32_t x_x_y_hi = x_hi * y_hi;
    uint32_t carry_bit = (x_x_y_mid & 0x0000ffff) +
                         (y_x_x_mid & 0x0000ffff) +
                         (x_x_y_lo >> 16);
    carry_bit >>= 16;
    uint64_t multhi = x_x_y_hi +
                      (x_x_y_mid >> 16) + (y_x_x_mid >> 16) +
                      carry_bit;
    return multhi;
}

ulong philox2x32_10(philox2x32_10_state& state, uint key){
	uint tmp, L0 = state.L, R0 = state.R;
	for(uint i=0;i<10;i++){
		uint tmp = R0 * PHILOX2X32_10_MULTIPLIER;
		R0 = mul_hi32(R0,PHILOX2X32_10_MULTIPLIER) ^ L0 ^ key;
		L0 = tmp;
		key += PHILOX2X32_10_KEY_INC;
	}
	state.L = L0;
	state.R = R0;
	return state.LR;
}

#define philox2x32_10_ulong(state) _philox2x32_10_ulong(&state)
ulong _philox2x32_10_ulong(philox2x32_10_state *state){
	state->LR++;
	return philox2x32_10(*state, 12345);
}

void philox2x32_10_seed(philox2x32_10_state *state, ulong j){
	state->LR = j;
}

#define philox2x32_10_uint(state) ((uint)philox2x32_10_ulong(state))

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
    clrand_initialize_prng(test, (*tmpStructPtr).target_device, (*tmpStructPtr).ctx, CLRAND_GENERATOR_PHILOX2X32_10);
    (*tmpStructPtr).queue = test->GetStreamQueue();

    err = test->SetupWorkConfigurations();
    if (err) {
        fprintf(stderr,"Unable to set the execution configuration!\n");
        return -1;
    }

    // Initialize the counters that tracks available random number generators
    size_t numPRNGs = test->GetNumberOfRNGs();
    size_t bufMult = 1;

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
    philox2x32_10_state* state_mem = new philox2x32_10_state[numPRNGs];
    if (stateMemSize == numPRNGs * sizeof(philox2x32_10_state)) {
        err = test->CopyStateToHost((void*)(state_mem));
        if (err) {
            std::cout << "ERROR: unable to copy state buffer to host!" << std::endl;
        }
    } else {
        std::cout << "ERROR: something went wrong setting up memory sizes!" << std::endl;
        std::cout << "State Structure Size (host side): " << sizeof(philox2x32_10_state) << std::endl;
        std::cout << "State Structure Size (obj side): " << stateStructSize << std::endl;
        std::cout << "Number of PRNGs: " << numPRNGs << std::endl;
        std::cout << "Size of state buffer: " << stateMemSize << std::endl;
    }

    // Generate RNG states on host side
    philox2x32_10_state* golden_states = new philox2x32_10_state[numPRNGs];
    ulong init_seedVal = test->GetSeed();
    uint err_counts = 0;
    for (uint idx = 0; idx < numPRNGs; idx++) {
        ulong newSeed = (ulong)(idx);
        newSeed <<= 1;
        newSeed += init_seedVal;
        if (newSeed == 0) {
            newSeed += 1;
        }
        philox2x32_10_seed(&golden_states[idx], newSeed);
        if (golden_states[idx].LR != state_mem[idx].LR) {
            err_counts++;
            std::cout << "Mismatch in LR at idx = " << idx << std::endl;
            continue;
        }
    }
    if (err_counts == 0) {
        std::cout << "No errors detected after seeding!" << std::endl;
    } else {
        std::cout << err_counts << " errors detected after seeding!" << std::endl;
        return -2;
    }

    std::cout << "Attempting to generate random uint on device..." << std::endl;
    err = test->FillBuffer();
    if (err) {
        std::cout << "ERROR: unable to fill temporary buffer with random numbers" << std::endl;
        return err;
    }

    uint* deviceRandomNumbers = new uint[numPRNGs];
    cl_mem deviceRandomBuffer = clCreateBuffer((*tmpStructPtr).ctx, CL_MEM_READ_WRITE, test->GetNumBufferEntries() * sizeof(uint), NULL, &err);
    if (err) {
        std::cout << "ERROR: unable to create buffer to extract random uint!" << std::endl;
        return err;
    }
    
    err = test->CopyBufferEntries(deviceRandomBuffer, 0, test->GetNumBufferEntries());
    if (err) {
        std::cout << "ERROR: unable to perform buffer-to-buffer copy to extract random uint!" << std::endl;
        return err;
    }
    err = clEnqueueReadBuffer((*tmpStructPtr).queue, deviceRandomBuffer, true, 0, test->GetNumBufferEntries() * sizeof(uint), deviceRandomNumbers, 0, NULL, &event);
    if (err) {
        std::cout << "ERROR: unable to enqueue read buffer to extract random uint!" << std::endl;
        return err;
    }
    err = clWaitForEvents(1, &event);
    if (err) {
        std::cout << "ERROR: unable to wait for reading buffer to extract random uint!" << std::endl;
        return err;
    }
    std::cout << "Attempting to generate random uint on host..." << std::endl;

    err_counts = 0;
    uint* hostRandomNumbers = new uint[numPRNGs];
    for (uint idx = 0; idx < numPRNGs; idx++) {
        hostRandomNumbers[idx] = philox2x32_10_uint(golden_states[idx]);
        if (hostRandomNumbers[idx] != deviceRandomNumbers[idx]) {
            std::cout << "ERROR: numbers do not match at idx = " << idx << std::endl;
            err_counts++;
        }
    }
    if (err_counts == 0) {
        std::cout << "No errors detected after random number generation!" << std::endl;
    } else {
        std::cout << err_counts << " errors detected after seeding!" << std::endl;
        return -2;
    }
   
    // Completed checks...
    std::cout << "Checks completed!..." << std::endl;
    delete [] state_mem;
    delete [] golden_states;
    delete [] deviceRandomNumbers;
    delete [] hostRandomNumbers;
    free(tmpStructPtr);
    return res;
}
