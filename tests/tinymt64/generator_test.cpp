#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include "../../src/clrand.hpp"
#include "../../generator/tinymt64.hpp"
#include "utils.h"

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#define TINYMT64J_MAT1 0xfa051f40U
#define TINYMT64J_MAT2 0xffd0fff4U;
#define TINYMT64J_TMAT UINT64_C(0x58d02ffeffbfffbc)

#define TINYMT64_SHIFT0 12
#define TINYMT64_SHIFT1 11
#define TINYMT64_MIN_LOOP 8

const ulong tinymt64_mask = 0x7fffffffffffffffUL;

inline static void
tinymt64_next_state(tinymt64wp_t * tiny)
{
    ulong x;

    tiny->s0 &= tinymt64_mask;
    x = tiny->s0 ^ tiny->s1;
    x ^= x << TINYMT64_SHIFT0;
    x ^= x >> 32;
    x ^= x << 32;
    x ^= x << TINYMT64_SHIFT1;
    tiny->s0 = tiny->s1;
    tiny->s1 = x;
    if (x & 1) {
        tiny->s0 ^= tiny->mat1;
        tiny->s1 ^= (ulong)tiny->mat2 << 32;
    }
}

inline static ulong
tinymt64_temper(tinymt64wp_t * tiny)
{
    ulong x;
    x = tiny->s0 + tiny->s1;
    x ^= tiny->s0 >> 8;
    if (x & 1) {
        x ^= tiny->tmat;
    }
    return x;
}

inline static ulong
tinymt64_uint64(tinymt64wp_t * tiny)
{
    tinymt64_next_state(tiny);
    return tinymt64_temper(tiny);
}

inline static void
tinymt64_period_certification(tinymt64wp_t * tiny)
{
    if ((tiny->s0 & tinymt64_mask) == 0 &&
        tiny->s1 == 0) {
        tiny->s0 = 'T';
        tiny->s1 = 'M';
    }
}

inline static void
tinymt64_init(tinymt64wp_t * tiny, ulong seed)
{
    ulong status[2];
    status[0] = seed ^ ((ulong)tiny->mat1 << 32);
    status[1] = tiny->mat2 ^ tiny->tmat;
    for (int i = 1; i < TINYMT64_MIN_LOOP; i++) {
        status[i & 1] ^= i + 6364136223846793005UL
            * (status[(i - 1) & 1] ^ (status[(i - 1) & 1] >> 62));
    }
    tiny->s0 = status[0];
    tiny->s1 = status[1];
    tinymt64_period_certification(tiny);
}

#define tinymt64_ulong(state) tinymt64_uint64(&state)

void tinymt64_seed(tinymt64_state* state, ulong seed){
	state->mat1=TINYMT64J_MAT1;
	state->mat2=TINYMT64J_MAT2;
	state->tmat=TINYMT64J_TMAT;
	tinymt64_init(state, seed);
}

#define tinymt64_uint(state) ((uint)tinymt64_ulong(state))

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
    clrand_initialize_prng(test, (*tmpStructPtr).target_device, (*tmpStructPtr).ctx, CLRAND_GENERATOR_TINYMT64);
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
    tinymt64_state* state_mem = new tinymt64_state[numPRNGs];
    if (stateMemSize == numPRNGs * sizeof(tinymt64_state)) {
        err = test->CopyStateToHost((void*)(state_mem));
        if (err) {
            std::cout << "ERROR: unable to copy state buffer to host!" << std::endl;
        }
    } else {
        std::cout << "ERROR: something went wrong setting up memory sizes!" << std::endl;
        std::cout << "State Structure Size (host side): " << sizeof(tinymt64_state) << std::endl;
        std::cout << "State Structure Size (obj side): " << stateStructSize << std::endl;
        std::cout << "Number of PRNGs: " << numPRNGs << std::endl;
        std::cout << "Size of state buffer: " << stateMemSize << std::endl;
    }

    // Generate RNG states on host side
    tinymt64_state* golden_states = new tinymt64_state[numPRNGs];
    ulong init_seedVal = test->GetSeed();
    uint err_counts = 0;
    for (int idx = 0; idx < numPRNGs; idx++) {
        ulong newSeed = (ulong)(idx);
        newSeed <<= 1;
        newSeed += init_seedVal;
        if (newSeed == 0) {
            newSeed += 1;
        }
        tinymt64_seed(&golden_states[idx], newSeed);
        if (golden_states[idx].s0 != state_mem[idx].s0) {
            err_counts++;
            std::cout << "Mismatch in s0 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].s1 != state_mem[idx].s1) {
            err_counts++;
            std::cout << "Mismatch in s1 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].mat1 != state_mem[idx].mat1) {
            err_counts++;
            std::cout << "Mismatch in mat1 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].mat2 != state_mem[idx].mat2) {
            err_counts++;
            std::cout << "Mismatch in mat2 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].tmat != state_mem[idx].tmat) {
            err_counts++;
            std::cout << "Mismatch in tmat at idx = " << idx << std::endl;
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
    for (int idx = 0; idx < numPRNGs; idx++) {
        hostRandomNumbers[idx] = tinymt64_uint(golden_states[idx]);
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
