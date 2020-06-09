#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include "../../src/clrand.hpp"
#include "../../generator/tinymt32.hpp"
#include "utils.h"

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#define TINYMT32J_MAT1 0x8f7011eeU
#define TINYMT32J_MAT2 0xfc78ff1fU
#define TINYMT32J_TMAT 0x3793fdffU

#define TINYMT32_SHIFT0 1
#define TINYMT32_SHIFT1 10
#define TINYMT32_SHIFT8 8
#define TINYMT32_MIN_LOOP 8
#define TINYMT32_PRE_LOOP 8
const uint tinymt32_mask = 0x7fffffff;

inline static void
tinymt32_next_state(tinymt32wp_t * tiny)
{
    uint x = (tiny->s0 & tinymt32_mask) ^ tiny->s1 ^ tiny->s2;
    uint y = tiny->s3;
    uint t0, t1;
    x ^= x << TINYMT32_SHIFT0;
    y ^= (y >> TINYMT32_SHIFT0) ^ x;
    tiny->s0 = tiny->s1;
    tiny->s1 = tiny->s2;
    tiny->s2 = x ^ (y << TINYMT32_SHIFT1);
    tiny->s3 = y;
    if (y & 1) {
	tiny->s1 ^= tiny->mat1;
	tiny->s2 ^= tiny->mat2;
    }
}

inline static uint
tinymt32_temper(tinymt32wp_t * tiny)
{
    uint t0, t1;
    t0 = tiny->s3;
    t1 = tiny->s0 + (tiny->s2 >> TINYMT32_SHIFT8);
    t0 ^= t1;
    if (t1 & 1) {
	t0 ^= tiny->tmat;
    }
    return t0;
}

inline static uint
tinymt32_uint32(tinymt32wp_t * tiny)
{
    tinymt32_next_state(tiny);
    return tinymt32_temper(tiny);
}

inline static void
tinymt32_period_certification(tinymt32wp_t * tiny)
{
    if ((tiny->s0 & tinymt32_mask) == 0 &&
        tiny->s1 == 0 &&
        tiny->s2 == 0 &&
        tiny->s3 == 0) {
        tiny->s0 = 'T';
        tiny->s1 = 'I';
        tiny->s2 = 'N';
        tiny->s3 = 'Y';
    }
}

inline static void
tinymt32_init(tinymt32wp_t * tiny, uint seed)
{
    uint status[4];
    status[0] = seed;
    status[1] = tiny->mat1;
    status[2] = tiny->mat2;
    status[3] = tiny->tmat;
    for (int i = 1; i < TINYMT32_MIN_LOOP; i++) {
        status[i & 3] ^= i + 1812433253U
            * (status[(i - 1) & 3]
               ^ (status[(i - 1) & 3] >> 30));
    }
    tiny->s0 = status[0];
    tiny->s1 = status[1];
    tiny->s2 = status[2];
    tiny->s3 = status[3];
    tinymt32_period_certification(tiny);
    for (int i = 0; i < TINYMT32_PRE_LOOP; i++) {
        tinymt32_uint32(tiny);
    }
}

void tinymt32_seed(tinymt32_state* state, ulong seed){
	state->mat1=TINYMT32J_MAT1;
	state->mat2=TINYMT32J_MAT2;
	state->tmat=TINYMT32J_TMAT;
	tinymt32_init(state, seed);
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
    clrand_initialize_prng(test, (*tmpStructPtr).target_device, (*tmpStructPtr).ctx, CLRAND_GENERATOR_TINYMT32);

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
    tinymt32_state* state_mem = new tinymt32_state[numPRNGs];
    if (stateMemSize == numPRNGs * sizeof(tinymt32_state)) {
        err = test->CopyStateToHost((void*)(state_mem));
        if (err) {
            std::cout << "ERROR: unable to copy state buffer to host!" << std::endl;
        }
    } else {
        std::cout << "ERROR: something went wrong setting up memory sizes!" << std::endl;
        std::cout << "State Structure Size (host side): " << sizeof(tinymt32_state) << std::endl;
        std::cout << "State Structure Size (obj side): " << stateStructSize << std::endl;
        std::cout << "Number of PRNGs: " << numPRNGs << std::endl;
        std::cout << "Size of state buffer: " << stateMemSize << std::endl;
    }

    // Generate RNG states on host side
    tinymt32_state* golden_states = new tinymt32_state[numPRNGs];
    ulong init_seedVal = test->GetSeed();
    uint err_counts = 0;
    for (int idx = 0; idx < numPRNGs; idx++) {
        ulong newSeed = (ulong)(idx);
        newSeed <<= 1;
        newSeed += init_seedVal;
        if (newSeed == 0) {
            newSeed += 1;
        }
        tinymt32_seed(&golden_states[idx], newSeed);
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
        if (golden_states[idx].s2 != state_mem[idx].s2) {
            err_counts++;
            std::cout << "Mismatch in s2 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].s3 != state_mem[idx].s3) {
            err_counts++;
            std::cout << "Mismatch in s3 at idx = " << idx << std::endl;
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
        std::cout << "No errors detected!" << std::endl;
    }

    // Completed checks...
    std::cout << "Checks completed!..." << std::endl;
    delete [] state_mem;
    delete [] golden_states;
    free(tmpStructPtr);
    return res;
}
