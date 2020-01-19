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

#define LCG12864_MULTI_HIGH 2549297995355413924UL
#define LCG12864_MULTI_LOW 4865540595714422341UL
#define LCG12864_INC_HIGH 6364136223846793005UL
#define LCG12864_INC_LOW 1442695040888963407UL

#define UINT_MAX 0xffffffff

ulong hadd(ulong x, ulong y)
{
    return (x >> 1) + (y >> 1) + (x & y & 1);
}
    
ulong mul_hi(ulong x, ulong y)
{
    ulong f, o, i;
    ulong l;

    //Move the high/low halves of x/y into the lower 32-bits of variables so
    //that we can multiply them without worrying about overflow.
    ulong x_hi = x >> 32;
    ulong x_lo = x & UINT_MAX;
    ulong y_hi = y >> 32;
    ulong y_lo = y & UINT_MAX;

    //Multiply all of the components according to FOIL method
    f = x_hi * y_hi;
    o = x_hi * y_lo;
    i = x_lo * y_hi;
    l = x_lo * y_lo;

    //Now add the components back together, taking care to respect the fact that:
    //F: doesn't need to be modified
    //O/I: Need to be added together.
    //L: Shift right by 32-bits, then add into the sum of O and I
    //Once O/I/L are summed up, then shift the sum by 32-bits and add to F.
    //
    //We use hadd to give us a bit of extra precision for the intermediate sums
    //but as a result, we shift by 31 bits instead of 32
    return (f + (hadd(o, (i + (l>>32))) >> 31));
}

#define lcg12864_ulong(state) ( \
	state.high = state.high * LCG12864_MULTI_LOW + state.low * LCG12864_MULTI_HIGH + mul_hi(state.low, LCG12864_MULTI_LOW), \
	state.low = state.low * LCG12864_MULTI_LOW, \
	state.low += LCG12864_INC_LOW, \
	state.high += state.low < LCG12864_INC_LOW, \
	state.high += LCG12864_INC_HIGH, \
	state.high \
)

#define lcg12864_uint(state) ((uint)lcg12864_ulong(state))

void lcg12864_seed(lcg12864_state* state, ulong j){
	state->low=j;
	state->high=j^0xda3e39cb94b95bdbUL;
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
    clrand_initialize_prng(test, (*tmpStructPtr).target_device, (*tmpStructPtr).ctx, CLRAND_GENERATOR_LCG12864);
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
    lcg12864_state* state_mem = new lcg12864_state[numPRNGs];
    if (stateMemSize == numPRNGs * sizeof(lcg12864_state)) {
        err = test->CopyStateToHost((void*)(state_mem));
        if (err) {
            std::cout << "ERROR: unable to copy state buffer to host!" << std::endl;
        }
    } else {
        std::cout << "ERROR: something went wrong setting up memory sizes!" << std::endl;
        std::cout << "State Structure Size (host side): " << sizeof(isaac_state) << std::endl;
        std::cout << "State Structure Size (obj side): " << stateStructSize << std::endl;
        std::cout << "Number of PRNGs: " << numPRNGs << std::endl;
        std::cout << "Size of state buffer: " << stateMemSize << std::endl;
    }

    // Generate RNG states on host side
    lcg12864_state* golden_states = new lcg12864_state[numPRNGs];
    ulong init_seedVal = test->GetSeed();
    uint err_counts = 0;
    for (int idx = 0; idx < numPRNGs; idx++) {
        ulong newSeed = (ulong)(idx);
        newSeed <<= 1;
        newSeed += init_seedVal;
        if (newSeed == 0) {
            newSeed += 1;
        }
        lcg12864_seed(&golden_states[idx], newSeed);
        if (golden_states[idx].low != state_mem[idx].low) {
            err_counts++;
            std::cout << "Mismatch in low at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].high != state_mem[idx].high) {
            err_counts++;
            std::cout << "Mismatch in high at idx = " << idx << std::endl;
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
        hostRandomNumbers[idx] = lcg12864_uint(golden_states[idx]);
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
