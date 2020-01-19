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

#define XORSHIFT1024_WARPSIZE 32
#define XORSHIFT1024_WORD 32
#define XORSHIFT1024_WORDSHIFT 10
#define XORSHIFT1024_RAND_A 9
#define XORSHIFT1024_RAND_B 27
#define XORSHIFT1024_RAND_C 24

uint xorshift1024_uint(local xorshift1024_state* stateblock){
	/* Indices. */
	int tid = get_local_id(0) + get_local_size(0) * (get_local_id(1) + get_local_size(1) * get_local_id(2));
	int wid = tid / XORSHIFT1024_WARPSIZE; // Warp index in block
	int lid = tid % XORSHIFT1024_WARPSIZE; // Thread index in warp
	int woff = wid * (XORSHIFT1024_WARPSIZE + XORSHIFT1024_WORDSHIFT + 1) + XORSHIFT1024_WORDSHIFT + 1;
	// warp offset
	/* Shifted indices. */
	int lp = lid + XORSHIFT1024_WORDSHIFT; // Left word shift
	int lm = lid - XORSHIFT1024_WORDSHIFT; // Right word shift

	uint state;

	/* << A. */
	state = stateblock[woff + lid]; // Read states
	state ^= stateblock[woff + lp] << XORSHIFT1024_RAND_A; // Left part
	state ^= stateblock[woff + lp + 1] >> (XORSHIFT1024_WORD - XORSHIFT1024_RAND_A); // Right part
	barrier(CLK_LOCAL_MEM_FENCE);

	/* >> B. */
	stateblock[woff + lid] = state; // Share states
	barrier(CLK_LOCAL_MEM_FENCE);
	state ^= stateblock[woff + lm - 1] << (XORSHIFT1024_WORD - XORSHIFT1024_RAND_B); // Left part
	state ^= stateblock[woff + lm] >> XORSHIFT1024_RAND_B; // Right part
	barrier(CLK_LOCAL_MEM_FENCE);

	/* << C. */
	stateblock[woff + lid] = state; // Share states
	barrier(CLK_LOCAL_MEM_FENCE);
	state ^= stateblock[woff + lp] << XORSHIFT1024_RAND_C; // Left part
	state ^= stateblock[woff + lp + 1] >> (XORSHIFT1024_WORD - XORSHIFT1024_RAND_C); // Right part
	barrier(CLK_LOCAL_MEM_FENCE);

	stateblock[woff + lid] = state; // Share states
	barrier(CLK_LOCAL_MEM_FENCE);

	return state;
}

void xorshift1024_seed(local xorshift1024_state* stateblock, ulong seed){
	int tid = get_local_id(0) + get_local_size(0) * (get_local_id(1) + get_local_size(1) * get_local_id(2));
	int wid = tid / XORSHIFT1024_WARPSIZE; // Warp index in block
	int lid = tid % XORSHIFT1024_WARPSIZE; // Thread index in warp
	int woff = wid * (XORSHIFT1024_WARPSIZE + XORSHIFT1024_WORDSHIFT + 1) + XORSHIFT1024_WORDSHIFT + 1;
	//printf("tid: %d, lid %d, wid %d, woff %d \n", tid, (uint)get_local_id(0), wid, woff);

	uint mem = (XORSHIFT1024_WARPSIZE + XORSHIFT1024_WORDSHIFT + 1) * (get_local_size(0) * get_local_size(1) * get_local_size(2) / XORSHIFT1024_WARPSIZE) + XORSHIFT1024_WORDSHIFT + 1;

	if(lid==13 && (uint)seed==0){ //shouldnt be seeded with all zeroes in wrap, but such check is simpler
		seed=1;
	}

	if(lid<XORSHIFT1024_WORDSHIFT + 1){
		//printf("%d setting %d to 0\n",(uint)get_global_id(0), woff - XORSHIFT1024_WORDSHIFT - 1 + lid);
		stateblock[woff - XORSHIFT1024_WORDSHIFT - 1 + lid] = 0;
	}
	if(tid<XORSHIFT1024_WORDSHIFT + 1){
		//printf("%d setting2 %d to 0\n",(uint)get_global_id(0), mem - 1 - tid);
		stateblock[mem - 1 - tid] = 0;
	}
	stateblock[woff + lid] = (uint)seed;
	//printf("%d seed set\n",(uint)get_local_id(0));
	barrier(CLK_LOCAL_MEM_FENCE);
	//printf("%d after barrier\n",(uint)get_local_id(0));
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
    clrand_initialize_prng(test, (*tmpStructPtr).target_device, (*tmpStructPtr).ctx, CLRAND_GENERATOR_XORSHIFT6432STAR);
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
    xorshift6432star_state* state_mem = new xorshift6432star_state[numPRNGs];
    if (stateMemSize == numPRNGs * sizeof(xorshift6432star_state)) {
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
    xorshift6432star_state* golden_states = new xorshift6432star_state[numPRNGs];
    ulong init_seedVal = test->GetSeed();
    uint err_counts = 0;
    for (int idx = 0; idx < numPRNGs; idx++) {
        ulong newSeed = (ulong)(idx);
        newSeed <<= 1;
        newSeed += init_seedVal;
        if (newSeed == 0) {
            newSeed += 1;
        }
        xorshift6432star_seed(&golden_states[idx], newSeed);
        if (golden_states[idx] != state_mem[idx]) {
            err_counts++;
            std::cout << "Mismatch at idx = " << idx << std::endl;
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
        hostRandomNumbers[idx] = xorshift6432star_uint(golden_states[idx]);
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
