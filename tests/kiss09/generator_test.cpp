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

#define RNG32

#define ISAAC_RANDSIZL   (8)
#define ISAAC_RANDSIZ    (1<<ISAAC_RANDSIZL)

typedef unsigned char uchar;
#define ind(mm,x) (*(uint *)((uchar *)(mm) + ((x) & (255 << 2))))
//#define ind(mm,x) (*(uint *)((uint *)(mm) + (((x) >> 2) & 255)))
#define rngstep(mix,a,b,mm,m,m2,r,x) \
{\
	x = *m; \
	a = (a ^ (mix)) + *(m2++); \
	*(m++) = y = ind(mm, x) + a + b; \
	*(r++) = b = ind(mm, y >> 8) + x; \
}

void isaac_seed(isaac_state* state, ulong j){
	state->aa = j;
	state->bb = j ^ 123456789;
	state->cc = j + 123456789;
	state->idx = ISAAC_RANDSIZ;
	for(int i=0;i<ISAAC_RANDSIZ;i++){
		j=6906969069UL * j + 1234567UL; //LCG
		state->mm[i]=j;
		//isaac_advance(state);
	}
}

void isaac_advance(isaac_state* state){
	uint a, b, x, y, *m, *m2, *r, *mend;
	m = state->mm;
	r = state->rr;
	a = state->aa;
	b = state->bb + (++state->cc);
	for (m = state->mm, mend = m2 = m+(ISAAC_RANDSIZ/2); m < mend; ){
		rngstep(a << 13, a, b, state->mm, m, m2, r, x);
		rngstep(a >> 6 , a, b, state->mm, m, m2, r, x);
		rngstep(a << 2 , a, b, state->mm, m, m2, r, x);
		rngstep(a >> 16, a, b, state->mm, m, m2, r, x);
	}
	for (m2 = state->mm; m2 < mend; ){
		rngstep(a << 13, a, b, state->mm, m, m2, r, x);
		rngstep(a >> 6 , a, b, state->mm, m, m2, r, x);
		rngstep(a << 2 , a, b, state->mm, m, m2, r, x);
		rngstep(a >> 16, a, b, state->mm, m, m2, r, x);
	}
	state->bb = b;
	state->aa = a;
}

#define isaac_uint(state) _isaac_uint(&state)
uint _isaac_uint(isaac_state* state){
	//printf("%d\n", get_global_id(0));
	if(state->idx == ISAAC_RANDSIZ){
		isaac_advance(state);
		state->idx=0;
	}
	return state->rr[state->idx++];
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
    clrand_initialize_prng(test, (*tmpStructPtr).target_device, (*tmpStructPtr).ctx, CLRAND_GENERATOR_ISAAC);
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
    isaac_state* state_mem = new isaac_state[numPRNGs];
    if (stateMemSize == numPRNGs * sizeof(isaac_state)) {
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
    isaac_state* golden_states = new isaac_state[numPRNGs];
    ulong init_seedVal = test->GetSeed();
    uint err_counts = 0;
    for (int idx = 0; idx < numPRNGs; idx++) {
        ulong newSeed = (ulong)(idx);
        newSeed <<= 1;
        newSeed += init_seedVal;
        if (newSeed == 0) {
            newSeed += 1;
        }
        isaac_seed(&golden_states[idx], newSeed);
        if (golden_states[idx].aa != state_mem[idx].aa) {
            err_counts++;
            std::cout << "Mismatch in aa at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].bb != state_mem[idx].bb) {
            err_counts++;
            std::cout << "Mismatch in bb at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].cc != state_mem[idx].cc) {
            err_counts++;
            std::cout << "Mismatch in cc at idx = " << idx << std::endl;
            continue;
        }
        for (int idx1 = 0; idx1 < ISAAC_RANDSIZ ; idx1++) {
            if (golden_states[idx].mm[idx1] != state_mem[idx].mm[idx1]) {
                err_counts++;
                std::cout << "Mismatch in mm at idx = " << idx << std::endl;
                break;
            }
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
        hostRandomNumbers[idx] = isaac_uint(golden_states[idx]);
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
