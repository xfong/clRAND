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

#define   IM1 2147483563
#define   IM2 2147483399
#define   AM (1.0/IM1)
#define   IMM1  (IM1-1)
#define   IA1 40014
#define   IA2 40692
#define   IQ1 53668
#define   IQ2 52774
#define   IR1 12211
#define   IR2 3791
#define   NTAB 32
#define   NDIV (1+IMM1/NTAB)
#define   EPS 1.2e-7
#define   RNMX (1.0-EPS)

#define ran2_uint(state) (_ran2_uint(&state)<<1)
ulong _ran2_uint(ran2_state* state){

	int k = state->idum / IQ1;
	state->idum = IA1 * (state->idum - k*IQ1) - k*IR1;
	if(state->idum < 0){
		state->idum += IM1;
	}

	k = state->idum2 / IQ2;
	state->idum2 = IA2 * (state->idum2 - k*IQ2) - k*IR2;
	if(state->idum2 < 0){
		state->idum2 += IM2;
	}

	short j = state->iy / NDIV;
	state->iy = state->iv[j] - state->idum2;
	state->iv[j] = state->idum;
	if(state->iy < 1){
		state->iy += IMM1;
	}
	return state->iy;
	/*float temp = AM * state->iy;
	if(temp > RNMX){
		return RNMX;
	}
	else {
		return temp;
	}*/
}

void ran2_seed(ran2_state* state, ulong seed){
	if(seed == 0){
		seed = 1;
	}
	state->idum = seed;
	state->idum2 = seed>>32;
	for(int j = NTAB + 7; j >= 0; j--){
		short k = state->idum / IQ1;
		state->idum = IA1 * (state->idum - k*IQ1) - k*IR1;
		if(state->idum < 0){
			state->idum += IM1;
		}
		if(j < NTAB){
			state->iv[j] = state->idum;
		}
	}
	state->iy = state->iv[0];
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
    clrand_initialize_prng(test, (*tmpStructPtr).target_device, (*tmpStructPtr).ctx, CLRAND_GENERATOR_RAN2);
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
    ran2_state* state_mem = new ran2_state[numPRNGs];
    if (stateMemSize == numPRNGs * sizeof(ran2_state)) {
        err = test->CopyStateToHost((void*)(state_mem));
        if (err) {
            std::cout << "ERROR: unable to copy state buffer to host!" << std::endl;
        }
    } else {
        std::cout << "ERROR: something went wrong setting up memory sizes!" << std::endl;
        std::cout << "State Structure Size (host side): " << sizeof(ran2_state) << std::endl;
        std::cout << "State Structure Size (obj side): " << stateStructSize << std::endl;
        std::cout << "Number of PRNGs: " << numPRNGs << std::endl;
        std::cout << "Size of state buffer: " << stateMemSize << std::endl;
    }

    // Generate RNG states on host side
    ran2_state* golden_states = new ran2_state[numPRNGs];
    ulong init_seedVal = test->GetSeed();
    uint err_counts = 0;
    for (int idx = 0; idx < numPRNGs; idx++) {
        ulong newSeed = (ulong)(idx);
        newSeed <<= 1;
        newSeed += init_seedVal;
        if (newSeed == 0) {
            newSeed += 1;
        }
        ran2_seed(&golden_states[idx], newSeed);
        if (golden_states[idx].idum != state_mem[idx].idum) {
            err_counts++;
            std::cout << "Mismatch in idum at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].idum2 != state_mem[idx].idum2) {
            err_counts++;
            std::cout << "Mismatch in idum2 at idx = " << idx << std::endl;
            continue;
        }
        if (golden_states[idx].iy != state_mem[idx].iy) {
            err_counts++;
            std::cout << "Mismatch in iy at idx = " << idx << std::endl;
            continue;
        }
        for (int idx1 = 0; idx1 < NTAB ; idx1++) {
            if (golden_states[idx].iv[idx1] != state_mem[idx].iv[idx1]) {
                err_counts++;
                std::cout << "Mismatch in iv at idx = " << idx << std::endl;
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
        hostRandomNumbers[idx] = ran2_uint(golden_states[idx]);
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
