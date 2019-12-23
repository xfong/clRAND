#include "clPRNG.hpp"

// Main C interface functions

extern "C" ClPRNG create_clPRNG_stream() {
    return new ClPRNG{};
}

extern "C" void init_prng(ClPRNG* p, cl_device_id dev_id, const char *name) {
    p->Init(dev_id, name);
}

// Default constructor and destructor
ClPRNG::~CPRNG() {
    init_flag = false;
}

ClPRNG::~ClPRNG() {
    cl_int err;
    err = clReleaseMemObject(stateBuffer);
    err = clReleaseMemObject(tmpOutputBuffer);
    err = clReleaseKernel(generate_bitstream);
    err = clReleaseKernel(seed_rng);
    err = clReleaseProgram(rng_program);
    err = clReleaseCommandQueue(com_queue);
    err = clReleaseContext(context);
    err = clReleaseDevice(device);
}

void ClPRNG::Init(cl_device_id dev_id, const char *name) {
    device = dev_id;
    cl_int err;
    context = clCreateContext(NULL, cl_uint(1), &{ dev_id }, NULL, NULL, &err);
}
