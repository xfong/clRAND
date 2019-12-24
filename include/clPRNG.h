#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#ifndef __CLPRNG_H
    #define __CLPRNG_H
    #define CLPRNG_VERSION_MAJOR 0
    #define CLPRNG_VERSION_MINOR 0
    #define CLPRNG_VERSION_REV   1
#endif

typedef ClPRNG;

// Create PRNG object
ClPRNG create_clPRNG_stream();

// Initialize the PRNG
void init_prng(ClPRNG* p, cl_device_id dev_id, const char *name);
