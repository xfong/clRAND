#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#ifndef __CLPRNG_HPP
    #define __CLPRNG_HPP
#endif

typedef ClPRNG;

// Create PRNG object
ClPRNG create_clPRNG_stream();

// Initialize the PRNG
void init_prng(ClPRNG* p, cl_device_id dev_id, const char *name);

