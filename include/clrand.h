#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#ifndef __CLRAND_H
    #define __CLRAND_H
    #define CLRAND_VERSION_MAJOR 0
    #define CLRAND_VERSION_MINOR 0
    #define CLRAND_VERSION_REV   1
#endif

#if defined( __WIN32 )
    #if defined( CLRAND_STATIC )
        #define CLRAND_DLL
    #elif  defined( CLRAND_EXPORT )
        #define CLRAND_DLL __declspec(dllexport)
    #else
        #define CLRAND_DLL __declspec(dllimport)
    #endif
#else
    #define CLRAND_DLL
#endif

typedef
    struct clRAND
        clRAND;

#ifdef __cplusplus
extern "C" {
#endif
// Create PRNG object
CLRAND_DLL clRAND* clrand_create_stream();

// Initialize the PRNG
CLRAND_DLL cl_int clrand_initialize_prng(clRAND* p, cl_device_id dev_id, const char *name);

// Get the precision setting of the PRNG
CLRAND_DLL const char * clrand_get_prng_precision(clRAND* p);

// Set the precision setting of the PRNG
CLRAND_DLL int clrand_set_prng_precision(clRAND* p, const char* precision);

// Get the name setting of the PRNG
CLRAND_DLL const char * clrand_get_prng_name(clRAND* p);

// Set the name setting of the PRNG
CLRAND_DLL cl_int clrand_set_prng_name(clRAND* p, const char* name);

// Seeds the random number generator in the stream object
CLRAND_DLL void clrand_set_prng_seed(clRAND* p, ulong seedNum);

// Readies the stream object for random number generation
CLRAND_DLL cl_int clrand_ready_stream(clRAND* p);

// Generate random number using the stream object
CLRAND_DLL cl_int clrand_generate_stream(clRAND* p, int count, cl_mem dst);

#ifdef __cplusplus
}
#endif

