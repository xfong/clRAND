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

#if defined( __WIN32 )
    #if defined( CLPRNG_STATIC )
        #define CLPRNG_DLL
    #elif  defined( CLPRNG_EXPORT )
        #define CLPRNG_DLL __declspec(dllexport)
    #else
        #define CLPRNG_DLL __declspec(dllimport)
    #endif
#else
    #define CLPRNG_DLL
#endif

typedef
    struct ClPRNG
        ClPRNG;

#ifdef __cplusplus
extern "C" {
#endif
// Create PRNG object
CLPRNG_DLL ClPRNG* create_clPRNG_stream();

// Initialize the PRNG
CLPRNG_DLL void initialize_prng(ClPRNG* p, cl_device_id dev_id, const char *name);

// Get the precision setting of the PRNG
CLPRNG_DLL const char * get_precision(ClPRNG* p);

// Get the name setting of the PRNG
CLPRNG_DLL const char * get_name(ClPRNG* p);

// Set the precision setting of the PRNG
CLPRNG_DLL int set_precision(ClPRNG* p, const char* precision);

// Set the name setting of the PRNG
CLPRNG_DLL void set_name(ClPRNG* p, const char* name);

#ifdef __cplusplus
}
#endif

