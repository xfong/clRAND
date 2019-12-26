#include <iostream>
#include <map>
#include <string>

#include "../generator/isaac.hpp"
#include "../generator/kiss09.hpp"
#include "../generator/kiss99.hpp"
#include "../generator/lcg6432.hpp"
#include "../generator/lcg12864.hpp"
#include "../generator/lfib.hpp"
#include "../generator/mrg31k3p.hpp"
#include "../generator/mrg63k3a.hpp"
#include "../generator/msws.hpp"
#include "../generator/mt19937.hpp"
#include "../generator/mwc64x.hpp"
#include "../generator/pcg6432.hpp"
#include "../generator/philox2x32_10.hpp"
#include "../generator/ran2.hpp"
#include "../generator/tinymt32.hpp"
#include "../generator/tinymt64.hpp"
#include "../generator/tyche.hpp"
#include "../generator/tyche_i.hpp"
#include "../generator/well512.hpp"
#include "../generator/xorshift1024.hpp"
#include "../generator/xorshift6432star.hpp"

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#ifndef __CLPRNG_HPP
    #define __CLPRNG_HPP
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

// Prototype class
CLPRNG_DLL class ClPRNG {
    private:
        cl::Device        device;
        cl::Context       context;
        cl::CommandQueue  com_queue;

        cl::Program       rng_program;
        cl::Kernel        seed_rng;
        cl::Kernel        generate_bitstream;

        cl::Buffer        stateBuffer;
        cl::Buffer        tmpOutputBuffer;
        size_t            valid_cnt;

        size_t            wkgrp_size;
        size_t            wkgrp_count;

        const char*       rng_name;
        const char*       rng_precision;
        std::string       rng_source;

        bool              source_ready;
        bool              init_flag;

        int LookupPRNG(std::string name);
        void generateBufferKernel(std::string name, std::string type, std::string src);

    public:
        void Init(cl_device_id dev_id, const char * name);
        void BuildSource();
        cl_int BuildKernelProgram();
        void Seed(uint32_t seed);
        void GenerateStream(cl_mem OutputBuffer);
        bool IsSourceReady() { return source_ready; }
        bool IsInitialized() { return init_flag; }
        std::string GetPrecision() { return std::string(rng_precision); }
        int SetPrecision(const char * precision);
        std::string GetName() { return std::string(rng_name); }
        void SetName(const char * name) { rng_name = name; }
        std::string GetSource() { return rng_source; }
        ClPRNG();
        ~ClPRNG();
};

// External functions
#ifdef __cplusplus
extern "C" {
#endif
CLPRNG_DLL void initialize_prng(ClPRNG* p, cl_device_id dev_id, const char *name);

CLPRNG_DLL ClPRNG* create_clPRNG_stream();

CLPRNG_DLL const char * get_precision(ClPRNG* p) {
    return (*p).GetPrecision().c_str();
}

CLPRNG_DLL int set_precision(ClPRNG* p, const char* precision) {
    return (*p).SetPrecision(precision);
}

CLPRNG_DLL const char * get_name(ClPRNG* p) {
    return (*p).GetName().c_str();
}

CLPRNG_DLL void set_name(ClPRNG* p, const char* name) {
    (*p).SetName(name);
}

CLPRNG_DLL cl_int buildPRNGKernelProgram(ClPRNG* p);

#ifdef __cplusplus
}
#endif
