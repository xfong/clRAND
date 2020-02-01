#include <iostream>
#include <map>
#include <string>

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#include "../generator/isaac.hpp"
#include "../generator/kiss09.hpp"
#include "../generator/kiss99.hpp"
#include "../generator/lcg6432.hpp"
#include "../generator/lcg12864.hpp"
#include "../generator/lfib.hpp"
#include "../generator/mrg31k3p.hpp"
#include "../generator/mrg32k3a.hpp"
#include "../generator/mrg63k3a.hpp"
#include "../generator/msws.hpp"
#include "../generator/mt19937.hpp"
#include "../generator/mwc64x.hpp"
#include "../generator/pcg6432.hpp"
#include "../generator/philox2x32_10.hpp"
#include "../generator/philox4x32_10.hpp"
#include "../generator/ran2.hpp"
#include "../generator/sobol32.hpp"
#include "../generator/threefry.hpp"
#include "../generator/tinymt32.hpp"
#include "../generator/tinymt64.hpp"
#include "../generator/tyche.hpp"
#include "../generator/tyche_i.hpp"
#include "../generator/well512.hpp"
#include "../generator/xorshift1024.hpp"
#include "../generator/xorshift6432star.hpp"
#include "../generator/xorwow.hpp"

#ifndef __CLRAND_HPP
    #define __CLRAND_HPP
    #define CLRAND_VERSION_MAJOR 0
    #define CLRAND_VERSION_MINOR 1
    #define CLRAND_VERSION_REV   0
#endif

#if defined( __WIN32 )
    #if defined( clRAND_STATIC )
        #define CLRAND_DLL
    #elif  defined( clRAND_EXPORT )
        #define CLRAND_DLL __declspec(dllexport)
    #else
        #define CLRAND_DLL __declspec(dllimport)
    #endif
#else
    #define CLRAND_DLL
#endif

enum clrandRngType {
    CLRAND_GENERATOR_ISAAC            = 1,
    CLRAND_GENERATOR_KISS09           = 2,
    CLRAND_GENERATOR_KISS99           = 3,
    CLRAND_GENERATOR_LCG6432          = 4,
    CLRAND_GENERATOR_LCG12864         = 5,
    CLRAND_GENERATOR_LFIB             = 6,
    CLRAND_GENERATOR_MRG31K3P         = 7,
    CLRAND_GENERATOR_MRG32K3A         = 8,
    CLRAND_GENERATOR_MRG63K3A         = 9,
    CLRAND_GENERATOR_MSWS             = 10,
    CLRAND_GENERATOR_MT11213          = 11,
    CLRAND_GENERATOR_MT19937          = 12,
    CLRAND_GENERATOR_MT23209          = 13,
    CLRAND_GENERATOR_MT44497          = 14,
    CLRAND_GENERATOR_MWC64X           = 15,
    CLRAND_GENERATOR_PCG6432          = 16,
    CLRAND_GENERATOR_PHILOX2X32_10    = 17,
    CLRAND_GENERATOR_PHILOX4X32_10    = 18,
    CLRAND_GENERATOR_RAN2             = 19,
    CLRAND_GENERATOR_SOBOL32          = 20,
    CLRAND_GENERATOR_THREEFRY         = 21,
    CLRAND_GENERATOR_TINYMT32         = 22,
    CLRAND_GENERATOR_TINYMT64         = 23,
    CLRAND_GENERATOR_TYCHE            = 24,
    CLRAND_GENERATOR_TYCHE_I          = 25,
    CLRAND_GENERATOR_WELL512          = 26,
    CLRAND_GENERATOR_XORSHIFT1024     = 27,
    CLRAND_GENERATOR_XORSHIFT6432STAR = 28,
    CLRAND_GENERATOR_XORWOW           = 29
};

// Prototype class
CLRAND_DLL class clRAND {
    private:
        cl::Device        device;              // OpenCL C++ API
        cl_device_id      device_id;           // OpenCL C API (to support buffer copy)

        cl::Context       context;             // OpenCL C++ API
        cl_context        context_id;          // OpenCL C API (to support buffer copy)

        cl::CommandQueue  com_queue;           // OpenCL C++ API
        cl_command_queue  com_queue_id;        // OpenCL C API (to support buffer copy)

        cl::Program       rng_program;         // OpenCL C++ API
        cl::Kernel        seed_rng;            // OpenCL C++ API
        cl::Kernel        seed_rng_array;      // OpenCL C++ API
        cl::Kernel        generate_bitstream;  // OpenCL C++ API

        cl::Buffer        stateBuffer;         // OpenCL C++ API
        cl_mem            stateBuffer_id;      // OpenCL C API (to support buffer copy)

        cl::Buffer        tmpOutputBuffer;     // OpenCL C++ API
        cl_mem            tmpOutputBuffer_id;  // OpenCL C API (to support buffer copy)

        size_t            state_size;          // Information for PRNG state
        bool              loaded_state;        // Flag for whether PRNG states are loaded to device
        void *            local_state_mem;     // Host side storage of PRNG state

        size_t            total_count;         // Information for temporary output buffer
        size_t            valid_count;         // Information for temporary output buffer
        size_t            offset;              // Information for temporary output buffer

        cl_uint           wkgrp_size;          // For kernel launch configuration
        cl_uint           wkgrp_count;         // For kernel launch configuration

        clrandRngType     rng_type;            // Name of PRNG
        std::string       rng_name;            // Name of PRNG
        const char*       rng_precision;       // Precision of PRNG
        std::string       rng_source;          // Kernel source code of PRNG

        ulong             seedVal;             // Seed value used to generate seed array
        ulong*            seedArrayHost;       // Array of seed values used to seed the PRNGs (host side)
        cl_mem            seedArray_id;        // Array of seed values used to seed the PRNGs (device side)
        cl::Buffer        seedArray;           // Array of seed values used to seed the PRNGs (device side)

        bool              init_flag;           // Flag for whether stream object has been initialized
        bool              source_ready;        // Flag for whether kernel source is built
        bool              program_ready;       // Flag for whether kernel program is compiled
        bool              generator_ready;     // Flag for whether the stream object has been completely set up
        bool              seeded;              // Flag for whether PRNG has been seeded
        bool              buffers_ready;       // Flag for whether the temporary output buffers are ready

        void LookupPRNG();
        void generateBufferKernel(std::string type);
        cl_int fillBuffer();
        void SetStateSize();
        cl_int PrivateGenerateStream(); // To implement

    public:
        clRAND();
        ~clRAND();

        void Init(cl_device_id dev_id, cl_context ctx_id, clrandRngType rng_type_);
        cl_device_id GetStreamDevice() { return device_id; }
        cl_context GetStreamContext() { return context_id; }
        cl_command_queue GetStreamQueue() { return com_queue_id; }

        void BuildSource();
        std::string GetSource() { return this->rng_source; }

        cl_int BuildKernelProgram();
        cl_int ReadyGenerator();
        cl_int SetupWorkConfigurations();
        cl_int SeedGenerator();
        cl_int SeedGeneratorByArray();
        size_t GetNumberOfRNGs() { return (this->wkgrp_size * this->wkgrp_count); }

        cl_int SetupStreamBuffers(size_t bufMult, size_t numPRNGs);
        cl_int FillBuffer();

        bool GetStateOfStateBuffer() { return this->loaded_state; }
        size_t GetStateStructSize() { return this->state_size; }
        size_t GetStateBufferSize() { return (this->state_size * this->wkgrp_size * this->wkgrp_count); }
        cl_int CopyStateToDevice();
        cl_int CopyStateToHost(void* hostPtr);
        void* GetHostStatePtr() { return this->local_state_mem; }

        size_t GetNumBufferEntries() { return this->total_count; }
        void SetNumBufferEntries(size_t num) { this->total_count = num; }

        size_t GetNumValidEntries() { return this->valid_count; }
        void SetNumValidEntries(size_t num) { this->valid_count = num; }

        void SetBufferOffset(size_t ptr);
        size_t GetBufferOffset();

        std::string GetPrecision() { return std::string(this->rng_precision); }
        int SetPrecision(const char * precision);

        clrandRngType GetName() { return this->rng_type; }
        void SetRNGType(clrandRngType rng_type_);
        std::string GetRNGName() { return this->rng_name; }

        ulong GetSeed() { return this->seedVal; }
        void SetSeed(ulong seed);

        cl_uint GetWorkgroupSize() { return this->wkgrp_size; }
        cl_uint GetNumberOfWorkgroups() { return this->wkgrp_count; }

        bool IsInitialized() { return this->init_flag; }
        bool IsSourceReady() { return this->source_ready; }
        bool IsProgramReady() { return this->program_ready; }
        bool IsGeneratorReady() { return this->generator_ready; }
        bool IsSeeded() { return this->seeded; }

	cl_int CopyBufferEntries(cl_mem dst, size_t dst_offset, size_t count);
        bool SetReady() { this->generator_ready = true; }

};

// External functions
#ifdef __cplusplus
extern "C" {
#endif
CLRAND_DLL clRAND* clrand_create_stream();

CLRAND_DLL cl_int clrand_initialize_prng(clRAND* p, cl_device_id dev_id, cl_context ctx_id, clrandRngType rng_type_);

CLRAND_DLL const char * clrand_get_prng_precision(clRAND* p) {
    return (*p).GetPrecision().c_str();
}

CLRAND_DLL int clrand_set_prng_precision(clRAND* p, const char* precision) {
    return (*p).SetPrecision(precision);
}

CLRAND_DLL const char * clrand_get_prng_name(clRAND* p) {
    std::string tmp = (*p).GetRNGName();
    return tmp.c_str();
}

CLRAND_DLL cl_int clrand_set_prng_name(clRAND* p, clrandRngType rng_type) {
    (*p).SetRNGType(rng_type);
    (*p).BuildSource();
    return (*p).BuildKernelProgram();
}

CLRAND_DLL void clrand_set_prng_seed(clRAND* p, ulong seedNum) {
    (*p).SetSeed(seedNum);
}

CLRAND_DLL cl_int clrand_ready_stream(clRAND* p) {
    return (*p).ReadyGenerator();
}

CLRAND_DLL cl_int clrand_generate_stream(clRAND* p, int count, cl_mem dst);

#ifdef __cplusplus
}
#endif
