#include <iostream>
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
#endif

// Prototype class
class ClPRNG {
    private:
        cl_device_id      device;
        cl_context        context;
        cl_command_queue  com_queue;

        cl_program        rng_program;
        cl_kernel         seed_rng;
        cl_kernel         generate_bitstream;

        cl_mem            stateBuffer;
        cl_mem            tmpOutputBuffer;
        size_t            valid_cnt;

        size_t            wkgrp_size;
        size_t            wkgrp_count;

        const char*       rng_name;
        std::string       rng_source;

        bool              source_code;
        bool              init_flag;
		
		void generateBufferKernel(std::string name, std::string type, std::string src);
		void generateBufferKernelLocal(std::string name, std::string type, std::string src);

    public:
        void Init(cl_device_id dev_id, const char * name);
        void Seed(uint32_t seed);
        void GenerateStream(cl_mem OutputBuffer);
		bool IsSourceReady() { return source_code; }
		bool IsInitialized() { return init_flag; }
        ClPRNG();
        ~ClPRNG();
};

// Internal functions