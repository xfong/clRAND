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
		
		void generateBufferKernel(string name, string type, string src);
		void generateBufferKernelLocal(string name, string type, string src);

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