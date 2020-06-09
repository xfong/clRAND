#include "clrand.hpp"
#include "../generator/common/float.hpp"
#include "../generator/common/double.hpp"
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
#include "../generator/mtgp32.hpp"
#include "../generator/mtgp64.hpp"
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

// Main C interface functions

// Main call to create a stream object
clRAND* CLRAND_DLL clrand_create_stream() {
    return new clRAND;
}

// Main call to generate stream in the stream object
cl_int CLRAND_DLL clrand_generate_stream(clRAND* p, int count, cl_mem dst) {
    if (count < 0) {
        std::cout << "ERROR: count must be a positive integer!" << std::endl;
        return -1;
    }
    cl_int err;
    if (p->IsInitialized() && p->IsSourceReady() && p->IsProgramReady() && p->IsSeeded() == false) {
        std::cout << "ERROR: stream is not fully ready!" << std::endl;
        return -2;
    }
    for (; count > 0;) {
        if ((count == 0) || (dst == NULL)) {
            break;
        }
        if (p->GetNumValidEntries() <= 0) {
            if (p->GetStateOfStateBuffer() == false) {
                err = p->CopyStateToDevice();
                if (err) {
                    return err;
                }
            }
            err = p->FillBuffer();
            if (err) {
                std::cout << "ERROR: unable to generate random bit stream!" << std::endl;
                if (p->GetStateOfStateBuffer()) {
                    err = p->CopyStateToHost(p->GetHostStatePtr());
                    if (err) {
                        return err;
                    }
                }
                return err;
            }
            p->SetNumValidEntries(p->GetNumBufferEntries());
            p->SetBufferOffset(0);
        }
        size_t dst_offset = 0;
        if (count <= p->GetNumValidEntries()) {
	    err = p->CopyBufferEntries(dst, dst_offset, (size_t)(count));
            if (err) {
                std::cout << "ERROR: unable to copy random bit stream from buffer to dst!" << std::endl;
                if (p->GetStateOfStateBuffer()) {
                    err = p->CopyStateToHost(p->GetHostStatePtr());
                    if (err) {
                        return err;
                    }
                }
                return err;
            }
            p->SetNumValidEntries(p->GetNumValidEntries() - count);
            p->SetBufferOffset(count + p->GetBufferOffset());
            break;
        } else {
	    err = p->CopyBufferEntries(dst, dst_offset, p->GetNumValidEntries());
	    if (err) {
                if (p->GetStateOfStateBuffer()) {
                    err = p->CopyStateToHost(p->GetHostStatePtr());
                    if (err) {
                        return err;
                    }
                }
	        return err;
	    }
            count -= p->GetNumValidEntries();
            dst_offset += p->GetNumValidEntries();
            p->SetNumValidEntries(0);
        }
    }
    if (p->GetStateOfStateBuffer()) {
        err = p->CopyStateToHost(p->GetHostStatePtr());
        if (err) {
            return err;
        }
    }
    return err;
}

// Main call to initialize the stream object
cl_int CLRAND_DLL clrand_initialize_prng(clRAND* p, cl_device_id dev_id, cl_context ctx_id, clrandRngType rng_type_) {
    (*p).Init(dev_id, ctx_id, rng_type_);
    (*p).BuildSource();
    return (*p).BuildKernelProgram();
}

// Default constructor
clRAND::clRAND() {
    device = 0;
    context = 0;
    com_queue = 0;
    total_count = 0;
    valid_count = 0;
    seedVal = (ulong)(time(NULL));
    offset = 0;
    local_state_mem = NULL;
    wkgrp_size = 0;
    wkgrp_count = 0;
    init_flag = false;
    source_ready = false;
    rng_name = "mt19937";
    rng_precision = "ulong";
    rng_source = "";
}

// Default destructor
clRAND::~clRAND() {
}

// Internal function to initialize the stream object
void clRAND::Init(cl_device_id dev_id, cl_context ctx_id, clrandRngType rng_type_) {
    this->device_id = dev_id;
    this->device = device_id;
    this->context_id = ctx_id;
    this->context = context_id;
    cl_int err;
    cl_device_fp_config device_fp_config;
    err = clGetDeviceInfo(dev_id, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cl_device_fp_config), &device_fp_config, NULL);
    if (err) {
        std::cout << "ERROR: Unable to query device for cl_double support!" << std::endl;
        return;
    }
    this->fp64_flag = (device_fp_config > 0);
    this->com_queue_id = clCreateCommandQueue(context_id, device_id, NULL, &err);
    if (err) {
        std::cout << "ERROR: Unable to create command queue!" << std::endl;
        return;
    }
    this->com_queue = com_queue_id;
    this->SetRNGType(rng_type_);
    this->rng_precision = "uint";
    this->init_flag = true;
    this->source_ready = false;
    this->program_ready = false;
    this->generator_ready = false;
    this->seeded = false;
}

// Internal function to set the precision of the random
// numbers generated by the stream object
int clRAND::SetPrecision(const char * precision) {
    this->source_ready = false;
    this->program_ready = false;
    this->generator_ready = false;
    this->seeded = false;
    std::string str = std::string(precision);
    if ((str == "uint") || (str == "ulong") || (str == "float") || (str == "double")) {
        this->rng_precision = precision;
    } else {
        fprintf(stderr, "Can only generate numbers of types: uint, ulong, float, double!");
        return -1;
    }
    return 0;
}

void clRAND::SetRNGType(clrandRngType rng_type_) {
    this->rng_type = rng_type_;
    this->LookupPRNG();
}

// Internal function that maps RNG names to an integer id
void clRAND::LookupPRNG() {
    switch (this->rng_type) {
        case CLRAND_GENERATOR_ISAAC :
            this->rng_name = "isaac";
            break;
        case CLRAND_GENERATOR_KISS09 :
            this->rng_name = "kiss09";
            break;
        case CLRAND_GENERATOR_KISS99 :
            this->rng_name = "kiss99";
            break;
        case CLRAND_GENERATOR_LCG6432 :
            this->rng_name = "lcg6432";
            break;
        case CLRAND_GENERATOR_LCG12864 :
            this->rng_name = "lcg12864";
            break;
        case CLRAND_GENERATOR_LFIB :
            this->rng_name = "lfib";
            break;
        case CLRAND_GENERATOR_MRG31K3P :
            this->rng_name = "mrg31k3p";
            break;
        case CLRAND_GENERATOR_MRG32K3A :
            this->rng_name = "mrg32k3a";
            break;
        case CLRAND_GENERATOR_MRG63K3A :
            this->rng_name = "mrg63k3a";
            break;
        case CLRAND_GENERATOR_MSWS :
            this->rng_name = "msws";
            break;
        case CLRAND_GENERATOR_MT11213 :
            this->rng_name = "mt11213";
            break;
        case CLRAND_GENERATOR_MT19937 :
            this->rng_name = "mt19937";
            break;
        case CLRAND_GENERATOR_MT23209 :
            this->rng_name = "mt23209";
            break;
        case CLRAND_GENERATOR_MT44497 :
            this->rng_name = "mt44497";
            break;
        case CLRAND_GENERATOR_MWC64X :
            this->rng_name = "mwc64x";
            break;
        case CLRAND_GENERATOR_PCG6432 :
            this->rng_name = "pcg6432";
            break;
        case CLRAND_GENERATOR_PHILOX2X32_10 :
            this->rng_name = "philox2x32_10";
            break;
        case CLRAND_GENERATOR_PHILOX4X32_10 :
            this->rng_name = "philox4x32_10";
            break;
        case CLRAND_GENERATOR_RAN2 :
            this->rng_name = "ran2";
            break;
        case CLRAND_GENERATOR_SOBOL32 :
            this->rng_name = "sobol32";
            break;
        case CLRAND_GENERATOR_THREEFRY :
            this->rng_name = "threefry";
            break;
        case CLRAND_GENERATOR_TINYMT32 :
            this->rng_name = "tinymt32";
            break;
        case CLRAND_GENERATOR_TINYMT64 :
            this->rng_name = "tinymt64";
            break;
        case CLRAND_GENERATOR_TYCHE :
            this->rng_name = "tyche";
            break;
        case CLRAND_GENERATOR_TYCHE_I :
            this->rng_name = "tyche_i";
            break;
        case CLRAND_GENERATOR_WELL512 :
            this->rng_name = "well512";
            break;
        case CLRAND_GENERATOR_XORSHIFT1024 :
            this->rng_name = "xorshift1024";
            break;
        case CLRAND_GENERATOR_XORSHIFT6432STAR :
            this->rng_name = "xorshift6432star";
            break;
        case CLRAND_GENERATOR_XORWOW :
            this->rng_name = "xorwow";
            break;
        default :
            std::cout << "Unknown PRNG. No implementation found!" << std::endl;
            break;
    }
}

// Internal function to generate the kernel codes for the PRNGs
void clRAND::generateBufferKernel() {
    this->source_ready = false;
    this->program_ready = false;
    this->generator_ready = false;
    this->rng_source = std::string((this->fp64_flag == true) ? " #pragma OPENCL EXTENSION cl_khr_fp64 : enable \n" : "");
    switch(this->rng_type) {
        case CLRAND_GENERATOR_ISAAC :
            this->rng_source += isaac_prng_kernel;
            break;
        case CLRAND_GENERATOR_KISS09 :
            this->rng_source += kiss09_prng_kernel;
            break;
        case CLRAND_GENERATOR_KISS99 :
            this->rng_source += kiss99_prng_kernel;
            break;
        case CLRAND_GENERATOR_LCG6432 :
            this->rng_source += lcg6432_prng_kernel;
            break;
        case CLRAND_GENERATOR_LCG12864 :
            this->rng_source += lcg12864_prng_kernel;
            break;
        case CLRAND_GENERATOR_LFIB :
            this->rng_source += lfib_prng_kernel;
            break;
        case CLRAND_GENERATOR_MRG31K3P :
            this->rng_source += mrg31k3p_prng_kernel;
            break;
        case CLRAND_GENERATOR_MRG32K3A :
            this->rng_source += mrg32k3a_prng_kernel;
            break;
        case CLRAND_GENERATOR_MRG63K3A :
            this->rng_source += mrg63k3a_prng_kernel;
            break;
        case CLRAND_GENERATOR_MSWS :
            this->rng_source += msws_prng_kernel;
            break;
        case CLRAND_GENERATOR_MT11213 :
		    this->rng_source += "#define MTGP32_MEXP 11213\n";
            this->rng_source += mtgp32_prng_kernel;
		    this->rng_source += "#define MTGP64_MEXP 11213\n";
            this->rng_source += mtgp64_prng_kernel;
            break;
        case CLRAND_GENERATOR_MT19937 :
            this->rng_source += mt19937_prng_kernel;
            break;
        case CLRAND_GENERATOR_MT23209 :
		    this->rng_source += "#define MTGP32_MEXP 23209\n";
            this->rng_source += mtgp32_prng_kernel;
		    this->rng_source += "#define MTGP64_MEXP 23209\n";
            this->rng_source += mtgp64_prng_kernel;
            break;
        case CLRAND_GENERATOR_MT44497 :
		    this->rng_source += "#define MTGP32_MEXP 44497\n";
            this->rng_source += mtgp32_prng_kernel;
		    this->rng_source += "#define MTGP64_MEXP 44497\n";
            this->rng_source += mtgp64_prng_kernel;
            break;
        case CLRAND_GENERATOR_MWC64X :
            this->rng_source += mwc64x_prng_kernel;
            break;
        case CLRAND_GENERATOR_PCG6432 :
            this->rng_source += pcg6432_prng_kernel;
            break;
        case CLRAND_GENERATOR_PHILOX2X32_10 :
            this->rng_source += philox2x32_10_prng_kernel;
            break;
        case CLRAND_GENERATOR_PHILOX4X32_10 :
            this->rng_source += philox4x32_10_prng_kernel;
            break;
        case CLRAND_GENERATOR_RAN2 :
            this->rng_source += ran2_prng_kernel;
            break;
        case CLRAND_GENERATOR_SOBOL32 :
            this->rng_source += sobol32_prng_kernel;
            break;
        case CLRAND_GENERATOR_THREEFRY :
            this->rng_source += threefry_prng_kernel;
            break;
        case CLRAND_GENERATOR_TINYMT32 :
            this->rng_source += tinymt32_prng_kernel;
            break;
        case CLRAND_GENERATOR_TINYMT64 :
            this->rng_source += tinymt64_prng_kernel;
            break;
        case CLRAND_GENERATOR_TYCHE :
            this->rng_source += tyche_prng_kernel;
            break;
        case CLRAND_GENERATOR_TYCHE_I :
            this->rng_source += tyche_i_prng_kernel;
            break;
        case CLRAND_GENERATOR_WELL512 :
            this->rng_source += well512_prng_kernel;
            break;
        case CLRAND_GENERATOR_XORSHIFT1024 :
            this->rng_source += xorshift1024_prng_kernel;
            break;
        case CLRAND_GENERATOR_XORSHIFT6432STAR :
            this->rng_source += xorshift6432star_prng_kernel;
            break;
        case CLRAND_GENERATOR_XORWOW :
            this->rng_source += xorwow_prng_kernel;
            break;
        default :
            std::cout << "Unknown PRNG. No implementation found!" << std::endl;
            break;
    }
    switch(this->rng_type) {
        case CLRAND_GENERATOR_XORSHIFT1024 :
            this->rng_source += "\n"
                   "kernel void seed_prng_by_value(ulong seedVal, global " + this->rng_name + "_state* stateBuf, local " + this->rng_name + "_state* local_state){\n"
                   "    uint gid=get_global_id(0);\n"
                   "    ulong seed = (ulong)(gid);\n"
                   "    seed <<= 1;\n"
                   "    seed += seedVal;\n"
                   "    if (seed == 0) {\n"
                   "        seed += 1;\n"
                   "    }\n"
                   "\n"
                   "    " + this->rng_name + "_seed(local_state,seed);\n"
                   "    stateBuf[gid] = local_state[get_local_id(0)];\n"
                   "}\n"
                   "\n"
                   "kernel void seed_prng_by_array(global ulong* seedArr, global " + this->rng_name + "_state* stateBuf, local " + this->rng_name + "_state* local_state){\n"
                   "    uint gid=get_global_id(0);\n"
                   "    ulong seed = seedArr[gid];\n"
                   "\n"
                   "    " + this->rng_name + "_seed(local_state,seed);\n"
                   "    stateBuf[gid] = local_state[get_local_id(0)];\n"
                   "}\n"
                   "\n"
                   "kernel void generate_uint(uint num, global ulong* seed, global uint* res, global " + this->rng_name + "_state* stateBuf, local " + this->rng_name + "_state* state){\n"
                   "    uint gid=get_global_id(0);\n"
                   "    uint gsize=get_global_size(0);\n"
                   "\n"
                   "    state[get_local_id(0)] = stateBuf[gid];\n"
                   "    uint num_gsize = ((num - 1) / gsize + 1)*gsize; //next multiple of gsize, larger or equal to N\n"
                   "    for (int i = gid; i<num_gsize; i += gsize) {\n"
                   "        uint val = " + this->rng_name + "_uint(state); //all threads within workgroup must call generator, even if result is not needed!\n"
                   "        if (i<num) {\n"
                   "            res[i] = val;\n"
                   "        }\n"
                   "    }\n"
                   "}\n"
                   "\n"
                   "kernel void generate_ulong(uint num, global ulong* seed, global ulong* res, global " + this->rng_name + "_state* stateBuf, local " + this->rng_name + "_state* state){\n"
                   "    uint gid=get_global_id(0);\n"
                   "    uint gsize=get_global_size(0);\n"
                   "\n"
                   "    state[get_local_id(0)] = stateBuf[gid];\n"
                   "    uint num_gsize = ((num - 1) / gsize + 1)*gsize; //next multiple of gsize, larger or equal to N\n"
                   "    for (int i = gid; i<num_gsize; i += gsize) {\n"
                   "        ulong val = " + this->rng_name + "_ulong(state); //all threads within workgroup must call generator, even if result is not needed!\n"
                   "        if (i<num) {\n"
                   "            res[i] = val;\n"
                   "        }\n"
                   "    }\n"
                   "}";
                   break;
        case CLRAND_GENERATOR_MT11213 :
            break;
        case CLRAND_GENERATOR_MT23209 :
            break;
        case CLRAND_GENERATOR_MT44497 :
            break;
        default :
            this->rng_source += "\n"
                   "kernel void seed_prng_by_value(ulong seedVal, global " + this->rng_name + "_state* stateBuf){\n"
                   "    uint gid=get_global_id(0);\n"
                   "    ulong seed = (ulong)(gid);\n"
                   "    seed <<= 1;\n"
                   "    seed += seedVal;\n"
                   "    if (seed == 0) {\n"
                   "        seed += 1;\n"
                   "    }\n"
                   "    " + this->rng_name + "_state state;\n"
                   "    " + this->rng_name + "_seed(&state,seed);\n"
                   "    stateBuf[gid] = state;\n"
                   "}\n"
                   "\n"
                   "kernel void seed_prng_by_array(global ulong* seedArr, global " + this->rng_name + "_state* stateBuf){\n"
                   "    uint gid=get_global_id(0);\n"
                   "    ulong seed = seedArr[gid];\n"
                   "    " + this->rng_name + "_state state;\n"
                   "    " + this->rng_name + "_seed(&state,seed);\n"
                   "    stateBuf[gid] = state;\n"
                   "}\n"
                   "\n"
                   "kernel void generate_uint(uint num, global " + this->rng_name + "_state* stateBuf, global uint* res){\n"
                   "    uint gid=get_global_id(0);\n"
                   "    uint gsize=get_global_size(0);\n"
                   "    " + this->rng_name + "_state state;\n"
                   "    state = stateBuf[gid];\n"
                   "    for(uint i=gid;i<num;i+=gsize){\n"
                   "        res[i]=" + this->rng_name + "_uint(state);\n"
                   "    }\n"
                   "    stateBuf[gid] = state;\n"
                   "}\n"
                   "\n"
                   "kernel void generate_ulong(uint num, global " + this->rng_name + "_state* stateBuf, global ulong* res){\n"
                   "    uint gid=get_global_id(0);\n"
                   "    uint gsize=get_global_size(0);\n"
                   "    " + this->rng_name + "_state state;\n"
                   "    state = stateBuf[gid];\n"
                   "    for(uint i=gid;i<num;i+=gsize){\n"
                   "        res[i]=" + this->rng_name + "_ulong(state);\n"
                   "    }\n"
                   "    stateBuf[gid] = state;\n"
                   "}";
                   break;
    }
}

// Internal function to build the kernel source codes
// for the program of the stream object
void clRAND::BuildSource() {
    this->source_ready = false;
    this->program_ready = false;
    this->generator_ready = false;
    this->generateBufferKernel();
    this->source_ready = true;
}

// Internal function to compile and link the kernel source
// codes into the program of the stream object
cl_int clRAND::BuildKernelProgram() {
    this->program_ready = false;
    this->generator_ready = false;
    cl_int err;
    if (init_flag && source_ready) {
        cl::Program::Sources sources(1, std::make_pair(rng_source.c_str(), rng_source.length()));
#ifdef DEBUG1
        std::cout << "Preparing to build program..." << std::endl;
#endif
        this->rng_program = cl::Program(context, sources);
        std::string build_args = "-cl-std=CL1.2 -cl-kernel-arg-info";
        err = this->rng_program.build(std::vector<cl::Device>({device}), build_args.c_str());
#ifdef DEBUG1
        std::cout << "Built program..." << std::endl;
#endif
        if (err) {
            std::cout << "ERROR: Unable to build PRNG program!" << std::endl;
            return err;
        }
#ifdef DEBUG1
        std::cout << "Create kernel to seed PRNG by one value..." << std::endl;
#endif
        this->seed_rng = cl::Kernel(rng_program, "seed_prng_by_value");
#ifdef DEBUG1
        std::cout << "Create kernel to seed PRNG by array of values..." << std::endl;
#endif
        this->seed_rng_array = cl::Kernel(rng_program, "seed_prng_by_array");
#ifdef DEBUG1
        std::cout << "Create kernel to generate random bitstream..." << std::endl;
#endif
        this->generate_bitstream = cl::Kernel(rng_program, "generate_uint");
        this->generate_streamUL = cl::Kernel(rng_program, "generate_ulong");
        this->program_ready = true;
        return err;
    }
    std::cout << "ERROR: Stream object not initialized or kernel source code has not been built!" << std::endl;
    return -1;
}

// Internal function that is the second phase of the initialization step.
// The number of workitems per workgroup supported by the device is
// discovered. The buffers of the PRNG are created to support the number
// of workitems that can be launched in parallel. The kernel launch
// configuration is also built
cl_int clRAND::ReadyGenerator() {
    // Query number of workitems and workgroups supported by the device
    cl_int err;
    err = this->SetupWorkConfigurations();
    if (err) {
        std::cout << "ERROR: unable to set up workgroup configuration!" <<std::endl;
        return err;
    }

    // Initialize the counters that tracks available random number generators
    size_t numPRNGs = (size_t)(this->wkgrp_count * this->wkgrp_size);
    size_t bufMult = 34;

    err = this->SetupStreamBuffers(bufMult, numPRNGs);

    // At this point the buffers are set up...
    // Seed the PRNG
    this->generator_ready = true;
    err = this->SeedGenerator();
    if (err) {
        std::cout << "ERROR: failed to seed PRNG" << std::endl;
        this->generator_ready = false;
        return err;
    }

    // Generate a set of random numbers to fill the temporary buffer
    err = this->FillBuffer();
    if (err) {
        std::cout << "ERROR: failed to fill temporary buffer while readying PRNG" << std::endl;
        this->generator_ready = false;
        return err;
    }

    // Initialize counters that track number of valid random numbers
    // in the temporary buffer
    this->offset = 0;
    this->valid_count = bufMult * numPRNGs;

    return err;
}

cl_int clRAND::SetupWorkConfigurations() {
    // Query number of workitems and workgroups supported by the device
    cl_int err;
    this->wkgrp_size = this->device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);
    if (err) {
        std::cout << "ERROR: failed to get max workgroup size on device!" << std::endl;
        return err;
    }
    this->wkgrp_count = this->device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(&err);
    if (err) {
        std::cout << "ERROR: failed to get max number of compute unites on device!" << std::endl;
        return err;
    }

    // Upper limit for number of workitems per workgroup is set at 256...
    if (this->wkgrp_size > 256) {
        this->wkgrp_size = 256;
    }

    // Upper limit for number of workitems per workgroup is lowered if
    // the PRNG is based on xorshift1024
    if (std::string(this->rng_name) == "xorshift1024") {
        this->wkgrp_size = 32;
    }
    return err;
}

cl_int clRAND::SetupStreamBuffers(size_t bufMult, size_t numPRNGs) {
    cl_int err;
    // Determine the number of bytes for each random number generated
    // by a workitem
    size_t typeSize = 4;
    bool typeDetection = ((std::string)(this->rng_precision) == "uint") || 
                         ((std::string)(this->rng_precision) == "ulong") || 
                         ((std::string)(this->rng_precision) == "float") || 
                         ((std::string)(this->rng_precision) == "double");
    if (((std::string)(this->rng_precision) == "double") || ((std::string)(this->rng_precision) == "ulong")) {
        typeSize = 8;
    } else if (typeDetection != true) {
        std::cout << "ERROR: Unknown rng_precision detected!" << std::endl;
        return -1;
    }

    // Create the buffer storing the states of the PRNGs
    this->SetStateSize();
    size_t stateBufSize = numPRNGs * this->state_size;
    this->stateBuffer_id = clCreateBuffer(this->context_id, CL_MEM_READ_WRITE, stateBufSize, NULL, &err);
    if (err) {
        std::cout << "ERROR: Unable to create state buffer or PRNG!" << std::endl;
        return err;
    }
    this->stateBuffer = stateBuffer_id;

    // Create the temporary buffer in which random numbers are generated.
    // These numbers will be copied to the desired destination when required.
    this->total_count = bufMult * numPRNGs;
    this->tmpOutputBuffer_id = clCreateBuffer(this->context_id, CL_MEM_READ_WRITE, this->total_count * typeSize, NULL, &err);
    if (err) {
        std::cout << "ERROR: Unable to create temporary buffer or PRNG!" << std::endl;
        return err;
    }
    this->tmpOutputBuffer = tmpOutputBuffer_id;
    return err;
}

// Function to set the size of the state for each PRNG
void clRAND::SetStateSize() {
    this->generator_ready = false;
    switch(this->rng_type)
    {
        case CLRAND_GENERATOR_ISAAC:
            this->state_size = sizeof(isaac_state);
            break;
        case CLRAND_GENERATOR_KISS09:
            this->state_size = sizeof(kiss09_state);
            break;
        case CLRAND_GENERATOR_KISS99:
            this->state_size = sizeof(kiss99_state);
            break;
        case CLRAND_GENERATOR_LCG6432:
            this->state_size = sizeof(lcg6432_state);
            break;
        case CLRAND_GENERATOR_LCG12864:
            this->state_size = sizeof(lcg12864_state);
            break;
        case CLRAND_GENERATOR_LFIB:
            this->state_size = sizeof(lfib_state);
            break;
        case CLRAND_GENERATOR_MRG31K3P:
            this->state_size = sizeof(mrg31k3p_state);
            break;
        case CLRAND_GENERATOR_MRG32K3A:
            this->state_size = sizeof(mrg32k3a_state);
            break;
        case CLRAND_GENERATOR_MRG63K3A:
            this->state_size = sizeof(mrg63k3a_state);
            break;
        case CLRAND_GENERATOR_MSWS:
            this->state_size = sizeof(msws_state);
            break;
        case CLRAND_GENERATOR_MT19937:
            this->state_size = sizeof(mt19937_state);
            break;
        case CLRAND_GENERATOR_MWC64X:
            this->state_size = sizeof(mwc64x_state);
            break;
        case CLRAND_GENERATOR_PCG6432:
            this->state_size = sizeof(pcg6432_state);
            break;
        case CLRAND_GENERATOR_PHILOX2X32_10:
            this->state_size = sizeof(philox2x32_10_state);
            break;
        case CLRAND_GENERATOR_PHILOX4X32_10:
            this->state_size = sizeof(philox2x32_10_state);
            break;
        case CLRAND_GENERATOR_RAN2:
            this->state_size = sizeof(ran2_state);
            break;
        case CLRAND_GENERATOR_SOBOL32:
            this->state_size = sizeof(sobol32_state);
            break;
        case CLRAND_GENERATOR_THREEFRY:
            this->state_size = sizeof(threefry_state);
            break;
        case CLRAND_GENERATOR_TINYMT32:
            this->state_size = sizeof(tinymt32_state);
            break;
        case CLRAND_GENERATOR_TINYMT64:
            this->state_size = sizeof(tinymt64_state);
            break;
        case CLRAND_GENERATOR_TYCHE:
            this->state_size = sizeof(tyche_state);
            break;
        case CLRAND_GENERATOR_TYCHE_I:
            this->state_size = sizeof(tyche_i_state);
            break;
        case CLRAND_GENERATOR_WELL512:
            this->state_size = sizeof(well512_state);
            break;
        case CLRAND_GENERATOR_XORSHIFT1024:
            this->state_size = sizeof(xorshift1024_state);
            break;
        case CLRAND_GENERATOR_XORSHIFT6432STAR:
            this->state_size = sizeof(xorshift6432star_state);
            break;
        case CLRAND_GENERATOR_XORWOW:
            this->state_size = sizeof(xorwow_state);
            break;
        default :
            this->state_size = 0;
            break;
    }
}

// Sets the current offset of the temporary buffer store
// Entries preceding the offset are invalid
void clRAND::SetBufferOffset(size_t ptr) {
    if ((std::string)(this->rng_precision) == "double") {
        this->offset = ptr * 8;
    } else {
        this->offset = ptr * 4;
    }
}

// Returns the current offset of the temporary buffer
size_t clRAND::GetBufferOffset() {
    if ((std::string)(this->rng_precision) == "double") {
        return (this->offset / 8);
    }
    return (this->offset / 4);
}

// Function to copy random numbers from temporary buffer to desired destination
cl_int clRAND::CopyBufferEntries(cl_mem dst, size_t dst_offset, size_t count) {
    size_t numBytes = 4;
    if (((std::string)(this->rng_precision) == "double") || ((std::string)(this->rng_precision) == "ulong")) {
        numBytes = 8;
    }

    // Copy buffer data in device
    cl_event eventFlag;
    cl_int err = clEnqueueCopyBuffer(this->com_queue_id, this->tmpOutputBuffer_id, dst, this->offset, dst_offset * numBytes, count * numBytes, 0, NULL, &eventFlag);
    if (err) {
        std::cout << "ERROR: unable to copy buffer entries in CopyBufferEntries!" << std::endl;
        return err;
    }

    // Wait for copy in device side to complete before returning
    err = clWaitForEvents(1, &eventFlag);
    if (err) {
        std::cout << "ERROR: unable to wait for copy buffer to complete in CopyBufferEntries!" << std::endl;
        return err;
    }
    return err;
}

// Internal function that seeds the PRNGs in the
// stream object
cl_int clRAND::SeedGenerator() {
    this->seeded = false;
    if (this->init_flag != true) {
        std::cout << "ERROR: stream object has not been initialized!" << std::endl;
        return -1;
    }
    if (this->source_ready != true) {
        std::cout << "ERROR: source for stream object has not been built!" << std::endl;
        return -2;
    }
    if (this->program_ready != true) {
        std::cout << "ERROR: kernel programs for stream object has not been built!" << std::endl;
        return -3;
    }
    if (this->generator_ready != true) {
        std::cout << "ERROR: temporary buffers in stream object has not been set up!" << std::endl;
        return -4;
    }
    cl_int err;
#ifdef DEBUG1
    std::cout << "Setting seedVal" << std::endl;
#endif
    err = this->seed_rng.setArg<ulong>(0, this->seedVal);
    if (err != 0) {
        std::cout << "ERROR: Unable to set first argument to kernel to seed PRNG!" << std::endl;
        return err;
    }
#ifdef DEBUG1
    std::cout << "Setting stateBuffer" << std::endl;
#endif
    err = seed_rng.setArg<cl::Buffer>(1, this->stateBuffer);
    if (err) {
        std::cout << "ERROR: Unable to set second argument to kernel to seed PRNG!" << std::endl;
        return err;
    }
#ifdef DEBUG1
    std::cout << "Allocating local memory for XORSHIFT1024" << std::endl;
#endif
    switch(this->rng_type)
    {
        case CLRAND_GENERATOR_XORSHIFT1024:
            err = this->seed_rng.setArg<cl::LocalSpaceArg>(2, cl::Local((size_t)(xorshift1024_local_mem((size_t)(this->GetWorkgroupSize()))) * (size_t)(sizeof(xorshift1024_state))));
            if (err) {
                std::cout << "ERROR: Unable to set third argument to kernel to seed PRNG!" << std::endl;
                return err;
            }
            break;
        default:
            break;
    }
    cl_event event_id;
    cl::Event event;
#ifdef DEBUG1
    std::cout << "Executing kernel to seed generator" << std::endl;
#endif
    err = this->com_queue.enqueueNDRangeKernel(this->seed_rng, cl::NDRange(0), cl::NDRange((size_t)(this->wkgrp_count * this->wkgrp_size)), cl::NDRange((size_t)(this->wkgrp_size)), NULL, &event);
    if (err) {
        std::cout << "ERROR: Unable to enqueue kernel to seed PRNG!" << std::endl;
        return err;
    }
    std::vector<cl::Event> eventList = { event };
    err = cl::WaitForEvents(eventList);
    if (err) {
        std::cout << "ERROR: Unable to wait for kernel to seed PRNG!" << std::endl;
        return err;
    }
    if (local_state_mem == NULL) {
        static void *tmp_mem = malloc(state_size * (size_t)(this->wkgrp_size * this->wkgrp_count));
        this->local_state_mem = tmp_mem;
    } else {
        free(this->local_state_mem);
        static void *tmp_mem = malloc(state_size * (size_t)(this->wkgrp_size * this->wkgrp_count));
        this->local_state_mem = tmp_mem;
    }
    err = this->CopyStateToHost(this->local_state_mem);
    this->seeded = true;
#ifdef DEBUG1
    std::cout << "Done seeding generator" << std::endl;
#endif
    return err;
}

// Internal function that seeds the PRNGs in the
// stream object
cl_int clRAND::SeedGeneratorByArray() {
    this->seeded = false;
    if (this->init_flag != true) {
        std::cout << "ERROR: stream object has not been initialized!" << std::endl;
        return -1;
    }
    if (this->source_ready != true) {
        std::cout << "ERROR: source for stream object has not been built!" << std::endl;
        return -2;
    }
    if (this->program_ready != true) {
        std::cout << "ERROR: kernel programs for stream object has not been built!" << std::endl;
        return -3;
    }
    if (this->generator_ready != true) {
        std::cout << "ERROR: temporary buffers in stream object has not been set up!" << std::endl;
        return -4;
    }
    cl_int err;
#ifdef DEBUG1
    std::cout << "Setting seed array" << std::endl;
#endif
    err = this->seed_rng_array.setArg<cl::Buffer>(0, this->seedArray);
    if (err != 0) {
        std::cout << "ERROR: Unable to set first argument to kernel to seed PRNG!" << std::endl;
        return err;
    }
#ifdef DEBUG1
    std::cout << "Setting stateBuffer" << std::endl;
#endif
    err = seed_rng_array.setArg<cl::Buffer>(1, this->stateBuffer);
    if (err) {
        std::cout << "ERROR: Unable to set second argument to kernel to seed PRNG!" << std::endl;
        return err;
    }
    cl_event event_id;
    cl::Event event;
#ifdef DEBUG1
    std::cout << "Executing kernel to seed generator" << std::endl;
#endif
    err = this->com_queue.enqueueNDRangeKernel(this->seed_rng_array, cl::NDRange(0), cl::NDRange((size_t)(this->wkgrp_count * this->wkgrp_size)), cl::NDRange((size_t)(this->wkgrp_size)), NULL, &event);
    if (err) {
        std::cout << "ERROR: Unable to enqueue kernel to seed PRNG!" << std::endl;
        return err;
    }
    std::vector<cl::Event> eventList = { event };
    err = cl::WaitForEvents(eventList);
    if (err) {
        std::cout << "ERROR: Unable to wait for kernel to seed PRNG!" << std::endl;
        return err;
    }
    if (local_state_mem == NULL) {
        static void *tmp_mem = malloc(state_size * (size_t)(this->wkgrp_size * this->wkgrp_count));
        this->local_state_mem = tmp_mem;
    } else {
        free(this->local_state_mem);
        static void *tmp_mem = malloc(state_size * (size_t)(this->wkgrp_size * this->wkgrp_count));
        this->local_state_mem = tmp_mem;
    }
    err = this->CopyStateToHost(this->local_state_mem);
    this->seeded = true;
#ifdef DEBUG1
    std::cout << "Done seeding generator" << std::endl;
#endif
    return err;
}

// Internal function that copies the PRNG states from
// host side to device side
cl_int clRAND::CopyStateToDevice() {
    // Copy PRNG states from host side to device side
    cl::Event event;
    cl_int err = this->com_queue.enqueueWriteBuffer(this->stateBuffer, true, 0, this->state_size * (size_t)(this->wkgrp_count * this->wkgrp_size), this->local_state_mem, NULL, &event);
    if (err) {
        std::cout << "ERROR: unable to copy state from host to device!" << std::endl;
        return err;
    }
    std::vector<cl::Event> eventList = { event };
    err = cl::WaitForEvents(eventList);
    if (err) {
        std::cout << "ERROR: unable to wait for copy state from host to device to finish!" << std::endl;
        return err;
    }

}

// Internal function that copies the PRNG states from
// device side to host side
cl_int clRAND::CopyStateToHost(void* hostPtr) {
    // Copy PRNG states from device side back to host side
    cl::Event event;
    cl_int err = this->com_queue.enqueueReadBuffer(this->stateBuffer, true, 0, this->state_size * (size_t)(this->wkgrp_count * this->wkgrp_size), hostPtr, NULL, &event);
    if (err) {
        std::cout << "ERROR: unable to copy state from host to device!" << std::endl;
        return err;
    }
    std::vector<cl::Event> eventList = { event };
    err = cl::WaitForEvents(eventList);
    if (err) {
        std::cout << "ERROR: unable to wait for copy state from host to device to finish!" << std::endl;
        return err;
    }
    return err;
}

// Internal function that generates random stream of uint
// in the stream object by calling the kernel.
cl_int clRAND::FillBuffer() {
    // Set up kernel to generate random bitstream
#ifdef DEBUG1
    std::cout << "Setting total number of generators for kernel argument" << std::endl;
#endif
    cl_int err = this->generate_bitstream.setArg<uint>(0, (uint)(this->total_count));
    if (err) {
        std::cout << "ERROR: Unable to set first argument to kernel to generate bitstream!" << std::endl;
        return err;
    }
#ifdef DEBUG1
    std::cout << "Setting state buffer for kernel argument" << std::endl;
#endif
    err = this->generate_bitstream.setArg<cl::Buffer>(1, this->stateBuffer);
    if (err) {
        std::cout << "ERROR: Unable to set second argument to kernel to generate bitstream!" << std::endl;
        return err;
    }
#ifdef DEBUG1
    std::cout << "Setting output buffer for kernel argument" << std::endl;
#endif
    err = this->generate_bitstream.setArg<cl::Buffer>(2, this->tmpOutputBuffer);
    if (err) {
        std::cout << "ERROR: Unable to set third argument to kernel to generate bitstream!" << std::endl;
        return err;
    }

    // Execute kernel to generate random bitstream
    cl::Event event;
#ifdef DEBUG1
    std::cout << "Executing kernel" << std::endl;
#endif
    err = this->com_queue.enqueueNDRangeKernel(this->generate_bitstream, cl::NDRange(0), cl::NDRange((size_t)(this->wkgrp_count * this->wkgrp_size)), cl::NDRange((size_t)(this->wkgrp_size)), NULL, &event);
    if (err) {
        std::cout << "ERROR: Unable to enqueue kernel to generate bitstream!" << std::endl;
        return err;
    }
    std::vector<cl::Event> eventList = { event };
    err = cl::WaitForEvents(eventList);
    if (err) {
        std::cout << "ERROR: unable to wait for copy state from host to device to finish!" << std::endl;
    }
#ifdef DEBUG1
    std::cout << "Buffer of stream object is filled" << std::endl;
#endif
    return err;
}

// Internal function that generates random stream of ulong
// in the stream object by calling the kernel.
cl_int clRAND::FillBufferUL() {
    // Set up kernel to generate random bitstream
#ifdef DEBUG1
    std::cout << "Setting total number of generators for kernel argument" << std::endl;
#endif
    cl_int err = this->generate_streamUL.setArg<uint>(0, (uint)(this->total_count / 2));
    if (err) {
        std::cout << "ERROR: Unable to set first argument to kernel to generate bitstream!" << std::endl;
        return err;
    }
#ifdef DEBUG1
    std::cout << "Setting state buffer for kernel argument" << std::endl;
#endif
    err = this->generate_streamUL.setArg<cl::Buffer>(1, this->stateBuffer);
    if (err) {
        std::cout << "ERROR: Unable to set second argument to kernel to generate bitstream!" << std::endl;
        return err;
    }
#ifdef DEBUG1
    std::cout << "Setting output buffer for kernel argument" << std::endl;
#endif
    err = this->generate_streamUL.setArg<cl::Buffer>(2, this->tmpOutputBuffer);
    if (err) {
        std::cout << "ERROR: Unable to set third argument to kernel to generate bitstream!" << std::endl;
        return err;
    }

    // Execute kernel to generate random bitstream
    cl::Event event;
#ifdef DEBUG1
    std::cout << "Executing kernel" << std::endl;
#endif
    err = this->com_queue.enqueueNDRangeKernel(this->generate_streamUL, cl::NDRange(0), cl::NDRange((size_t)(this->wkgrp_count * this->wkgrp_size)), cl::NDRange((size_t)(this->wkgrp_size)), NULL, &event);
    if (err) {
        std::cout << "ERROR: Unable to enqueue kernel to generate bitstream!" << std::endl;
        return err;
    }
    std::vector<cl::Event> eventList = { event };
    err = cl::WaitForEvents(eventList);
    if (err) {
        std::cout << "ERROR: unable to wait for copy state from host to device to finish!" << std::endl;
    }
#ifdef DEBUG1
    std::cout << "Buffer of stream object is filled" << std::endl;
#endif
    return err;
}

void clRAND::SetSeed(ulong seed) {
    this->seedVal = seed;
    this->seeded = false;
    cl_int err;
    err = this->SeedGenerator();
    if (err) {
        std::cout << "ERROR: failed to seed generator after setting seed value!" << std::endl;
        return;
    }
    this->seeded = true;
}
