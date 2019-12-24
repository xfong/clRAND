#include "clPRNG.hpp"

// Main C interface functions

extern "C" ClPRNG create_clPRNG_stream() {
	static ClPRNG p;
    return p;
}

extern "C" void init_prng(ClPRNG* p, cl_device_id dev_id, const char *name) {
    p->Init(dev_id, name);
}

// Default constructor and destructor
ClPRNG::ClPRNG() {
    init_flag = false;
	source_code = false;
}

ClPRNG::~ClPRNG() {
    cl_int err;
	if (source_code) {
		err = clReleaseMemObject(stateBuffer);
		err = clReleaseMemObject(tmpOutputBuffer);
		err = clReleaseKernel(generate_bitstream);
		err = clReleaseKernel(seed_rng);
		err = clReleaseProgram(rng_program);
	}
	if (init_flag) {
		err = clReleaseCommandQueue(com_queue);
		err = clReleaseContext(context);
		err = clReleaseDevice(device);
	}
}

void ClPRNG::Init(cl_device_id dev_id, const char *name) {
    device = dev_id;
    cl_int err;
    context = clCreateContext(NULL, cl_uint(1), &dev_id, NULL, NULL, &err);
	if (err) {
		std::cout << "ERROR: Unable to create context!" << std::endl;
		return;
	}
	com_queue = clCreateCommandQueue(context, device, NULL, &err);
	if (err) {
		std::cout << "ERROR: Unable to create command queue!" << std::endl;
		return;
	}
	init_flag = true;
}

void ClPRNG::generateBufferKernel(std::string name, std::string type, std::string src) {
	static std::string tmp = std::string((type=="double") ? " #pragma OPENCL EXTENSION cl_khr_fp64 : enable \n" : "");
	tmp += "\n"
		"kernel void seed_prng(global ulong* seed, global " + name + "_state* stateBuf){\n"
		"    uint gid=get_global_id(0);\n"
		"    uint gsize=get_global_size(0);\n"
		"    " + name + "_state state;\n"
		"    " + name + "_seed(&state,seed[gid]);\n"
		"    stateBuf[gid] = state;\n" 
		"}"
		"\n"
		"kernel void generate(uint num, global " + name + "_state* stateBuf, global " + type + "* res){\n"
		"    uint gid=get_global_id(0);\n"
		"    uint gsize=get_global_size(0);\n"
		"    " + name + "_state state;\n"
		"    state = stateBuf[gid];\n"
		"    for(uint i=gid;i<num;i+=gsize){\n"
		"        res[i]=" + name + "_" + type + "(state);\n"
		"    }\n"
		"}";
	src = tmp;
}

void ClPRNG::generateBufferKernelLocal(std::string name, std::string type, std::string src) {
	static std::string tmp = std::string((type == "double") ? " #pragma OPENCL EXTENSION cl_khr_fp64 : enable \n" : "");
	tmp += "\n"
		"kernel void seed_prng(uint num, global ulong* seed, global " + name + "_state* state){\n"
		"    uint gid=get_global_id(0);\n"
		"    uint gsize=get_global_size(0);\n"
		"\n"
		"    " + name + "_seed(state,seed[gid]);\n"
		"}";
		"\n"
		"kernel void generate(uint num, global ulong* seed, global " + type + "* res, global " + name + "_state* stateBuf, local " + name + "_state* state){\n"
		"    uint gid=get_global_id(0);\n"
		"    uint gsize=get_global_size(0);\n"
		"\n"
		"    state = stateBuf;\n"
		"    uint num_gsize = ((num - 1) / gsize + 1)*gsize; //next multiple of gsize, larger or equal to N\n"
		"    for (int i = gid; i<num_gsize; i += gsize) {\n"
		"        " + type + " val = " + name + "_" + type + "(state); //all threads within workgroup must call generator, even if result is not needed!\n"
		"        if (i<num) {\n"
		"     	     res[i] = val;\n"
		"     	 }\n"
		"    }\n"
		"}";
	src = tmp;
}
