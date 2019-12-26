#include <stdio.h>
#include <unistd.h>
#include <getopt.h>

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

typedef struct oclStruct__ {
    cl_platform_id   target_platform;
    cl_device_id     target_device;
    cl_context       ctx;
    cl_command_queue queue;
    bool             init_flag;
} oclStruct;

// Function to select a particular GPU in all GPUs found
int chooseDeviceInPlatforms(int gpu_number, oclStruct* oclStructure);

// Consolidate information about selected GPU into a data structure
int makeOclStruct(int argc, char **argv, oclStruct* oclStructure);

// Output information about selected GPU
int printOclStructInfo(oclStruct* ComputeStructure);

// Create context and queue in structure
cl_int initStructure(oclStruct* ComputeStructure);

// Delete and free data structure
cl_int freeStructure(oclStruct* ComputeStructure);
