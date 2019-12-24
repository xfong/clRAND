// Utility functions for example files to
// demonstrate use of clPRNG library
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <getopt.h>
#include "util.h"

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

// Function to update data stucture with information about targeted
// GPU device from all GPU devices
int chooseDeviceInPlatforms(int gpu_number, oclStruct* oclStructure) {
    cl_int err;
    cl_platform_id target_platform;
    cl_device_id   target_device;

    cl_uint num_platforms = 0; // To track number of platforms found
    cl_uint device_num = 0;
    cl_uint num_devices = 0;   // To track devices found in each platform
    int device_count = 0;      // To track overall number of devices

    if (gpu_number < 1) {
        fprintf(stderr, "gpu_number must be a positive non-zero integer!\n");
        return -1;
    }

    // Find all platforms
    err = clGetPlatformIDs( 1, &target_platform, &num_platforms );
    if (err != 0) {
        printf("Cannot get platform!\n");
        return -1;
    }
    cl_platform_id* platform_list = (cl_platform_id*) malloc( num_platforms * sizeof(cl_platform_id) );
    err = clGetPlatformIDs( num_platforms, platform_list, NULL );
    // Go through each platform and find wanted GPU device
    for (cl_uint idx0 = 0; idx0 < num_platforms; idx0++) {
        // Find all devices on the platform
        err = clGetDeviceIDs(platform_list[idx0], CL_DEVICE_TYPE_GPU, 1, &target_device, &num_devices);
        if (err) {
            fprintf(stderr, "Error getting device IDs!\n");
                return -1;
        }
        if (num_devices == 0) { // Go to next platform if no GPUs found for current platform
            continue;
        }
        cl_device_id* device_list = (cl_device_id*) malloc(num_devices * sizeof(cl_device_id));
        err = clGetDeviceIDs(platform_list[idx0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);
        // Go through each device until we get to the one we want
        for (cl_uint idx1 = 0; idx1 < num_devices; idx1++) {
            device_num++; // Increment GPU found

            // We will always remember the first GPU found
            // If the index of another GPU we find matches the
            // GPU index that is targeted, we overwrite the earlier information
            if ((device_num == 1) || (device_num == gpu_number)) {
                target_platform = platform_list[idx0];
                target_device = device_list[idx1];

                // Stop going through other devices if we found the targeted GPU
                if (device_num == gpu_number) {
                    break;
                }
            }
        }
        // Release memory for device list of each platform
        free(device_list);
        if (device_num >= gpu_number) {
            break;
        }
    }
    // Release memory for list of platforms
    free(platform_list);

    // Output message if the total number of GPUs is fewer than
    // the GPU index that is being targeted
    if (device_num < gpu_number) {
        printf("Number of available GPUs is %i and less than wanted GPU...\n", device_num);
        printf("Falling back on first GPU found...\n");
    }

    // Output message if there are no GPUs
    if (num_devices == 0) {
        printf("No GPUs found!\n");
        return 0;
    }

    // Tranfer the information to the data structure
    oclStructure->target_platform = target_platform;
    oclStructure->target_device = target_device;
    return 0;
}

// Function to parse input arguments and call function to build
// data structure storing information about the GPU
int makeOclStruct(int argc, char **argv, oclStruct* oclStructure) {
    if (oclStructure == NULL) {
        fprintf(stderr, "Invalid pointer to oclStruct!\n");
        return -1;
    }
    int gpu_number = -1;
    int res;
    if (argc < 2) { // If no arguments were given to program...
        gpu_number = 1; // Target the first GPU found
    } else {
        // Search for '-n #' in list of arguments
        int c;
        char *gnum = NULL;
        while ((c = getopt (argc, argv, "n:")) != -1) {
            switch(c)
                {
                case 'n':
                    gnum = optarg;
                    gpu_number = atoi(gnum);
                    break;
                case '?':
                    if (optopt == 'n') {
                        fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                    } else {
                        if (isprint (optopt))
                            fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                        else
                            fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
                    }
                    return -1;
                default:
                    fprintf(stderr, "Usage: example1 <-n #>\n");
                    fprintf(stderr, "    <optional arguments>:\n");
                    fprintf(stderr, "    -n: must be followed by a positive non-zero number\n");
            }
        }

        // Error check input argument to wanted GPU
        if (gpu_number < 1) {
            fprintf(stderr, "Option -n must be followed by a positive non-zero number\n");
            return -1;
        }
    }
    return chooseDeviceInPlatforms(gpu_number, oclStructure); // Call function to update the data structure
}

// Function to print information about GPU in data structure
int printOclStructInfo(oclStruct* ComputeStructure) {
    // Error check input
    if (ComputeStructure == NULL) {
        fprintf(stderr, "Invalid pointer to oclStruct!\n");
        return -1;
    }

    cl_int err;
    char platform_name[128];
    char device_name[128];
    char vendor_name[128];

    size_t ret_param_size = 0;

    oclStruct tmpStruct = *ComputeStructure;
    err = clGetPlatformInfo(tmpStruct.target_platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, &ret_param_size);
    printf("Platform found: %s\n", platform_name);
    err = clGetDeviceInfo(tmpStruct.target_device, CL_DEVICE_NAME, sizeof(device_name), device_name, &ret_param_size);
    printf("Device found on the above platform: %s\n", device_name);
    err = clGetPlatformInfo(tmpStruct.target_platform, CL_PLATFORM_VENDOR, sizeof(vendor_name), vendor_name, &ret_param_size);
    printf("Vendor for the above platform: %s\n", vendor_name);

    return 0;
}

// Set up context and queue in the structure
cl_int initStructure(oclStruct* ComputeStructure) {
    if (ComputeStructure->init_flag) {
        return 0; // Already initialized and nothing to do
    }
    cl_int err = 0;
    ComputeStructure->ctx = clCreateContext( NULL, 1, &ComputeStructure->target_device, NULL, NULL, &err );
    if (err) {
        fprintf(stderr,"Error creating context!\n");
        return err;
    }
    ComputeStructure->queue = clCreateCommandQueue( ComputeStructure->ctx, ComputeStructure->target_device, NULL, &err);
    if (err) {
        fprintf(stderr,"Error creating command queue!\n");
        return err;
    }
    ComputeStructure->init_flag = true;
    return err;
}

// Delete and free data structure
cl_int freeStructure(oclStruct* ComputeStructure) {
    if (ComputeStructure->init_flag) {
        cl_int err;
        err = clReleaseCommandQueue(ComputeStructure->queue);
        if (err) {
            fprintf(stderr,"Error releasing command queue!\n");
            return err;
        }
        err = clReleaseContext(ComputeStructure->ctx);
        if (err) {
            fprintf(stderr,"Error releasing context!\n");
            return err;
        }
        err = clReleaseDevice(ComputeStructure->target_device);
        if (err) {
            fprintf(stderr,"Error releasing GPU device!\n");
            return err;
        }
        return err;
    }
    return 0;
}
