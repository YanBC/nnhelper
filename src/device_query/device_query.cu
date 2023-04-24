#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include "argparse.h"
#include "helper_cuda.h"

static const char *const usage[] = {
    "device_query [options]",
    NULL,
};

void pad_string(char *des, char *src, size_t target_length);

void print_general_info(cudaDeviceProp deviceProp);

void print_advanced_features(cudaDeviceProp deviceProp);

void print_programming_features(cudaDeviceProp deviceProp);


void print_device_info(int device_id) {
    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, device_id);
    if (err != cudaSuccess) {
        printf("fail to query gpu #%d\n", device_id);
        return;
    }

    printf("\nDevice %d\n", device_id);
    print_general_info(deviceProp);
    print_programming_features(deviceProp);
    print_advanced_features(deviceProp);
}

int main(int argc, const char **argv) {
    int device_id = -1;
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_INTEGER('d', "device", &device_id, "deivce id, default to 0"),
        OPT_END(),
    };
    struct argparse argparse;
    argparse_init(&argparse, options, usage, 0);
    int argc_remain = argparse_parse(&argparse, argc, argv);
    // printf("num of remaining arguments: %d\n", argc);
    // printf("device id: %d\n", device_id);

    int num_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&num_devices);
    if (err != cudaSuccess) {
        printf("fail to query CUDA devices\n");
        return -1;
    }
    if (device_id != -1 && device_id >= num_devices) {
        printf("GPU #%d does not exit: total number of CUDA devices: %d\n", device_id, num_devices);
        return 2;
    }

    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    if (device_id == -1) {
        for (device_id = 0; device_id < num_devices; device_id++) {
            print_device_info(device_id);
        }
    } else {
        print_device_info(device_id);
    }

    return 0;
}


void print_general_info(cudaDeviceProp deviceProp) {
    // name,
    // integrated,
    // major, minor,
    // multiProcessorCount,
    // clockRate,
    // memoryClockRate,
    // memoryBusWidth,
    // l2CacheSize,

    const int description_len = 68;
    char *padded = (char *)malloc(sizeof(char) * (description_len + 1));

    printf("%s (%s), ", deviceProp.name,
           _ConvertSMVer2ArchName(deviceProp.major, deviceProp.minor));
    if (deviceProp.integrated == 1) {
        printf("integrated\n");
    } else {
        printf("decrete\n");
    }

    pad_string(padded, "  CUDA Capability Major/Minor version number:", description_len);
    printf("%s%d.%d\n", padded, deviceProp.major, deviceProp.minor);

    printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:                        %d CUDA Cores\n",
           deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
           deviceProp.multiProcessorCount);

    pad_string(padded, "  GPU Max Clock rate:", description_len);
    printf("%s%.0f MHz (%0.2f GHz)\n", padded, deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

    pad_string(padded, "  Memory Clock rate:", description_len);
    printf("%s%.0f Mhz\n", padded, deviceProp.memoryClockRate * 1e-3f);

    pad_string(padded, "  Memory Bus Width:", description_len);
    printf("%s%d-bit\n", padded, deviceProp.memoryBusWidth);

    pad_string(padded, "  L2 Cache Size:", description_len);
    printf("%s%d bytes\n", padded, deviceProp.l2CacheSize);

    free(padded);
    return;
}

void print_advanced_features(cudaDeviceProp deviceProp) {
    // computePreemptionSupported,
    // streamPrioritiesSupported,
    // globalL1CacheSupported,
    // localL1CacheSupported,

    // unifiedAddressing,
    // canMapHostMemory,
    // managedMemory,
    // directManagedMemAccessFromHost,
    // asyncEngineCount,

    const int description_len = 68;
    char *padded = (char *)malloc(sizeof(char) * (description_len + 1));

    pad_string(padded, "  Device supports Compute Preemption:", description_len);
    printf("%s%s\n", padded, deviceProp.computePreemptionSupported ? "Yes" : "No");

    pad_string(padded, "  Device supports Stream Priority:", description_len);
    printf("%s%s\n", padded, deviceProp.streamPrioritiesSupported ? "Yes" : "No");

    pad_string(padded, "  Device supports storing globals in L1 cache:", description_len);
    printf("%s%s\n", padded, deviceProp.globalL1CacheSupported ? "Yes" : "No");

    pad_string(padded, "  Device supports storing locals in L1 cache:", description_len);
    printf("%s%s\n", padded, deviceProp.localL1CacheSupported ? "Yes" : "No");

    pad_string(padded, "  Device shares a unified memory address space with host:", description_len);
    printf("%s%s\n", padded, deviceProp.unifiedAddressing ? "Yes" : "No");

    pad_string(padded, "  Device supports mapping host memory into CUDA address space:", description_len);
    printf("%s%s\n", padded, deviceProp.canMapHostMemory ? "Yes" : "No");

    pad_string(padded, "  Device supports managed memory alllocation:", description_len);
    printf("%s%s\n", padded, deviceProp.managedMemory ? "Yes" : "No");

    pad_string(padded, "  Device allows host to access managed mameory directly:", description_len);
    printf("%s%s\n", padded, deviceProp.directManagedMemAccessFromHost ? "Yes" : "No");

    pad_string(padded, "  Device support concurrent memory copy and kernel execution: ", description_len);
    if (deviceProp.asyncEngineCount == 0) {
        printf("%sNo\n", padded);
    } else if (deviceProp.asyncEngineCount == 1) {
        printf("%sHalf duplex\n", padded);
    } else if (deviceProp.asyncEngineCount == 2) {
        printf("%sFull duplex\n", padded);
    }

    free(padded);
    return;
}

void print_programming_features(cudaDeviceProp deviceProp) {
    // warpSize,
    // regsPerBlock,
    // maxThreadsPerBlock,
    // maxThreadsDim,
    // maxGridSize,

    // regsPerMultiprocessor,
    // maxThreadsPerMultiProcessor,
    // sharedMemPerMultiprocessor,
    // maxBlocksPerMultiProcessor,

    const int description_len = 68;
    char *padded = (char *)malloc(sizeof(char) * (description_len + 1));

    pad_string(padded, "  Warp size:", description_len);
    printf("%s%d\n", padded, deviceProp.warpSize);

    pad_string(padded, "  Total number of registers available per block:", description_len);
    printf("%s%d\n", padded, deviceProp.regsPerBlock);

    pad_string(padded, "  Maximum number of threads per block:", description_len);
    printf("%s%d\n", padded, deviceProp.maxThreadsPerBlock);

    pad_string(padded, "  Max dimension size of a thread block (x,y,z):", description_len);
    printf("%s(%d, %d, %d)\n", padded, deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);

    pad_string(padded, "  Max dimension size of a grid size    (x,y,z):", description_len);
    printf("%s(%d, %d, %d)\n", padded, deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);

    pad_string(padded, "  Registers per multiprocessor:", description_len);
    printf("%s%d\n", padded, deviceProp.regsPerMultiprocessor);

    pad_string(padded, "  Maximum number of threads per multiprocessor:", description_len);
    printf("%s%d\n", padded, deviceProp.maxThreadsPerMultiProcessor);

    pad_string(padded, "  Total shared memory per multiprocessor:", description_len);
    printf("%s%ld bytes\n", padded, deviceProp.sharedMemPerMultiprocessor);

    pad_string(padded, "  Max blocks per multiprocessor:", description_len);
    printf("%s%d\n", padded, deviceProp.maxBlocksPerMultiProcessor);

    free(padded);
    return;
}

void pad_string(char *des, char *src, size_t target_length) {
    size_t string_length = strlen(src);

    if (string_length > target_length) {
        memcpy(des, src, target_length+1);
        return;
    }

    memset(des, 32, (target_length + 1));
    memcpy(des, src, string_length);
    des[target_length + 1] = '\0';
    return;
}
