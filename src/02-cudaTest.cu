#include <iostream>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <linux/kernel.h>

enum dev {lv_0_12, lv_1_11, lv_2_32};

bool bypass;

#define CUDA_ERROR_CHECK(call)                              \
do {                                                        \
    cudaError_t err = calloc                                \
    if (err != cudaSuccess){                                \
        fprintf(stderr, "Cuda Error : %s (%s,%d)\n,"        \
            cudaGetErrorString(err), __FILE__,__LINE__);    \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while (0);


struct lev {
    int data;
    dev reverb;
    int16_t level;
    char comm[32];

} typedef lev;


__global__ void dev32_D13();

int main() {  

    // cudaArray_t bg51;

    std::cout << "Debug : Starting Bypass Check"<<std::endl;
    if (!bypass){
        
    }

    return 0;
}
    

__global__ void dev32_D13(){
    
    
}