#include <iostream>
#include <ctime>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <curand_kernel.h>


// Macro for checking CUDA errors
#define CHECK_CUDA_ERROR(call)                                      \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA Error: %s (%s:%d)\n",             \
                    cudaGetErrorString(err), __FILE__, __LINE__);   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)



struct bodyCar{
    float3 pos;
    float3 vel;
} typedef bodyCar;


    // A program to simulate total position change over time
__global__ void moveBody(bodyCar *body, int len, int dt){

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < len){
        body[i].pos.x += body[i].vel.x * dt;
        body[i].pos.y += body[i].vel.y * dt;
        body[i].pos.z += body[i].vel.z * dt;

    }
    

}

__global__ void initRNG(curandState *states, unsigned long seed){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed + i, i ,10, &states[i]);

}

__global__ void initCars(bodyCar *body, curandState *states, int len){

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < len){

        curandState localstate = states[i];

        body[i].pos.x = curand_uniform(&localstate) * 100.0;
        body[i].pos.y = curand_uniform(&localstate) * 100.0;
        body[i].pos.z = curand_uniform(&localstate) * 100.0;

        body[i].vel.x = curand_uniform(&localstate) * 10.0;
        body[i].vel.y = curand_uniform(&localstate) * 10.0;
        body[i].vel.z = curand_uniform(&localstate) * 10.0;

        states[i] = localstate;

        if (i < 10) {
            printf("Generated pos: %f, %f, %f\n", body[i].pos.x, body[i].pos.y, body[i].pos.z);
        }

    }
}



int main(){

    int NUM = 329;
    int my_time = 5;

    bodyCar Cars[NUM];

    bodyCar *d_body;
    curandState *d_states;


    
    CHECK_CUDA_ERROR(cudaMalloc(&d_body, NUM * sizeof(bodyCar)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_states, NUM * sizeof(curandState)));

    int ThreadPerBlocks = 128;
    int BlocksPerGrid = (NUM + ThreadPerBlocks - 1) / ThreadPerBlocks;

    initRNG<<< BlocksPerGrid, ThreadPerBlocks>>>(d_states, time(NULL));
    
    initCars<<< BlocksPerGrid, ThreadPerBlocks>>>(d_body, d_states, NUM);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(Cars, d_body, NUM * sizeof(bodyCar), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 10; ++i) {
        int idx = rand() % 329;
        std::cout << "pos initially - Car["<< i <<"].pos.x =" << Cars[i].pos.x << std::endl;
    }
    
    moveBody<<< BlocksPerGrid, ThreadPerBlocks>>>(d_body, NUM, 45);
    
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(Cars, d_body, NUM * sizeof(bodyCar), cudaMemcpyDeviceToHost));


    srand(time(NULL));
    for (int i = 0; i < 10; ++i) {
        int idx = rand() % 329;
        std::cout << "pos initially - Car["<< i <<"].pos.x =" << Cars[i].pos.x << std::endl;
    }
    

    printf("pos: %f, %f, %f\n", Cars[0].pos.x, Cars[2].pos.y, Cars[8].pos.z);



    cudaFree(d_body);
    cudaFree(d_states);

    printf("Exec complete");
    return 0;
}