#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <math.h>

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




void genValue(float *a, int len, float mod)
{
    float gen = 0.0;
    for (int x = 0; x < len; x++){
	    for (int y = 0; y < len; y++){
        	gen = mod + x*y +  len/mod*y;
			a[x * len + y]=gen;
			gen=0.0;
    }}
};


__global__ void addVector(float* A, float* B, float* C,int sz)
{

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i<sz){
	C[i] = A[i] + B[i];
	}
}


int main(){

	int n = 100;
	int size = n*n;


	float h_A[size], h_B[size], h_C[size];

	genValue(h_A, n, 42);
	genValue(h_B, n, 23);

	float *d_A, *d_B, *d_C;

	CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size * sizeof(float)));

	CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice));

	// Launch the kernel (1 block with n threads in this case)
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	addVector<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

	//CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost));


    printf("Resultant vector:\n");
    for (int i = 0; i < 10; i++) {
		for (int j = 0; j< 10; j++){
        	printf("i(%d)(%d) \t A(%f) \t + \t B(%f) \t -> \t %f\n", i,j, h_A[i*n+ j], h_B[i *n + j], h_C[i*n +j]);
    	}
	}


	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
