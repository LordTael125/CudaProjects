#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CHECK_CUDA_ERROR(call)                             \
    do {                                                   \
        cudaError_t err = call;                            \
        if (err != cudaSuccess) {                          \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            exit(EXIT_FAILURE);                            \
        }                                                  \
    } while (0)

// CUDA kernel to add two vectors
__global__ void addVector(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Size of the vectors
    const int n = 5;

    // Host vectors using std::vector
    std::vector<float> h_A = {2.54, 3.41, 5.23, 3.345, 7.21};
    std::vector<float> h_B = {6.32, 3.34, 4.35, 2.21, 1.34};
    std::vector<float> h_C(n, 0.0f); // Result vector initialized to 0

    // Device pointers
    float *d_A, *d_B, *d_C;

    // Allocate memory on the device
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, n * sizeof(float)));

    // Copy input data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    addVector<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C.data(), d_C, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the result
    std::cout << "Resultant vector:" << std::endl;
    for (const auto& value : h_C) {
        std::cout << value << std::endl;
    }

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    return 0;
}
