#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

// Simulation Parameters
#define ROAD_LENGTH 100
#define NUM_CARS 20
#define MAX_VELOCITY 5
#define PROB_SLOW 0.3f
#define TICKS 30
#define BLOCK_SIZE 256

// Error handling macro
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", __FILE__, __LINE__, \
              err, cudaGetErrorString(err));                                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void setup_kernel(curandState *state, unsigned long seed) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < NUM_CARS) {
    curand_init(seed, id, 0, &state[id]);
  }
}

__global__ void update_velocity(int *positions, int *velocities,
                                curandState *state, int num_cars,
                                int road_length) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= num_cars)
    return;

  int v = velocities[id];
  int p = positions[id];

  // 1. Acceleration
  if (v < MAX_VELOCITY) {
    v++;
  }

  // Find distance to next car
  int next_car_idx = (id + 1) % num_cars;
  int next_p = positions[next_car_idx];

  int distance;
  if (next_p > p) {
    distance = next_p - p;
  } else {
    // Wrap around case
    distance = (next_p + road_length) - p;
  }

  // 2. Deceleration (due to other cars)
  // distance is gaps + 1 (if distance is 1, gap is 0, collision imminent)
  // Avoid collision: we want to travel at most 'distance - 1' cells
  if (v >= distance) {
    v = distance - 1;
  }

  // 3. Randomization
  // Randomly slow down with probability p
  float rand_val = curand_uniform(&state[id]);
  if (rand_val < PROB_SLOW && v > 0) {
    v--;
  }

  velocities[id] = v;
}

__global__ void update_position(int *positions, int *velocities, int num_cars,
                                int road_length) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= num_cars)
    return;

  int p = positions[id];
  int v = velocities[id];

  p = (p + v) % road_length;
  positions[id] = p;
}

void print_road(const std::vector<int> &positions) {
  std::string road(ROAD_LENGTH, '.');
  for (int p : positions) {
    road[p] = '>'; // Car symbol
  }
  std::cout << "|" << road << "|" << std::endl;
}

int main() {
  int *d_positions, *d_velocities;
  curandState *d_state;

  // Allocate device memory
  CHECK_CUDA(cudaMalloc(&d_positions, NUM_CARS * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_velocities, NUM_CARS * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_state, NUM_CARS * sizeof(curandState)));

  // Initialize host data (sorted positions for simplicity initially, though
  // kernel handles sort-agnostic if we scanned, but simple logic assumes sorted
  // somewhat implicitly if we just use index i and i+1. Wait,
  // Nagel-Schreckenberg requires processing cars in order or just verifying
  // distance to *next* car. For simplicity, let's initialize them evenly spaced
  // and correctly ordered.
  std::vector<int> h_positions(NUM_CARS);
  std::vector<int> h_velocities(NUM_CARS, 0);

  for (int i = 0; i < NUM_CARS; ++i) {
    h_positions[i] = (i * (ROAD_LENGTH / NUM_CARS));
  }

  // Copy to device
  CHECK_CUDA(cudaMemcpy(d_positions, h_positions.data(), NUM_CARS * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_velocities, h_velocities.data(),
                        NUM_CARS * sizeof(int), cudaMemcpyHostToDevice));

  // Setup RNG
  setup_kernel<<<(NUM_CARS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
      d_state, time(NULL));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::cout << "Traffic Simulation (Nagel-Schreckenberg)" << std::endl;
  std::cout << "Road Length: " << ROAD_LENGTH << ", Cars: " << NUM_CARS
            << std::endl;

  for (int t = 0; t < TICKS; ++t) {
    // Copy back for display
    CHECK_CUDA(cudaMemcpy(h_positions.data(), d_positions,
                          NUM_CARS * sizeof(int), cudaMemcpyDeviceToHost));
    print_road(h_positions);

    // Ideally we should sort cars if they overtake, but in single lane N-S
    // model with v < dist, overtaking shouldn't happen by logic (velocity
    // restricted to distance-1). Cars should maintain order 0..N-1 effectively
    // in circular buffer. There is a edge case where index N-1 is behind index
    // 0 (wrap around). Our update_velocity logic: `next_car_idx = (id + 1) %
    // num_cars;` assumes array is sorted by position. Since cars cannot
    // overtake, this sorted order is invariant.

    update_velocity<<<(NUM_CARS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_positions, d_velocities, d_state, NUM_CARS, ROAD_LENGTH);
    CHECK_CUDA(cudaDeviceSynchronize());

    update_position<<<(NUM_CARS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_positions, d_velocities, NUM_CARS, ROAD_LENGTH);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Cleanup
  cudaFree(d_positions);
  cudaFree(d_velocities);
  cudaFree(d_state);

  return 0;
}
