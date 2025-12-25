# CUDA Projects Repository

This repository contains a collection of CUDA programs exploring various aspects of GPGPU computing, from basic memory management to complex cellular automaton simulations.

## Project Structure

The source code is located in the `src/` directory:

| File | Description |
|------|-------------|
| **`src/00-cudaTest.cu`** | **Standard Vector Addition**: Basic implementation using `std::vector` and CUDA C++ APIs. Demonstrates host-device memory transfer. |
| **`src/01-cudaTest.cu`** | **Algorithmic Vector Addition**: Generates large datasets on the host using a custom formula before processing on the GPU. Uses C-style arrays. |
| **`src/02-cudaTest.cu`** | **Kinematics Simulation**: A 3D particle physics simulation. Uses `curand` to initialize random positions and velocities for multiple bodies and updates their states in parallel. |
| **`src/03-trafficSimulation.cu`** | **Traffic Simulation (Nagel-Schreckenberg)**: A cellular automaton model simulating traffic flow on a single-lane circular road. Features acceleration, collision avoidance, and random braking logic to generate emergent traffic jams. |
| **`src/04-cudaTest.cu`** | **Experimental Structures**: Definitions for audio/device control structures and enums. Currently serves as a prototype/scaffold. |

## Getting Started

### Prerequisites
*   NVIDIA GPU with CUDA Compute Capability.
*   CUDA Toolkit (`nvcc`).
*   Linux OS (tested on Linux).

### Compilation and Running

You can compile any of the source files using `nvcc`.

#### Running the Traffic Simulation
This is the feature project of the repository.

1.  **Compile:**
    ```bash
    nvcc src/03-trafficSimulation.cu -o traffic
    ```

2.  **Run:**
    ```bash
    ./traffic
    ```

    You will see an ASCII visualization of the traffic flow in your terminal:
    ```text
    |.....>........>.....>>.>...>..>...>|
    ```

#### Running Vector Addition
```bash
nvcc src/00-cudaTest.cu -o vector_add
./vector_add
```
