cmake_minimum_required(VERSION 3.18)
project(CUDAProjects LANGUAGE CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)



set(SOURCES
    src/03-trafficSimulation.cu
    include/0303-trafficSimulation.hpp)

set(CUDA_ARCHITECTURES "86")

