# set the minimum required cmake version
cmake_minimum_required(VERSION 3.20.0)

# set project name
project(gemm-101 VERSION 0.0.1 LANGUAGES CXX CUDA)

# set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add subdirectory
add_subdirectory(src)

