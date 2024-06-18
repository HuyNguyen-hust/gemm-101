# set cmake version
cmake_minimum_required(VERSION 3.20.0)

# set project name
project(gemm-101 VERSION 0.0.1 LANGUAGES CXX CUDA)

# set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find cuda toolkit
find_package(CUDAToolkit REQUIRED)

# find include dir of this project
find_path(CUDA_GEMM_INCLUDE_DIR cuda_gemm.hpp HINTS ${CMAKE_SOURCE_DIR}/include)
# CUDA_GEMM_INCLUDE_DIR == include/

# setup cuda architectures

# add libraries
add_library(cuda_gemm
            SHARED
            00_non_coalesced_global_memory_access.cu
            01_coalesced_global_memory_access.cu
            profile_utils.cu
            utils.cu)
# include directories for cuda_gemm
target_include_directories(cuda_gemm PUBLIC ${CUDA_GEMM_INCLUDE_DIR})
# set gpu properties
set_target_properties(cuda_gemm PROPERTIES CUDA_ARCHITECTURES 60 70 75 80 90)

# add executables
add_executable(profile_cuda_gemm_fp32 profile_cuda_gemm_fp32.cu)
# link libraries
target_link_libraries(profile_cuda_gemm_fp32 cuda_gemm CUDA::cublas)
# set gpu properties
set_target_properties(profile_cuda_gemm_fp32 PROPERTIES CUDA_ARCHITECTURES 60 70 75 80 90)

# add executables
add_executable(profile_cuda_gemm_fp16 profile_cuda_gemm_fp16.cu)
# link libraries
target_link_libraries(profile_cuda_gemm_fp16 cuda_gemm CUDA::cublas)
# set gpu properties
set_target_properties(profile_cuda_gemm_fp16 PROPERTIES CUDA_ARCHITECTURES 60 70 75 80 90)