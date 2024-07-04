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
# CUDA_GEMM_INCLUDE_DIR == /include

# add libraries
add_library(cuda_gemm
            SHARED
            00_non_coalesced_global_memory_access.cu
            01_coalesced_global_memory_access.cu
            02_2d_block_tiling.cu
            02_2d_block_tiling_vma.cu
            03_2d_block_tiling_1d_thread_tiling.cu
            03_2d_block_tiling_1d_thread_tiling_vma.cu
            04_2d_block_tiling_2d_thread_tiling.cu
            04_2d_block_tiling_2d_thread_tiling_vma.cu
            05_2d_block_tiling_2d_thread_tiling_transposed.cu
            05_2d_block_tiling_2d_thread_tiling_transposed_vma.cu
            06_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_transposed.cu
            06_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_transposed_vma.cu
            07_2d_block_tiling_2d_warp_tiling_transposed_wmma.cu
            07_2d_block_tiling_2d_warp_tiling_transposed_wmma_vma.cu
            profile_utils.cu
            cuda_utils.cu)
# include directories for cuda_gemm
target_include_directories(cuda_gemm PUBLIC ${CUDA_GEMM_INCLUDE_DIR})
# set gpu properties
set_target_properties(cuda_gemm PROPERTIES CUDA_ARCHITECTURES 70)

# add executables
add_executable(profile_cuda_gemm_fp32 profile_cuda_gemm_fp32.cu)
# link libraries
target_link_libraries(profile_cuda_gemm_fp32 cuda_gemm CUDA::cublas)
# set gpu properties
set_target_properties(profile_cuda_gemm_fp32 PROPERTIES CUDA_ARCHITECTURES 70)

# add executables
add_executable(profile_cuda_gemm_fp16 profile_cuda_gemm_fp16.cu)
# link libraries
target_link_libraries(profile_cuda_gemm_fp16 cuda_gemm CUDA::cublas)
# set gpu properties
set_target_properties(profile_cuda_gemm_fp16 PROPERTIES CUDA_ARCHITECTURES 70)