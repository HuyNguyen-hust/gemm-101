# GEMM-101: CUDA Matrix Multiplication Optimization

## Overview

This repository provides a step-by-step implementation of optimized General Matrix Multiplication (GEMM) using CUDA. It's based on the excellent article by Lei Mao: [CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/).

## Implementations

The repository includes several versions of GEMM implementations, each building upon the previous one with additional optimizations:

0. Naive Implementation (v00)
1. Coalesced Memory Access (v01)
2. Shared Memory Usage, 2D Block Tiling (v02)
3. 2D Block Tiling, 1D Thread Tiling (v03)
4. 2D Block Tiling, 2D Thread Tiling (v04)
5. 2D Block Tiling, 2D Thread Tiling, load transposed A to enable VMA (v05)
6. 2D Block Tiling, 2D Warp Tiling, 2D Thread Tiling, load transposed A to enable VMA (v06)
7. Use WMMA API to leverage Tensor Core (v07)
8. Use Cutlass CuTe to leverage Tensor Core (v08, v09)

## Important Note

A small but crucial fix has been applied to versions v06 and v07. The Lei Mao's original code used `__syncwarp()`, which could potentially lead to race conditions. This has been replaced with `__syncthreads()` to ensure proper synchronization across all threads in a block.

## Build and Benchmark

```bash
git submodule init
git submodule update
cmake -B build
cmake --build build
./build/csrc/profile_cuda_gemm_fp32
./build/csrc/profile_cuda_gemm_fp16
```

## Credits

This project is based on the work of [Lei Mao](https://leimao.github.io/) and [Siboehm](https://siboehm.com/articles/22/CUDA-MMM). I am grateful for their detailed explanations and implementations.
