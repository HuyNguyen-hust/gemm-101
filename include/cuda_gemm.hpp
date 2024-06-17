#pragma once
#include <cuda_runtime.h>

// define all the gemm kernel launch functions

template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k,
                            const T* alpha,
                            const T* A, size_t lda,
                            const T* B, size_t ldb,
                            const T* beta,
                            T *C, size_t ldc,
                            cudaStream_t stream);
