#pragma once
#include <cuda_runtime.h>

// define all the gemm kernel launch functions

template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k,
                            const T *alpha,
                            const T *A, size_t lda,
                            const T *B, size_t ldb,
                            const T *beta,
                            T *C, size_t ldc,
                            cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v01(size_t m, size_t n, size_t k,
                            const T *alpha,
                            const T *A, size_t lda,
                            const T *B, size_t ldb,
                            const T *beta,
                            T *C, size_t ldc,
                            cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v02(size_t m, size_t n, size_t k,
                            const T *alpha,
                            const T *A, size_t lda,
                            const T *B, size_t ldb,
                            const T *beta,
                            T *C, size_t ldc,
                            cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v02_vectorized(size_t m, size_t n, size_t k,
                                       const T *alpha,
                                       const T *A, size_t lda,
                                       const T *B, size_t ldb,
                                       const T *beta,
                                       T *C, size_t ldc,
                                       cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v03(size_t m, size_t n, size_t k,
                            const T *alpha,
                            const T *A, size_t lda,
                            const T *B, size_t ldb,
                            const T *beta,
                            T *C, size_t ldc,
                            cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v03_vectorized(size_t m, size_t n, size_t k,
                                       const T *alpha,
                                       const T *A, size_t lda,
                                       const T *B, size_t ldb,
                                       const T *beta,
                                       T *C, size_t ldc,
                                       cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v04(size_t m, size_t n, size_t k,
                            const T *alpha,
                            const T *A, size_t lda,
                            const T *B, size_t ldb,
                            const T *beta,
                            T *C, size_t ldc,
                            cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v04_vectorized(size_t m, size_t n, size_t k,
                                       const T *alpha,
                                       const T *A, size_t lda,
                                       const T *B, size_t ldb,
                                       const T *beta,
                                       T *C, size_t ldc,
                                       cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v05(size_t m, size_t n, size_t k,
                            const T *alpha,
                            const T *A, size_t lda,
                            const T *B, size_t ldb,
                            const T *beta,
                            T *C, size_t ldc,
                            cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v05_vectorized(size_t m, size_t n, size_t k,
                                       const T *alpha,
                                       const T *A, size_t lda,
                                       const T *B, size_t ldb,
                                       const T *beta,
                                       T *C, size_t ldc,
                                       cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v06(size_t m, size_t n, size_t k,
                            const T *alpha,
                            const T *A, size_t lda,
                            const T *B, size_t ldb,
                            const T *beta,
                            T *C, size_t ldc,
                            cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v06_vectorized(size_t m, size_t n, size_t k,
                                       const T *alpha,
                                       const T *A, size_t lda,
                                       const T *B, size_t ldb,
                                       const T *beta,
                                       T *C, size_t ldc,
                                       cudaStream_t stream);