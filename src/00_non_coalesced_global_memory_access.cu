#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_utils.hpp"

// kernel
template <typename T>
__global__ void gemm_v00(size_t m, size_t n, size_t k,
                         const T alpha,
                         const T *A, size_t lda,
                         const T *B, size_t ldb,
                         const T beta,
                         T *C, size_t ldc)
{
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t i{0}; i < k; ++i)
        {
            sum += A[C_row_idx * lda + i] * B[i * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] = alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

// launch
template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k,
                            const T *alpha,
                            const T *A, size_t lda,
                            const T *B, size_t ldb,
                            const T *beta,
                            T *C, size_t ldc,
                            cudaStream_t stream)
{
    const dim3 block{32U, 32U, 1U};
    const dim3 grid{
        (static_cast<unsigned int>(m) + block.x - 1U) / block.x,
        (static_cast<unsigned int>(n) + block.y - 1U) / block.y, 1U
    };

    gemm_v00<T><<<grid, block, 0, stream>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);

    CHECK_LAST_CUDA_ERROR();
}

// explicit instantiation
template void launch_gemm_kernel_v00<float>(size_t m, size_t n, size_t k,
                                   const float *alpha,
                                   const float *A, size_t lda,
                                   const float *B, size_t ldb,
                                   const float *beta,
                                   float *C, size_t ldc,
                                   cudaStream_t stream);

template void launch_gemm_kernel_v00<double>(size_t m, size_t n, size_t k,
                                    const double *alpha,
                                    const double *A, size_t lda,
                                    const double *B, size_t ldb,
                                    const double *beta,
                                    double *C, size_t ldc,
                                    cudaStream_t stream);

template void launch_gemm_kernel_v00<__half>(size_t m, size_t n, size_t k,
                                    const __half *alpha,
                                    const __half *A, size_t lda,
                                    const __half *B, size_t ldb,
                                    const __half *beta,
                                    __half *C, size_t ldc,
                                    cudaStream_t stream);
