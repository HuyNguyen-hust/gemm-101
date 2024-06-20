#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_utils.hpp"
#include "gemm_utils.cuh"

// kernel
template <
    typename T,
    const size_t BLOCK_TILE_SIZE_M,
    const size_t BLOCK_TILE_SIZE_N,
    const size_t BLOCK_TILE_SIZE_K
    >
__global__ void gemm_v02(size_t m, size_t n, size_t k,
                            const T alpha,
                            const T *A, size_t lda,
                            const T *B, size_t ldb,
                            const T beta,
                            T *C, size_t ldc)
{   
    // init shared memory
    // shared memory must be allocated outside the conditional statements
    __shared__ T A_thread_block_tile_shared[BLOCK_TILE_SIZE_M][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];

    const size_t C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    const size_t C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    if (C_col_idx < n && C_row_idx < m)
    {
        size_t num_AB_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1U) / BLOCK_TILE_SIZE_K};

        T sum{static_cast<T>(0.0)};
        for (size_t i{0U}; i < num_AB_thread_block_tiles; ++i)
        {
            // copy A and B to shared memory
            size_t AB_thread_block_tile_idx{i};
            load_data_to_shared_memory<T, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N>(
                A, lda,
                B, ldb,
                A_thread_block_tile_shared,
                B_thread_block_tile_shared,
                C_row_idx, C_col_idx,
                AB_thread_block_tile_idx,
                m, n, k
            );

            __syncthreads();

            // compute
            #pragma unroll        
            for (size_t j{0U}; j < BLOCK_TILE_SIZE_K; ++j)
            {
                sum += A_thread_block_tile_shared[threadIdx.y][j] * B_thread_block_tile_shared[j][threadIdx.x];
            }

            __syncthreads();
        }
        // accumulate C
        C[C_row_idx * ldc + C_col_idx] = alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

// launch
template <typename T>
void launch_gemm_kernel_v02(size_t m, size_t n, size_t k,
                            const T* alpha,
                            const T *A, size_t lda,
                            const T *B, size_t ldb,
                            const T* beta,
                            T *C, size_t ldc,
                            cudaStream_t stream)
{   
    constexpr size_t BLOCK_TILE_SIZE_M{32U};
    constexpr size_t BLOCK_TILE_SIZE_N{32U};
    constexpr size_t BLOCK_TILE_SIZE_K{32U};

    dim3 block{32U, 32U, 1U};
    dim3 grid{
        (static_cast<unsigned int>(m) + block.x - 1U) / block.x,
        (static_cast<unsigned int>(n) + block.y - 1U) / block.y, 1U 
    };

    gemm_v02<T, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K>
                <<<grid, block, 0, stream>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);

    CHECK_LAST_CUDA_ERROR();
}                            

// explicit instantiation
template void launch_gemm_kernel_v02<float>(size_t m, size_t n, size_t k,
                                    const float *alpha,
                                    const float *A, size_t lda,
                                    const float *B, size_t ldb,
                                    const float *beta,
                                    float *C, size_t ldc,
                                    cudaStream_t stream);

template void launch_gemm_kernel_v02<double>(size_t m, size_t n, size_t k,
                                     const double *alpha,
                                     const double *A, size_t lda,
                                     const double *B, size_t ldb,
                                     const double *beta,
                                     double *C, size_t ldc,
                                     cudaStream_t stream);
                                     
template void launch_gemm_kernel_v02<__half>(size_t m, size_t n, size_t k,
                                    const __half *alpha,
                                    const __half *A, size_t lda,
                                    const __half *B, size_t ldb,
                                    const __half *beta,
                                    __half *C, size_t ldc,
                                    cudaStream_t stream);                                     