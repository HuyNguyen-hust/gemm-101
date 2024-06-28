#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_utils.hpp"
#include "gemm_utils.cuh"

// kernel
template <
    typename T,
    const size_t BLOCK_TILE_SIZE_M,
    const size_t BLOCK_TILE_SIZE_N,
    const size_t BLOCK_TILE_SIZE_K,
    const size_t THREAD_TILE_SIZE_M,
    const size_t THREAD_TILE_SIZE_N
>
__global__ void gemm_v05_vectorized(size_t m, size_t n, size_t k,
                            const T alpha,
                            const T *A, size_t lda,
                            const T *B, size_t ldb,
                            const T beta,
                            T *C, size_t ldc)
{
    constexpr size_t NUM_THREADS{(BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N) / (THREAD_TILE_SIZE_M * THREAD_TILE_SIZE_N)};
    constexpr size_t NUM_THREADS_PER_BLOCK_N{BLOCK_TILE_SIZE_N / THREAD_TILE_SIZE_N};

    const size_t thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    __shared__ T A_thread_block_tile_shared_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M];
    __shared__ T B_thread_block_tile_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];

    size_t num_AB_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1U) / BLOCK_TILE_SIZE_K};

    constexpr size_t NUM_VECTOR_UNITS{sizeof(int4) / sizeof(T)};
    // check if an integer number of units can fit into a vector
    static_assert(sizeof(int4) % sizeof(T) == 0U);

    // check if an integer number of vector can fit into A_thread_block_tile_shared and B_thread_block_tile_shared
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_N % NUM_VECTOR_UNITS == 0U);

    // check if an integer number of vector can fit into A_tmp and B_tmp
    constexpr size_t VECTORIZED_THREAD_TILE_SIZE_N{THREAD_TILE_SIZE_N / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_THREAD_TILE_SIZE_M{THREAD_TILE_SIZE_M / NUM_VECTOR_UNITS};
    static_assert(THREAD_TILE_SIZE_N % NUM_VECTOR_UNITS == 0U);
    static_assert(THREAD_TILE_SIZE_M % NUM_VECTOR_UNITS == 0U);

    T sum[THREAD_TILE_SIZE_M][THREAD_TILE_SIZE_N] = {static_cast<T>(0.0)};
    T A_tmp[THREAD_TILE_SIZE_M] = {static_cast<T>(0.0)};
    T B_tmp[THREAD_TILE_SIZE_N] = {static_cast<T>(0.0)};
    
    for (size_t i{0U}; i < num_AB_thread_block_tiles; ++i)
    {
        size_t AB_thread_block_tile_idx{i};

        // copy A, B to shared memory
        load_data_to_shared_memory_transposed_vectorized<T, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, NUM_THREADS>(
            A, lda,
            B, ldb,
            A_thread_block_tile_shared_transposed,
            B_thread_block_tile_shared,
            AB_thread_block_tile_idx,
            thread_linear_idx,
            m, n, k
        );

        __syncthreads();

        // compute
        #pragma unroll
        for (size_t j{0U}; j < BLOCK_TILE_SIZE_K; ++j)
        {
            // load necessary A data
            size_t A_thread_block_tile_row_idx{(thread_linear_idx / NUM_THREADS_PER_BLOCK_N) * THREAD_TILE_SIZE_M};
            size_t A_thread_block_tile_col_idx{j};

            #pragma unroll
            for (size_t thread_tile_row_idx{0U}; thread_tile_row_idx < VECTORIZED_THREAD_TILE_SIZE_M; ++thread_tile_row_idx)
            {
                *reinterpret_cast<int4*>(&A_tmp[thread_tile_row_idx*NUM_VECTOR_UNITS])
                    = *reinterpret_cast<const int4*>(
                        &A_thread_block_tile_shared_transposed[A_thread_block_tile_col_idx][A_thread_block_tile_row_idx + thread_tile_row_idx*NUM_VECTOR_UNITS]
                    );
            }

            // load necessary B data
            size_t B_thread_block_tile_row_idx{j};
            size_t B_thread_block_tile_col_idx{(thread_linear_idx % NUM_THREADS_PER_BLOCK_N) * THREAD_TILE_SIZE_N};

            #pragma unroll
            for (size_t thread_tile_col_idx{0U}; thread_tile_col_idx < VECTORIZED_THREAD_TILE_SIZE_N; ++thread_tile_col_idx)
            {
                *reinterpret_cast<int4*>(&B_tmp[thread_tile_col_idx*NUM_VECTOR_UNITS])
                    = *reinterpret_cast<const int4*>(
                        &B_thread_block_tile_shared[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx + thread_tile_col_idx*NUM_VECTOR_UNITS]
                    );
            }

            #pragma unroll
            for (size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_SIZE_M; ++thread_tile_row_idx)
            {
                #pragma unroll
                for (size_t thread_tile_col_idx{0U}; thread_tile_col_idx < THREAD_TILE_SIZE_N; ++thread_tile_col_idx)
                {
                    sum[thread_tile_row_idx][thread_tile_col_idx] += A_tmp[thread_tile_row_idx] * B_tmp[thread_tile_col_idx];
                }
            }
        }
        
        __syncthreads();
    }

    // write C to global memory
    #pragma unroll
    for (size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_SIZE_M; ++thread_tile_row_idx)
    {   
        #pragma unroll
        for (size_t thread_tile_col_idx{0U}; thread_tile_col_idx < THREAD_TILE_SIZE_N; ++thread_tile_col_idx)
        {
            size_t C_row_idx{(blockIdx.y * BLOCK_TILE_SIZE_M) + thread_linear_idx / NUM_THREADS_PER_BLOCK_N * THREAD_TILE_SIZE_M + thread_tile_row_idx};
            size_t C_col_idx{(blockIdx.x * BLOCK_TILE_SIZE_N) + thread_linear_idx % NUM_THREADS_PER_BLOCK_N * THREAD_TILE_SIZE_N + thread_tile_col_idx};
            if (C_row_idx < m && C_col_idx < n)
            {
                C[C_row_idx * ldc + C_col_idx] = alpha * sum[thread_tile_row_idx][thread_tile_col_idx] + beta * C[C_row_idx * ldc + C_col_idx];
            }
        }
    }
}                            


// launch
template <typename T>
void launch_gemm_kernel_v05_vectorized(size_t m, size_t n, size_t k,
                            const T *alpha,
                            const T *A, size_t lda,
                            const T *B, size_t ldb,
                            const T *beta,
                            T *C, size_t ldc,
                            cudaStream_t stream)
{   
    constexpr size_t BLOCK_TILE_SIZE_M{128U};
    constexpr size_t BLOCK_TILE_SIZE_N{128U};
    constexpr size_t BLOCK_TILE_SIZE_K{8U};
    
    constexpr size_t THREAD_TILE_SIZE_M{8U};
    constexpr size_t THREAD_TILE_SIZE_N{8U};

    constexpr size_t NUM_THREADS_PER_BLOCK{(BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N) / (THREAD_TILE_SIZE_M * THREAD_TILE_SIZE_N)};

    // check if threads can fit into a block (because each thread now is a small column)
    static_assert(BLOCK_TILE_SIZE_M % THREAD_TILE_SIZE_M == 0U);
    static_assert(BLOCK_TILE_SIZE_N % THREAD_TILE_SIZE_N == 0U);

    // check if each thread within a block can be assigned the same workload
    static_assert(BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_K % NUM_THREADS_PER_BLOCK == 0U);
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_N % NUM_THREADS_PER_BLOCK == 0U);

    dim3 block{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 grid{
        (static_cast<unsigned int>(n) + static_cast<unsigned int>(BLOCK_TILE_SIZE_N) - 1U) / static_cast<unsigned int>(BLOCK_TILE_SIZE_N),
        (static_cast<unsigned int>(m) + static_cast<unsigned int>(BLOCK_TILE_SIZE_M) - 1U) / static_cast<unsigned int>(BLOCK_TILE_SIZE_M),
        1U
    };

    gemm_v05_vectorized<T, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_M, THREAD_TILE_SIZE_N>
                <<<grid, block, 0, stream>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    
    CHECK_LAST_CUDA_ERROR();
}                            


// explicit instantiation

template void launch_gemm_kernel_v05_vectorized<float>(size_t m, size_t n, size_t k,
                                   const float *alpha,
                                   const float *A, size_t lda,
                                   const float *B, size_t ldb,
                                   const float *beta,
                                   float *C, size_t ldc,
                                   cudaStream_t stream);

template void launch_gemm_kernel_v05_vectorized<double>(size_t m, size_t n, size_t k,
                                    const double *alpha,
                                    const double *A, size_t lda,
                                    const double *B, size_t ldb,
                                    const double *beta,
                                    double *C, size_t ldc,
                                    cudaStream_t stream);                                   

template void launch_gemm_kernel_v05_vectorized<__half>(size_t m, size_t n, size_t k,
                                   const __half *alpha,
                                   const __half *A, size_t lda,
                                   const __half *B, size_t ldb,
                                   const __half *beta,
                                   __half *C, size_t ldc,
                                   cudaStream_t stream);                                   