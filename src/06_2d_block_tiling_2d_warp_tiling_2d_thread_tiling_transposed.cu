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
    const size_t WARP_TILE_SIZE_M,
    const size_t WARP_TILE_SIZE_N,
    const size_t THREAD_TILE_SIZE_M,
    const size_t THREAD_TILE_SIZE_N,
    const size_t NUM_THREADS_PER_WARP_M,
    const size_t NUM_THREADS_PER_WARP_N
>
__global__ void gemm_v06(size_t m, size_t n, size_t k,
                            const T alpha,
                            const T *A, size_t lda,
                            const T *B, size_t ldb,
                            const T beta,
                            T *C, size_t ldc)
{
    // get NUM_THREADS_PER_WARP
    constexpr size_t NUM_THREADS_PER_WARP{NUM_THREADS_PER_WARP_M * NUM_THREADS_PER_WARP_N};

    // get NUM_THREADS
    constexpr size_t NUM_WARPS_M{BLOCK_TILE_SIZE_M / WARP_TILE_SIZE_M};
    constexpr size_t NUM_WARPS_N{BLOCK_TILE_SIZE_N / WARP_TILE_SIZE_N};
    constexpr size_t NUM_THREADS_M{NUM_THREADS_PER_WARP_M * NUM_WARPS_M};
    constexpr size_t NUM_THREADS_N{NUM_THREADS_PER_WARP_N * NUM_WARPS_N};
    constexpr size_t NUM_THREADS{NUM_THREADS_M * NUM_THREADS_N};

    // get NUM_THREAD_TILES_PER_WARP
    constexpr size_t NUM_THREAD_TILES_PER_WARP_M{WARP_TILE_SIZE_M / (NUM_THREADS_PER_WARP_M * THREAD_TILE_SIZE_M)};
    constexpr size_t NUM_THREAD_TILES_PER_WARP_N{WARP_TILE_SIZE_N / (NUM_THREADS_PER_WARP_N * THREAD_TILE_SIZE_N)};

    // get thread id
    const size_t thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};
    const size_t warp_linear_idx{thread_linear_idx / NUM_THREADS_PER_WARP};
    const size_t warp_row_idx{warp_linear_idx / NUM_WARPS_N};
    const size_t warp_col_idx{warp_linear_idx % NUM_WARPS_N};
    const size_t thread_linear_idx_in_warp{thread_linear_idx % NUM_THREADS_PER_WARP};
    const size_t thread_linear_row_idx_in_warp{thread_linear_idx_in_warp / NUM_THREADS_PER_WARP_N};
    const size_t thread_linear_col_idx_in_warp{thread_linear_idx_in_warp % NUM_THREADS_PER_WARP_N};

    __shared__ T A_thread_block_tile_shared_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M];
    __shared__ T B_thread_block_tile_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];

    size_t num_AB_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1U) / BLOCK_TILE_SIZE_K};

    T sum[NUM_THREAD_TILES_PER_WARP_M][NUM_THREAD_TILES_PER_WARP_N][THREAD_TILE_SIZE_M][THREAD_TILE_SIZE_N] = {static_cast<T>(0.0)};
    T A_tmp[NUM_THREAD_TILES_PER_WARP_M][THREAD_TILE_SIZE_M] = {static_cast<T>(0.0)};
    T B_tmp[NUM_THREAD_TILES_PER_WARP_N][THREAD_TILE_SIZE_N] = {static_cast<T>(0.0)};
    for (size_t i{0U}; i < num_AB_thread_block_tiles; ++i)
    {
        size_t AB_thread_block_tile_idx{i};

        load_data_to_shared_memory_transposed<T, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, NUM_THREADS>(
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
            // load A, B to register            
            size_t A_thread_block_tile_col_idx{j};

            #pragma unroll
            for (size_t thread_tile_repeat_row_idx{0U}; thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_M; ++thread_tile_repeat_row_idx)
            {   
                size_t A_thread_block_tile_row_idx{warp_row_idx * WARP_TILE_SIZE_M +
                    thread_tile_repeat_row_idx * NUM_THREADS_PER_WARP_M * THREAD_TILE_SIZE_M + 
                    thread_linear_row_idx_in_warp * THREAD_TILE_SIZE_M};
                
                #pragma unroll
                for (size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_SIZE_M; ++thread_tile_row_idx)
                {
                    A_tmp[thread_tile_repeat_row_idx][thread_tile_row_idx] =
                        A_thread_block_tile_shared_transposed[A_thread_block_tile_col_idx][A_thread_block_tile_row_idx + thread_tile_row_idx];
                }
            }
            
            size_t B_thread_block_tile_row_idx{j};

            #pragma unroll
            for (size_t thread_tile_repeat_col_idx{0U}; thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_N; ++thread_tile_repeat_col_idx)
            {   
                
            size_t B_thread_block_tile_col_idx{warp_col_idx * WARP_TILE_SIZE_N +
                thread_tile_repeat_col_idx * NUM_THREADS_PER_WARP_N * THREAD_TILE_SIZE_N +
                thread_linear_col_idx_in_warp * THREAD_TILE_SIZE_N};
            
                #pragma unroll
                for (size_t thread_tile_col_idx{0U}; thread_tile_col_idx < THREAD_TILE_SIZE_N; ++thread_tile_col_idx)
                {
                    B_tmp[thread_tile_repeat_col_idx][thread_tile_col_idx] = 
                        B_thread_block_tile_shared[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx + thread_tile_col_idx];
                }
            }

            #pragma unroll
            for (size_t thread_tile_repeat_row_idx{0U}; thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_M; ++thread_tile_repeat_row_idx)
            {
                #pragma unroll
                for (size_t thread_tile_repeat_col_idx{0U}; thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_N; ++thread_tile_repeat_col_idx)
                {
                    #pragma unroll
                    for (size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_SIZE_M; ++thread_tile_row_idx)
                    {
                        #pragma unroll
                        for (size_t thread_tile_col_idx{0U}; thread_tile_col_idx < THREAD_TILE_SIZE_N; ++thread_tile_col_idx)
                        {
                            sum[thread_tile_repeat_row_idx][thread_tile_repeat_col_idx][thread_tile_row_idx][thread_tile_col_idx] +=
                                A_tmp[thread_tile_repeat_row_idx][thread_tile_row_idx] * B_tmp[thread_tile_repeat_col_idx][thread_tile_col_idx];
                        }
                    }
                }
            }
        }

        __syncthreads();
    }
    
    // write C to global memory
    #pragma unroll
    for (size_t thread_tile_repeat_row_idx{0U}; thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_M; ++thread_tile_repeat_row_idx)
    {
        #pragma unroll
        for (size_t thread_tile_repeat_col_idx{0U}; thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_N; ++thread_tile_repeat_col_idx)
        {
            #pragma unroll
            for (size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_SIZE_M; ++thread_tile_row_idx)
            {
                #pragma unroll
                for (size_t thread_tile_col_idx{0U}; thread_tile_col_idx < THREAD_TILE_SIZE_N; ++thread_tile_col_idx)
                {
                    size_t C_row_idx{blockIdx.y * BLOCK_TILE_SIZE_M + 
                        warp_row_idx * WARP_TILE_SIZE_M + thread_tile_repeat_row_idx * NUM_THREADS_PER_WARP_M * THREAD_TILE_SIZE_M + 
                        thread_linear_row_idx_in_warp * THREAD_TILE_SIZE_M + 
                        thread_tile_row_idx};
                    size_t C_col_idx{blockIdx.x * BLOCK_TILE_SIZE_N + 
                        warp_col_idx * WARP_TILE_SIZE_N + thread_tile_repeat_col_idx * NUM_THREADS_PER_WARP_N * THREAD_TILE_SIZE_N + 
                        thread_linear_col_idx_in_warp * THREAD_TILE_SIZE_N + 
                        thread_tile_col_idx};
                    if (C_row_idx < m && C_col_idx < n)
                    {
                        C[C_row_idx * ldc + C_col_idx] = sum[thread_tile_repeat_row_idx][thread_tile_repeat_col_idx][thread_tile_row_idx][thread_tile_col_idx];
                    }
                }                    
            }
        }                                    
    }
}                        

// launch
template <typename T>
void launch_gemm_kernel_v06(size_t m, size_t n, size_t k,
                            const T *alpha,
                            const T *A, size_t lda,
                            const T *B, size_t ldb,
                            const T *beta,
                            T *C, size_t ldc,
                            cudaStream_t stream)
{
    constexpr size_t BLOCK_TILE_SIZE_M{128U};
    constexpr size_t BLOCK_TILE_SIZE_N{128U};
    constexpr size_t BLOCK_TILE_SIZE_K{16U};

    constexpr size_t WARP_TILE_SIZE_M{64U};
    constexpr size_t WARP_TILE_SIZE_N{32U};
    constexpr size_t NUM_WARPS_M{BLOCK_TILE_SIZE_M / WARP_TILE_SIZE_M};
    constexpr size_t NUM_WARPS_N{BLOCK_TILE_SIZE_N / WARP_TILE_SIZE_N};
    // check if warps can fit into a block
    static_assert(BLOCK_TILE_SIZE_M % WARP_TILE_SIZE_M == 0U);
    static_assert(BLOCK_TILE_SIZE_N % WARP_TILE_SIZE_N == 0U);

    constexpr size_t THREAD_TILE_SIZE_M{8U};
    constexpr size_t THREAD_TILE_SIZE_N{8U};

    constexpr size_t NUM_THREADS_PER_WARP_M{8U};
    constexpr size_t NUM_THREADS_PER_WARP_N{4U};
    // check if a warp has 32 threads
    static_assert(NUM_THREADS_PER_WARP_M * NUM_THREADS_PER_WARP_N == 32U);

    // check if an integer number of matrices of size (8 x THREAD_TILE_SIZE_M) x (4 x THREAD_TILE_SIZE_N) can fit into a warp
    static_assert(WARP_TILE_SIZE_M % (NUM_THREADS_PER_WARP_M * THREAD_TILE_SIZE_M) == 0U);
    static_assert(WARP_TILE_SIZE_N % (NUM_THREADS_PER_WARP_N * THREAD_TILE_SIZE_N) == 0U);

    constexpr size_t NUM_THREADS_M{NUM_THREADS_PER_WARP_M * NUM_WARPS_M};
    constexpr size_t NUM_THREADS_N{NUM_THREADS_PER_WARP_N * NUM_WARPS_N};

    constexpr size_t NUM_THREADS_PER_BLOCK{NUM_THREADS_M * NUM_THREADS_N};

    dim3 block{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 grid{
        (static_cast<unsigned int>(n) + static_cast<unsigned int>(BLOCK_TILE_SIZE_N) - 1U) / static_cast<unsigned int>(BLOCK_TILE_SIZE_N),
        (static_cast<unsigned int>(m) + static_cast<unsigned int>(BLOCK_TILE_SIZE_M) - 1U) / static_cast<unsigned int>(BLOCK_TILE_SIZE_M),
        1U
    };

    gemm_v06<
        T, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K,
        WARP_TILE_SIZE_M, WARP_TILE_SIZE_N,
        THREAD_TILE_SIZE_M, THREAD_TILE_SIZE_N,
        NUM_THREADS_PER_WARP_M, NUM_THREADS_PER_WARP_N
    ><<<grid, block, 0, stream>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);

    CHECK_LAST_CUDA_ERROR();
}                            

// explicit instantiation
template void launch_gemm_kernel_v06<float>(size_t m, size_t n, size_t k,
                                              const float *alpha,
                                              const float *A, size_t lda,
                                              const float *B, size_t ldb,
                                              const float *beta,
                                              float *C, size_t ldc,
                                              cudaStream_t stream);

template void launch_gemm_kernel_v06<__half>(size_t m, size_t n, size_t k,
                                              const __half *alpha,
                                              const __half *A, size_t lda,
                                              const __half *B, size_t ldb,
                                              const __half *beta,
                                              __half *C, size_t ldc,
                                              cudaStream_t stream);

template void launch_gemm_kernel_v06<double>(size_t m, size_t n, size_t k,
                                              const double *alpha,
                                              const double *A, size_t lda,
                                              const double *B, size_t ldb,
                                              const double *beta,
                                              double *C, size_t ldc,
                                              cudaStream_t stream);                                              