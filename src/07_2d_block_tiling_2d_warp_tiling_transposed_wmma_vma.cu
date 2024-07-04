#include <cuda_fp16.h>
#include <mma.h>

#include "cuda_gemm.hpp"
#include "cuda_utils.hpp"
#include "gemm_utils.cuh"

// kernel
template <
    typename T,
    const size_t BLOCK_TILE_SIZE_M, const size_t BLOCK_TILE_SIZE_N, const size_t BLOCK_TILE_SIZE_K,
    const size_t BLOCK_TILE_SKEW_SIZE_M, const size_t BLOCK_TILE_SKEW_SIZE_N,
    const size_t WARP_TILE_SIZE_M, const size_t WARP_TILE_SIZE_N,
    const size_t WMMA_TILE_SIZE_M, const size_t WMMA_TILE_SIZE_N, const size_t WMMA_TILE_SIZE_K,
    const size_t NUM_THREADS
>
__global__ void gemm_v07_vectorized(size_t m, size_t n, size_t k,
                            const T alpha,
                            const T *A, size_t lda,
                            const T *B, size_t ldb,
                            const T beta,
                            T *C, size_t ldc)
{
    constexpr size_t NUM_WMMA_TILES_M{WARP_TILE_SIZE_M / WMMA_TILE_SIZE_M};
    constexpr size_t NUM_WMMA_TILES_N{WARP_TILE_SIZE_N / WMMA_TILE_SIZE_N};
    constexpr size_t NUM_WMMA_TILES_K{BLOCK_TILE_SIZE_K / WMMA_TILE_SIZE_K}; // BLOCK_TILE_SIZE_K == WARP_TILE_SIZE_K

    constexpr size_t NUM_WARPS_N{BLOCK_TILE_SIZE_N / WARP_TILE_SIZE_N};

    // get thread id
    const size_t thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};
    const size_t warp_linear_idx{thread_linear_idx / 32U};
    const size_t warp_row_idx{warp_linear_idx / NUM_WARPS_N};
    const size_t warp_col_idx{warp_linear_idx % NUM_WARPS_N};

    __shared__ T A_thread_block_tile_shared_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M + BLOCK_TILE_SKEW_SIZE_M];
    __shared__ T B_thread_block_tile_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N + BLOCK_TILE_SKEW_SIZE_N];

    const size_t num_AB_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1U) / BLOCK_TILE_SIZE_K};

    // declare fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 
                        WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, T, 
                        nvcuda::wmma::col_major> A_fragments[NUM_WMMA_TILES_M];
    // currently the A tile is transpoded so we need to specify A fragment in col major, so it can read the fragment right                        
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                        WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, T,
                        nvcuda::wmma::row_major> B_fragments[NUM_WMMA_TILES_N];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                        WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, T> 
                        acc_fragments[NUM_WMMA_TILES_M][NUM_WMMA_TILES_N];    
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                        WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, T> C_fragment;   
                        
    // set accumulator to zero
    #pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_M; ++wmma_tile_row_idx)
    {
        #pragma unroll
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_N; ++wmma_tile_col_idx)
        {
            nvcuda::wmma::fill_fragment(acc_fragments[wmma_tile_row_idx][wmma_tile_col_idx], static_cast<T>(0));
        }
    }

    for (size_t i{0U}; i < num_AB_thread_block_tiles; ++i)
    {
        size_t AB_thread_block_tile_idx{i};
        load_data_to_shared_memory_transposed_vectorized<
            T, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K,
            NUM_THREADS, int4,
            BLOCK_TILE_SKEW_SIZE_M, BLOCK_TILE_SKEW_SIZE_N
        >(
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
        for (size_t j{0U}; j < NUM_WMMA_TILES_K; ++j)
        {
            // load A fragments
            const size_t A_thread_block_tile_col_idx{j * WMMA_TILE_SIZE_K};
            #pragma unroll
            for (size_t A_fragment_row_idx{0U}; A_fragment_row_idx < NUM_WMMA_TILES_M; ++A_fragment_row_idx)
            {   
                const size_t A_thread_block_tile_row_idx{warp_row_idx * WARP_TILE_SIZE_M + A_fragment_row_idx * WMMA_TILE_SIZE_M};
                nvcuda::wmma::load_matrix_sync(
                    A_fragments[A_fragment_row_idx], 
                    &A_thread_block_tile_shared_transposed[A_thread_block_tile_col_idx][A_thread_block_tile_row_idx],
                    BLOCK_TILE_SIZE_M + BLOCK_TILE_SKEW_SIZE_M
                );

                // load B fragments
                const size_t B_thread_block_tile_row_idx{j * WMMA_TILE_SIZE_K};
                #pragma unroll
                for (size_t B_fragment_col_idx{0U}; B_fragment_col_idx < NUM_WMMA_TILES_N; ++B_fragment_col_idx)
                {   
                    const size_t B_thread_block_tile_col_idx{warp_col_idx * WARP_TILE_SIZE_N + B_fragment_col_idx * WMMA_TILE_SIZE_N};
                    nvcuda::wmma::load_matrix_sync(
                        B_fragments[B_fragment_col_idx], 
                        &B_thread_block_tile_shared[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx],
                        BLOCK_TILE_SIZE_N + BLOCK_TILE_SKEW_SIZE_N
                    );

                    nvcuda::wmma::mma_sync(
                        acc_fragments[A_fragment_row_idx][B_fragment_col_idx], 
                        A_fragments[A_fragment_row_idx],
                        B_fragments[B_fragment_col_idx], 
                        acc_fragments[A_fragment_row_idx][B_fragment_col_idx]
                    );
                }
            }

        }

        __syncthreads();
    }

    // write C to global memory
    #pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_M; ++wmma_tile_row_idx)
    {
        #pragma unroll
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_N; ++wmma_tile_col_idx)
        {   
            const size_t C_row_idx{blockIdx.y * BLOCK_TILE_SIZE_M + warp_row_idx * WARP_TILE_SIZE_M + wmma_tile_row_idx * WMMA_TILE_SIZE_M};
            const size_t C_col_idx{blockIdx.x * BLOCK_TILE_SIZE_N + warp_col_idx * WARP_TILE_SIZE_N + wmma_tile_col_idx * WMMA_TILE_SIZE_N};
            nvcuda::wmma::load_matrix_sync(
                C_fragment,
                &C[C_row_idx * ldc + C_col_idx],
                ldc,
                nvcuda::wmma::mem_row_major
            );

            for (size_t i{0U}; i < C_fragment.num_elements; ++i)
            {
                C_fragment.x[i] = alpha * acc_fragments[wmma_tile_row_idx][wmma_tile_col_idx].x[i] + beta * C_fragment.x[i];
            }

            nvcuda::wmma::store_matrix_sync(
                &C[C_row_idx * ldc + C_col_idx],
                C_fragment,
                ldc,
                nvcuda::wmma::mem_row_major
            );
        }
    }
}                            

// launch
template <typename T>
void launch_gemm_kernel_v07_vectorized(size_t m, size_t n, size_t k,
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

    constexpr size_t BLOCK_TILE_SKEW_SIZE_M{16U};
    constexpr size_t BLOCK_TILE_SKEW_SIZE_N{16U};

    constexpr size_t WARP_TILE_SIZE_M{64U};
    constexpr size_t WARP_TILE_SIZE_N{32U};

    constexpr size_t WMMA_TILE_SIZE_M{16U};
    constexpr size_t WMMA_TILE_SIZE_N{16U};
    constexpr size_t WMMA_TILE_SIZE_K{16U};
    
    // check if wmma tiles (wmma fragments) can fit into a warp
    static_assert(BLOCK_TILE_SIZE_M % WARP_TILE_SIZE_M == 0U);
    static_assert(BLOCK_TILE_SIZE_N % WARP_TILE_SIZE_N == 0U);
    static_assert(BLOCK_TILE_SIZE_K % WMMA_TILE_SIZE_K == 0U);

    constexpr size_t NUM_WARPS_M{BLOCK_TILE_SIZE_M / WARP_TILE_SIZE_M};
    constexpr size_t NUM_WARPS_N{BLOCK_TILE_SIZE_N / WARP_TILE_SIZE_N};
    // check if warps can fit into a block
    static_assert(BLOCK_TILE_SIZE_M % WARP_TILE_SIZE_M == 0U);
    static_assert(BLOCK_TILE_SIZE_N % WARP_TILE_SIZE_N == 0U);
    
    constexpr size_t NUM_THREADS_PER_BLOCK{NUM_WARPS_M * NUM_WARPS_N * 32U};

    dim3 block{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 grid{
        (static_cast<unsigned int>(n) + static_cast<unsigned int>(BLOCK_TILE_SIZE_N) - 1U) / static_cast<unsigned int>(BLOCK_TILE_SIZE_N),
        (static_cast<unsigned int>(m) + static_cast<unsigned int>(BLOCK_TILE_SIZE_M) - 1U) / static_cast<unsigned int>(BLOCK_TILE_SIZE_M),
        1U
    };

    gemm_v07_vectorized<
        T,
        BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K,
        BLOCK_TILE_SKEW_SIZE_M, BLOCK_TILE_SKEW_SIZE_N,
        WARP_TILE_SIZE_M, WARP_TILE_SIZE_N,
        WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K,
        NUM_THREADS_PER_BLOCK
    ><<<grid, block, 0, stream>>>(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);

    CHECK_LAST_CUDA_ERROR();
}                            

// explicit instantiation
template void launch_gemm_kernel_v07_vectorized<__half>(size_t m, size_t n, size_t k,
                                    const __half *alpha,
                                    const __half *A, size_t lda,
                                    const __half *B, size_t ldb,
                                    const __half *beta,
                                    __half *C, size_t ldc,
                                    cudaStream_t stream);                                  