#pragma once

// define all util functions for gemm

template <typename T,
        const size_t BLOCK_TILE_SIZE_M,
        const size_t BLOCK_TILE_SIZE_K,
        const size_t BLOCK_TILE_SIZE_N>
__device__ void load_data_to_shared_memory(const T* A, size_t lda,
                                        const T* B, size_t ldb,
                                        T A_thread_block_tile_shared[BLOCK_TILE_SIZE_M][BLOCK_TILE_SIZE_K],
                                        T B_thread_block_tile_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N],
                                        size_t C_row_idx, size_t C_col_idx,
                                        size_t AB_thread_block_tile_idx,
                                        size_t m, size_t n, size_t k
                                        ) 
{   
    #pragma unroll
    for (size_t load_idx{0U}; load_idx < ((BLOCK_TILE_SIZE_K + BLOCK_TILE_SIZE_N - 1U) / BLOCK_TILE_SIZE_N); ++load_idx)
    {   

        size_t A_thread_block_tile_row_idx{C_row_idx % BLOCK_TILE_SIZE_M};
        size_t A_thread_block_tile_col_idx{(C_col_idx % BLOCK_TILE_SIZE_N) + load_idx * BLOCK_TILE_SIZE_N};
        
        size_t A_row_idx{C_row_idx};
        size_t A_col_idx{AB_thread_block_tile_idx * BLOCK_TILE_SIZE_K + A_thread_block_tile_col_idx};

        if (A_col_idx < k && A_row_idx < m)
        {
            A_thread_block_tile_shared[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx] = A[A_row_idx * lda + A_col_idx];
        }
    }

    #pragma unroll
    for (size_t load_idx{0U}; load_idx < ((BLOCK_TILE_SIZE_K + BLOCK_TILE_SIZE_M - 1U) / BLOCK_TILE_SIZE_M); ++load_idx)
    {
        size_t B_thread_block_tile_row_idx{(C_row_idx % BLOCK_TILE_SIZE_M) + load_idx * BLOCK_TILE_SIZE_M};
        size_t B_thread_block_tile_col_idx{C_col_idx % BLOCK_TILE_SIZE_N};

        size_t B_row_idx{AB_thread_block_tile_idx * BLOCK_TILE_SIZE_K + B_thread_block_tile_row_idx};
        size_t B_col_idx{C_col_idx};

        if (B_col_idx < n && B_row_idx < k)
        {
            B_thread_block_tile_shared[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = B[B_row_idx * ldb + B_col_idx];
        }
    }
}