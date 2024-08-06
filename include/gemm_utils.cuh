#pragma once

// define all util functions for gemm

template <
    typename T,
    const size_t BLOCK_TILE_SIZE_M,
    const size_t BLOCK_TILE_SIZE_N,
    const size_t BLOCK_TILE_SIZE_K,
    const size_t NUM_THREADS,
    const size_t BLOCK_TILE_SKEW_SIZE_N = 0U,
    const size_t BLOCK_TILE_SKEW_SIZE_K = 0U
>
__device__ void load_data_to_shared_memory(const T* A, size_t lda,
                                        const T* B, size_t ldb,
                                        T A_thread_block_tile_shared[BLOCK_TILE_SIZE_M][BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
                                        T B_thread_block_tile_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N + BLOCK_TILE_SKEW_SIZE_N],
                                        size_t AB_thread_block_tile_idx,
                                        size_t thread_linear_idx,
                                        size_t m, size_t n, size_t k
                                        )
{   
    // thread_linear_idx is the linear index of thread RELATIVE to the thread block
    constexpr size_t A_THREAD_BLOCK_TILE_SIZE{BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_K};

    #pragma unroll
    for (size_t load_idx{0U}; load_idx < ((A_THREAD_BLOCK_TILE_SIZE + NUM_THREADS - 1U) / NUM_THREADS); ++load_idx)
    {   

        size_t A_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
        size_t A_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};
        
        size_t A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_M + A_thread_block_tile_row_idx};
        size_t A_col_idx{AB_thread_block_tile_idx * BLOCK_TILE_SIZE_K + A_thread_block_tile_col_idx};

        T val{static_cast<T>(0.0)};
        if (A_col_idx < k && A_row_idx < m)
        {
           val = A[A_row_idx * lda + A_col_idx];
        }
        A_thread_block_tile_shared[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx] = val;
    }

    constexpr size_t B_THREAD_BLOCK_TILE_SIZE{BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_N};

    #pragma unroll
    for (size_t load_idx{0U}; load_idx < ((B_THREAD_BLOCK_TILE_SIZE + NUM_THREADS - 1U) / NUM_THREADS); ++load_idx)
    {   
        size_t B_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_N};
        size_t B_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_N};
        
        size_t B_row_idx{AB_thread_block_tile_idx * BLOCK_TILE_SIZE_K + B_thread_block_tile_row_idx};
        size_t B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_N + B_thread_block_tile_col_idx};
        
        T val{static_cast<T>(0.0)};
        if (B_col_idx < n && B_row_idx < k)
        {
            val = B[B_row_idx * ldb + B_col_idx];
        }
        B_thread_block_tile_shared[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
    }
}

template <
    typename T,
    const size_t BLOCK_TILE_SIZE_M,
    const size_t BLOCK_TILE_SIZE_N,
    const size_t BLOCK_TILE_SIZE_K,
    const size_t NUM_THREADS,
    const size_t BLOCK_TILE_SKEW_SIZE_M = 0U,
    const size_t BLOCK_TILE_SKEW_SIZE_N = 0U
>
__device__ void load_data_to_shared_memory_transposed(const T* A, size_t lda,
                                                    const T* B, size_t ldb,
                                                    T A_thread_block_tile_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M + BLOCK_TILE_SKEW_SIZE_M],
                                                    T B_thread_block_tile_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N + BLOCK_TILE_SKEW_SIZE_N],
                                                    size_t AB_thread_block_tile_idx,
                                                    size_t thread_linear_idx,
                                                    size_t m, size_t n, size_t k
                                                    )
{
    // thread_linear_idx is the linear index of thread RELATIVE to the thread block

    constexpr size_t A_THREAD_BLOCK_TILE_SIZE{BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_M};

    #pragma unroll
    for (size_t load_idx{0U}; load_idx < ((A_THREAD_BLOCK_TILE_SIZE + NUM_THREADS - 1U) / NUM_THREADS); ++load_idx)
    {

        size_t A_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
        size_t A_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};
        
        size_t A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_M + A_thread_block_tile_row_idx};
        size_t A_col_idx{AB_thread_block_tile_idx * BLOCK_TILE_SIZE_K + A_thread_block_tile_col_idx};
        
        T val{static_cast<T>(0.0)};
        if (A_col_idx < k && A_row_idx < m)
        {
            val = A[A_row_idx * lda + A_col_idx];
        }
        A_thread_block_tile_shared[A_thread_block_tile_col_idx][A_thread_block_tile_row_idx] = val;
    }

    constexpr size_t B_THREAD_BLOCK_TILE_SIZE{BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_N};

    #pragma unroll
    for (size_t load_idx{0U}; load_idx < ((B_THREAD_BLOCK_TILE_SIZE + NUM_THREADS - 1U) / NUM_THREADS); ++load_idx)
    {   
        size_t B_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_N};
        size_t B_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_N};
        
        size_t B_row_idx{AB_thread_block_tile_idx * BLOCK_TILE_SIZE_K + B_thread_block_tile_row_idx};
        size_t B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_N + B_thread_block_tile_col_idx};
        
        T val{static_cast<T>(0.0)};
        if (B_col_idx < n && B_row_idx < k)
        {
            val = B[B_row_idx * ldb + B_col_idx];
        }
        B_thread_block_tile_shared[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
    }
}

template <
    typename T,
    const size_t BLOCK_TILE_SIZE_M,
    const size_t BLOCK_TILE_SIZE_N,
    const size_t BLOCK_TILE_SIZE_K,
    const size_t NUM_THREADS,
    typename VECTOR_TYPE = int4,
    const size_t BLOCK_TILE_SKEW_SIZE_K = 0U,
    const size_t BLOCK_TILE_SKEW_SIZE_N = 0U
>
__device__ void load_data_to_shared_memory_vectorized(const T* A, size_t lda,
                                             const T* B, size_t ldb,
                                             T A_thread_block_tile_shared[BLOCK_TILE_SIZE_M][BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
                                             T B_thread_block_tile_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N + BLOCK_TILE_SKEW_SIZE_N],
                                             size_t AB_thread_block_tile_idx,
                                             size_t thread_linear_idx,
                                             size_t m, size_t n, size_t k
                                             )
{   
    // thread_linear_idx is the linear index of thread RELATIVE to the thread block
    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(T)};

    static_assert((BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_N + BLOCK_TILE_SKEW_SIZE_N) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);

    // load A thread block tile
    constexpr size_t VECTORIZED_THREAD_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS};
    constexpr size_t A_VECTORIZED_THREAD_BLOCK_TILE_SIZE{BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS * BLOCK_TILE_SIZE_M};

    #pragma unroll
    for (size_t load_idx{0U}; load_idx < ((A_VECTORIZED_THREAD_BLOCK_TILE_SIZE + NUM_THREADS - 1U) / NUM_THREADS); ++load_idx)
    {
        size_t A_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) / VECTORIZED_THREAD_BLOCK_TILE_SIZE_K};
        size_t A_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) % VECTORIZED_THREAD_BLOCK_TILE_SIZE_K * NUM_VECTOR_UNITS};
        
        size_t A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_M + A_thread_block_tile_row_idx};
        size_t A_col_idx{AB_thread_block_tile_idx * BLOCK_TILE_SIZE_K + A_thread_block_tile_col_idx};
        
        VECTOR_TYPE A_row_vector_vals{0, 0, 0, 0};
        if (A_col_idx < k && A_row_idx < m)
        {
            A_row_vector_vals = *reinterpret_cast<const VECTOR_TYPE*>(&A[A_row_idx * lda + A_col_idx]);
        }
        if (A_col_idx + NUM_VECTOR_UNITS > k)
        {
            size_t num_invalid_units{A_col_idx + NUM_VECTOR_UNITS - k};
            T* A_row_vector_vals_ptr{reinterpret_cast<T*>(&A_row_vector_vals)};
            for (size_t i{NUM_VECTOR_UNITS - num_invalid_units}; i < NUM_VECTOR_UNITS; ++i)
            {
                A_row_vector_vals_ptr[i] = static_cast<T>(0.0);
            }
        }

        if (A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K
            && A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_M)
        {
            *reinterpret_cast<VECTOR_TYPE*>(&A_thread_block_tile_shared[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx]) = A_row_vector_vals;
        }
    }

    // load B thread block tile
    constexpr size_t VECTORIZED_THREAD_BLOCK_TILE_SIZE_N{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS};
    constexpr size_t B_VECTORIZED_THREAD_BLOCK_TILE_SIZE{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS * BLOCK_TILE_SIZE_K};

    #pragma unroll
    for (size_t load_idx{0U}; load_idx < ((B_VECTORIZED_THREAD_BLOCK_TILE_SIZE + NUM_THREADS - 1U) / NUM_THREADS); ++load_idx)
    {
        size_t B_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) / VECTORIZED_THREAD_BLOCK_TILE_SIZE_N};
        size_t B_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) % VECTORIZED_THREAD_BLOCK_TILE_SIZE_N * NUM_VECTOR_UNITS};
        
        size_t B_row_idx{AB_thread_block_tile_idx * BLOCK_TILE_SIZE_K + B_thread_block_tile_row_idx};
        size_t B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_N + B_thread_block_tile_col_idx};
        
        VECTOR_TYPE B_row_vector_vals{0, 0, 0, 0};
        if (B_col_idx < n && B_row_idx < k)
        {
            B_row_vector_vals = *reinterpret_cast<const VECTOR_TYPE*>(&B[B_row_idx * ldb + B_col_idx]);
        }
        if (B_col_idx + NUM_VECTOR_UNITS > n)
        {
            size_t num_invalid_units{B_col_idx + NUM_VECTOR_UNITS - n};
            T* B_row_vector_vals_ptr{reinterpret_cast<T*>(&B_row_vector_vals)};
            for (size_t i{NUM_VECTOR_UNITS - num_invalid_units}; i < NUM_VECTOR_UNITS; ++i)
            {
                B_row_vector_vals_ptr[i] = static_cast<T>(0.0);
            }
        }

        if (B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_N
            && B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K)
        {
            *reinterpret_cast<VECTOR_TYPE*>(&B_thread_block_tile_shared[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx]) = B_row_vector_vals;
        }
    }
}

template <
    typename T,
    const size_t BLOCK_TILE_SIZE_M,
    const size_t BLOCK_TILE_SIZE_N,
    const size_t BLOCK_TILE_SIZE_K,
    const size_t NUM_THREADS,
    typename VECTOR_TYPE = int4,
    const size_t BLOCK_TILE_SKEW_SIZE_M = 0U,
    const size_t BLOCK_TILE_SKEW_SIZE_N = 0U
>
__device__ void load_data_to_shared_memory_transposed_vectorized(const T* A, size_t lda,
                                             const T* B, size_t ldb,
                                             T A_thread_block_tile_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M + BLOCK_TILE_SKEW_SIZE_M],
                                             T B_thread_block_tile_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N + BLOCK_TILE_SKEW_SIZE_N],
                                             size_t AB_thread_block_tile_idx,
                                             size_t thread_linear_idx,
                                             size_t m, size_t n, size_t k
                                             )
{   
    // thread_linear_idx is the linear index of thread RELATIVE to the thread block

    static_assert((BLOCK_TILE_SIZE_M + BLOCK_TILE_SKEW_SIZE_M) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_N + BLOCK_TILE_SKEW_SIZE_N) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);

    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(T)};

    // load A thread block tile
    constexpr size_t VECTORIZED_THREAD_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS};
    constexpr size_t A_VECTORIZED_THREAD_BLOCK_TILE_SIZE{BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS * BLOCK_TILE_SIZE_M};

    #pragma unroll
    for (size_t load_idx{0U}; load_idx < ((A_VECTORIZED_THREAD_BLOCK_TILE_SIZE + NUM_THREADS - 1U) / NUM_THREADS); ++load_idx)
    {
        size_t A_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) / VECTORIZED_THREAD_BLOCK_TILE_SIZE_K};
        size_t A_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) % VECTORIZED_THREAD_BLOCK_TILE_SIZE_K * NUM_VECTOR_UNITS};
        
        size_t A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_M + A_thread_block_tile_row_idx};
        size_t A_col_idx{AB_thread_block_tile_idx * BLOCK_TILE_SIZE_K + A_thread_block_tile_col_idx};
        
        VECTOR_TYPE A_row_vector_vals{0, 0, 0, 0};
        if (A_col_idx < k && A_row_idx < m)
        {
            A_row_vector_vals = *reinterpret_cast<const VECTOR_TYPE*>(&A[A_row_idx * lda + A_col_idx]);
        }
        if (A_col_idx + NUM_VECTOR_UNITS > k)
        {
            size_t num_invalid_units{A_col_idx + NUM_VECTOR_UNITS - k};
            T* A_row_vector_vals_ptr{reinterpret_cast<T*>(&A_row_vector_vals)};
            for (size_t i{NUM_VECTOR_UNITS - num_invalid_units}; i < NUM_VECTOR_UNITS; ++i)
            {
                A_row_vector_vals_ptr[i] = static_cast<T>(0.0);
            }
        }

        if (A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K
            && A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_M)
        {
            for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i)
            {
                A_thread_block_tile_shared[A_thread_block_tile_col_idx + i][A_thread_block_tile_row_idx]
                    = reinterpret_cast<const T*>(&A_row_vector_vals)[i];
            }
        }
    }

    // load B thread block tile
    constexpr size_t VECTORIZED_THREAD_BLOCK_TILE_SIZE_N{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS};
    constexpr size_t B_VECTORIZED_THREAD_BLOCK_TILE_SIZE{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS * BLOCK_TILE_SIZE_K};

    #pragma unroll
    for (size_t load_idx{0U}; load_idx < ((B_VECTORIZED_THREAD_BLOCK_TILE_SIZE + NUM_THREADS - 1U) / NUM_THREADS); ++load_idx)
    {
        size_t B_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) / VECTORIZED_THREAD_BLOCK_TILE_SIZE_N};
        size_t B_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) % VECTORIZED_THREAD_BLOCK_TILE_SIZE_N * NUM_VECTOR_UNITS};
        
        size_t B_row_idx{AB_thread_block_tile_idx * BLOCK_TILE_SIZE_K + B_thread_block_tile_row_idx};
        size_t B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_N + B_thread_block_tile_col_idx};
        
        VECTOR_TYPE B_row_vector_vals{0, 0, 0, 0};
        if (B_col_idx < n && B_row_idx < k)
        {
            B_row_vector_vals = *reinterpret_cast<const VECTOR_TYPE*>(&B[B_row_idx * ldb + B_col_idx]);
        }
        if (B_col_idx + NUM_VECTOR_UNITS > n)
        {
            size_t num_invalid_units{B_col_idx + NUM_VECTOR_UNITS - n};
            T* B_row_vector_vals_ptr{reinterpret_cast<T*>(&B_row_vector_vals)};
            for (size_t i{NUM_VECTOR_UNITS - num_invalid_units}; i < NUM_VECTOR_UNITS; ++i)
            {
                B_row_vector_vals_ptr[i] = static_cast<T>(0.0);
            }
        }

        if (B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_N
            && B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K)
        {
            *reinterpret_cast<VECTOR_TYPE*>(&B_thread_block_tile_shared[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx]) = B_row_vector_vals;
        }
    }
}