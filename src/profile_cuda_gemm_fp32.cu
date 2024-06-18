#include <iostream>
#include <cuda_runtime.h>

#include "profile_utils.cuh"
#include "cuda_gemm.hpp"

int main() {

    // print device information
    print_device_info();

    constexpr size_t num_repeats{1U};
    constexpr size_t num_warmups{1U};

    float const fp32_abs_tol{1.0e-3f};
    double const fp32_rel_tol{0.0e-4f};

    constexpr size_t m{4096U};
    constexpr size_t n{4096U};
    constexpr size_t k{4096U};

    constexpr size_t lda{(m + 16U - 1U) / 16U * 16U};
    constexpr size_t ldb{(k + 16U - 1U) / 16U * 16U};
    constexpr size_t ldc{(m + 16U - 1U) / 16U * 16U};

    static_assert(lda >= k);
    static_assert(ldb >= n);
    static_assert(ldc >= n);

    std::cout << "Matrix size: " << m << " x " << n << " x " << k << std::endl;
    std::cout << "Matrix A: " << m << " x " << k << " Leading Dimension Size " << lda << std::endl;
    std::cout << "Matrix B: " << k << " x " << n << " Leading Dimension Size " << ldb << std::endl;
    std::cout << "Matrix C: " << m << " x " << n << " Leading Dimension Size " << ldc << std::endl;

    // Define all the gemm kernel launch functions
    std::vector<
        std::pair<
            std::string, 
            std::function<void(size_t, size_t, size_t, 
                                const float*, 
                                const float*, size_t,
                                const float*, size_t, 
                                const float*,
                                float*, size_t, 
                                cudaStream_t)>>> const gemm_kernel_launch_functions {
                                    {"Custom gemm kernel V00", launch_gemm_kernel_v00<float>},
                                    {"Custom gemm kernel V01", launch_gemm_kernel_v01<float>},
                                };

    for (auto gemm_kernel_launch_function : gemm_kernel_launch_functions) {
        std::cout << gemm_kernel_launch_function.first << std::endl;
        std::pair<float, float> gemm_kernel_profile_result{
            profile_gemm<float>(m, n, k, lda, ldb, ldc, gemm_kernel_launch_function.second,
                                fp32_abs_tol, fp32_rel_tol, num_repeats, num_warmups)};
        std::cout << std::endl;
    }
    return 0;
}