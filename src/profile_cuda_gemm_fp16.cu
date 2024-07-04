#include <iostream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "profile_utils.cuh"
#include "cuda_gemm.hpp"

int main() 
{

    // print device information
    print_device_info();

    constexpr size_t num_repeats{1U};
    constexpr size_t num_warmups{1U};

    __half const fp16_abs_tol{__float2half(5.0e-2f)};
    double const fp16_rel_tol{1.0e-1f};

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
                                const __half*,
                                const __half*, size_t, 
                                const __half*, size_t,
                                const __half*,
                                __half*, size_t,
                                cudaStream_t)>>> const gemm_kernel_launch_functions {
                                    {"Custom gemm kernel V00", launch_gemm_kernel_v00<__half>},
                                    {"Custom gemm kernel V01", launch_gemm_kernel_v01<__half>},
                                    {"Custom gemm kernel V02", launch_gemm_kernel_v02<__half>},
                                    {"Custom gemm kernel V02 vectorized", launch_gemm_kernel_v02_vectorized<__half>},
                                    {"custom gemm kernel V03", launch_gemm_kernel_v03<__half>},
                                    {"custom gemm kernel V03 vectorized", launch_gemm_kernel_v03_vectorized<__half>},
                                    {"custom gemm kernel V04", launch_gemm_kernel_v04<__half>},
                                    {"custom gemm kernel V04 vectorized", launch_gemm_kernel_v04_vectorized<__half>},
                                    {"custom gemm kernel V05", launch_gemm_kernel_v05<__half>},
                                    {"custom gemm kernel V05_vectorized", launch_gemm_kernel_v05_vectorized<__half>},
                                    {"custom gemm kernel V06", launch_gemm_kernel_v06<__half>},
                                    {"custom gemm kernel V06_vectorized", launch_gemm_kernel_v06_vectorized<__half>},
                                    {"custom gemm kernel V07", launch_gemm_kernel_v07<__half>},
                                    {"custom gemm kernel V07_vectorized", launch_gemm_kernel_v07_vectorized<__half>}
                                };

    for (auto gemm_kernel_launch_function : gemm_kernel_launch_functions) {
        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << gemm_kernel_launch_function.first << std::endl;
        std::pair<float, float> gemm_kernel_profile_result{
            profile_gemm<__half>(m, n, k, lda, ldb, ldc, gemm_kernel_launch_function.second,
                                fp16_abs_tol, fp16_rel_tol, num_repeats, num_warmups)};
        std::cout << std::endl;
    }
    return 0;
}

