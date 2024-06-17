#pragma once
#include <cuda_runtime.h>

// define cuda util functions, e.g., print device information and error checking

#define CHECK_CUDA_ERROR(val) check_cuda_error(val, #val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() check_last_cuda_error(__FILE__, __LINE__)

void check_cuda_error(cudaError_t error, const char *const func, const char *const file, int const line);
// const T *: pointer to a fix T value
// T* const: a fix pointer to a T value
// const char* const: a fix pointer to a fix string
void check_last_cuda_error(const char *const file, int const line);