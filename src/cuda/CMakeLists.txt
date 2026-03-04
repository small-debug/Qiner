cmake_minimum_required(VERSION 3.21)

# CUDA library for score_addition GPU port
add_library(score_addition_cuda STATIC
    score_addition_cuda.cu
)

# Use C++17 for CUDA; nvcc can miscompile or fail with C++20 on some setups
set_target_properties(score_addition_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

target_include_directories(score_addition_cuda
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

# Support multiple GPU types. CUDA 13.x only supports sm_75+ (Turing and newer).
# 75=Turing (RTX 20xx), 80=Ampere (RTX 30xx), 89=Ada (RTX 40xx). For one GPU use: -DCUDA_ARCHITECTURES="75"
set(CUDA_ARCHITECTURES_DEFAULT "75;80;89")
set(CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES_DEFAULT}" CACHE STRING "CUDA architectures to build for (semicolon-separated, e.g. 75 or 75;80;89)")
# If cache had old 60/70 (unsupported in CUDA 13.x), replace with supported list
if(CUDA_ARCHITECTURES MATCHES "60|70")
    set(CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES_DEFAULT}" CACHE STRING "CUDA architectures (75+ for CUDA 13.x)" FORCE)
    message(STATUS "CUDA_ARCHITECTURES: reset to ${CUDA_ARCHITECTURES} (60/70 not supported in CUDA 13.x)")
endif()
set_target_properties(score_addition_cuda PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")

set_target_properties(score_addition_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Disable MSVC /RTC1 for CUDA (nvcc + host cl.exe often fail with runtime checks on Windows)
if(MSVC)
    set_target_properties(score_addition_cuda PROPERTIES
        CUDA_NVCC_FLAGS_DEBUG "-Xcompiler=/RTC-"
    )
endif()
