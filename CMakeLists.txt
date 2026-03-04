cmake_minimum_required(VERSION 3.21)
project(Qiner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
endif()

OPTION(PORTABLE "Build for portable x86-64 (no RDRAND/BMI)" OFF)

# Additional cmake flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(PORTABLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=x86-64")
        add_compile_definitions(PORTABLE)
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mrdrnd -mbmi")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mrdrnd -mbmi")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(PORTABLE)
        add_compile_definitions(PORTABLE)
    endif()
else()
    message(FATAL "Unsupported compiler.")
endif()

# Compiled output
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Options
OPTION(BUILD_TOOLS "Build Qiner tools" OFF)
OPTION(BUILD_CUDA "Build CUDA components (score_addition GPU port)" OFF)

# Main executable
add_executable(${PROJECT_NAME} src/Qiner.cpp src/keyUtils.cpp)

include_directories(${CMAKE_SOURCE_DIR}/src)

# Linking
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)
    target_link_libraries(${PROJECT_NAME} PRIVATE wsock32 ws2_32)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
endif()

# Other projects
add_subdirectory(test)
if (BUILD_CUDA)
    # Help CMake/VS find CUDA: set compiler and VS toolset CUDA dir (fixes empty CudaToolkitDir in MSBuild)
    if (DEFINED CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/bin/nvcc.exe")
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler")
        set(CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR "${CUDAToolkit_ROOT}" CACHE PATH "CUDA toolkit dir for VS generator")
    elseif (DEFINED ENV{CUDAToolkit_ROOT} AND EXISTS "$ENV{CUDAToolkit_ROOT}/bin/nvcc.exe")
        set(CMAKE_CUDA_COMPILER "$ENV{CUDAToolkit_ROOT}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler")
        set(CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR "$ENV{CUDAToolkit_ROOT}" CACHE PATH "CUDA toolkit dir for VS generator")
    endif()
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        # Ensure VS generator gets toolkit path (CudaToolkitDir in .vcxproj)
        if (CMAKE_GENERATOR MATCHES "Visual Studio" AND CUDAToolkit_ROOT)
            set(CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR "${CUDAToolkit_ROOT}" CACHE PATH "CUDA toolkit dir for VS" FORCE)
            # TryCompile projects don't get CudaToolkitDir; skip the test so configure can complete.
            set(CMAKE_CUDA_COMPILER_WORKS 1 CACHE INTERNAL "Skip CUDA TryCompile (VS generator)")
        endif()
        enable_language(CUDA)
        message(STATUS "Enable CUDA (score_addition GPU port).")
        add_subdirectory(src/cuda)
        target_link_libraries(${PROJECT_NAME} PRIVATE score_addition_cuda CUDA::cudart)
        target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_CUDA_ADDITION)
    else()
        message(WARNING "BUILD_CUDA=ON but CUDA toolkit not found. Disabling CUDA.")
        set(BUILD_CUDA OFF CACHE BOOL "Build CUDA components (score_addition GPU port)" FORCE)
    endif()
endif()
if (BUILD_TOOLS)
    message(STATUS "Enable build Qiner tools.")
    add_subdirectory(tools)
endif()
