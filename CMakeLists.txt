cmake_minimum_required(VERSION 4.3.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
project(NeRF VERSION 1.0.0 LANGUAGES CXX CUDA)

option(NERF_BUILD_BENCHMARK "Build benchmark applications" OFF)

# ============================================================================
# C++ Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)


# ============================================================================
# CUDA Configuration
# ============================================================================
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ============================================================================
# Dependencies
# ============================================================================
find_package(CUDAToolkit 13.2 REQUIRED)

function(download_if_needed url out_path)
    if (EXISTS "${out_path}")
        message(STATUS "Using cached file: ${out_path}")
        return()
    endif ()

    get_filename_component(out_dir "${out_path}" DIRECTORY)
    file(MAKE_DIRECTORY "${out_dir}")

    message(STATUS "Downloading ${url}")
    file(DOWNLOAD
            "${url}"
            "${out_path}"
            TLS_VERIFY ON
            STATUS dl_status
    )

    list(GET dl_status 0 dl_code)
    list(GET dl_status 1 dl_msg)
    if (NOT dl_code EQUAL 0)
        message(FATAL_ERROR "Failed to download ${url}: [${dl_code}] ${dl_msg}")
    endif ()
endfunction()

download_if_needed(
        "https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp"
        "${CMAKE_CURRENT_BINARY_DIR}/vendor/json/json.hpp"
)


# ============================================================================
# Libraries
# ============================================================================
add_library(nerf SHARED)
target_sources(
        nerf
        PRIVATE
        backend-cuda.cu
        PUBLIC
        nerf.h
)
target_link_libraries(nerf PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::cublas)
target_compile_definitions(nerf PRIVATE NOMINMAX NERF_BUILD_SHARED NERF_BUILD_EXPORTS)
target_include_directories(nerf PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vendor")


# ============================================================================
# Benchmark
# ============================================================================
if (NERF_BUILD_BENCHMARK)
    download_if_needed(
            "https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image.h"
            "${CMAKE_CURRENT_BINARY_DIR}/vendor/stb/stb_image.h"
    )
    add_executable(benchmark benchmark.cpp)
    target_link_libraries(benchmark PRIVATE nerf)
    target_include_directories(benchmark PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vendor")
    set_property(TARGET benchmark PROPERTY CXX_MODULE_STD ON)
endif ()
