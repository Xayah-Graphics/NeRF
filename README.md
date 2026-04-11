# [ECCV 2020 Oral] NeRF. Ben Mildenhall, et al.
[![Linux Build (Arch)](https://github.com/Xayah-Graphics/NeRF/actions/workflows/arch-build.yml/badge.svg)](https://github.com/Xayah-Graphics/NeRF/actions/workflows/arch-build.yml)
[![Windows Build](https://github.com/Xayah-Graphics/NeRF/actions/workflows/windows-build.yml/badge.svg)](https://github.com/Xayah-Graphics/NeRF/actions/workflows/windows-build.yml)

## 1. Algorithm Pipeline

On dev...

## 2. Build Instruction

#### Build C ABI library
- CMake 4.3.0 or higher
- Ninja build system (for CXX std module support)
- A C++23 compliant compiler (tested on Arch Linux with gcc/g++ 15.2.1, Windows with MSVC 17.14.29)
- NVIDIA CUDA 13.2 or higher

```
cmake -B build -S . -G Ninja
cmake --build build --parallel
```

#### Build Benchmark Executable

```
cmake -B build -S . -G Ninja -DNERF_BUILD_BENCHMARK=ON
cmake --build build --parallel
```
