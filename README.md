# TNT - Templated Numeric Tensors Library

[![Build Status](https://travis-ci.org/JordanCheney/tnt.svg?branch=master)](https://travis-ci.org/JordanCheney/tnt)

TNT is a C++11 library for creating and manipulating N-dimensional tensors on the
CPU. The library is a side project and is in very early
stages of development. It is not suitable for production.

The library offers a large test suite, continuous integration and terrible documentation.
The latter will be improved.

## Design Decisions

The library is designed with performance and simplicity of use and 
implementation as primary design considerations, leaning towards
simplicity where trade-offs are considered. Benchmarks against Eigen
and OpenCV are available for many operations.

The main contribution of the library is an N-Dimensional tensor object
with a numeric type-

* unsigned: `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`
* signed: `int8_t`, `int16_t`, `int32_t`, `int64_t`
* floating: `float`, `double`

Half precision floats may be considered in the future. 
To enable both simplicity of implementation and high-performance, 
tensors always contain a single, contiguous block of memory.
Non-contiguous slices of a tensor are represented as a non-owning view,
which offer a subset of utility functions.

## Functionality

* Core operations
    - [x] Per-element access
    - [x] Range based n-dimensional slicing
    - [x] Bidirectional iterators
    - [x] Copy and Move constructors
    - [x] Aligned memory allocation for SIMD
    - [x] SIMD accelerated Mask operations (<, <=, >, >=, ==, !=)

* Math operations
    - [x] SIMD accelerated element operations (+, -, *, /)
    - [] SIMD accelerated global and per axis summarization statistics (mean, median, mode, min, max)
    - [x] BLAS accelerated matrix multiplication

* Linear algebra
    - [x] Eigenvector and Eigenvalue computation
    - [] Discrete Fourier Transform
    - [] Discrete Cosine Transform
    - [] N-D Convolution
    - [] Winograd's convolution algorithm for small kernels

* Image processing
    - [] JPEG compression / decompression

## Building

The library is header-only but has executables for testing and benchmarking. 
It uses CMake 3.X as its build system. To build and run the unit tests run-

1. `cd <tnt_root>`
2. `mkdir build && cd build`
3. `cmake ..`
4. `make -j8`
5. `./tnt_tests`

To run the benchmarks do-

1. `cd <tnt_root>`
2. `mkdir build && cd build`
3. `cmake -DBUILD_TNT_BENCHMARKS=ON ..`
4. `make -j8`
5. `./benchmark/tnt_benchmarks`
