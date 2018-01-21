# TNT - Templated Numerical Tensors Library

[![Build Status](https://travis-ci.org/JordanCheney/tnt.svg?branch=master)](https://travis-ci.org/JordanCheney/tnt)

TNT is a C++11 library for creating and manipulating N-dimensional tensors on the
CPU and, in the future, GPU. The library is a side project and is in very early
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

and backened (`CPU` or `GPU`). Half precision floats may be considered
in the future. To enable both simplicity of implementation and high-
performance, tensors always contain a single, contiguous block of memory.
Non-contiguous slices of a tensor are represented as a non-owning view,
which offer a subset of utility functions.

## Functionality

* Core operations
    - [x] Per-element access
    - [x] Range based n-dimensional slicing
    - [x] Bidirectional iterators
    - [x] Copy and Move constructors
    - [x] Aligned memory allocation for SIMD
    - [] SIMD accelerated Mask operations (<, <=, >, >=, ==, !=)

* Math operations
    - [] SIMD accelerated element operations (+, -, *, /)
    - [] SIMD accelerated global and per axis summarization statistics (mean, median, mode, min, max)
    - [] BLAS accelerated matrix multiplication

* Linear algebra
    - [x] Eigenvector and Eigenvalue computation
    - [] Discrete Fourier Transform
    - [] Discrete Cosine Transform
    - [] N-D Convolution
    - [] Winograd's convolution algorithm for small kernels

* Image processing
    - [] JPEG compression / decompression

