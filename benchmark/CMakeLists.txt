option(BUILD_TNT_BENCHMARKS "Build the TNT benchmarks" OFF)
if (BUILD_TNT_BENCHMARKS)
    # Add Google Benchmarking
    add_subdirectory(benchmark-1.2.0)
    include_directories(benchmark-1.2.0/include)

    # Set Benchmark options
    set(BENCHMARK_BUILD_32_BITS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_EXCEPTIONS ON CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_LTO OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_USE_LIBCXX OFF CACHE BOOL "" FORCE)

    # Mark Benchmark options as advanced
    mark_as_advanced(BENCHMARK_BUILD_32_BITS)
    mark_as_advanced(BENCHMARK_ENABLE_EXCEPTIONS)
    mark_as_advanced(BENCHMARK_ENABLE_LTO)
    mark_as_advanced(BENCHMARK_ENABLE_TESTING)
    mark_as_advanced(BENCHMARK_USE_LIBCXX)

    # Link against OpenCV libraries
    find_package(OpenCV 3 REQUIRED)

    # Link against BLAS
    find_package(BLAS REQUIRED)

    # Gather benchmarks
    set(TNT_BENCHMARKS src/math/add.cpp
                       src/math/multiply.cpp
                       src/math/divide.cpp
                       src/linear/matrix_multiply.cpp)

    # Build the benchmark executable
    add_executable(tnt_benchmarks run_benchmarks.cpp ${TNT_BENCHMARKS})
    target_compile_options(tnt_benchmarks PRIVATE "-DDOCTEST_CONFIG_DISABLE" "-Wno-unused-parameter" "-Wno-unused-variable" "-Wno-unused-local-typedef")
    target_link_libraries(tnt_benchmarks benchmark ${BLAS_LIBRARIES} opencv_core)
    target_include_directories(tnt_benchmarks PRIVATE include ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/tnt/deps/simd)

    install(TARGETS tnt_benchmarks
            RUNTIME DESTINATION bin)
endif()
