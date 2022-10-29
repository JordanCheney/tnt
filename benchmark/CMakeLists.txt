option(BUILD_TNT_BENCHMARKS "Build the TNT benchmarks" OFF)
if (BUILD_TNT_BENCHMARKS)
    # Add Google Benchmarking
    set(BENCHMARK_BUILD_32_BITS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_EXCEPTIONS ON CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_LTO OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_USE_LIBCXX OFF CACHE BOOL "" FORCE)

    add_subdirectory(${TNT_THIRDPARTY_DIR}/benchmark ${CMAKE_BINARY_DIR}/3rdparty/benchmark)

    # Link against OpenCV libraries
    find_package(OpenCV 4 REQUIRED)

    # Link against BLAS
    find_package(BLAS REQUIRED)

    # Gather benchmarks
    set(TNT_BENCHMARKS src/math/add.cpp
                       src/math/multiply.cpp
                       src/math/divide.cpp
                       src/linear/matrix_multiply.cpp)

    # Build the benchmark executable
    add_executable(tnt_benchmarks run_benchmarks.cpp ${TNT_BENCHMARKS})
    target_compile_options(tnt_benchmarks PRIVATE
        "-DDOCTEST_CONFIG_DISABLE"
        "-Wno-unused-parameter"
        "-Wno-unused-variable"
        "-Wno-unused-local-typedef")
    target_include_directories(tnt_benchmarks PRIVATE include ${TNT_THIRDPARTY_DIR}/eigen)
    target_link_libraries(tnt_benchmarks PRIVATE tnt benchmark::benchmark ${BLAS_LIBRARIES} opencv_core)
    #target_link_libraries(tnt_benchmarks LINK_PUBLIC tnt)

    install(TARGETS tnt_benchmarks
            RUNTIME DESTINATION bin)
endif()
