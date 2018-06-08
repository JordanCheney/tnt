project(tnt)
cmake_minimum_required(VERSION 3.0.0)

include(cmake/version.cmake)
include(cmake/flags.cmake)
include(cmake/coverage.cmake)
include(cmake/sanitize.cmake)
include(cmake/docs.cmake)

file(GLOB_RECURSE TNT_HEADERS "include/tnt/*.h" "include/tnt/*.hpp")

# Add interface library for convenience
add_library(tnt INTERFACE)
target_sources(tnt INTERFACE ${TNT_HEADERS})
target_include_directories(tnt INTERFACE include include/tnt/deps/simd)

# Build the unit tests
add_executable(tnt_tests src/test.cpp)
target_include_directories(tnt_tests PUBLIC include include/tnt/deps/simd)

# Install the headers
install(DIRECTORY include/tnt
        DESTINATION include)

option(TNT_WITH_TEST_COVERAGE OFF)
if (${TNT_WITH_TEST_COVERAGE})
  target_compile_options(tnt_tests PRIVATE "-fprofile-arcs -ftest-coverage")
#  file(GLOB_RECURSE TNT_CORE_HEADERS "include/tnt/core/*.hpp"
#	                             "include/tnt/math/*.hpp"
#				     "include/tnt/linear/*.hpp")
#  tnt_add_coverage(${TNT_CORE_HEADERS} TRUE) 
endif()

#option(TNT_WITH_SANITIZERS OFF)
#if (${TNT_WITH_SANITIZERS})
#  tnt_add_sanitizers(tnt_tests) 
#endif()

# Optionally build the benchmarks
add_subdirectory(benchmark)
