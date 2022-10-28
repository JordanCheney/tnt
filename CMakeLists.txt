project(tnt)
cmake_minimum_required(VERSION 3.1.0)
set(CMAKE_CXX_STANDARD 11)

include(cmake/version.cmake)
include(cmake/coverage.cmake)
include(cmake/sanitize.cmake)
include(cmake/docs.cmake)
include(cmake/3rdparty.cmake)

# Add interface library for convenience
add_library(tnt INTERFACE)
target_include_directories(tnt INTERFACE include)
target_link_libraries(tnt INTERFACE simdpp doctest)

# Build the unit tests
add_executable(tnt_tests src/test.cpp)
target_link_libraries(tnt_tests LINK_PUBLIC tnt doctest)

# Install the headers
install(DIRECTORY include/tnt
        DESTINATION include)

option(TNT_WITH_TEST_COVERAGE OFF)
if (${TNT_WITH_TEST_COVERAGE})
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(tnt_tests PRIVATE "-fprofile-arcs -ftest-coverage")
  else()
    message(WARNING "Code coverage is currently only supported with GCC")
  endif()
endif()

#option(TNT_WITH_SANITIZERS OFF)
#if (${TNT_WITH_SANITIZERS})
#  tnt_add_sanitizers(tnt_tests)
#endif()

# Optionally build the benchmarks
add_subdirectory(benchmark)
