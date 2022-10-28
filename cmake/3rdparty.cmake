# ---- libsimdpp

include(cmake/SimdppMultiarch.cmake)
message(STATUS "Checking instruction set support in the compiler...")
simdpp_get_compilable_archs(COMPILABLE_ARCHS)
message(STATUS "Checking instruction sets to run tests for on this host...")
simdpp_get_runnable_archs(NATIVE_ARCHS)

add_library(simdpp INTERFACE)
target_include_directories(simdpp INTERFACE 3rdparty/libsimdpp)
