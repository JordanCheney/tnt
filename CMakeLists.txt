project(tnt)
cmake_minimum_required(VERSION 3.0.0)

include(cmake/version.cmake)
include(cmake/flags.cmake)
include(cmake/blas.cmake)
include(cmake/coverage.cmake)
include(cmake/docs.cmake)

# Include headers
include_directories(include)
include_directories(include/tnt/deps/simd/)

set(TNT_HEADERS include/tnt/tnt.hpp
 
                include/tnt/core/core.hpp
                include/tnt/core/aligned_ptr.hpp
                include/tnt/core/impl/aligned_ptr_impl.hpp
                include/tnt/core/index.hpp
                include/tnt/core/impl/index_impl.hpp
                include/tnt/core/range.hpp
                include/tnt/core/impl/range_impl.hpp
                include/tnt/core/shape.hpp
                include/tnt/core/impl/shape_impl.hpp
                include/tnt/core/stride.hpp
                include/tnt/core/impl/stride_impl.hpp
                include/tnt/core/tensor.hpp
                include/tnt/core/impl/tensor_impl.hpp
                include/tnt/core/tensor_view.hpp
                include/tnt/core/impl/tensor_view_impl.hpp

                include/tnt/utils/utils.hpp
                include/tnt/utils/macros.hpp
                include/tnt/utils/errors.hpp
                include/tnt/utils/testing.hpp
                include/tnt/utils/simd.hpp

                include/tnt/deps/cblas.hpp
                include/tnt/deps/doctest.hpp

                include/tnt/math/math.hpp
                include/tnt/math/bitwise_ops.hpp
                include/tnt/math/impl/bitwise_not_impl.hpp
                include/tnt/math/impl/bitwise_and_impl.hpp
                include/tnt/math/impl/bitwise_or_impl.hpp
                include/tnt/math/impl/bitwise_xor_impl.hpp

                include/tnt/math/compare_ops.hpp
                include/tnt/math/impl/compare_equal_impl.hpp
                include/tnt/math/impl/compare_not_equal_impl.hpp
                include/tnt/math/impl/compare_less_than_impl.hpp
                include/tnt/math/impl/compare_greater_than_impl.hpp
                include/tnt/math/impl/compare_less_or_equal_impl.hpp
                include/tnt/math/impl/compare_greater_or_equal_impl.hpp

                include/tnt/math/arithmetic_ops.hpp
                include/tnt/math/impl/add_impl.hpp
                include/tnt/math/impl/divide_impl.hpp
                include/tnt/math/impl/multiply_impl.hpp
                include/tnt/math/impl/subtract_impl.hpp

                include/tnt/linear/linear.hpp
                include/tnt/linear/eigen.hpp
                include/tnt/linear/impl/eigen_impl.hpp
                include/tnt/linear/matrix_multiply.hpp
                include/tnt/linear/impl/matrix_multiply_impl.hpp
)

# Build the unit tests
add_executable(tnt src/test.cpp ${TNT_HEADERS})
target_link_libraries(tnt ${BLAS_LIBRARIES})

# Install the headers
install(FILES ${TNT_HEADERS}
        DESTINATION include)

# Optionally build the benchmarks
add_subdirectory(benchmark)
