#ifndef TNT_MATH_BITWISE_NOT_IMPL_HPP
#define TNT_MATH_BITWISE_NOT_IMPL_HPP

#include <tnt/math/bitwise_ops.hpp>
#include <tnt/utils/testing.hpp>
#include <tnt/utils/simd.hpp>

namespace tnt
{

namespace detail
{

template <typename LeftType>
struct OptimizedBitwiseNot<LeftType>
{
    static void eval(Tensor<LeftType>& tensor)
    {
        LeftType* ptr = tensor.data.data;

        int offset = 0, num_blocks = AlignSIMDType<LeftType>::num_aligned_blocks(tensor.shape.total());
        for ( ; num_blocks--; offset += OptimalSIMDSize<LeftType>::value) {
            auto block = LoadSIMDType<LeftType, LeftType>::load(ptr + offset);
            auto result = simdpp::bit_not(block);
            simdpp::store(ptr + offset, result);
        }
    }
};

} // namespace detail

// ----------------------------------------------------------------------------
// Unit tests

TEST_CASE_TEMPLATE("bitwise_not(Tensor<unsigned>&)", T, test_unsigned_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [](const Shape& shape) {
        REQUIRE((~TensorType(shape, 0)  == TensorType(shape, std::numeric_limits<T>::max())));
        REQUIRE((~TensorType(shape, 8)  == TensorType(shape, std::numeric_limits<T>::max() - static_cast<T>(8))));
        REQUIRE((~TensorType(shape, 67) == TensorType(shape, std::numeric_limits<T>::max() - static_cast<T>(67))));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("bitwise_not(Tensor<signed>&)", T, test_signed_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [](const Shape& shape) {
        REQUIRE((~TensorType(shape, 0)  == TensorType(shape, -1)));
        REQUIRE((~TensorType(shape, 8)  == TensorType(shape, -9)));
        REQUIRE((~TensorType(shape, 67) == TensorType(shape, -68)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

} // namespace tnt

#endif // TNT_MATH_BITWISE_NOT_IMPL_HPP
