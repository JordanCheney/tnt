#ifndef TNT_MATH_BITWISE_XOR_IMPL_HPP
#define TNT_MATH_BITWISE_XOR_IMPL_HPP

#include <tnt/math/bitwise_ops.hpp>
#include <tnt/utils/testing.hpp>
#include <tnt/utils/simd.hpp>

#include <simdpp/simd.h>

namespace tnt
{

namespace detail
{

template <typename LeftType, typename RightType>
struct OptimizedBitwiseXor<LeftType, RightType>
{
    static void eval(Tensor<LeftType>& tensor, const RightType& _scalar)
    {
        LeftType* ptr = tensor.data.data;

        LeftType scalar = static_cast<LeftType>(_scalar);
        auto scalar_vec = simdpp::load_splat<typename SIMDType<LeftType>::VecType>(&scalar);

        int offset = 0, num_blocks = AlignSIMDType<LeftType>::num_aligned_blocks(tensor.shape.total());
        for ( ; num_blocks--; offset += OptimalSIMDSize<LeftType>::value) {
            auto block = LoadSIMDType<LeftType, LeftType>::load(ptr + offset);
            auto result = simdpp::bit_xor(block, scalar_vec);
            simdpp::store(ptr + offset, result);
        }
    }

    static void eval(Tensor<LeftType>& left, const Tensor<RightType>& right)
    {
        LeftType*  l_ptr = left.data.data;
        RightType* r_ptr = right.data.data;

        int offset = 0, num_blocks = AlignSIMDType<LeftType>::num_aligned_blocks(left.shape.total());
        for ( ; num_blocks--; offset += OptimalSIMDSize<LeftType>::value) {
            auto l_block = LoadSIMDType<LeftType, LeftType>::load(l_ptr + offset);
            auto r_block = LoadSIMDType<LeftType, RightType>::load(r_ptr + offset);
            auto result = simdpp::bit_xor(l_block, r_block);
            simdpp::store(l_ptr + offset, result);
        }
    }
};

} // namespace detail

// ----------------------------------------------------------------------------
// Unit tests

TEST_CASE_TEMPLATE("bitwise_xor(Tensor<unsigned>&, Scalar)", T, test_unsigned_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [](const Shape& shape) {
        REQUIRE(((TensorType(shape, 0)   ^ 0)  == TensorType(shape, 0)));
        REQUIRE(((TensorType(shape, 1)   ^ 1)  == TensorType(shape, 0)));
        REQUIRE(((TensorType(shape, 1)   ^ 2)  == TensorType(shape, 3)));
        REQUIRE(((TensorType(shape, 1)   ^ 3)  == TensorType(shape, 2)));
        REQUIRE(((TensorType(shape, 170) ^ 85) == TensorType(shape, 255)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("bitwise_xor(Tensor<signed>&, Scalar)", T, test_signed_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [](const Shape& shape) {
        REQUIRE(((TensorType(shape, 0)   ^  0)  == TensorType(shape, 0)));
        REQUIRE(((TensorType(shape, 1)   ^  1)  == TensorType(shape, 0)));
        REQUIRE(((TensorType(shape, -1)  ^  2)  == TensorType(shape, -3)));
        REQUIRE(((TensorType(shape, 1)   ^  3)  == TensorType(shape, 2)));
        REQUIRE(((TensorType(shape, -86) ^  85) == TensorType(shape, -1)));
        REQUIRE(((TensorType(shape, -8)  ^ -16) == TensorType(shape, 8)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

} // namespace tnt

#endif // TNT_MATH_BITWISE_XOR_IMPL_HPP
