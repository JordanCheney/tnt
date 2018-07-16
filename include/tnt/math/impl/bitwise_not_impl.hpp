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
    using VecType = typename SIMDType<LeftType>::VecType;

    static void eval(Tensor<LeftType>& tensor)
    {
        constexpr int num_regs = 10;
        VecType regs[num_regs];

        LeftType* ptr = tensor.data.data;

        int offset = 0, num_blocks = AlignSIMDType<LeftType>::num_aligned_blocks(tensor.shape.total());
        for ( ; num_blocks; ) {
            const int block_size = std::min(num_regs, num_blocks);

            VecType* reg = regs;
            for (int i = 0; i < block_size; ++i, ++reg) {
                *reg = LoadSIMDType<LeftType, LeftType>::load(ptr + offset + i * OptimalSIMDSize<LeftType>::value);
                *reg = simdpp::bit_not(*reg);
            }

            for (int i = 0; i < block_size; ++i) {
                simdpp::store(ptr + offset + i * OptimalSIMDSize<LeftType>::value, regs[i]);
            }

            offset += block_size * OptimalSIMDSize<LeftType>::value;
            num_blocks -= block_size;
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
