#ifndef TNT_MATH_BITWISE_OR_IMPL_HPP
#define TNT_MATH_BITWISE_OR_IMPL_HPP

#include <tnt/math/bitwise_ops.hpp>
#include <tnt/utils/testing.hpp>
#include <tnt/utils/simd.hpp>

namespace tnt
{

namespace detail
{

template <typename LeftType, typename RightType>
struct OptimizedBitwiseOr<LeftType, RightType>
{
    using VecType = typename SIMDType<LeftType>::VecType;

    static void eval(Tensor<LeftType>& tensor, const RightType& _scalar)
    {
        constexpr int num_regs = 10;
        VecType regs[num_regs];

        LeftType* ptr = tensor.data.data;

        LeftType scalar = static_cast<LeftType>(_scalar);
        auto scalar_vec = simdpp::load_splat<typename SIMDType<LeftType>::VecType>(&scalar);

        int offset = 0, num_blocks = AlignSIMDType<LeftType>::num_aligned_blocks(tensor.shape.total());
        for ( ; num_blocks; ) {
            const int block_size = std::min(num_regs, num_blocks);

            VecType* reg = regs;
            for (int i = 0; i < block_size; ++i, ++reg) {
                *reg = LoadSIMDType<LeftType, LeftType>::load(ptr + offset + i * OptimalSIMDSize<LeftType>::value);
                *reg = simdpp::bit_or(*reg, scalar_vec);
            }

            for (int i = 0; i < block_size; ++i) {
                simdpp::store(ptr + offset + i * OptimalSIMDSize<LeftType>::value, regs[i]);
            }

            offset += block_size * OptimalSIMDSize<LeftType>::value;
            num_blocks -= block_size;
        }
    }

    static void eval(Tensor<LeftType>& left, const Tensor<RightType>& right)
    {
        constexpr int num_regs = 6;
        VecType l_regs[num_regs];
        VecType r_regs[num_regs];

        LeftType*  l_ptr = left.data.data;
        RightType* r_ptr = right.data.data;

        int offset = 0, num_blocks = AlignSIMDType<LeftType>::num_aligned_blocks(left.shape.total());
        for ( ; num_blocks; ) {
            const int block_size = std::min(num_regs, num_blocks);

            VecType* lreg = l_regs, * rreg = r_regs;
            for (int i = 0; i < block_size; ++i, ++lreg, ++rreg) {
                const int block_offset = offset + i * OptimalSIMDSize<LeftType>::value;

                *lreg = LoadSIMDType<LeftType, LeftType>::load(l_ptr + block_offset);
                *rreg = LoadSIMDType<LeftType, LeftType>::load(r_ptr + block_offset);
                *lreg = simdpp::bit_or(*lreg, *rreg);
            }

            for (int i = 0; i < block_size; ++i) {
                simdpp::store(l_ptr + offset + i * OptimalSIMDSize<LeftType>::value, l_regs[i]);
            }

            offset += block_size * OptimalSIMDSize<LeftType>::value;
            num_blocks -= block_size;
        }
    }
};

} // namespace detail

// ----------------------------------------------------------------------------
// Unit tests

TEST_CASE_TEMPLATE("bitwise_or(Tensor<unsigned>&, Scalar)", T, test_unsigned_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [](const Shape& shape) {
        REQUIRE(((TensorType(shape, 0)   | 0)  == TensorType(shape, 0)));
        REQUIRE(((TensorType(shape, 1)   | 1)  == TensorType(shape, 1)));
        REQUIRE(((TensorType(shape, 1)   | 2)  == TensorType(shape, 3)));
        REQUIRE(((TensorType(shape, 1)   | 3)  == TensorType(shape, 3)));
        REQUIRE(((TensorType(shape, 170) | 85) == TensorType(shape, 255)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("bitwise_or(Tensor<signed>&, Scalar)", T, test_signed_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [](const Shape& shape) {
        REQUIRE(((TensorType(shape, 0)   |  0)  == TensorType(shape, 0)));
        REQUIRE(((TensorType(shape, 1)   |  1)  == TensorType(shape, 1)));
        REQUIRE(((TensorType(shape, -1)  |  2)  == TensorType(shape, -1)));
        REQUIRE(((TensorType(shape, 1)   |  3)  == TensorType(shape, 3)));
        REQUIRE(((TensorType(shape, -86) |  85) == TensorType(shape, -1)));
        REQUIRE(((TensorType(shape, -8)  | -16) == TensorType(shape, -8)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

} // namespace tnt

#endif // TNT_MATH_BITWISE_OR_CPU_IMPL_HPP
