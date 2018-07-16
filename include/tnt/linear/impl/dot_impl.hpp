#ifndef TNT_LINEAR_DOT_IMPL_HPP
#define TNT_LINEAR_DOT_IMPL_HPP

#include <tnt/linear/dot.hpp>

#include <tnt/utils/testing.hpp>
#include <tnt/utils/simd.hpp>

namespace tnt
{

namespace detail
{

template <typename LeftType, typename RightType>
struct OptimizedDot<LeftType, RightType,
            typename std::enable_if<std::is_same<LeftType, uint8_t>::value
                                    || std::is_same<LeftType, int8_t>::value
                                    || std::is_same<LeftType, uint64_t>::value
                                    || std::is_same<LeftType, int64_t>::value>::type>
{
    static LeftType eval(const Tensor<LeftType>& left, const Tensor<RightType>& right)
    {
        const LeftType*  l_ptr = left.data.data;
        const RightType* r_ptr = right.data.data;

        LeftType sum = 0;

        int i = 0, total = left.shape.total();
        for ( ; total--; ++i) {
            sum += l_ptr[i] * static_cast<LeftType>(r_ptr[i]);
        }

        return sum;
    }
};

template <typename LeftType, typename RightType>
struct OptimizedDot<LeftType, RightType,
            typename std::enable_if<std::is_same<LeftType, uint16_t>::value
                                    || std::is_same<LeftType, int16_t>::value
                                    || std::is_same<LeftType, uint32_t>::value
                                    || std::is_same<LeftType, int32_t>::value>::type>
{
    using VecType = typename SIMDType<LeftType>::VecType;

    static LeftType eval(const Tensor<LeftType>& left, const Tensor<RightType>& right)
    {
        constexpr int num_regs = 6;
        VecType l_regs[num_regs];
        VecType r_regs[num_regs];

        const LeftType*  l_ptr = left.data.data;
        const RightType* r_ptr = right.data.data;

        LeftType sum = 0;

        int offset = 0, num_blocks = AlignSIMDType<LeftType>::num_aligned_blocks(left.shape.total());
        for ( ; num_blocks; ) {
            const int block_size = std::min(num_regs, num_blocks);

            VecType* lreg = l_regs, * rreg = r_regs;
            for (int i = 0; i < block_size; ++i, ++lreg, ++rreg) {
                const int block_offset = offset + i * OptimalSIMDSize<LeftType>::value;

                *lreg = LoadSIMDType<LeftType, LeftType>::load(l_ptr + block_offset);
                *rreg = LoadSIMDType<LeftType, RightType>::load(r_ptr + block_offset);
                *lreg = ConvertSIMDType<LeftType>::convert(simdpp::mull(*lreg, *rreg));
            }

            for (int i = 0; i < block_size; ++i) {
                sum += (LeftType) simdpp::reduce_add(l_regs[i]);
            }

            offset += block_size * OptimalSIMDSize<LeftType>::value;
            num_blocks -= block_size;
        }

        return sum;
    }
};

template <typename LeftType, typename RightType>
struct OptimizedDot<LeftType, RightType,
            typename std::enable_if<std::is_floating_point<LeftType>::value>::type>
{
    using VecType = typename SIMDType<LeftType>::VecType;

    static LeftType eval(const Tensor<LeftType>& left, const Tensor<RightType>& right)
    {
        constexpr int num_regs = 6;
        VecType l_regs[num_regs];
        VecType r_regs[num_regs];

        const LeftType*  l_ptr = left.data.data;
        const RightType* r_ptr = right.data.data;

        LeftType sum = 0;

        int offset = 0, num_blocks = AlignSIMDType<LeftType>::num_aligned_blocks(left.shape.total());
        for ( ; num_blocks; ) {
            const int block_size = std::min(num_regs, num_blocks);

            VecType* lreg = l_regs, * rreg = r_regs;
            for (int i = 0; i < block_size; ++i, ++lreg, ++rreg) {
                const int block_offset = offset + i * OptimalSIMDSize<LeftType>::value;

                *lreg = LoadSIMDType<LeftType, LeftType>::load(l_ptr + block_offset);
                *rreg = LoadSIMDType<LeftType, RightType>::load(r_ptr + block_offset);
                *lreg = simdpp::mul(*lreg, *rreg);
            }

            for (int i = 0; i < block_size; ++i) {
                sum += (LeftType) simdpp::reduce_add(l_regs[i]);
            }

            offset += block_size * OptimalSIMDSize<LeftType>::value;
            num_blocks -= block_size;
        }

        return sum;
    }
};

} // namespace detail

// ----------------------------------------------------------------------------
// Unit tests

TEST_CASE_TEMPLATE("dot(const Tensor<unsigned>&, const Tensor<unsigned>&)", T, test_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        REQUIRE(tnt::dot(TensorType(shape, 1), TensorType(shape, 2)) == shape.total() * 2);
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{1, 2, 3, 4});
}

} // namespace tnt

#endif // TNT_LINEAR_DOT_IMPL_HPP
