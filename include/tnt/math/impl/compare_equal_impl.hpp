#ifndef TNT_MATH_COMPARE_EQUAL_IMPL_HPP
#define TNT_MATH_COMPARE_EQUAL_IMPL_HPP

#include <tnt/math/compare_ops.hpp>
#include <tnt/utils/testing.hpp>
#include <tnt/utils/simd.hpp>

namespace tnt
{

namespace detail
{

template <typename LeftType, typename RightType>
struct OptimizedCompareEqual<LeftType, RightType>
{
    using VecType         = typename SIMDType<LeftType>::VecType;
    using UnsignedType    = typename VecType::uint_element_type;
    using UnsignedVecType = typename VecType::uint_vector_type;

    constexpr static int Size = OptimalSIMDSize<LeftType>::value;

    static Tensor<uint8_t> eval(const Tensor<LeftType>& tensor, const RightType& _scalar)
    {
        Tensor<uint8_t> mask(tensor.shape);

        LeftType* l_ptr = tensor.data.data;
        uint8_t*  m_ptr = mask.data.data;

        LeftType scalar = static_cast<LeftType>(_scalar);
        auto scalar_vec = simdpp::load_splat<VecType>(&scalar);

        int offset = 0, num_blocks = AlignSIMDType<LeftType>::num_aligned_blocks(tensor.shape.total());
        for ( ; num_blocks--; offset += Size) {
            auto block = LoadSIMDType<LeftType, LeftType>::load(l_ptr + offset);
            auto result = simdpp::bit_cast<UnsignedVecType>(simdpp::cmp_eq(block, scalar_vec));

            UnsignedType temp[Size];
            simdpp::store(temp, result);

            for (int i = 0; i < Size; ++i)
                (m_ptr + offset)[i] = static_cast<uint8_t>(temp[i]);
        }

        return mask;
    }
};

} // namespace detail

// ----------------------------------------------------------------------------
// Unit tests

TEST_CASE_TEMPLATE("compare_equal(Tensor<T>&, Scalar)", T, test_data_types)
{
    { // 2x2
        T input[4] = {1, 0, 2, 1};
        Tensor<T> tensor(Shape{2, 2}, AlignedPtr<T>(input, 4));

        uint8_t is_zero[4]  = {0, 255, 0, 0};
        uint8_t is_one[4]   = {255, 0, 0, 255};
        uint8_t is_two[4]   = {0, 0, 255, 0};
        uint8_t is_three[4] = {0, 0, 0, 0};

        REQUIRE(((tensor == 0) == Tensor<uint8_t>(Shape{2, 2}, AlignedPtr<uint8_t>(is_zero,  4))));
        REQUIRE(((tensor == 1) == Tensor<uint8_t>(Shape{2, 2}, AlignedPtr<uint8_t>(is_one,   4))));
        REQUIRE(((tensor == 2) == Tensor<uint8_t>(Shape{2, 2}, AlignedPtr<uint8_t>(is_two,   4))));
        REQUIRE(((tensor == 3) == Tensor<uint8_t>(Shape{2, 2}, AlignedPtr<uint8_t>(is_three, 4))));
    }

    { // 3x2x1
        T input[6] = {1, 2, 3, 3, 2, 1};
        Tensor<T> tensor(Shape{3, 2, 1}, AlignedPtr<T>(input, 6));

        uint8_t is_zero[6]  = {0, 0, 0, 0, 0, 0};
        uint8_t is_one[6]   = {255, 0, 0, 0, 0, 255};
        uint8_t is_two[6]   = {0, 255, 0, 0, 255, 0};
        uint8_t is_three[6] = {0, 0, 255, 255, 0, 0};
        uint8_t is_four[6]  = {0, 0, 0, 0, 0, 0};

        REQUIRE(((tensor == 0) == Tensor<uint8_t>(Shape{3, 2, 1}, AlignedPtr<uint8_t>(is_zero,  6))));
        REQUIRE(((tensor == 1) == Tensor<uint8_t>(Shape{3, 2, 1}, AlignedPtr<uint8_t>(is_one,   6))));
        REQUIRE(((tensor == 2) == Tensor<uint8_t>(Shape{3, 2, 1}, AlignedPtr<uint8_t>(is_two,   6))));
        REQUIRE(((tensor == 3) == Tensor<uint8_t>(Shape{3, 2, 1}, AlignedPtr<uint8_t>(is_three, 6))));
        REQUIRE(((tensor == 4) == Tensor<uint8_t>(Shape{3, 2, 1}, AlignedPtr<uint8_t>(is_four,  6))));
    }
}

} // namespace tnt

#endif // TNT_MATH_COMPARE_EQUAL_IMPL_HPP
