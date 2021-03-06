#ifndef TNT_MATH_ADD_IMPL_HPP
#define TNT_MATH_ADD_IMPL_HPP

#include <tnt/math/arithmetic_ops.hpp>
#include <tnt/utils/testing.hpp>
#include <tnt/utils/simd.hpp>

namespace tnt
{

namespace detail
{

template <typename LeftType, typename RightType>
struct OptimizedAdd<LeftType, RightType>
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
                *reg = simdpp::add(*reg, scalar_vec);
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
                *lreg = simdpp::add(*lreg, *rreg);
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

TEST_CASE_TEMPLATE("add(Tensor<unsigned>&, Scalar)", T, test_unsigned_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [](const Shape& shape) {
        TensorType tensor(shape, 0);

        REQUIRE(((tensor +  3)   == TensorType(shape, 3)));
        REQUIRE(((tensor += 5.5) == TensorType(shape, 5)));
        REQUIRE(((tensor +  1)   == TensorType(shape, 6)));
        REQUIRE(((tensor += 4.8) == TensorType(shape, 9)));

        REQUIRE((tensor.template as<uint8_t>() + 127        == Tensor<uint8_t>(shape, 136)));
        REQUIRE((tensor.template as<int>()     + -4         == Tensor<int>    (shape, 5)));
        REQUIRE((tensor.template as<float>()   + 1.3        == Tensor<float>  (shape, 10.3)));
        REQUIRE((tensor.template as<double>()  + -208.71875 == Tensor<double> (shape, -199.71875)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("add(Tensor<signed>&, Scalar)", T, test_signed_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [](const Shape& shape) {
        TensorType tensor(shape, 0);

        REQUIRE(((tensor +  3)    == TensorType(shape, 3)));
        REQUIRE(((tensor += 5.5)  == TensorType(shape, 5)));
        REQUIRE(((tensor +  -10)  == TensorType(shape, -5)));
        REQUIRE(((tensor += -4.8) == TensorType(shape, 1)));

        add(tensor, -3);
        REQUIRE((tensor == TensorType(shape, -2)));

        add(tensor, 2);
        REQUIRE((tensor == TensorType(shape, 0)));

        REQUIRE((tensor.template as<uint8_t>() + 127        == Tensor<uint8_t>(shape, 127)));
        REQUIRE((tensor.template as<int>()     + -4         == Tensor<int>    (shape, -4)));
        REQUIRE((tensor.template as<float>()   + 1.3        == Tensor<float>  (shape, 1.3)));
        REQUIRE((tensor.template as<double>()  + -208.71875 == Tensor<double> (shape, -208.71875)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("add(Tensor<floating>&, Scalar)", T, test_float_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [](const Shape& shape) {
        TensorType tensor(shape, 0);

        REQUIRE(((tensor +  3.72)    == TensorType(shape, 3.72)));
        REQUIRE(((tensor += 4.9999)  == TensorType(shape, 4.9999)));
        REQUIRE(((tensor +  -10)     == TensorType(shape, -5.0001)));
        REQUIRE(((tensor += -4.9999) == TensorType(shape, 0)));

        add(tensor, -3);
        REQUIRE((tensor == TensorType(shape, -3)));

        add(tensor, 3);
        REQUIRE((tensor == TensorType(shape, 0)));

        REQUIRE((tensor.template as<uint8_t>() + 127        == Tensor<uint8_t>(shape, 127)));
        REQUIRE((tensor.template as<int>()     + -4         == Tensor<int>    (shape, -4)));
        REQUIRE((tensor.template as<float>()   + 1.3        == Tensor<float>  (shape, 1.3)));
        REQUIRE((tensor.template as<double>()  + -208.71875 == Tensor<double> (shape, -208.71875)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("add(Tensor<unsigned>&, const Tensor<unsigned>&)", T, test_unsigned_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [](const Shape& shape) {
        TensorType tensor(shape, 0);

        REQUIRE(((tensor +  TensorType(shape, 3)) == TensorType(shape, 3)));
        REQUIRE(((tensor += TensorType(shape, 5)) == TensorType(shape, 5)));
        REQUIRE(((tensor +  TensorType(shape, 1)) == TensorType(shape, 6)));
        REQUIRE(((tensor += TensorType(shape, 4)) == TensorType(shape, 9)));

        REQUIRE((tensor.template as<uint8_t>() + Tensor<uint8_t>(shape, 127)        == Tensor<uint8_t>(shape, 136)));
        REQUIRE((tensor.template as<int>()     + Tensor<int>    (shape, -4)         == Tensor<int>    (shape, 5)));
        REQUIRE((tensor.template as<float>()   + Tensor<float>  (shape, 1.3)        == Tensor<float>  (shape, 10.3)));
        REQUIRE((tensor.template as<double>()  + Tensor<double> (shape, -208.71875) == Tensor<double> (shape, -199.71875)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("add(Tensor<signed>&, const Tensor<signed>&)", T, test_signed_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [](const Shape& shape) {
        TensorType tensor(shape, 0);

        REQUIRE(((tensor +  TensorType(shape, 3))   == TensorType(shape, 3)));
        REQUIRE(((tensor += TensorType(shape, 5))   == TensorType(shape, 5)));
        REQUIRE(((tensor +  TensorType(shape, -10)) == TensorType(shape, -5)));
        REQUIRE(((tensor += TensorType(shape, -4))  == TensorType(shape, 1)));

        add(tensor, TensorType(shape, -3));
        REQUIRE((tensor == TensorType(shape, -2)));

        add(tensor, TensorType(shape, 2));
        REQUIRE((tensor == TensorType(shape, 0)));

        REQUIRE((tensor.template as<uint8_t>() + Tensor<uint8_t>(shape, 127)        == Tensor<uint8_t>(shape, 127)));
        REQUIRE((tensor.template as<int>()     + Tensor<int>    (shape, -4)         == Tensor<int>    (shape, -4)));
        REQUIRE((tensor.template as<float>()   + Tensor<float>  (shape, 1.3)        == Tensor<float>  (shape, 1.3)));
        REQUIRE((tensor.template as<double>()  + Tensor<double> (shape, -208.71875) == Tensor<double> (shape, -208.71875)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("add(Tensor<floating>&, const Tensor<floating>&)", T, test_float_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [](const Shape& shape) {
        TensorType tensor(shape, 0);

        REQUIRE(((tensor +  TensorType(shape, 3.72))    == TensorType(shape, 3.72)));
        REQUIRE(((tensor += TensorType(shape, 4.9999))  == TensorType(shape, 4.9999)));
        REQUIRE(((tensor +  TensorType(shape, -10))     == TensorType(shape, -5.0001)));
        REQUIRE(((tensor += TensorType(shape, -4.9999)) == TensorType(shape, 0)));

        add(tensor, TensorType(shape, -3));
        REQUIRE((tensor == TensorType(shape, -3)));

        add(tensor, TensorType(shape, 3));
        REQUIRE((tensor == TensorType(shape, 0)));

        REQUIRE((tensor.template as<uint8_t>() + Tensor<uint8_t>(shape, 127)        == Tensor<uint8_t>(shape, 127)));
        REQUIRE((tensor.template as<int>()     + Tensor<int>    (shape, -4)         == Tensor<int>    (shape, -4)));
        REQUIRE((tensor.template as<float>()   + Tensor<float>  (shape, 1.3)        == Tensor<float>  (shape, 1.3)));
        REQUIRE((tensor.template as<double>()  + Tensor<double> (shape, -208.71875) == Tensor<double> (shape, -208.71875)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

} // namespace tnt

#endif // TNT_MATH_ADD_IMPL_HPP
