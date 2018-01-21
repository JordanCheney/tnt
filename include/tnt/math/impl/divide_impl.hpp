#ifndef TNT_MATH_DIVIDE_IMPL_HPP
#define TNT_MATH_DIVIDE_IMPL_HPP

#include <tnt/math/arithmetic_ops.hpp>
#include <tnt/utils/simd.hpp>
#include <tnt/utils/testing.hpp>

namespace tnt
{

namespace detail
{

template <typename LeftType, typename RightType>
struct OptimizedDivide<LeftType, RightType,
            typename std::enable_if<std::is_integral<LeftType>::value>::type>
{
    static void eval(Tensor<LeftType>& tensor, const RightType& _scalar)
    {
        LeftType* ptr = tensor.data.data;

        const LeftType scalar = static_cast<LeftType>(_scalar);

        int i = 0, total = tensor.shape.total();
        for ( ; total--; ++i)
            ptr[i] /= scalar;
    }

    static void eval(Tensor<LeftType>& left, const Tensor<RightType>& right)
    {
        LeftType* l_ptr        = left.data.data;
        const RightType* r_ptr = right.data.data;

        int i = 0, total = left.shape.total();
        for ( ; total--; ++i)
            l_ptr[i] /= static_cast<LeftType>(r_ptr[i]);
    }
};

template <typename LeftType, typename RightType>
struct OptimizedDivide<LeftType, RightType,
            typename std::enable_if<std::is_floating_point<LeftType>::value>::type>
{
    static void eval(Tensor<LeftType>& tensor, const RightType& _scalar)
    {
        LeftType* ptr = tensor.data.data;

        LeftType scalar = static_cast<LeftType>(_scalar);
        auto scalar_vec = simdpp::load_splat<typename SIMDType<LeftType>::VecType>(&scalar);

        int offset = 0, num_blocks = AlignSIMDType<LeftType>::num_aligned_blocks(tensor.shape.total());
        for ( ; num_blocks--; offset += OptimalSIMDSize<LeftType>::value) {
            auto block = LoadSIMDType<LeftType, LeftType>::load(ptr + offset);
            auto result = simdpp::div(block, scalar_vec);
            simdpp::store(ptr + offset, result);
        }
    }

    static void eval(Tensor<LeftType>& left, const Tensor<RightType>& right)
    {
        LeftType* l_ptr      = left.data.data;
        RightType* r_ptr = right.data.data;

        int offset = 0, num_blocks = AlignSIMDType<LeftType>::num_aligned_blocks(left.shape.total());
        for ( ; num_blocks--; offset += OptimalSIMDSize<LeftType>::value) {
            auto l_block = LoadSIMDType<LeftType, LeftType>::load(l_ptr + offset);
            auto r_block = LoadSIMDType<LeftType, RightType>::load(r_ptr + offset);
            auto result = simdpp::div(l_block, r_block);
            simdpp::store(l_ptr + offset, result);
        }
    }
};

} // namespace detail

// ----------------------------------------------------------------------------
// Unit tests

TEST_CASE_TEMPLATE("divide(Tensor<unsigned>&, Scalar)", T, test_unsigned_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 100);

        REQUIRE(((tensor /  4)   == TensorType(shape, 25)));
        REQUIRE(((tensor /= 5)   == TensorType(shape, 20)));
        REQUIRE(((tensor /= 2.2) == TensorType(shape, 10)));

        divide(tensor, 5);
        REQUIRE((tensor == TensorType(shape, 2)));

        tensor = 24;

        REQUIRE((tensor.template as<uint8_t>() / 6      == Tensor<uint8_t>(shape, 4)));
        REQUIRE((tensor.template as<int>()     / -6     == Tensor<int>    (shape, -4)));
        REQUIRE((tensor.template as<float>()   / 0.5    == Tensor<float>  (shape, 48)));
        REQUIRE((tensor.template as<double>()  / -0.125 == Tensor<double> (shape, -192)));

        REQUIRE_THROWS(tensor / 0);
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("divide(Tensor<signed>&, Scalar)", T, test_signed_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 100);

        REQUIRE(((tensor /   4)   == TensorType(shape, 25)));
        REQUIRE(((tensor /= -5)   == TensorType(shape, -20)));
        REQUIRE(((tensor /=  2.2) == TensorType(shape, -10)));

        divide(tensor, -5);
        REQUIRE((tensor == TensorType(shape, 2)));

        tensor = 24;

        REQUIRE((tensor.template as<uint8_t>() / 6      == Tensor<uint8_t>(shape, 4)));
        REQUIRE((tensor.template as<int>()     / -6     == Tensor<int>(shape, -4)));
        REQUIRE((tensor.template as<float>()   / 0.5    == Tensor<float>(shape, 48)));
        REQUIRE((tensor.template as<double>()  / -0.125 == Tensor<double>(shape, -192)));

        REQUIRE_THROWS(tensor / 0);
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("divide(Tensor<floating>&, Scalar)", T, test_float_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 1.0);

        REQUIRE(((tensor /   4  ) == TensorType(shape, 0.25)));
        REQUIRE(((tensor /= -0.5) == TensorType(shape, -2.0)));
        REQUIRE(((tensor /=  2.5) == TensorType(shape, -0.8)));

        divide(tensor, -5);
        REQUIRE((tensor == TensorType(shape, 0.16)));

        tensor = 24;

        REQUIRE((tensor.template as<uint8_t>() / 6      == Tensor<uint8_t>(shape, 4)));
        REQUIRE((tensor.template as<int>()     / -6     == Tensor<int>(shape, -4)));
        REQUIRE((tensor.template as<float>()   / 0.5    == Tensor<float>(shape, 48)));
        REQUIRE((tensor.template as<double>()  / -0.125 == Tensor<double>(shape, -192)));

        REQUIRE_THROWS(tensor / 0);
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("divide(Tensor<unsigned>&, const Tensor<unsigned>&)", T, test_unsigned_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 100);

        REQUIRE(((tensor /  TensorType(shape, 4)) == TensorType(shape, 25)));
        REQUIRE(((tensor /= TensorType(shape, 5)) == TensorType(shape, 20)));
        REQUIRE(((tensor /= TensorType(shape, 2)) == TensorType(shape, 10)));

        divide(tensor, TensorType(shape, 5));
        REQUIRE((tensor == TensorType(shape, 2)));

        tensor = 24;

        REQUIRE((tensor.template as<uint8_t>() / Tensor<uint8_t>(shape, 6)      == Tensor<uint8_t>(shape, 4)));
        REQUIRE((tensor.template as<int>()     / Tensor<int>    (shape, -6)     == Tensor<int>    (shape, -4)));
        REQUIRE((tensor.template as<float>()   / Tensor<float>  (shape, 0.5)    == Tensor<float>  (shape, 48)));
        REQUIRE((tensor.template as<double>()  / Tensor<double> (shape, -0.125) == Tensor<double> (shape, -192)));

        REQUIRE_THROWS(tensor / 0);
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("divide(Tensor<signed>&, const Tensor<signed>&)", T, test_signed_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 100);

        REQUIRE(((tensor /  TensorType(shape,  4)) == TensorType(shape, 25)));
        REQUIRE(((tensor /= TensorType(shape, -5)) == TensorType(shape, -20)));
        REQUIRE(((tensor /= TensorType(shape,  2)) == TensorType(shape, -10)));

        divide(tensor, TensorType(shape, -5));
        REQUIRE((tensor == TensorType(shape, 2)));

        tensor = 24;

        REQUIRE((tensor.template as<uint8_t>() / Tensor<uint8_t>(shape, 6)      == Tensor<uint8_t>(shape, 4)));
        REQUIRE((tensor.template as<int>()     / Tensor<int>    (shape, -6)     == Tensor<int>    (shape, -4)));
        REQUIRE((tensor.template as<float>()   / Tensor<float>  (shape, 0.5)    == Tensor<float>  (shape, 48)));
        REQUIRE((tensor.template as<double>()  / Tensor<double> (shape, -0.125) == Tensor<double> (shape, -192)));

        REQUIRE_THROWS(tensor / 0);
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("divide(Tensor<floating>&, const Tensor<float>&)", T, test_float_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 1.0);

        REQUIRE(((tensor /  TensorType(shape,  4))   == TensorType(shape, 0.25)));
        REQUIRE(((tensor /= TensorType(shape, -0.5)) == TensorType(shape, -2.0)));
        REQUIRE(((tensor /= TensorType(shape,  2.5)) == TensorType(shape, -0.8)));

        divide(tensor, TensorType(shape, -5));
        REQUIRE((tensor == TensorType(shape, 0.16)));

        tensor = 24;

        REQUIRE((tensor.template as<uint8_t>() / Tensor<uint8_t>(shape,  6)     == Tensor<uint8_t>(shape, 4)));
        REQUIRE((tensor.template as<int>()     / Tensor<int>    (shape, -6)     == Tensor<int>    (shape, -4)));
        REQUIRE((tensor.template as<float>()   / Tensor<float>  (shape,  0.5)   == Tensor<float>  (shape, 48)));
        REQUIRE((tensor.template as<double>()  / Tensor<double> (shape, -0.125) == Tensor<double> (shape, -192)));

        REQUIRE_THROWS(tensor / 0);
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

} // namespace tnt

#endif // TNT_MATH_DIVIDE_IMPL_HPP
