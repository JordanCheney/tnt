#ifndef TNT_MATH_SUBTRACT_IMPL_HPP
#define TNT_MATH_SUBTRACT_IMPL_HPP

#include <tnt/math/arithmetic_ops.hpp>
#include <tnt/utils/testing.hpp>

namespace tnt
{

namespace detail
{

template <typename LeftType, typename RightType>
struct OptimizedSubtract<LeftType, RightType>
{
    static void eval(Tensor<LeftType>& tensor, const RightType& _scalar)
    {
        LeftType* ptr = tensor.data.data;

        LeftType scalar = static_cast<LeftType>(_scalar);
        auto scalar_vec = simdpp::load_splat<typename SIMDType<LeftType>::VecType>(&scalar);

        int offset = 0, num_blocks = AlignSIMDType<LeftType>::num_aligned_blocks(tensor.shape.total());
        for ( ; num_blocks--; offset += OptimalSIMDSize<LeftType>::value) {
            auto block = LoadSIMDType<LeftType, LeftType>::load(ptr + offset);
            auto result = simdpp::sub(block, scalar_vec);
            simdpp::store(ptr + offset, result);
        }
    }

    static void eval(Tensor<LeftType>& left, const Tensor<RightType>& right)
    {
        LeftType* l_ptr        = left.data.data;
        RightType* r_ptr = right.data.data;

        int offset = 0, num_blocks = AlignSIMDType<LeftType>::num_aligned_blocks(left.shape.total());
        for ( ; num_blocks--; offset += OptimalSIMDSize<LeftType>::value) {
            auto l_block = LoadSIMDType<LeftType, LeftType>::load(l_ptr + offset);
            auto r_block = LoadSIMDType<LeftType, RightType>::load(r_ptr + offset);
            auto result = simdpp::sub(l_block, r_block);
            simdpp::store(l_ptr + offset, result);
        }
    }
};

} // namespace detail

// ----------------------------------------------------------------------------
// Unit tests

TEST_CASE_TEMPLATE("subtract(Tensor<unsigned>&, Scalar)", T, test_unsigned_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 255);

        REQUIRE(((tensor -  3)   == TensorType(shape, 252)));
        REQUIRE(((tensor -= 5)   == TensorType(shape, 250)));
        REQUIRE(((tensor -  1)   == TensorType(shape, 249)));
        REQUIRE(((tensor -= 4)   == TensorType(shape, 246)));

        subtract(tensor, 1);
        REQUIRE((tensor == TensorType(shape, 245)));

        subtract(tensor, 0);
        REQUIRE((tensor == TensorType(shape, 245)));

        REQUIRE((tensor.template as<int>() - 4 == Tensor<int>(shape, 241)));
        REQUIRE((approx_equal(tensor.template as<float>() - 1.3, Tensor<float>(shape, 243.7))));
        REQUIRE((approx_equal(tensor.template as<double>() - 38.98473, Tensor<double>(shape, 206.01527))));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("subtract(Tensor<signed>&, Scalar)", T, test_signed_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 0);

        REQUIRE(((tensor -  3)   == TensorType(shape, -3)));
        REQUIRE(((tensor -= -30) == TensorType(shape, 30)));
        REQUIRE(((tensor -  1)   == TensorType(shape, 29)));
        REQUIRE(((tensor -= 4)   == TensorType(shape, 26)));

        subtract(tensor, 1);
        REQUIRE((tensor == TensorType(shape, 25)));

        subtract(tensor, 0);
        REQUIRE((tensor == TensorType(shape, 25)));

        REQUIRE((tensor.template as<int>() - 4 == Tensor<int>(shape, 21)));
        REQUIRE((approx_equal(tensor.template as<float>() - 1.3, Tensor<float>(shape, 23.7))));
        REQUIRE((approx_equal(tensor.template as<double>() - 38.98473, Tensor<double>(shape, -13.98473))));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("subtract(Tensor<float>&, Scalar)", T, test_float_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 0.5);

        REQUIRE(((tensor -  0.25)  == TensorType(shape, 0.25)));
        REQUIRE(((tensor -= 1.0)   == TensorType(shape, -0.5)));
        REQUIRE(((tensor -  -0.5)  == TensorType(shape, 0.0)));
        REQUIRE(((tensor -= 1.333) == TensorType(shape, -1.833)));

        subtract(tensor, -1.033);
        REQUIRE((approx_equal(tensor, TensorType(shape, -0.8))));

        subtract(tensor, -20.8);
        REQUIRE((approx_equal(tensor, TensorType(shape, 20))));

        REQUIRE((tensor.template as<int>() - 4 == Tensor<int>(shape, 16)));
        REQUIRE((approx_equal(tensor.template as<float>() - 1.3, Tensor<float>(shape, 18.7))));
        REQUIRE((approx_equal(tensor.template as<double>() - 38.98473, Tensor<double>(shape, -18.98473))));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("subtract(Tensor<unsigned>&, const Tensor<unsigned>&)", T, test_unsigned_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 255);

        REQUIRE(((tensor -  TensorType(shape, 3)) == TensorType(shape, 252)));
        REQUIRE(((tensor -= TensorType(shape, 5)) == TensorType(shape, 250)));
        REQUIRE(((tensor -  TensorType(shape, 1)) == TensorType(shape, 249)));
        REQUIRE(((tensor -= TensorType(shape, 4)) == TensorType(shape, 246)));

        subtract(tensor, TensorType(shape, 1));
        REQUIRE((tensor == TensorType(shape, 245)));

        subtract(tensor, TensorType(shape, 0));
        REQUIRE((tensor == TensorType(shape, 245)));

        REQUIRE((tensor.template as<int>() - Tensor<int>(shape, 4) == Tensor<int>(shape, 241)));
        REQUIRE((approx_equal(tensor.template as<float>()  - Tensor<float>(shape, 1.3),       Tensor<float>(shape, 243.7))));
        REQUIRE((approx_equal(tensor.template as<double>() - Tensor<double>(shape, 38.98473), Tensor<double>(shape, 206.01527))));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("subtract(Tensor<signed>&, const Tensor<signed>&)", T, test_signed_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 0);

        REQUIRE(((tensor -  TensorType(shape, 3))   == TensorType(shape, -3)));
        REQUIRE(((tensor -= TensorType(shape, -30)) == TensorType(shape, 30)));
        REQUIRE(((tensor -  TensorType(shape, 1))   == TensorType(shape, 29)));
        REQUIRE(((tensor -= TensorType(shape, 4))   == TensorType(shape, 26)));

        subtract(tensor, TensorType(shape, 1));
        REQUIRE((tensor == TensorType(shape, 25)));

        subtract(tensor, TensorType(shape, 0));
        REQUIRE((tensor == TensorType(shape, 25)));

        REQUIRE((tensor.template as<int>() - Tensor<int>(shape, 4) == Tensor<int>(shape, 21)));
        REQUIRE((approx_equal(tensor.template as<float>()  - Tensor<float>(shape, 1.3),       Tensor<float>(shape, 23.7))));
        REQUIRE((approx_equal(tensor.template as<double>() - Tensor<double>(shape, 38.98473), Tensor<double>(shape, -13.98473))));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("subtract(Tensor<float>&, Scalar)", T, test_float_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 0.5);

        REQUIRE(((tensor -  TensorType(shape, 0.25))  == TensorType(shape, 0.25)));
        REQUIRE(((tensor -= TensorType(shape, 1.0))   == TensorType(shape, -0.5)));
        REQUIRE(((tensor -  TensorType(shape, -0.5))  == TensorType(shape, 0.0)));
        REQUIRE(((tensor -= TensorType(shape, 1.333)) == TensorType(shape, -1.833)));

        subtract(tensor, TensorType(shape, -1.033));
        REQUIRE((approx_equal(tensor, TensorType(shape, -0.8))));

        subtract(tensor, TensorType(shape, -20.8));
        REQUIRE((approx_equal(tensor, TensorType(shape, 20))));

        REQUIRE((tensor.template as<int>() - Tensor<int>(shape, 4) == Tensor<int>(shape, 16)));
        REQUIRE((approx_equal(tensor.template as<float>()  - Tensor<float>(shape, 1.3),      Tensor<float>(shape, 18.7))));
        REQUIRE((approx_equal(tensor.template as<double>() - Tensor<double>(shape, 38.98473), Tensor<double>(shape, -18.98473))));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

} // namespace tnt

#endif // TNT_MATH_SUBTRACT_IMPL_HPP
