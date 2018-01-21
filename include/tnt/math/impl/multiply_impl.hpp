#ifndef TNT_MATH_MULTIPLY_IMPL_HPP
#define TNT_MATH_MULTIPLY_IMPL_HPP

#include <tnt/math/arithmetic_ops.hpp>

#include <tnt/utils/testing.hpp>
#include <tnt/utils/simd.hpp>

namespace tnt
{

namespace detail
{

template <typename LeftType, typename RightType>
struct OptimizedMultiply<LeftType, RightType,
            typename std::enable_if<std::is_same<LeftType, uint8_t>::value
                                    || std::is_same<LeftType, int8_t>::value
                                    || std::is_same<LeftType, uint64_t>::value
                                    || std::is_same<LeftType, int64_t>::value>::type>
{
    static void eval(Tensor<LeftType>& tensor, const RightType& _scalar)
    {
        LeftType* ptr = tensor.data.data;

        const LeftType scalar = static_cast<LeftType>(_scalar);

        int i = 0, total = tensor.shape.total();
        for ( ; total--; ++i)
            ptr[i] *= scalar;
    }

    static void eval(Tensor<LeftType>& left, const Tensor<RightType>& right)
    {
        LeftType* l_ptr        = left.data.data;
        const RightType* r_ptr = right.data.data;

        int i = 0, total = left.shape.total();
        for ( ; total--; ++i)
            l_ptr[i] *= static_cast<LeftType>(r_ptr[i]);
    }
};

template <typename LeftType, typename RightType>
struct OptimizedMultiply<LeftType, RightType,
            typename std::enable_if<std::is_same<LeftType, uint16_t>::value
                                    || std::is_same<LeftType, int16_t>::value
                                    || std::is_same<LeftType, uint32_t>::value
                                    || std::is_same<LeftType, int32_t>::value>::type>
{
    static void eval(Tensor<LeftType>& tensor, const RightType& _scalar)
    {
        LeftType* ptr = tensor.data.data;

        LeftType scalar = static_cast<LeftType>(_scalar);
        auto scalar_vec = simdpp::load_splat<typename SIMDType<LeftType>::VecType>(&scalar);

        int offset = 0, num_blocks = AlignSIMDType<LeftType>::num_aligned_blocks(tensor.shape.total());
        for ( ; num_blocks--; offset += OptimalSIMDSize<LeftType>::value) {
            auto block = LoadSIMDType<LeftType, LeftType>::load(ptr + offset);
            auto result = ConvertSIMDType<LeftType>::convert(simdpp::mull(block, scalar_vec));
            simdpp::store(ptr + offset, result);
        }
    }

    static void eval(Tensor<LeftType>& left, const Tensor<RightType>& right)
    {
        LeftType* l_ptr  = left.data.data;
        RightType* r_ptr = right.data.data;

        int offset = 0, num_blocks = AlignSIMDType<LeftType>::num_aligned_blocks(left.shape.total());
        for ( ; num_blocks--; offset += OptimalSIMDSize<LeftType>::value) {
            auto l_block = LoadSIMDType<LeftType, LeftType>::load(l_ptr + offset);
            auto r_block = LoadSIMDType<LeftType, RightType>::load(r_ptr + offset);
            auto result = ConvertSIMDType<LeftType>::convert(simdpp::mull(l_block, r_block));
            simdpp::store(l_ptr + offset, result);
        }
    }
};

template <typename LeftType, typename RightType>
struct OptimizedMultiply<LeftType, RightType,
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
            auto result = simdpp::mul(block, scalar_vec);
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
            auto result = simdpp::mul(l_block, r_block);
            simdpp::store(l_ptr + offset, result);
        }
    }
};

} // namespace detail

// ----------------------------------------------------------------------------
// Unit tests

TEST_CASE_TEMPLATE("multiply(Tensor<unsigned>&, Scalar)", T, test_unsigned_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 1);

        REQUIRE(((tensor *  4)   == TensorType(shape, 4)));
        REQUIRE(((tensor *= 8)   == TensorType(shape, 8)));
        REQUIRE(((tensor *  0.5) == TensorType(shape, 0)));
        REQUIRE(((tensor *= 5)   == TensorType(shape, 40)));

        tensor = 1; // Reset to 1

        REQUIRE((tensor.template as<uint8_t>() *  64        == Tensor<uint8_t>(shape, 64)));
        REQUIRE((tensor.template as<int>()     * -4         == Tensor<int>    (shape, -4)));
        REQUIRE((tensor.template as<float>()   *  1.3       == Tensor<float>  (shape, 1.3)));
        REQUIRE((tensor.template as<double>()  * -208.71875 == Tensor<double> (shape, -208.71875)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("multiply(Tensor<signed>&, Scalar)", T, test_signed_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 1);

        REQUIRE(((tensor *  4)   == TensorType(shape, 4)));
        REQUIRE(((tensor *= -8)  == TensorType(shape, -8)));
        REQUIRE(((tensor *  0.5) == TensorType(shape, 0)));
        REQUIRE(((tensor *= 5)   == TensorType(shape, -40)));

        tensor = 1; // Reset to 1

        REQUIRE((tensor.template as<uint8_t>() *  64        == Tensor<uint8_t>(shape, 64)));
        REQUIRE((tensor.template as<int>()     * -4         == Tensor<int>    (shape, -4)));
        REQUIRE((tensor.template as<float>()   *  1.3       == Tensor<float>  (shape, 1.3)));
        REQUIRE((tensor.template as<double>()  * -208.71875 == Tensor<double> (shape, -208.71875)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("multiply(Tensor<floating>&, Scalar)", T, test_float_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 1);

        REQUIRE(((tensor *  4.)  == TensorType(shape, 4)));
        REQUIRE(((tensor *= -8)  == TensorType(shape, -8)));
        REQUIRE(((tensor *  0.5) == TensorType(shape, -4)));
        REQUIRE(((tensor *= 5)   == TensorType(shape, -40)));

        multiply(tensor, -0.125);
        REQUIRE((tensor == TensorType(shape, 5)));

        multiply(tensor, 0.2);
        REQUIRE((tensor == TensorType(shape, 1)));

        REQUIRE((tensor.template as<uint8_t>() *  64        == Tensor<uint8_t>(shape, 64)));
        REQUIRE((tensor.template as<int>()     * -4         == Tensor<int>    (shape, -4)));
        REQUIRE((tensor.template as<float>()   *  1.3       == Tensor<float>  (shape, 1.3)));
        REQUIRE((tensor.template as<double>()  * -208.71875 == Tensor<double> (shape, -208.71875)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("multiply(Tensor<unsigned>&, const Tensor<unsigned>&)", T, test_unsigned_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 1);

        REQUIRE(((tensor *  TensorType(shape, 4)) == TensorType(shape, 4)));
        REQUIRE(((tensor *= TensorType(shape, 8)) == TensorType(shape, 8)));
        REQUIRE(((tensor *  TensorType(shape, 0)) == TensorType(shape, 0)));
        REQUIRE(((tensor *= TensorType(shape, 5)) == TensorType(shape, 40)));

        tensor = 1; // Reset to 1

        REQUIRE((tensor.template as<uint8_t>() * Tensor<uint8_t>(shape,  64)        == Tensor<uint8_t>(shape, 64)));
        REQUIRE((tensor.template as<int>()     * Tensor<int>    (shape, -4)         == Tensor<int>    (shape, -4)));
        REQUIRE((tensor.template as<float>()   * Tensor<float>  (shape,  1.3)       == Tensor<float>  (shape, 1.3)));
        REQUIRE((tensor.template as<double>()  * Tensor<double> (shape, -208.71875) == Tensor<double> (shape, -208.71875)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("multiply(Tensor<signed>&, const Tensor<signed>&)", T, test_signed_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 1);

        REQUIRE(((tensor *  TensorType(shape,  4)) == TensorType(shape, 4)));
        REQUIRE(((tensor *= TensorType(shape, -8)) == TensorType(shape, -8)));
        REQUIRE(((tensor *  TensorType(shape,  0)) == TensorType(shape, 0)));
        REQUIRE(((tensor *= TensorType(shape,  5)) == TensorType(shape, -40)));

        tensor = 1; // Reset to 1

        REQUIRE((tensor.template as<uint8_t>() * Tensor<uint8_t>(shape, 64)         == Tensor<uint8_t>(shape, 64)));
        REQUIRE((tensor.template as<int>()     * Tensor<int>    (shape, -4)         == Tensor<int>    (shape, -4)));
        REQUIRE((tensor.template as<float>()   * Tensor<float>  (shape, 1.3)        == Tensor<float>  (shape, 1.3)));
        REQUIRE((tensor.template as<double>()  * Tensor<double> (shape, -208.71875) == Tensor<double> (shape, -208.71875)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

TEST_CASE_TEMPLATE("multiply(Tensor<floating>&, const Tensor<floating>&)", T, test_float_data_types)
{
    using TensorType = Tensor<T>;

    auto test_shape = [&](const Shape& shape) {
        TensorType tensor(shape, 1);

        REQUIRE(((tensor *  TensorType(shape, 4.))  == TensorType(shape, 4)));
        REQUIRE(((tensor *= TensorType(shape, -8))  == TensorType(shape, -8)));
        REQUIRE(((tensor *  TensorType(shape, 0.5)) == TensorType(shape, -4)));
        REQUIRE(((tensor *= TensorType(shape, 5))   == TensorType(shape, -40)));

        multiply(tensor, TensorType(shape, -0.125));
        REQUIRE((tensor == TensorType(shape, 5)));

        multiply(tensor, TensorType(shape, 0.2));
        REQUIRE((tensor == TensorType(shape, 1)));

        REQUIRE((tensor.template as<uint8_t>() * Tensor<uint8_t>(shape, 64)         == Tensor<uint8_t>(shape, 64)));
        REQUIRE((tensor.template as<int>()     * Tensor<int>    (shape, -4)         == Tensor<int>    (shape, -4)));
        REQUIRE((tensor.template as<float>()   * Tensor<float>  (shape, 1.3)        == Tensor<float>  (shape, 1.3)));
        REQUIRE((tensor.template as<double>()  * Tensor<double> (shape, -208.71875) == Tensor<double> (shape, -208.71875)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

} // namespace tnt

#endif // TNT_MATH_MULTIPLY_IMPL_HPP
