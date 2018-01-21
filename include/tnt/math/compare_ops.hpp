#ifndef TNT_MATH_COMPARE_OPS_HPP
#define TNT_MATH_COMPARE_OPS_HPP

#include <tnt/core/tensor.hpp>

namespace tnt
{

namespace detail
{

template <typename LeftType, typename RightType, typename Enable = void>
struct OptimizedCompareEqual
{
    static Tensor<uint8_t> eval(const Tensor<LeftType>&, const RightType&);
};

template <typename LeftType, typename RightType, typename Enable = void>
struct OptimizedCompareNotEqual
{
    static Tensor<uint8_t> eval(const Tensor<LeftType>&, const RightType&);
};

template <typename LeftType, typename RightType, typename Enable = void>
struct OptimizedCompareLessThan
{
    static Tensor<uint8_t> eval(const Tensor<LeftType>&, const RightType&);
};

template <typename LeftType, typename RightType, typename Enable = void>
struct OptimizedCompareGreaterThan
{
    static Tensor<uint8_t> eval(const Tensor<LeftType>&, const RightType&);
};

template <typename LeftType, typename RightType, typename Enable = void>
struct OptimizedCompareLessOrEqual
{
    static Tensor<uint8_t> eval(const Tensor<LeftType>&, const RightType&);
};

template <typename LeftType, typename RightType, typename Enable = void>
struct OptimizedCompareGreaterOrEqual
{
    static Tensor<uint8_t> eval(const Tensor<LeftType>&, const RightType&);
};

} // namespace detail

/// \brief Check equality of a tensor and scalar elementwise
///
/// \param tensor A immutable tensor.
/// \param right A scalar
/// \returns A mask tensor with DataType `uint8_t`. The mask will contain `255`
/// where the condition is true and `0` everywhere else.
template <typename LeftType, typename RightType>
inline Tensor<uint8_t> compare_equal(const Tensor<LeftType>& left, const RightType& scalar)
{
    return detail::OptimizedCompareEqual<LeftType, RightType>::eval(left, scalar);
}

/// \brief Check inequality of a tensor and scalar elementwise
///
/// \param tensor A immutable tensor.
/// \param right A scalar
/// \returns A mask tensor with DataType `uint8_t`. The mask will contain `255`
/// where the condition is true and `0` everywhere else.
template <typename LeftType, typename RightType>
inline Tensor<uint8_t> compare_not_equal(const Tensor<LeftType>& left, const RightType& scalar)
{
    return detail::OptimizedCompareNotEqual<LeftType, RightType>::eval(left, scalar);
}

/// \brief Check if a scalar is less than a tensor elementwise
///
/// \param tensor A immutable tensor.
/// \param right A scalar
/// \returns A mask tensor with DataType `uint8_t`. The mask will contain `255`
/// where the condition is true and `0` everywhere else.
template <typename LeftType, typename RightType>
inline Tensor<uint8_t> compare_less_than(const Tensor<LeftType>& left, const RightType& scalar)
{
    return detail::OptimizedCompareLessThan<LeftType, RightType>::eval(left, scalar);
}

/// \brief Check if a scalar is greater than a tensor elementwise
///
/// \param tensor A immutable tensor.
/// \param right A scalar
/// \returns A mask tensor with DataType `uint8_t`. The mask will contain `255`
/// where the condition is true and `0` everywhere else.
template <typename LeftType, typename RightType>
inline Tensor<uint8_t> compare_greater_than(const Tensor<LeftType>& left, const RightType& scalar)
{
    return detail::OptimizedCompareGreaterThan<LeftType, RightType>::eval(left, scalar);
}

/// \brief Check if a scalar is less than or equal to a tensor elementwise
///
/// \param tensor A immutable tensor.
/// \param right A scalar
/// \returns A mask tensor with DataType `uint8_t`. The mask will contain `255`
/// where the condition is true and `0` everywhere else.
template <typename LeftType, typename RightType>
inline Tensor<uint8_t> compare_less_or_equal(const Tensor<LeftType>& left, const RightType& scalar)
{
    return detail::OptimizedCompareLessOrEqual<LeftType, RightType>::eval(left, scalar);
}

/// \brief Check if a scalar is greater than or equal to a tensor elementwise
///
/// \param tensor A immutable tensor.
/// \param right A scalar
/// \returns A mask tensor with DataType `uint8_t`. The mask will contain `255`
/// where the condition is true and `0` everywhere else.
template <typename LeftType, typename RightType>
inline Tensor<uint8_t> compare_greater_or_equal(const Tensor<LeftType>& left, const RightType& scalar)
{
    return detail::OptimizedCompareGreaterOrEqual<LeftType, RightType>::eval(left, scalar);
}

} // namespace tnt

#endif // TNT_MATH_COMPARE_OPS_HPP
