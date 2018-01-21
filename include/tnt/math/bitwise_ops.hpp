#ifndef TNT_MATH_BITWISE_OPS_HPP
#define TNT_MATH_BITWISE_OPS_HPP

#include <tnt/core/tensor.hpp>

namespace tnt
{

namespace detail
{

template <typename LeftType, typename Enable = void>
struct OptimizedBitwiseNot
{
    static void eval(Tensor<LeftType>&) noexcept;
};

template <typename LeftType, typename RightType, typename Enable = void>
struct OptimizedBitwiseAnd
{
    static void eval(Tensor<LeftType>&, const RightType&) noexcept;
    static void eval(Tensor<LeftType>&, const Tensor<RightType>&) noexcept;
};

template <typename LeftType, typename RightType, typename Enable = void>
struct OptimizedBitwiseOr
{
    static void eval(Tensor<LeftType>&, const RightType&) noexcept;
    static void eval(Tensor<LeftType>&, const Tensor<RightType>&) noexcept;
};

template <typename LeftType, typename RightType, typename Enable = void>
struct OptimizedBitwiseXor
{
    static void eval(Tensor<LeftType>&, const RightType&) noexcept;
    static void eval(Tensor<LeftType>&, const Tensor<RightType>&) noexcept;
};

} // namespace detail

/// \brief Compute the 1's complement of an integer tensor
///
/// The 1's complement is computed in-place on the tensor
/// \param tensor A mutable tensor. The complement is taken in-place
/// \requires Type `LeftType` shall be an integer
template <typename LeftType>
inline void bitwise_not(Tensor<LeftType>& tensor) noexcept
{
    static_assert(std::is_integral<LeftType>::value,
                  "Bitwise not is only meaningful for integer types");

    detail::OptimizedBitwiseNot<LeftType>::eval(tensor);
}

/// \brief Compute the bitwise and between a tensor and a scalar elementwise
///
/// The bitwise and is computed in place on the tensor
/// \param tensor A mutable tensor. The bitwise and is done in-place
/// \param right A scalar
/// \requires Type `LeftType` shall be an integer
template <typename LeftType, typename RightType>
inline void bitwise_and(Tensor<LeftType>& left, const RightType& scalar) noexcept
{
    static_assert(std::is_integral<LeftType>::value,
                  "Bitwise and is only meaningful for integer types");

    detail::OptimizedBitwiseAnd<LeftType, RightType>::eval(left, scalar);
}

/// \brief Compute the bitwise and between a tensor and a tensor elementwise
///
/// The bitwise and is computed in place on the left tensor.
/// \param left A mutable tensor. The bitwise and is done in-place
/// \param right An immutable tensor of the same size as [left](*::left).
/// \requires [left](*::left) and [right](*::right) shall have the same shape
/// \requires [left](*::left) and [right](*::right) shall have the same device
/// \requires Type `LeftType` shall be an integer
/// \notes This function asserts that [left](*::left) and [right](*::right)
/// have the same shape and will throw an exception if they do not. This check
/// can be disabled by `#define DISABLE_CHECKS` before calling the function.
template <typename LeftType, typename RightType>
inline void bitwise_and(Tensor<LeftType>& left, const Tensor<RightType>& right)
{
    static_assert(std::is_integral<LeftType>::value,
                  "Bitwise and is only meaningful for integer types");

    TNT_ASSERT(left.shape == right.shape,
               InvalidParameterException("tnt::bitwise_and()", __FILE__, __LINE__,
                   "Element-wise bitwise and of two tensors requires that those tensors be of the same size"));

    detail::OptimizedBitwiseAnd<LeftType, RightType>::eval(left, right);
}

/// \brief Compute the bitwise or between an integer tensor and a scalar elementwise
///
/// The bitwise or is computed in place on the tensor
/// \param tensor A mutable tensor. The bitwise or is done in-place
/// \param right A scalar
/// \requires Type `LeftType` shall be an integer
template <typename LeftType, typename RightType>
inline void bitwise_or(Tensor<LeftType>& left, const RightType& scalar) noexcept
{
    static_assert(std::is_integral<LeftType>::value,
                  "Bitwise or is only meaningful for integer types");

    detail::OptimizedBitwiseOr<LeftType, RightType>::eval(left, scalar);
}

/// \brief Compute the bitwise or between an integer tensor and a tensor elementwise
///
/// The bitwise or is computed in place on the left tensor.
/// \param left A mutable tensor. The bitwise or is done in-place
/// \param right An immutable tensor of the same size as [left](*::left).
/// \requires [left](*::left) and [right](*::right) shall have the same shape
/// \requires [left](*::left) and [right](*::right) shall have the same device
/// \requires Type `LeftType` shall be an integer
/// \notes This function asserts that [left](*::left) and [right](*::right)
/// have the same shape and will throw an exception if they do not. This check
/// can be disabled by `#define DISABLE_CHECKS` before calling the function.
template <typename LeftType, typename RightType>
inline void bitwise_or(Tensor<LeftType>& left, const Tensor<RightType>& right)
{
    static_assert(std::is_integral<LeftType>::value,
                  "Bitwise or is only meaningful for integer types");

    TNT_ASSERT(left.shape == right.shape,
               InvalidParameterException("tnt::bitwise_or()", __FILE__, __LINE__,
                   "Element-wise bitwise or of two tensors requires that those tensors be of the same size"));

    detail::OptimizedBitwiseOr<LeftType, RightType>().run(left, right);
}

/// \brief Compute the bitwise xor between a tensor and a scalar elementwise
///
/// The bitwise xor is computed in place on the tensor
/// \param tensor A mutable tensor. The bitwise xor is done in-place
/// \param right A scalar
/// \requires Type `LeftType` shall be an integer
template <typename LeftType, typename RightType>
inline void bitwise_xor(Tensor<LeftType>& left, const RightType& scalar) noexcept
{
    static_assert(std::is_integral<LeftType>::value,
                  "Bitwise xor is only meaningful for integer types");

    detail::OptimizedBitwiseXor<LeftType, RightType>::eval(left, scalar);
}

/// \brief Compute the bitwise xor between a tensor and a tensor elementwise
///
/// The bitwise xor is computed in place on the left tensor.
/// \param left A mutable tensor. The bitwise xor is done in-place
/// \param right An immutable tensor of the same size as [left](*::left).
/// \requires [left](*::left) and [right](*::right) shall have the same shape
/// \requires [left](*::left) and [right](*::right) shall have the same device
/// \requires Type `LeftType` shall be an integer
/// \notes This function asserts that [left](*::left) and [right](*::right)
/// have the same shape and will throw an exception if they do not. This check
/// can be disabled by `#define DISABLE_CHECKS` before calling the function.
template <typename LeftType, typename RightType>
inline void bitwise_xor(Tensor<LeftType>& left, const Tensor<RightType>& right)
{
    static_assert(std::is_integral<LeftType>::value,
                  "Bitwise xor is only meaningful for integer types");

    TNT_ASSERT(left.shape == right.shape,
               InvalidParameterException("tnt::bitwise_xor()", __FILE__, __LINE__,
                   "Element-wise bitwise xor of two tensors requires that those tensors be of the same size"));

    detail::OptimizedBitwiseXor<LeftType, RightType>().run(left, right);
}

} // namespace tnt

#endif // TNT_MATH_BITWISE_OPS_HPP
