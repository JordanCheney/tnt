#ifndef TNT_MATH_BITWISE_XOR_HPP
#define TNT_MATH_BITWISE_XOR_HPP

#include <tnt/core/tensor.hpp>

namespace tnt
{

namespace detail
{

template <
          typename LeftType,
          typename RightType,
          typename DeviceType
         >
struct OptimizedBitwiseXor
{
    void run(Tensor<LeftType, DeviceType>&, const RightType&);
    void run(Tensor<LeftType, DeviceType>&, const Tensor<RightType, DeviceType>&);
};

} // namespace detail

/// \brief Compute the bitwise xor between a tensor and a scalar elementwise
///
/// The bitwise xor is computed in place on the tensor
/// \param tensor A mutable tensor. The bitwise xor is done in-place
/// \param right A scalar
/// \requires Type `LeftType` shall be an integer
template <
          typename LeftType,
          typename RightType,
          typename DeviceType
         >
inline void bitwise_xor(Tensor<LeftType, DeviceType>& left, const RightType& scalar)
{
    static_assert(std::is_integral<LeftType>::value,
                  "Bitwise xor is only meaningful for integer types");

    detail::OptimizedBitwiseXor<LeftType, RightType, DeviceType>().run(left, scalar);
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
template <
          typename LeftType,
          typename RightType,
          typename DeviceType
         >
inline void bitwise_xor(Tensor<LeftType, DeviceType>& left, const Tensor<RightType, DeviceType>& right)
{
    static_assert(std::is_integral<LeftType>::value,
                  "Bitwise xor is only meaningful for integer types");

    TNT_ASSERT(left.shape == right.shape,
               InvalidParameterException("tnt::bitwise_xor()", __FILE__, __LINE__,
                   "Element-wise bitwise xor of two tensors requires that those tensors be of the same size"));

    detail::OptimizedBitwiseXor<LeftType, RightType, DeviceType>().run(left, right);
}

} // namespace tnt

#endif // TNT_MATH_BITWISE_XOR_HPP
