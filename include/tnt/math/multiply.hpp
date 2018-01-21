#ifndef TNT_MATH_MULTIPLY_HPP
#define TNT_MATH_MULTIPLY_HPP

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
struct OptimizedMultiply
{
    void run(Tensor<LeftType, DeviceType>&, const RightType&);
    void run(Tensor<LeftType, DeviceType>&, const Tensor<RightType, DeviceType>&);
};

} // namespace detail

/// \brief Multiply a tensor by a scalar elementwise
///
/// The multiplication is computed in place on the tensor
/// \param tensor A mutable tensor. Multiplication is done in-place
/// \param right A scalar
template <
          typename LeftType,
          typename RightType,
          typename DeviceType
         >
inline void multiply(Tensor<LeftType, DeviceType>& tensor, const RightType& scalar) noexcept
{
    detail::OptimizedMultiply<LeftType, RightType, DeviceType>().run(tensor, scalar);
}

/// \brief Multiply a tensor with a tensor elementwise
///
/// The multiplication is computed in place on the left tensor.
/// \param left A mutable tensor. Multiplication is done in-place
/// \param right An immutable tensor of the same size as [left](*::left).
/// \requires [left](*::left) and [right](*::right) have the same shape
/// \requires [left](*::left) and [right](*::right) shall have the same device
/// \notes This function asserts that [left](*::left) and [right](*::right)
/// have the same shape and will throw an exception if they do not. This check
/// can be disabled by `#define DISABLE_CHECKS` before calling the function.
template <
          typename LeftType,
          typename RightType,
          typename DeviceType
         >
inline void multiply(Tensor<LeftType, DeviceType>& left, const Tensor<RightType, DeviceType>& right)
{
    TNT_ASSERT(left.shape == right.shape,
               InvalidParameterException("tnt::multiply()", __FILE__, __LINE__,
                   "Element-wise multiplication of two tensors requires that those tensors be of the same size"));

    detail::OptimizedMultiply<LeftType, RightType, DeviceType>().run(left, right);
}


} // namespace tnt

#endif // TNT_MATH_MULTIPLY_HPP
