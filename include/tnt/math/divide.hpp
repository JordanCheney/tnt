#ifndef TNT_MATH_DIVIDE_HPP
#define TNT_MATH_DIVIDE_HPP

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
struct OptimizedDivide
{
    void run(Tensor<LeftType, DeviceType>&, const RightType&);
    void run(Tensor<LeftType, DeviceType>&, const Tensor<RightType, DeviceType>&);
};

} // namespace detail

/// \brief Divide a tensor by a scalar elementwise
///
/// The division is computed in place on the tensor
/// \param tensor A mutable tensor. Division is done in-place
/// \param scalar A non-zero scalar
/// \notes This function asserts that the [scalar](*::scalar) is non-zero and.
/// will throw an exception if it is. This check can be disabled by
/// `#define DISABLE_CHECKS` before calling the function.
template <
          typename LeftType,
          typename RightType,
          typename DeviceType
         >
inline void divide(Tensor<LeftType, DeviceType>& tensor, const RightType& scalar)
{
    TNT_ASSERT(scalar != 0,
               InvalidParameterException("tnt::divide()", __FILE__, __LINE__,
                   "Cannot divide by 0"))

    detail::OptimizedDivide<LeftType, RightType, DeviceType>().run(tensor, scalar);
}

/// \brief Divide a tensor by a tensor elementwise
///
/// The division is computed in place on the left tensor.
/// \param left A mutable tensor. Division is done in-place
/// \param right An immutable tensor of the same size as [left](*::left).
/// \requires [left](*::left) and [right](*::right) shall have the same shape
/// \requires [left](*::left) and [right](*::right) shall have the same device
/// \notes This function asserts that [left](*::left) and [right](*::right)
/// have the same shape and will throw an exception if they do not. This check
/// can be disabled by `#define DISABLE_CHECKS` before calling the function.
template <
          typename LeftType,
          typename RightType,
          typename DeviceType
         >
inline void divide(Tensor<LeftType, DeviceType>& left, const Tensor<RightType, DeviceType>& right)
{
    TNT_ASSERT(left.shape == right.shape,
               InvalidParameterException("tnt::divide()", __FILE__, __LINE__,
                   "Element-wise division of two tensors requires that those tensors be of the same size"))

    detail::OptimizedDivide<LeftType, RightType, DeviceType>().run(left, right);
}

} // namespace tnt

#endif // TNT_MATH_DIVIDE_HPP
