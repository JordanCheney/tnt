#ifndef TNT_MATH_ADD_HPP
#define TNT_MATH_ADD_HPP

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
struct OptimizedAdd
{
    void run(Tensor<LeftType, DeviceType>&, const RightType&);
    void run(Tensor<LeftType, DeviceType>&, const Tensor<RightType, DeviceType>&);
};

} // namespace detail

/// \brief Add a scalar to a tensor elementwise
///
/// The addition is computed in place on the tensor
/// \param tensor A mutable tensor. Addition is done in-place
/// \param right A scalar
template <
          typename LeftType,
          typename RightType,
          typename DeviceType
         >
inline void add(Tensor<LeftType, DeviceType>& left, const RightType& scalar)
{
    detail::OptimizedAdd<LeftType, RightType, DeviceType>().run(left, scalar);
}

/// \brief Add a tensor to a tensor elementwise
///
/// The addition is computed in place on the left tensor.
/// \param left A mutable tensor. Addition is done in-place
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
inline void add(Tensor<LeftType, DeviceType>& left, const Tensor<RightType, DeviceType>& right)
{
    TNT_ASSERT(left.shape == right.shape,
               InvalidParameterException("tnt::add()", __FILE__, __LINE__,
                   "Element-wise addition of two tensors requires that those tensors be of the same size"));

    detail::OptimizedAdd<LeftType, RightType, DeviceType>().run(left, right);
}

} // namespace tnt

#endif // TNT_MATH_ADD_HPP
