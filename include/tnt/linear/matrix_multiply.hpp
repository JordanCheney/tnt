#ifndef TNT_LINEAR_MATRIX_MULTIPLY_HPP
#define TNT_LINEAR_MATRIX_MULTIPLY_HPP

#include <tnt/core/tensor.hpp>

namespace tnt
{

namespace detail
{

template <typename LeftType, typename RightType, typename Enable = void>
struct OptimizedMatrixMultiply
{
    static Tensor<LeftType> eval(const Tensor<LeftType>&, const Tensor<RightType>&);
};

} // namespace detail

/// \brief Compute the matrix product of two tensors
///
/// \param left The left tensor. It has size `MxN`
/// \param right The right tensor. It has size `NxC`
/// \returns A tensor with the same type and device [left](*::left) of size `MxC`
/// \requires [left](*::left) and [right](*::right) shall be two dimensional
/// \requires The second dimension of [left](*::left) shall equal the first
/// dimension of [right](*::right)
/// \requires [left](*::left) and [right](*::right) shall have the same device
/// \notes This function asserts that [left](*::left) and [right](*::right)
/// have two dimensions and that the second dimension of [left](*::left) equals
/// the first dimension of [right](*::right). These checks
/// can be disabled by `#define DISABLE_CHECKS` before calling the function.
template <typename LeftType, typename RightType>
inline Tensor<LeftType> matrix_multiply(const Tensor<LeftType>& left, const Tensor<RightType>& right)
{
    TNT_ASSERT(left.shape.num_axes() == 2 && right.shape.num_axes() == 2,
               InvalidParameterException("tnt::matrix_multiply()", __FILE__, __LINE__,
                   "Matrix multiplication requires 2D tensors"))
    TNT_ASSERT(left.shape[1] == right.shape[0],
               InvalidParameterException("tnt::matrix_multiply()", __FILE__, __LINE__,
                   "Matrix multiplication requires NxM and MxK sized matrices"))

    return detail::OptimizedMatrixMultiply<LeftType, RightType>::eval(left, right);
}

} // namespace tnt

#endif // TNT_LINEAR_MATRIX_MULTIPLY_HPP
