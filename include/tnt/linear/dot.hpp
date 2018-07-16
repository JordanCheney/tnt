#ifndef TNT_LINEAR_DOT_HPP
#define TNT_LINEAR_DOT_HPP

#include <tnt/core/tensor.hpp>

namespace tnt
{

namespace detail
{

template <
          typename LeftType,
          typename RightType,
          typename Enable = void
         >
struct OptimizedDot
{
    static LeftType eval(const Tensor<LeftType>&, const Tensor<RightType>&) noexcept;
};

} // namespace detail

/// \brief Compute a dot product between 2 equal sized tensors
///
/// \param left A tensor. The return type of the function matches the type of this tensor.
/// \param right Another tenors
/// \returns A scalar value with the same type as [left](*::left).
/// \notes This function asserts that [left](*::left) and [right](*::right)
/// have the same shape and will throw an exception if they do not. This check
/// can be disabled by `#define DISABLE_CHECKS` before calling the function.
template <typename LeftType, typename RightType>
inline LeftType dot(const Tensor<LeftType>& left, const Tensor<RightType>& right)
{
    TNT_ASSERT(left.shape == right.shape,
               InvalidParameterException("tnt::dot()", __FILE__, __LINE__,
                   "The dot product of two tensors requires that those tensors be of the same size"));

    return detail::OptimizedDot<LeftType, RightType>::eval(left, right);
}

} // namespace tnt

#endif // TNT_LINEAR_DOT_HPP
