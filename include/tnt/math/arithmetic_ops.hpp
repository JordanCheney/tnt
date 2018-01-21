#ifndef TNT_MATH_ARITHMETIC_OPS_HPP
#define TNT_MATH_ARITHMETIC_OPS_HPP

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
struct OptimizedAdd
{
    static void eval(Tensor<LeftType>&, const RightType&) noexcept;
    static void eval(Tensor<LeftType>&, const Tensor<RightType>&) noexcept;
};

template <
          typename LeftType,
          typename RightType,
          typename Enable = void
         >
struct OptimizedSubtract
{
    static void eval(Tensor<LeftType>&, const RightType&) noexcept;
    static void eval(Tensor<LeftType>&, const Tensor<RightType>&) noexcept;
};

template <
          typename LeftType,
          typename RightType,
          typename Enable = void
         >
struct OptimizedMultiply
{
    static void eval(Tensor<LeftType>&, const RightType&) noexcept;
    static void eval(Tensor<LeftType>&, const Tensor<RightType>&) noexcept;
};

template <
          typename LeftType,
          typename RightType,
          typename Enable = void
         >
struct OptimizedDivide
{
    static void eval(Tensor<LeftType>&, const RightType&) noexcept;
    static void eval(Tensor<LeftType>&, const Tensor<RightType>&) noexcept;
};

} // namespace detail

/// \brief Add a scalar to a tensor elementwise
///
/// The addition is computed in place on the tensor
/// \param tensor A mutable tensor. Addition is done in-place
/// \param right A scalar
template <typename LeftType, typename RightType>
inline void add(Tensor<LeftType>& left, const RightType& scalar) noexcept
{
    detail::OptimizedAdd<LeftType, RightType>::eval(left, scalar);
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
template <typename LeftType, typename RightType>
inline void add(Tensor<LeftType>& left, const Tensor<RightType>& right)
{
    TNT_ASSERT(left.shape == right.shape,
               InvalidParameterException("tnt::add()", __FILE__, __LINE__,
                   "Element-wise addition of two tensors requires that those tensors be of the same size"));

    detail::OptimizedAdd<LeftType, RightType>::eval(left, right);
}

/// \brief Subtract a scalar from a tensor elementwise
///
/// The subtraction is computed in place on the tensor
/// \param tensor A mutable tensor. Subtraction is done in-place
/// \param right A scalar
template <typename LeftType, typename RightType>
inline void subtract(Tensor<LeftType>& tensor, const RightType& scalar) noexcept
{
    detail::OptimizedSubtract<LeftType, RightType>::eval(tensor, scalar);
}

/// \brief Subtract a tensor from a tensor elementwise
///
/// The subtraction is computed in place on the left tensor.
/// \param left A mutable tensor. Subtraction is done in-place
/// \param right An immutable tensor of the same size as [left](*::left).
/// \requires [left](*::left) and [right](*::right) shall have the same shape
/// \requires [left](*::left) and [right](*::right) shall have the same device
/// \notes This function asserts that [left](*::left) and [right](*::right)
/// have the same shape and will throw an exception if they do not. This check
/// can be disabled by `#define DISABLE_CHECKS` before calling the function.
template <typename LeftType, typename RightType>
inline void subtract(Tensor<LeftType>& left, const Tensor<RightType>& right)
{
    TNT_ASSERT(left.shape == right.shape,
               InvalidParameterException("tnt::subtraction()", __FILE__, __LINE__,
                   "Element-wise subtraction of two tensors requires that those tensors be of the same size"));

    detail::OptimizedSubtract<LeftType, RightType>::eval(left, right);
}

/// \brief Multiply a tensor by a scalar elementwise
///
/// The multiplication is computed in place on the tensor
/// \param tensor A mutable tensor. Multiplication is done in-place
/// \param right A scalar
template <typename LeftType, typename RightType>
inline void multiply(Tensor<LeftType>& tensor, const RightType& scalar) noexcept
{
    detail::OptimizedMultiply<LeftType, RightType>::eval(tensor, scalar);
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
template <typename LeftType, typename RightType>
inline void multiply(Tensor<LeftType>& left, const Tensor<RightType>& right)
{
    TNT_ASSERT(left.shape == right.shape,
               InvalidParameterException("tnt::multiply()", __FILE__, __LINE__,
                   "Element-wise multiplication of two tensors requires that those tensors be of the same size"));

    detail::OptimizedMultiply<LeftType, RightType>::eval(left, right);
}


/// \brief Divide a tensor by a scalar elementwise
///
/// The division is computed in place on the tensor
/// \param tensor A mutable tensor. Division is done in-place
/// \param scalar A non-zero scalar
/// \notes This function asserts that the [scalar](*::scalar) is non-zero and.
/// will throw an exception if it is. This check can be disabled by
/// `#define DISABLE_CHECKS` before calling the function.
template <typename LeftType, typename RightType>
inline void divide(Tensor<LeftType>& tensor, const RightType& scalar)
{
    TNT_ASSERT(scalar != 0,
               InvalidParameterException("tnt::divide()", __FILE__, __LINE__,
                   "Cannot divide by 0"))

    detail::OptimizedDivide<LeftType, RightType>::eval(tensor, scalar);
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
template <typename LeftType, typename RightType>
inline void divide(Tensor<LeftType>& left, const Tensor<RightType>& right)
{
    TNT_ASSERT(left.shape == right.shape,
               InvalidParameterException("tnt::divide()", __FILE__, __LINE__,
                   "Element-wise division of two tensors requires that those tensors be of the same size"))

    detail::OptimizedDivide<LeftType, RightType>::eval(left, right);
}

} // namespace tnt

#endif // TNT_MATH_ARITHMETIC_OPS_HPP
