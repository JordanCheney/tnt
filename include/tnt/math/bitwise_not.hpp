#ifndef TNT_MATH_BITWISE_NOT_HPP
#define TNT_MATH_BITWISE_NOT_HPP

#include <tnt/core/tensor.hpp>

namespace tnt
{

namespace detail
{

template <
          typename LeftType,
          typename DeviceType
         >
struct OptimizedBitwiseNot
{
    void run(Tensor<LeftType, DeviceType>&);
};

} // namespace detail

/// \brief Compute the 1's complement of an integer tensor
///
/// The 1's complement is computed in-place on the tensor
/// \param tensor A mutable tensor. The complement is taken in-place
/// \requires Type `LeftType` shall be an integer
template <
          typename LeftType,
          typename DeviceType
         >
inline void bitwise_not(Tensor<LeftType, DeviceType>& tensor) noexcept
{
    static_assert(std::is_integral<LeftType>::value,
                  "Bitwise not is only meaningful for integer types");

    detail::OptimizedBitwiseNot<LeftType, DeviceType>().run(tensor);
}

} // namespace tnt

#endif // TNT_MATH_BITWISE_NOT_HPP
