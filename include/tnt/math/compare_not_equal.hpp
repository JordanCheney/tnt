#ifndef TNT_MATH_COMPARE_NEQ_HPP
#define TNT_MATH_COMPARE_NEQ_HPP

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
struct OptimizedCompareNotEqual
{
    Tensor<uint8_t, DeviceType> run(const Tensor<LeftType, DeviceType>&, const RightType&);
};

} // namespace detail

/// \brief Check inequality of a tensor and scalar elementwise
///
/// \param tensor A immutable tensor.
/// \param right A scalar
/// \returns A mask tensor with DataType `uint8_t`. The mask will contain `255`
/// where the condition is true and `0` everywhere else.
template <
          typename LeftType,
          typename RightType,
          typename DeviceType
         >
inline Tensor<uint8_t, DeviceType> compare_not_equal(const Tensor<LeftType, DeviceType>& left, const RightType& scalar)
{
    return detail::OptimizedCompareNotEqual<LeftType, RightType, DeviceType>().run(left, scalar);
}

} // namespace tnt

#endif // TNT_MATH_COMPARE_NEQ_HPP
