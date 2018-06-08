#ifndef TNT_LINEAR_DISCRETE_COSINE_TRANSFORM_HPP
#define TNT_LINEAR_DISCRETE_COSINE_TRANSFORM_HPP

#include <tnt/core/tensor.hpp>

namespace tnt
{

namespace detail
{

template <typename DataType, typename Enable = void>
struct OptimizedDCT
{
    static Tensor<DataType> eval(const Tensor<DataType>& tensor);
};

} // namespace detail

template <typename DataType>
inline Tensor<DataType> dct(const Tensor<DataType>& tensor)
{
    TNT_ASSERT(tensor.shape.num_axes() == 2,
               InvalidParamterException("tnt::dct()",
                                        __FILE__,
                                        __LINE__,
                                        "The discrete cosine transform requires a 2D tensor"))

    return detail::OptimizedDCT::eval(tensor);
}

} // namespace tnt

#endif // TNT_LINEAR_DISCRETE_COSINE_TRANSFORM_HPP
