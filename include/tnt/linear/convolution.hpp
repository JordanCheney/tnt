#ifndef TNT_LINEAR_CONVOLUTION_HPP
#define TNT_LINEAR_CONVOLUTION_HPP

#include <tnt/core/tensor.hpp>

namespace tnt
{

namespace detail
{

template <typename DataType, typename Enable = void>
struct OptimizedConvolution3D
{
    static Tensor<DataType> eval(const Tensor<DataType>& tensor,
                                 const Tensor<DataType>& kernel,
                                 int pad,
                                 DataType pad_value,
                                 int stride);
};

template <typename DataType, typename Enable = void>
struct OptimizedWinogradConvoluton
{
    static Tensor<DataType> eval(const Tensor<DataType>& tensor,
                                 const Tensor<DataType>& kernel,
                                 int pad,
                                 DataType pad_value,
                                 int stride);
};

} // namespace detail

template <typename DataType>
inline Tensor<DataType> conv3D(const Tensor<DataType>& tensor,
                               const Tensor<DataType>& kernel,
                               int pad = 0,
                               DataType pad_value = 0,
                               int stride = 1)
{
    TNT_ASSERT(tensor.shape.num_axes() == 3,
               InvalidParameterException("tnt::conv3D()",
                                         __FILE__,
                                         __LINE__,
                                         "3D convolution requires a 3D tensor"))

    TNT_ASSERT(kernel.shape.num_axes() == 3,
               InvalidParameterException("tnt::conv3D()",
                                         __FILE__,
                                         __LINE__,
                                         "3D convolution requires a 3D kernel"))

    TNT_ASSERT(tensor.shape[2] == kernel.shape[2],
               InvalidParameterException("tnt::conv3D()",
                                         __FILE__,
                                         __LINE__,
                                         "3D convolution requires the tensor and kernel to have the same size least significant dimension"))

    TNT_ASSERT(tensor.shape[0] + pad >= kernel.shape[0]
                && tensor.shape[1] + pad >= kernel.shape[1],
               InvalidParameterException("tnt::conv3D()",
                                         __FILE__,
                                         __LINE__,
                                         "3D convolution requires the tensor + padding to be larger than the kernel"))

    return detail::OptimizedConvolution3D<DataType>::eval(tensor, kernel, pad, pad_value, stride);
}

} // namespace tnt

#endif // TNT_LINEAR_CONVOLUTION_HPP
