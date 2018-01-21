#ifndef TNT_LINEAR_CONVOLUTION_3D_IMPL_HPP
#define TNT_LINEAR_CONVOLUTION_3D_IMPL_HPP

#include <tnt/linear/convolution.hpp>
#include <tnt/utils/testing.hpp>
#include <tnt/utils/simd.hpp>

namespace tnt
{

namespace detail
{

template <typename DataType>
struct OptimizedConvolution3D
{
    using VecType = typename SIMDType<DataType>::VecType;

    static Tensor<DataType> eval(const Tensor<DataType>& tensor, const Tensor<DataType>& kernel, int, DataType, int)
    {
        //size_t output_rows = (tensor.shape[0] + 2 * pad - kernel.shape[0] - 1) / stride;
        //size_t output_cols = (tensor.shape[1] + 2 * pad - kernel.shape[1] - 1) / stride;
        size_t output_rows = tensor.shape[0] - kernel.shape[0] + 1;
        size_t output_cols = tensor.shape[1] - kernel.shape[1] + 1;
        Tensor<DataType> output(Shape{output_rows, output_cols});

        const DataType* t_ptr = tensor.data.data;
        const DataType* k_ptr = kernel.data.data;

        const size_t kernel_block_size = AlignSIMDType<DataType>::num_aligned_blocks(kernel.shape[1] * kernel.shape[2]);

        VecType** kernel_blocks = new VecType*[kernel.shape[2]];
        for (int i = 0; i < kernel.shape[2]; ++i) {
            kernel_blocks[i] = new VecType[kernel_block_size];

            int offset = i * kernel.shape[1] * kernel.shape[2];
            for (int j = 0; j < kernel_blocks; offset += OptimalSIMDSize<DataType>::value)
                kernel_blocks[i][j] = LoadSIMDType<DataType, DataType>::load(k_ptr + offset);
        }

        for (int r = 0; r < output_rows; ++r) {
            for (int c = 0; c < output_cols; ++c) {
                DataType total = 0;

            }
        }
    }
};

} // namespace detail

} // namespace tnt

#endif // TNT_LINEAR_CONVOLUTION_3D_IMPL_HPP
