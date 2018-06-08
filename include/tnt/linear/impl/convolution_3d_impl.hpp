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
struct OptimizedConvolution3D<DataType, void>
{
    using VecType = typename SIMDType<DataType>::VecType;

public:
    static Tensor<DataType> eval(const Tensor<DataType>& tensor, const Tensor<DataType>& kernel, int pad, int stride)
    {
        int output_rows = (tensor.shape[0] - kernel.shape[0] + 2 * pad) / stride + 1;
        int output_cols = (tensor.shape[1] - kernel.shape[1] + 2 * pad) / stride + 1;
        Tensor<DataType> output(Shape{output_rows, output_cols});

        const DataType* t_ptr = tensor.data.data;
        const DataType* k_ptr = kernel.data.data;
        DataType* o_ptr       = output.data.data;

        const int row_step = tensor.shape[1] * tensor.shape[2];
        const int col_step = tensor.shape[2];

        const int channel_steps     = kernel.shape[2] / OptimalSIMDSize<DataType>::value;
        const int channel_remainder = kernel.shape[2] - (channel_steps * OptimalSIMDSize<DataType>::value);

        int start_row = -pad, start_col = -pad;

        int steps = output_rows * output_cols;
        for (int i = 0; --steps; ++i) {
            o_ptr[i] = 0;

            int kernel_rows = kernel.shape[0];
            for (int row = start_row; --kernel_rows; ++row) {
                if (row < 0) {
                    continue;
                }

                int kernel_cols = kernel.shape[1];
                for (int col = start_col; --kernel_cols; ++col) {
                    if (col < 0) {
                        continue;
                    }

                    int t_offset = row * row_step + col * col_step;

                    int channel_steps_remaining = channel_steps;
                    for (; --channel_steps_remaining; t_offset += OptimalSIMDSize<DataType>::value) {

                    }
                    auto t_block = LoadSIMDType<DataType, DataType>::load(t_ptr + t_offset);
                    //auto r_block = LoadSIMDType<LeftType, RightType>::load(r_ptr + offset);
                    //auto result = ConvertSIMDType<LeftType>::convert(simdpp::mull(l_block, r_block));
                }
            }
        }        
    }
};

} // namespace detail

} // namespace tnt

#endif // TNT_LINEAR_CONVOLUTION_3D_IMPL_HPP
