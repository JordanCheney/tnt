#ifndef TNT_LINEAR_DISCRETE_COSINE_TRANSFORM_IMPL_HPP
#define TNT_LINEAR_DISCRETE_COSINE_TRANSFORM_IMPL_HPP

#include <tnt/linear/discrete_cosine_transform.hpp>
#include <tnt/utils/testing.hpp>
#include <tnt/utils/simd.hpp>

namespace tnt
{

namespace detail
{

constexpr static int dct_forward_coeffs[64] = {
       1,    1,    1,    1,    1,    1,    1,    1,
      15,  101,   35,    1,   -1,  -35, -101,  -15,
       3,    1,   -1,   -3,   -3,   -1,    1,    3,
       1,    3,  -11,   -1,    1,   11,   -3,   -1,
       1,   -1,   -1,    1,    1,   -1,   -1,    1,
       1,  -23,   -1,    1,   -1,    1,   23,   -1,
       1,   -1,    1,   -1,   -1,    1,   -1,    1,
       1,  -21,   13,   -1,    1,  -13,   21,   -1
};

constexpr static int dct_forward_shifts[64] = {
       0,    0,    0,    0,    0,    0,    0,    0,
       4,    7,    6,    2,    2,    6,    7,    4,  
       2,    1,    1,    2,    2,    1,    1,    2,
       1,    5,    4,    1,    1,    4,    5,    1,
       1,    1,    1,    1,    1,    1,    1,    1,
       0,    4,    3,    0,    0,    3,    4,    0,
       1,    0,    0,    1,    1,    0,    0,    1,
       2,    5,    4,    0,    0,    4,    5,    2
};


template <typename DataType>
struct OptimizedDCT<DataType, void>
{
    using VecType = typename SIMDType<DataType>::VecType;

    static Tensor<DataType> eval(const Tensor<DataType>& tensor)
    {
        
    }
};

} // namespace detail

} // namespace tnt

#endif // TNT_LINEAR_DISCRETE_COSINE_TRANSFORM_IMPL_HPP
