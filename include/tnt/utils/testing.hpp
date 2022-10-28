#ifndef TNT_TESTING_HPP
#define TNT_TESTING_HPP

#include <tnt/utils/doctest.hpp>
#include <tnt/core/tensor.hpp>

#include <cstdint>

namespace tnt
{

// ----------------------------------------------------------------------------
// Vanilla data types

typedef doctest::Types<uint8_t,
                       uint16_t,
                       uint32_t,
                       uint64_t,
                       int8_t,
                       int16_t,
                       int32_t,
                       int64_t,
                       float,
                       double> test_data_types;

typedef doctest::Types<uint8_t, uint16_t, uint32_t, uint64_t> test_unsigned_data_types;
typedef doctest::Types<int8_t, int16_t, int32_t, int64_t> test_signed_data_types;
typedef doctest::Types<uint8_t, uint16_t, uint32_t, uint64_t,
                        int8_t,  int16_t,  int32_t,  int64_t> test_integer_data_types;
typedef doctest::Types<float, double> test_float_data_types;

// ----------------------------------------------------------------------------
// Approximately equal for floating point comparisons

template <typename DataType>
inline bool approx_equal(const tnt::Tensor<DataType>& left, const tnt::Tensor<DataType>& right, float epsilon = 0.0001)
{
    if (left.shape != right.shape)
        return false;

    const DataType* l_ptr = left.data.data;
    const DataType* r_ptr = right.data.data;

    int i = 0, total = left.shape.total();
    for ( ; total--; ++i)
        if (fabs(l_ptr[i] - r_ptr[i]) > epsilon)
            return false;

    return true;
}

} // namespace tnt

#endif // TNT_TESTING_HPP
