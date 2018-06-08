#ifndef TNT_SIMD_HPP
#define TNT_SIMD_HPP

#include <tnt/utils/macros.hpp>

#include <tnt/deps/simd/simd.h>

#include <sstream>

namespace tnt
{

/// \brief Utility struct with the optimal size of a SIMD vector for a given type
/// on the current architecture
///
/// \requires Type `T` is arithmetic
template <typename T>
struct OptimalSIMDSize
{
    constexpr static int value = 0;
};

template <> struct OptimalSIMDSize<uint8_t>  { constexpr static int value = SIMDPP_FAST_INT8_SIZE;  };
template <> struct OptimalSIMDSize<uint16_t> { constexpr static int value = SIMDPP_FAST_INT16_SIZE; };
template <> struct OptimalSIMDSize<uint32_t> { constexpr static int value = SIMDPP_FAST_INT32_SIZE; };
template <> struct OptimalSIMDSize<uint64_t> { constexpr static int value = SIMDPP_FAST_INT64_SIZE; };

template <> struct OptimalSIMDSize<int8_t>   { constexpr static int value = SIMDPP_FAST_INT8_SIZE;  };
template <> struct OptimalSIMDSize<int16_t>  { constexpr static int value = SIMDPP_FAST_INT16_SIZE; };
template <> struct OptimalSIMDSize<int32_t>  { constexpr static int value = SIMDPP_FAST_INT32_SIZE; };
template <> struct OptimalSIMDSize<int64_t>  { constexpr static int value = SIMDPP_FAST_INT64_SIZE; };

template <> struct OptimalSIMDSize<float>    { constexpr static int value = SIMDPP_FAST_FLOAT32_SIZE; };
template <> struct OptimalSIMDSize<double>   { constexpr static int value = SIMDPP_FAST_FLOAT64_SIZE; };

/// \brief Utility struct with a typedef for the SIMD vector of a given type
/// and size
///
/// \requires Type `T` is arithmentic
template <typename T, int Size>
struct FullSIMDType {};

template <int _Size> struct FullSIMDType<uint8_t, _Size>
{
    using VecType = simdpp::uint8<_Size>;
    constexpr static int Size = _Size;
};
template <int _Size> struct FullSIMDType<uint16_t, _Size>
{
    using VecType = simdpp::uint16<_Size>;
    constexpr static int Size = _Size;
};
template <int _Size> struct FullSIMDType<uint32_t, _Size>
{
    using VecType = simdpp::uint32<_Size>;
    constexpr static int Size = _Size;
};
template <int _Size> struct FullSIMDType<uint64_t, _Size>
{
    using VecType = simdpp::uint64<_Size>;
    constexpr static int Size = _Size;
};

template <int _Size> struct FullSIMDType<int8_t, _Size>
{
    using VecType = simdpp::int8<_Size>;
    constexpr static int Size = _Size;
};
template <int _Size> struct FullSIMDType<int16_t, _Size>
{
    using VecType = simdpp::int16<_Size>;
    constexpr static int Size = _Size;
};
template <int _Size> struct FullSIMDType<int32_t, _Size>
{
    using VecType = simdpp::int32<_Size>;
    constexpr static int Size = _Size;
};
template <int _Size> struct FullSIMDType<int64_t, _Size>
{
    using VecType = simdpp::int64<_Size>;
    constexpr static int Size = _Size;
};

template <int _Size> struct FullSIMDType<float, _Size>
{
    using VecType = simdpp::float32<_Size>;
    constexpr static int Size = _Size;
};
template <int _Size> struct FullSIMDType<double, _Size>
{
    using VecType = simdpp::float64<_Size>;
    constexpr static int Size = _Size;
};

/// \brief Shortcut struct for a [FullSIMDType]() of a given type with the
/// optimal size for that type on the current architecture
///
/// \requires Type `T` is arithmetic
template <typename T>
struct SIMDType : public FullSIMDType<T, OptimalSIMDSize<T>::value>
{
};

/// \brief Struct to convert a SIMD type to type `T`.
///
/// \requires Type `T` is arithmetic
template <typename T>
struct ConvertSIMDType
{
    /// \brief Convert a SIMD type to another SIMD type
    ///
    /// \requires Type `U` is a SIMD Type
    /// \effects Returns a SIMD type of type `T` with the same size as type `U`.
    template <typename U>
    static TNT_INL typename SIMDType<T>::VecType convert(U) {}
};

template <> struct ConvertSIMDType<uint8_t>  { template <typename U> static TNT_INL SIMDType<uint8_t>::VecType  convert(U value) { return simdpp::to_uint8(value);  } };
template <> struct ConvertSIMDType<uint16_t> { template <typename U> static TNT_INL SIMDType<uint16_t>::VecType convert(U value) { return simdpp::to_uint16(value); } };
template <> struct ConvertSIMDType<uint32_t> { template <typename U> static TNT_INL SIMDType<uint32_t>::VecType convert(U value) { return simdpp::to_uint32(value); } };
template <> struct ConvertSIMDType<uint64_t> { template <typename U> static TNT_INL SIMDType<uint64_t>::VecType convert(U value) { return simdpp::to_uint64(value); } };

template <> struct ConvertSIMDType<int8_t>   { template <typename U> static TNT_INL SIMDType<int8_t>::VecType   convert(U value) { return simdpp::to_int8(value);  } };
template <> struct ConvertSIMDType<int16_t>  { template <typename U> static TNT_INL SIMDType<int16_t>::VecType  convert(U value) { return simdpp::to_int16(value); } };
template <> struct ConvertSIMDType<int32_t>  { template <typename U> static TNT_INL SIMDType<int32_t>::VecType  convert(U value) { return simdpp::to_int32(value); } };
template <> struct ConvertSIMDType<int64_t>  { template <typename U> static TNT_INL SIMDType<int64_t>::VecType  convert(U value) { return simdpp::to_int64(value); } };

template <> struct ConvertSIMDType<float>    { template <typename U> static TNT_INL SIMDType<float>::VecType    convert(U value) { return simdpp::to_float32(value); } };
template <> struct ConvertSIMDType<double>   { template <typename U> static TNT_INL SIMDType<double>::VecType   convert(U value) { return simdpp::to_float64(value); } };

/// \brief Struct to load a pointer of type `U` into a SIMD register.
///
/// \requires Type `T` is arithmetic
/// \requires Type `U` is arithmetic
template <typename T, typename U>
struct LoadSIMDType
{
    /// \brief Load a pointer of type `U` into a SIMD register
    ///
    /// \requires `ptr` have at least N elements where N is the optimal SIMD
    /// size of type `T`
    static TNT_INL typename SIMDType<T>::VecType load(const U* ptr)
    {
        typedef typename std::remove_cv<U>::type CleanU;
        typedef typename FullSIMDType<CleanU, OptimalSIMDSize<T>::value>::VecType OtherVecType;

        return ConvertSIMDType<T>::convert(simdpp::load<OtherVecType>(ptr));
    }
};

// Special Cases
#define SPECIAL_LOAD_CASE(LEFT_TYPE, RIGHT_TYPE)                               \
template <>                                                                    \
struct LoadSIMDType<LEFT_TYPE, RIGHT_TYPE>                                     \
{                                                                              \
    static TNT_INL typename SIMDType<LEFT_TYPE>::VecType load(const RIGHT_TYPE* ptr) \
    {                                                                          \
        LEFT_TYPE buffer[OptimalSIMDSize<LEFT_TYPE>::value];                   \
        for (int i = 0; i < OptimalSIMDSize<LEFT_TYPE>::value; ++i)            \
            buffer[i] = (LEFT_TYPE) ptr[i];                                    \
        return simdpp::load<SIMDType<LEFT_TYPE>::VecType>(buffer);             \
    }                                                                          \
};                                                                             \
                                                                               \
template <>                                                                    \
struct LoadSIMDType<RIGHT_TYPE, LEFT_TYPE>                                     \
{                                                                              \
    static TNT_INL typename SIMDType<RIGHT_TYPE>::VecType load(const LEFT_TYPE* ptr) \
    {                                                                          \
        RIGHT_TYPE buffer[OptimalSIMDSize<RIGHT_TYPE>::value];                 \
        for (int i = 0; i < OptimalSIMDSize<RIGHT_TYPE>::value; ++i)           \
            buffer[i] = (RIGHT_TYPE) ptr[i];                                   \
        return simdpp::load<SIMDType<RIGHT_TYPE>::VecType>(buffer);            \
    }                                                                          \
};

SPECIAL_LOAD_CASE(float, uint64_t)
SPECIAL_LOAD_CASE(float, int64_t)
SPECIAL_LOAD_CASE(double, uint32_t)
SPECIAL_LOAD_CASE(double, uint64_t)
SPECIAL_LOAD_CASE(double, int64_t)

/// \brief Hardcoded lookup for the log of powers of 2 up to 256.
///
/// \notes `log2(0)` == 0 in this implementation.
template <int N>
struct FastLog2
{
    constexpr static int value = -1;
};

template <> struct FastLog2<0> { constexpr static int value = 0; };
template <> struct FastLog2<1> { constexpr static int value = 0; };
template <> struct FastLog2<2> { constexpr static int value = 1; };
template <> struct FastLog2<4> { constexpr static int value = 2; };
template <> struct FastLog2<8> { constexpr static int value = 3; };
template <> struct FastLog2<16> { constexpr static int value = 4; };
template <> struct FastLog2<32> { constexpr static int value = 5; };
template <> struct FastLog2<64> { constexpr static int value = 6; };
template <> struct FastLog2<128> { constexpr static int value = 7; };
template <> struct FastLog2<256> { constexpr static int value = 8; };

/// \brief Struct to compute aligned sizes for SIMD allocations
///
/// \requires Type `T` is arithmetic
template <typename T>
struct AlignSIMDType
{
    static_assert(std::is_arithmetic<T>::value, "AlignSIMDType requires an arithmetic type");
    static_assert(FastLog2<OptimalSIMDSize<T>::value>::value > 0, "Invalid size passed to FastLog2()");

    constexpr static int Shift = FastLog2<OptimalSIMDSize<T>::value>::value;

    /// \brief Calculate the minimum number of SIMD blocks that are required to
    /// processes the given number of bytes. A single SIMD operation can
    /// process 1 block.
    static TNT_INL int num_aligned_blocks(size_t bytes)
    {
        return ((bytes - 1) >> Shift) + 1;
    }

    /// \brief Calculate the number of bytes required for the minimum number of blocks
    /// to processes the given number of bytes.
    static TNT_INL int aligned_buffer_size(size_t bytes)
    {
        return num_aligned_blocks(bytes) << Shift;
    }
};

/// \brief SIMDPP multiplication increases integer size to avoid overflow. This type
///        is a convienience type to promote an integer input to its expected output
template <typename T>
struct IntegerMultiplicationResultType : public FullSIMDType<T, OptimalSIMDSize<T>::value>
{
    static_assert(sizeof(T) == -1, "IntegerMultiplicationResultType is valid only for [`uint8_t`, `uint16_t`, `uint32_t`, `int8_t`, `int16_t`, `int32_t`]");
};

template <> struct IntegerMultiplicationResultType<uint16_t> : public FullSIMDType<uint32_t, OptimalSIMDSize<uint16_t>::value> {};
template <> struct IntegerMultiplicationResultType<uint32_t> : public FullSIMDType<uint64_t, OptimalSIMDSize<uint32_t>::value> {};

template <> struct IntegerMultiplicationResultType<int16_t> : public FullSIMDType<int32_t, OptimalSIMDSize<int16_t>::value> {};
template <> struct IntegerMultiplicationResultType<int32_t> : public FullSIMDType<int64_t, OptimalSIMDSize<int32_t>::value> {};

/// \brief Provide a consistent interface for multiplication
template <typename T>
struct MultiplySIMD
{
    static_assert(std::is_arithmetic<T>::value,      "MultiplySIMD requires an arithmetic type");
    static_assert(!std::is_same<T, uint8_t>::value,  "Unsigned 8 bit multiplication is not supported");
    static_assert(!std::is_same<T, uint64_t>::value, "Unsigned 64 bit multiplication is not supported (overflow)");
    static_assert(!std::is_same<T, int8_t>::value,   "Signed 8 bit multiplication is not supported");
    static_assert(!std::is_same<T, int64_t>::value,  "Signed 64 bit multiplication is not supported (overflow)");

    using VecType = typename SIMDType<T>::VecType;

    static TNT_INL VecType run(const VecType& left, const VecType& right)
    {
        using IntegerResult = typename IntegerMultiplicationResultType<T>::VecType;

        IntegerResult result = simdpp::mull(left, right);
        return ConvertSIMDType<T>::convert(result);
    }
};

template <> struct MultiplySIMD<float>
{
    using VecType = typename SIMDType<float>::VecType;

    static TNT_INL VecType run(const VecType& left, const VecType& right)
    {
        return simdpp::mul(left, right);
    }
};

template <> struct MultiplySIMD<double>
{
    using VecType = typename SIMDType<double>::VecType;

    static TNT_INL VecType run(const VecType& left, const VecType& right)
    {
        return simdpp::mul(left, right);
    }
};

/// \brief Utility struct to print out type information
///
/// This struct provides a single character for type and an integer for size.
/// Valid types are `'u'`, `'i'`, and `'f'` for unsigned, signed and floating
/// respectively. Size is the number of bits in the type.
/// \requires Type `T` is arithmentic
/// \notes `printf("%c%d", TypeInfo<float>::type, TypeInfo<float>::bits); // prints f32`
template <typename T>
struct TypeInfo
{
    static_assert(std::is_arithmetic<T>::value, "TypeInfo requires an arithmetic type");

    constexpr static char type = 'n';
    constexpr static int bits = 0;
};

template <> struct TypeInfo<uint8_t>  { constexpr static char type = 'u'; constexpr static int bits = 8; };
template <> struct TypeInfo<uint16_t> { constexpr static char type = 'u'; constexpr static int bits = 16; };
template <> struct TypeInfo<uint32_t> { constexpr static char type = 'u'; constexpr static int bits = 32; };
template <> struct TypeInfo<uint64_t> { constexpr static char type = 'u'; constexpr static int bits = 64; };

template <> struct TypeInfo<int8_t>   { constexpr static char type = 'i'; constexpr static int bits = 8; };
template <> struct TypeInfo<int16_t>  { constexpr static char type = 'i'; constexpr static int bits = 16; };
template <> struct TypeInfo<int32_t>  { constexpr static char type = 'i'; constexpr static int bits = 32; };
template <> struct TypeInfo<int64_t>  { constexpr static char type = 'i'; constexpr static int bits = 64; };

template <> struct TypeInfo<float>    { constexpr static char type = 'f'; constexpr static int bits = 32; };
template <> struct TypeInfo<double>   { constexpr static char type = 'f'; constexpr static int bits = 64; };

/// \brief Recursive helper to convert an arbitrary sized SIMD register to a
/// string for debugging.
template <unsigned N, typename T>
struct SIMDToStringHelper
{
    static_assert(N <= 32, "N is too big");

    TNT_INL static std::string to_string(const T& value)
    {
        return SIMDToStringHelper<N - 1, T>::to_string(value) + "," + std::to_string(simdpp::extract<N - 1>(value));
    }
};

template <typename T>
struct SIMDToStringHelper<1, T>
{
    TNT_INL static std::string to_string(const T& value)
    {
        return std::to_string(simdpp::extract<0>(value));
    }
};

/// \brief Convert the contents of a SIMD register to a string.
///
/// The string is formatted like [r0,r1,r2...rN]
template <typename T>
TNT_INL std::string simd_to_string(const T& value)
{
    return "[" + SIMDToStringHelper<T::length, T>::to_string(value) + "]";
}

} // namespace tnt

#endif // TNT_SIMD_HPP
