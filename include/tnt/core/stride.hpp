#ifndef TNT_STRIDE_HPP
#define TNT_STRIDE_HPP

#include <tnt/core/shape.hpp>

#include <vector>
#include <initializer_list>

namespace tnt
{

// ----------------------------------------------------------------------------
// Utility class representing the length of multiple dimensions

class TNT_EXPORT Stride
{
public:

// ----------------------------------------------------------------------------
// Constructors

    Stride();
    Stride(const std::initializer_list<int>& values);
    Stride(const std::vector<int>& values);
    Stride(const Shape& shape);

    Stride(const Stride& other) = default;
    Stride& operator =(const Stride& other) = default;

    Stride(Stride&& other) = default;
    Stride& operator =(Stride&& other) = default;

// ----------------------------------------------------------------------------
// Operators

    bool operator ==(const Stride& other) const;
    bool operator !=(const Stride& other) const;

    int  operator [](int index) const;
    int& operator [](int index);

// ----------------------------------------------------------------------------
// Functions

    int num_axes() const;

// ----------------------------------------------------------------------------
// Members

    std::vector<int> strides;
};

// ----------------------------------------------------------------------------

} // namespace tnt

// ----------------------------------------------------------------------------

#endif // TNT_STRIDE_HPP
