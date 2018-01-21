#ifndef TNT_SHAPE_HPP
#define TNT_SHAPE_HPP

#include <tnt/utils/errors.hpp>

#include <vector>
#include <initializer_list>

namespace tnt
{

// ----------------------------------------------------------------------------
// Utility class to hold the size of a tensor

class TNT_EXPORT Shape
{
public:

// ----------------------------------------------------------------------------
// Constructors

    Shape() noexcept;
    Shape(const std::initializer_list<int>& axes);
    Shape(const std::vector<int>& axes);

    Shape(const Shape& other) = default;
    Shape& operator =(const Shape& other) = default;

    Shape(Shape&& other) = default;
    Shape& operator =(Shape&& other) = default;

// ----------------------------------------------------------------------------
// Operators

    bool operator== (const Shape& other) const noexcept;
    bool operator!= (const Shape& other) const noexcept;

    int  operator[] (int index) const;
    int& operator[] (int index);

// ----------------------------------------------------------------------------
// Functions

    int num_axes() const noexcept;
    int total(int from_axis = 0, int to_axis = -1) const;

// ----------------------------------------------------------------------------
// Members

    std::vector<int> axes;
};

// ----------------------------------------------------------------------------

} // namespace tnt

// ----------------------------------------------------------------------------

#endif // TNT_SHAPE_HPP

