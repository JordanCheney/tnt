#ifndef TNT_INDEX_HPP
#define TNT_INDEX_HPP

#include <tnt/core/shape.hpp>
#include <tnt/core/stride.hpp>

namespace tnt
{

// ----------------------------------------------------------------------------
// Utility class to hold a position in a tensor

class TNT_EXPORT Index
{
public:

// ----------------------------------------------------------------------------
// Constructors

    Index() noexcept;
    Index(const Shape& shape);

    Index(const Index& other) = default;
    Index& operator =(const Index& other) = default;

    Index(Index&& other) = default;
    Index& operator =(Index&& other) = default;

// ----------------------------------------------------------------------------
// Operators

    bool operator== (const Index& other) const noexcept;
    bool operator!= (const Index& other) const noexcept;

    int  operator[] (int index) const;
    int& operator[] (int index);

    Index& operator++() noexcept;
    Index operator++(int) noexcept;

// ----------------------------------------------------------------------------
// Functions

    int num_axes() const noexcept;
    int distance(const Stride& stride) const;

// ----------------------------------------------------------------------------
// Members

    std::vector<int> loc;
    Shape shape;
};

// ----------------------------------------------------------------------------

} // namespace tnt

// ----------------------------------------------------------------------------

#endif // TNT_INDEX_HPP

