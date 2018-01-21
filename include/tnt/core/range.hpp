#ifndef TNT_RANGE_HPP
#define TNT_RANGE_HPP

#include <tnt/utils/errors.hpp>

namespace tnt
{

// ----------------------------------------------------------------------------
// Utility class to slice a tensor

class TNT_EXPORT Range
{
public:

// ----------------------------------------------------------------------------
// Constructors

    Range() noexcept;
    Range(int location) noexcept;
    Range(int begin, int end) noexcept;

    Range(const Range& other) = default;
    Range& operator=(const Range& other) = default;

    Range(Range&& other) = default;
    Range& operator=(Range&& other) = default;

// ----------------------------------------------------------------------------
// Functions

    bool operator== (const Range& other) const noexcept;
    bool operator!= (const Range& other) const noexcept;

// ----------------------------------------------------------------------------
// Members

    int begin;
    int end;
};

// ----------------------------------------------------------------------------
// Utility functions to construct a vector<Range> object with variadic templates

std::vector<Range> make_range_list(int location);
std::vector<Range> make_range_list(Range range);

template <typename ... RangeConstructable>
std::vector<Range> make_range_list(int location, RangeConstructable... ranges);

template <typename ... RangeConstructable>
std::vector<Range> make_range_list(Range range, RangeConstructable... ranges);

// ----------------------------------------------------------------------------

} // namespace tnt

// ----------------------------------------------------------------------------

#endif // TNT_RANGE_HPP

