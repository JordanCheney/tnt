#ifndef TNT_RANGE_IMPL_HPP
#define TNT_RANGE_IMPL_HPP

#include <tnt/core/range.hpp>

namespace tnt
{

// ----------------------------------------------------------------------------
// Constructors

inline Range::Range() noexcept
{
    this->begin = 0;
    this->end = -1;
}

inline Range::Range(int location) noexcept
{
    this->begin = location;
    this->end = location + 1;
}

inline Range::Range(int begin, int end) noexcept
{
    this->begin = begin;
    this->end = end;
}

// ----------------------------------------------------------------------------
// Operators

inline bool Range::operator ==(const Range& other) const noexcept
{
    return (this->begin == other.begin)
            && (this->end == other.end);
}

inline bool Range::operator !=(const Range& other) const noexcept
{
    return (this->begin != other.begin)
            || (this->end != other.end);
}

// ----------------------------------------------------------------------------
// Range Stream Operator

TNT_EXPORT inline std::ostream& operator<<(std::ostream& stream, const Range& range)
{
    return stream << "Range: {"
                  << std::to_string(range.begin)
                  << " -> "
                  << std::to_string(range.end)
                  << "}";
}

// ----------------------------------------------------------------------------
// Utility variadic constructor

TNT_EXPORT inline std::vector<Range> make_range_list(int location)
{
    return std::vector<Range>{Range(location)};
}

TNT_EXPORT inline std::vector<Range> make_range_list(Range range)
{
    return std::vector<Range>{range};
}

template <typename ... RangeConstructable>
TNT_EXPORT inline std::vector<Range> make_range_list(int location, RangeConstructable... ranges)
{
    std::vector<Range> left{Range(location)};
    std::vector<Range> right = make_range_list(ranges...);
    left.insert(left.end(), right.begin(), right.end());

    return left;
}

template <typename ... RangeConstructable>
TNT_EXPORT inline std::vector<Range> make_range_list(Range range, RangeConstructable... ranges)
{
    std::vector<Range> left{range};
    std::vector<Range> right = make_range_list(ranges...);
    left.insert(left.end(), right.begin(), right.end());

    return left;
}

} // namespace tnt

#endif // TNT_RANGE_IMPL_HPP
