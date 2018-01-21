#ifndef TNT_STRIDE_IMPL_HPP
#define TNT_STRIDE_IMPL_HPP

#include <tnt/core/stride.hpp>

namespace tnt
{

// ----------------------------------------------------------------------------
// Constructors

inline Stride::Stride() {}

inline Stride::Stride(const std::initializer_list<int>& strides)
{
    this->strides = strides;
}

inline Stride::Stride(const std::vector<int> &strides)
{
    this->strides = strides;
}


inline Stride::Stride(const Shape &shape)
{
    this->strides.push_back(1);
    for (int i = 1; i < shape.num_axes(); ++i)
        this->strides.insert(this->strides.begin(), this->strides.front() * shape[shape.num_axes() - i]);
}

// ----------------------------------------------------------------------------
// Operators

inline bool Stride::operator ==(const Stride& other) const
{
    return this->strides == other.strides;
}

inline bool Stride::operator !=(const Stride& other) const
{
    return this->strides != other.strides;
}

inline int Stride::operator [](int index) const
{
    BOUNDS_CHECK("Stride::operator[]", index, 0, this->num_axes());
    return this->strides[index];
}

inline int& Stride::operator [](int index)
{
    BOUNDS_CHECK("Stride::operator[]", index, 0, this->num_axes());
    return this->strides[index];
}

// ----------------------------------------------------------------------------
// Functions

inline int Stride::num_axes() const
{
    return this->strides.size();
}

// ----------------------------------------------------------------------------
// Stride Stream Operator

TNT_EXPORT inline std::ostream& operator<<(std::ostream& stream, const Stride& stride)
{
    if (stride.num_axes() == 0)
        return stream << "Stride: {}";

    std::string output = "Stride: {";
    for (int step : stride.strides)
        output += std::to_string(step) + ",";
    return stream << output.substr(0, output.size() - 1) + "}";
}

// ----------------------------------------------------------------------------

} // namespace tnt

// ----------------------------------------------------------------------------

#endif // TNT_STRIDE_IMPL_HPP
