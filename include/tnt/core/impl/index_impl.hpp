#ifndef TNT_INDEX_IMPL_HPP
#define TNT_INDEX_IMPL_HPP

#include <tnt/core/index.hpp>
#include <tnt/utils/testing.hpp>

#include <iostream>

namespace tnt
{

inline Index::Index() noexcept {}

inline Index::Index(const Shape& shape)
{
    this->loc = std::vector<int>(shape.num_axes(), 0);
    this->shape = shape;
}

inline bool Index::operator== (const Index& other) const noexcept
{
    return this->loc == other.loc &&
           this->shape == other.shape;
}

inline bool Index::operator!= (const Index& other) const noexcept
{
    return this->loc != other.loc &&
           this->shape == other.shape;
}

inline int Index::operator[] (int index) const
{
    BOUNDS_CHECK("Index::operator[]", index, 0, this->shape.num_axes() - 1);
    return this->loc[index];
}

inline int& Index::operator[] (int index)
{
    BOUNDS_CHECK("Index::operator[]", index, 0, this->shape.num_axes() - 1);
    return this->loc[index];
}

inline Index& Index::operator++ () noexcept
{
    for (int i = this->num_axes() - 1; i >= 0; i--) {
        if ((++this->loc[i]) == this->shape[i]) {
            if (i == 0) // Special case, end of the index
                break; // Leave the index in an invalid state

            this->loc[i] = 0;
        } else {
            break;
        }
    }

    return *this;
}

TEST_CASE("Index::operator++()")
{
    Index index(Shape{2, 2, 2});

    REQUIRE((index.loc     == std::vector<int>{0, 0, 0}));
    REQUIRE(((++index).loc == std::vector<int>{0, 0, 1}));
    REQUIRE(((++index).loc == std::vector<int>{0, 1, 0}));
    REQUIRE(((++index).loc == std::vector<int>{0, 1, 1}));
    REQUIRE(((++index).loc == std::vector<int>{1, 0, 0}));
    REQUIRE(((++index).loc == std::vector<int>{1, 0, 1}));
    REQUIRE(((++index).loc == std::vector<int>{1, 1, 0}));
    REQUIRE(((++index).loc == std::vector<int>{1, 1, 1}));
    REQUIRE(((++index).loc == std::vector<int>{2, 0, 0}));
}

inline Index Index::operator++ (int) noexcept
{
    Index temp = *this;
    this->operator ++();
    return temp;
}

TEST_CASE("Index::operator++(int)")
{
    Index index(Shape{2, 2, 2});

    REQUIRE((index.loc     == std::vector<int>{0, 0, 0}));
    REQUIRE(((index++).loc == std::vector<int>{0, 0, 0}));
    REQUIRE(((index++).loc == std::vector<int>{0, 0, 1}));
    REQUIRE(((index++).loc == std::vector<int>{0, 1, 0}));
    REQUIRE(((index++).loc == std::vector<int>{0, 1, 1}));
    REQUIRE(((index++).loc == std::vector<int>{1, 0, 0}));
    REQUIRE(((index++).loc == std::vector<int>{1, 0, 1}));
    REQUIRE(((index++).loc == std::vector<int>{1, 1, 0}));
    REQUIRE(((index++).loc == std::vector<int>{1, 1, 1}));
    REQUIRE(((index).loc == std::vector<int>{2, 0, 0}));
}

inline int Index::num_axes() const noexcept
{
    return this->shape.num_axes();
}

TEST_CASE("Index::num_axes()")
{
    REQUIRE(Index().num_axes()                  == 0);
    REQUIRE(Index(Shape{1}).num_axes()          == 1);
    REQUIRE(Index(Shape{1, 1}).num_axes()       == 2);
    REQUIRE(Index(Shape{1, 1, 1, 1}).num_axes() == 4);
}

inline int Index::distance(const Stride& stride) const
{
    TNT_ASSERT(stride.num_axes() == this->shape.num_axes(),
               std::runtime_error("Error in Index::Distance(). Given stride has a different number of axes then the index"));

    int dist = 0;
    for (int i = 0; i < this->num_axes(); ++i)
        dist += this->loc[i] * stride[i];

    return dist;
}

TEST_CASE("Index::distance(const Stride&)")
{
    Index index(Shape{2, 2, 2});
    index.loc = {1, 0, 1};

    REQUIRE(index.distance(Stride{4, 4, 1}) == 5);
    REQUIRE(index.distance(Stride{1, 1, 1}) == 2);
    REQUIRE(index.distance(Stride{0, 0, 0}) == 0);
    REQUIRE(index.distance(Stride{2, 1, 2}) == 4);

    REQUIRE_THROWS(index.distance(Stride{2, 2}));
}

// ----------------------------------------------------------------------------
// Index Stream Operator

TNT_EXPORT inline std::ostream& operator<<(std::ostream& stream, const Index& index)
{
    if (index.num_axes() == 0)
        return stream << "Index: {}";

    std::string output = "Index: {";
    for (int dim : index.loc)
        output += std::to_string(dim) + ",";
    return stream << output.substr(0, output.size() - 1) + "}";
}

} // namespace tnt

#endif // TNT_INDEX_IMPL_HPP
