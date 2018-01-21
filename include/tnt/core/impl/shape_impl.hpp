#ifndef TNT_SHAPE_IMPL_HPP
#define TNT_SHAPE_IMPL_HPP

#include <tnt/core/shape.hpp>
#include <tnt/utils/testing.hpp>

namespace tnt
{

// ----------------------------------------------------------------------------
// Constructors

inline Shape::Shape() noexcept {}

TEST_CASE("Shape()")
{
    Shape shape;

    REQUIRE(shape.num_axes() == 0);
    REQUIRE(shape.total() == 0);

    REQUIRE_THROWS(shape[1]);
}

inline Shape::Shape(const std::initializer_list<int>& axes)
{
    this->axes = axes;
}

TEST_CASE("Shape(std::initializer_list<int>&)")
{
    {
        Shape shape{1, 2, 3, 4};

        REQUIRE(shape.num_axes() == 4);
        REQUIRE(shape.total() == 24);
        REQUIRE(shape.total(2) == 12);

        REQUIRE(shape[0] == 1);
        REQUIRE(shape[1] == 2);
        REQUIRE(shape[2] == 3);
        REQUIRE(shape[3] == 4);

        REQUIRE_THROWS(shape[4]);
    }

    {
        Shape shape{5, 6, 7};

        REQUIRE(shape.num_axes() == 3);
        REQUIRE(shape.total() == 210);
        REQUIRE(shape.total(2) == 7);

        REQUIRE(shape[0] == 5);
        REQUIRE(shape[1] == 6);
        REQUIRE(shape[2] == 7);

        REQUIRE_THROWS(shape[3]);
    }
}

inline Shape::Shape(const std::vector<int>& axes)
{
    this->axes = axes;
}

TEST_CASE("Shape(const std::vector<int>&)")
{
    {
        Shape shape(std::vector<int>{1, 2, 3, 4});

        REQUIRE(shape.num_axes() == 4);
        REQUIRE(shape.total() == 24);
        REQUIRE(shape.total(2) == 12);

        REQUIRE(shape[0] == 1);
        REQUIRE(shape[1] == 2);
        REQUIRE(shape[2] == 3);
        REQUIRE(shape[3] == 4);

        REQUIRE_THROWS(shape[4]);
    }

    {
        Shape shape(std::vector<int>{5, 6, 7});

        REQUIRE(shape.num_axes() == 3);
        REQUIRE(shape.total() == 210);
        REQUIRE(shape.total(2) == 7);

        REQUIRE(shape[0] == 5);
        REQUIRE(shape[1] == 6);
        REQUIRE(shape[2] == 7);

        REQUIRE_THROWS(shape[3]);
    }
}

// ----------------------------------------------------------------------------
// Operators

inline bool Shape::operator== (const Shape& other) const noexcept
{
    return this->axes == other.axes;
}

inline bool Shape::operator!= (const Shape& other) const noexcept
{
    return this->axes != other.axes;
}

TEST_CASE("Shape::operator== / Shape::operator!=")
{
    Shape shape1;
    Shape shape2{1, 2, 3};
    Shape shape3{2, 2, 3};
    Shape shape4(std::vector<int>{1, 2, 3});
    Shape shape5(std::vector<int>{2, 2, 2, 2});

    REQUIRE(shape1 != shape2);
    REQUIRE(shape1 != shape3);
    REQUIRE(shape1 != shape4);
    REQUIRE(shape1 != shape5);

    REQUIRE(shape2 != shape1);
    REQUIRE(shape2 != shape3);
    REQUIRE(shape2 == shape4);
    REQUIRE(shape2 != shape5);

    REQUIRE(shape3 != shape1);
    REQUIRE(shape3 != shape2);
    REQUIRE(shape3 != shape4);
    REQUIRE(shape3 != shape5);

    REQUIRE(shape4 != shape1);
    REQUIRE(shape4 == shape2);
    REQUIRE(shape4 != shape3);
    REQUIRE(shape4 != shape5);

    REQUIRE(shape5 != shape1);
    REQUIRE(shape5 != shape2);
    REQUIRE(shape5 != shape3);
    REQUIRE(shape5 != shape4);
}

inline int Shape::operator[] (int index) const
{
    BOUNDS_CHECK("Shape::operator[]", index, 0, this->num_axes())
    return this->axes[index];
}

inline int& Shape::operator[] (int index)
{
    BOUNDS_CHECK("Shape::operator[]", index, 0, this->num_axes())
    return this->axes[index];
}

TEST_CASE("Shape::operator[]")
{
    Shape shape{1, 2, 3};

    REQUIRE(shape[0] == 1);
    REQUIRE(shape[1] == 2);
    REQUIRE(shape[2] == 3);

    REQUIRE_THROWS(shape[3]);

    shape[1] = 10;

    REQUIRE(shape.num_axes() == 3);

    REQUIRE(shape[0] == 1);
    REQUIRE(shape[1] == 10);
    REQUIRE(shape[2] == 3);

    REQUIRE_THROWS(shape[3]);
}

// ----------------------------------------------------------------------------
// Functions

inline int Shape::num_axes() const noexcept
{
    return this->axes.size();
}

TEST_CASE("Shape::num_axes()")
{
    REQUIRE(Shape().num_axes() == 0);
    REQUIRE(Shape({1}).num_axes() == 1);
    REQUIRE(Shape(std::vector<int>{1, 1, 1, 1, 1, 1}).num_axes() == 6);
}

inline int Shape::total(int from_axis, int to_axis) const
{
    if (this->num_axes() == 0)
        return 0;

    if (to_axis == -1)
        to_axis = this->num_axes();

    BOUNDS_CHECK("Shape::total()", from_axis, 0, this->num_axes())
    BOUNDS_CHECK("Shape::total()", to_axis,   0, this->num_axes() + 1)

    TNT_ASSERT(from_axis <= to_axis,
               InvalidParameterException("Shape::total()", __FILE__, __LINE__, "The from_axis must be less than or equal to to_axis. from_axis: " + std::to_string(from_axis) + " to_axis: " + std::to_string(to_axis)))

    int count = 1;
    for (int i = from_axis; i < to_axis; ++i)
        count *= this->axes[i];

    return count;
}

TEST_CASE("Shape::total()")
{
    Shape shape{1, 2, 3, 4, 5, 6};

    REQUIRE(shape.total() == 720);
    REQUIRE(shape.total(1) == 720);
    REQUIRE(shape.total(2) == 360);
    REQUIRE(shape.total(3) == 120);
    REQUIRE(shape.total(4) == 30);
    REQUIRE(shape.total(5) == 6);

    REQUIRE_THROWS(shape.total(6));

    REQUIRE(shape.total(0, 1) == 1);
    REQUIRE(shape.total(1, 3) == 6);
    REQUIRE(shape.total(2, 5) == 60);
    REQUIRE(shape.total(2, 6) == 360);

    REQUIRE_THROWS(shape.total(2, 7));
}

// ----------------------------------------------------------------------------
// Shape Stream Operator

TNT_EXPORT inline std::ostream& operator<<(std::ostream& stream, const Shape& shape)
{
    if (shape.num_axes() == 0)
        return stream << "Shape: {}";

    std::string output = "Shape: {";
    for (int axis : shape.axes)
        output += std::to_string(axis) + "x";
    return stream << output.substr(0, output.size() - 1) + "}";
}

} // namespace tnt

#endif // TNT_SHAPE_IMPL_HPP
