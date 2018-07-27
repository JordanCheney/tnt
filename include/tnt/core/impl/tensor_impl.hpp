#ifndef TNT_TENSOR_IMPL_HPP
#define TNT_TENSOR_IMPL_HPP

#include <tnt/core/tensor.hpp>

#include <tnt/math/bitwise_ops.hpp>
#include <tnt/math/compare_ops.hpp>
#include <tnt/math/arithmetic_ops.hpp>

#include <tnt/linear/matrix_multiply.hpp>

#include <tnt/utils/testing.hpp>
#include <tnt/utils/simd.hpp>

#include <iostream>
#include <sstream>
#include <type_traits>
#include <random>

namespace tnt
{

// ----------------------------------------------------------------------------
// Constructors

template <typename DataType>
inline Tensor<DataType>::Tensor() noexcept {}

TEST_CASE_TEMPLATE("Tensor()", T, test_data_types)
{
    Tensor<T> t1;
    REQUIRE(t1.data == AlignedPtr<T>());
    REQUIRE(t1.shape == Shape());
}

template <typename DataType>
inline Tensor<DataType>::Tensor(const Shape& shape)
{
    this->shape  = shape;
    this->data   = AlignedPtr<DataType>(shape.total());
}

TEST_CASE_TEMPLATE("Tensor(Shape&)", T, test_data_types)
{
    /* generic Shape constructor test */
    Shape s1;
    Tensor<T> t1(s1);

    REQUIRE(t1.shape == s1);
    REQUIRE(t1.data.size == s1.total());

    /* Shape given initializer_list */
    Shape s2{0, 1, 2};
    Tensor<T> t2(s2);

    REQUIRE(t2.shape == s2);
    REQUIRE(t2.data.size == s2.total());

    /* Shape given vector */
    Shape s3(std::vector<int>{1, 2, 3, 4});
    Tensor<T> t3(s3);

    REQUIRE(t3.shape == s3);
    REQUIRE(t3.data.size == s3.total());

}

// Shape + Data
template <typename DataType>
inline Tensor<DataType>::Tensor(const Shape& shape, const PtrType& data)
{
    this->shape  = shape;
    this->data   = data;
}

TEST_CASE_TEMPLATE("Tensor(Shape&, PtrType&)", T, test_data_types)
{
    /* Generic Shape, only testing Aligned_ptr
     *
     * Ptr with no parameters
     */

    Shape s1;
    AlignedPtr<T> ptr1;
    Tensor<T> t1(s1, ptr1);
    REQUIRE(t1.shape == s1);
    REQUIRE(t1.data == ptr1);

    /* ptr with size parameter only */
    AlignedPtr<T> ptr2(10);
    Tensor<T> t2(s1, ptr2);
    REQUIRE(t2.data.size == 10);

    /* ptr with size and value parameters */
    T data[5] = {1, 2, 3, 4, 5};
    AlignedPtr<T> ptr3(data, 5);
    Tensor<T> t3(s1, ptr3);
    REQUIRE(t3.data.size == 5);
    REQUIRE(t3.data[2] == 3);
    REQUIRE(t3.data[4] == 5);

}

// Shape + Value
template <typename DataType>
inline Tensor<DataType>::Tensor(const Shape& shape, const DataType& value)
{
    this->shape  = shape;
    this->data   = AlignedPtr<DataType>(shape.total());

    this->operator=(value); // Set to value
}

TEST_CASE_TEMPLATE("Tensor(const Shape&, const DataType&)", T, test_data_types)
{
    Shape s1{2, 2};
    Tensor<T> t1(s1, (T) 4);
    AlignedPtr<T> test = t1.data;

    for (int i = 0; i < 4; i++) {
        REQUIRE(test[i] == 4);
    }
}

template <typename DataType>
inline Tensor<DataType>::Tensor(const SelfType& other)
{
    this->shape  = other.shape;
    this->data   = other.data;
}

TEST_CASE_TEMPLATE("Tensor(const SelfType&)", T, test_data_types)
{
    Shape s1{2, 2};
    T data[4] = {1, 2, 3, 4};
    AlignedPtr<T> ptr(data, 4);
    const Tensor<T> t1(s1, ptr);
    Tensor<T> t2(t1);

    REQUIRE(t2.shape == t1.shape);
    REQUIRE(t2.data == t1.data);
}

template <typename DataType>
inline Tensor<DataType>& Tensor<DataType>::operator=(const SelfType& other)
{
    this->shape  = other.shape;
    this->data   = other.data;

    return *this;
}

TEST_CASE_TEMPLATE("Tensor::operator=(const SelfType&)", T, test_data_types)
{
    Shape s1{2, 2};
    T data[4] = {1, 2, 3, 4};
    AlignedPtr<T> ptr(data, 4);
    Tensor<T> t1(s1, ptr);
    Tensor<T> t2 = t1;

    REQUIRE(t2.shape == t1.shape);
    REQUIRE(t2.data == t1.data);
}

template <typename DataType>
inline Tensor<DataType>::Tensor(SelfType&& other) noexcept
{
    this->shape  = other.shape;
    this->data   = other.data;

    // Invalidate the moved object
    other.shape  = Shape();
    other.data   = PtrType();
}

TEST_CASE_TEMPLATE("Tensor(const SelfType&&)", T, test_data_types)
{
    Shape s1{2, 2};
    T data[4] = {1, 2, 3, 4};
    AlignedPtr<T> ptr(data, 4);
    Tensor<T> t1(s1, ptr);
    Tensor<T> t2(std::move(t1));

    REQUIRE(t2.shape == s1);
    REQUIRE(t2.data == ptr);
    REQUIRE(t1.shape == Shape());
    REQUIRE(t1.data == AlignedPtr<T>());

}

template <typename DataType>
inline Tensor<DataType>& Tensor<DataType>::operator=(SelfType&& other) noexcept
{
    this->shape  = other.shape;
    this->data   = other.data;

    // Invalidate the moved object
    other.shape  = Shape();
    other.data   = PtrType();

    return *this;
}

TEST_CASE_TEMPLATE("Tensor::operator=(const SelfType&&)", T, test_data_types)
{
    Shape s1{2, 2};
    T data[4] = {1, 2, 3, 4};
    AlignedPtr<T> ptr(data, 4);
    Tensor<T> t1(s1, ptr);
    Tensor<T> t2 = std::move(t1);

    REQUIRE(t2.shape == s1);
    REQUIRE(t2.data == ptr);
    REQUIRE(t1.shape == Shape());
    REQUIRE(t1.data == AlignedPtr<T>());
}

// ----------------------------------------------------------------------------
// Iterators

template <typename DataType>
inline typename Tensor<DataType>::IteratorType Tensor<DataType>::begin() const noexcept
{
    return this->data.data;
}

TEST_CASE_TEMPLATE("Tensor::begin()", T, test_data_types)
{
    /* int */
    Shape s1{2, 2};
    T data[4] = {(T) 1, (T) 2, (T) 3, (T) 4};
    AlignedPtr<T> ptr(data, 4);
    Tensor<T> t1(s1, ptr);

    REQUIRE(*t1.begin() == 1);

    data[0] = (T) 5;
    AlignedPtr<T> ptr1(data, 4);
    Tensor<T> t2(s1, ptr1);
    REQUIRE(*t2.begin() == 5);

}

template <typename DataType>
inline typename Tensor<DataType>::IteratorType Tensor<DataType>::end() const noexcept
{
    return this->data.is_null() ? nullptr : this->data.data + this->shape.total();
}

TEST_CASE_TEMPLATE("Tensor::end()", T, test_data_types)
{
    Shape s1{2, 2};
    T data[4] = {(T) 1, (T) 2, (T) 3, (T) 4};
    AlignedPtr<T> ptr(data, 4);
    Tensor<T> t1(s1, ptr);
    T* it = t1.begin();

    int cnt = 1;
    while (*it != *t1.end()) {
        REQUIRE(*it == (T) cnt);
        it++;
        cnt++;
    }
}

template <typename DataType>
inline typename Tensor<DataType>::ConstIteratorType Tensor<DataType>::cbegin() const noexcept
{
    return this->data.data;
}

TEST_CASE_TEMPLATE("Tensor::cbegin()", T, test_data_types)
{
    /* int */
    Shape s1{2, 2};
    T data[4] = {(T) 1, (T) 2, (T) 3, (T) 4};
    AlignedPtr<T> ptr(data, 4);
    const Tensor<T> t1(s1, ptr);

    REQUIRE(*t1.begin() == 1);

    data[0] = (T) 5;
    AlignedPtr<T> ptr1(data, 4);
    const Tensor<T> t2(s1, ptr1);
    REQUIRE(*t2.begin() == 5);

}

template <typename DataType>
inline typename Tensor<DataType>::ConstIteratorType Tensor<DataType>::cend() const noexcept
{
    return this->data.is_null() ? nullptr : this->data.data + this->shape.total();
}

TEST_CASE_TEMPLATE("Tensor::cend()", T, test_data_types)
{
    Shape s1{2, 2};
    T data[4] = {(T) 1, (T) 2, (T) 3, (T) 4};
    AlignedPtr<T> ptr(data, 4);
    const Tensor<T> t1(s1, ptr);
    T* it = t1.begin();

    int cnt = 1;
    while (*it != *t1.end()) {
                REQUIRE(*it == (T) cnt);
        it++;
        cnt++;
    }
}

// ----------------------------------------------------------------------------
// Operators

template <typename DataType>
inline bool Tensor<DataType>::operator== (const SelfType& other) const noexcept
{
    return this->shape == other.shape
            && this->data == other.data;
}

TEST_CASE_TEMPLATE("Tensor::operator== (const SelfType&", T, test_data_types)
{
    Shape s1{2, 2};
    T data[3] = {1, 2, 3};
    AlignedPtr<T> ptr(data, 3);
    Tensor<T> t1(s1, ptr);
    Tensor<T> t2(s1, ptr);
    REQUIRE(t1 == t2);

}

template <typename DataType>
inline bool Tensor<DataType>::operator!= (const SelfType& other) const noexcept
{
    return this->shape != other.shape
            || this->data != other.data;
}

TEST_CASE_TEMPLATE("Tensor::operator== (const SelfType&", T, test_data_types)
{
    Shape s1{2, 2};
    T data[3] = {1, 2, 3};
    T data1[3] = {3, 2, 1};
    AlignedPtr<T> ptr(data, 3);
    AlignedPtr<T> ptr1(data1, 3);
    Tensor<T> t1(s1, ptr);
    Tensor<T> t2(s1, ptr1);
    REQUIRE(t1 != t2);
}

template <typename DataType>
inline Tensor<DataType>::operator DataType()
{
    TNT_ASSERT(!this->data.is_null(),
               OutOfBoundsAccessException("Tensor::operator DataType()",
                                          __FILE__,
                                          __LINE__,
                                          "Attempted type promotion on an empty tensor."))

    return this->data[0];
}

TEST_CASE_TEMPLATE("Tensor::operator DataType()", T, test_data_types)
{
    Shape s1{2, 2};
    T data[3] = {1, 2, 3};
    AlignedPtr<T> ptr(data, 3);
    Tensor<T> t1(s1, ptr);
}

template <typename DataType>
inline Tensor<DataType>::operator TensorView<DataType>()
{
    return ViewType(this->shape, Stride{this->shape}, 0, this->data.data);
}

template <typename DataType>
inline Tensor<DataType>& Tensor<DataType>::operator= (const DataType& scalar) noexcept
{
    for (IteratorType it = this->begin(); it != this->end(); ++it)
        *it = scalar;

    return *this;
}

TEST_CASE_TEMPLATE("Tensor::operator=()", T, test_data_types)
{
    auto test_shape = [](const Shape& shape) {
        Tensor<T> tensor(shape, 0);

        tensor = 1;
        REQUIRE((tensor == Tensor<T>(shape, 1)));

        tensor = 10;
        REQUIRE((tensor == Tensor<T>(shape, 10)));

        tensor = 100;
        REQUIRE((tensor == Tensor<T>(shape, 100)));
    };

    test_shape(Shape{3, 1, 3});
    test_shape(Shape{2, 1, 2, 1, 2});
    test_shape(Shape{4, 4, 4, 5});
}

template <typename DataType>
template <typename ... IndexType>
inline TensorView<DataType> Tensor<DataType>::operator() (IndexType... indices) const
{
    std::vector<Range> ranges = make_range_list(indices...);

    TNT_ASSERT(ranges.size() <= (size_t) this->shape.num_axes(),
               InvalidParameterException("Tensor::operator()",
                                         __FILE__,
                                         __LINE__,
                                         "The slice operation cannot accept more ranges then it has dimensions."))

    int offset = 0;

    Shape new_shape;
    Stride stride(this->shape);
    for (size_t i = 0; i < ranges.size(); ++i) {
        int start = ranges[i].begin, end = ranges[i].end;
        if (start < 0) start = this->shape[i] + start + 1;
        if (end   < 0) end   = this->shape[i] + end + 1;

        BOUNDS_CHECK("Tensor::operator()()", start, 0, this->shape[i] + 1)
        BOUNDS_CHECK("Tensor::operator()()", end,   0, this->shape[i] + 1)

        offset += start * stride[i];
        new_shape.axes.push_back(end - start);
    }

    for (size_t i = ranges.size(); i < (size_t) this->shape.num_axes(); ++i)
        new_shape.axes.push_back(this->shape[i]);

    return ViewType(new_shape, stride, offset, this->data.data);
}

TEST_CASE_TEMPLATE("Tensor::operator()()", T, test_data_types)
{
    T data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    Tensor<T> tensor(Shape{3, 3}, AlignedPtr<T>(data, 9));

    { // First row
        TensorView<T> view = tensor(0, Range{0, -1});
        REQUIRE((view.shape == Shape{1, 3}));
        REQUIRE(T(view(0, 0)) == 1);
        REQUIRE(T(view(0, 1)) == 2);
        REQUIRE(T(view(0, 2)) == 3);
    }

    { // Second column
        TensorView<T> view = tensor(Range(), 1);
        REQUIRE((view.shape == Shape{3, 1}));
        REQUIRE(T(view(0, 0)) == 2);
        REQUIRE(T(view(1, 0)) == 5);
        REQUIRE(T(view(2, 0)) == 8);
    }

    { // Bottom corner
        TensorView<T> view = tensor(Range(1, -1), Range(1, -1));
        REQUIRE((view.shape == Shape{2, 2}));
        REQUIRE(T(view(0, 0)) == 5);
        REQUIRE(T(view(0, 1)) == 6);
        REQUIRE(T(view(1, 0)) == 8);
        REQUIRE(T(view(1, 1)) == 9);
    }
}

// Bitwise not
template <typename DataType>
inline Tensor<DataType> Tensor<DataType>::operator~() const
{
    Tensor<DataType> temp = *this;
    bitwise_not(temp);

    return temp;
}

// Bitwise and
template <typename DataType>
template <typename OtherType>
inline Tensor<DataType> Tensor<DataType>::operator& (const OtherType& scalar) const
{
    Tensor<DataType> temp = *this;
    bitwise_and(temp, scalar);

    return temp;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType>& Tensor<DataType>::operator&= (const OtherType& scalar) noexcept
{
    bitwise_and(*this, scalar);
    return *this;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType> Tensor<DataType>::operator& (const Tensor<OtherType>& other) const
{
    Tensor<DataType> temp = *this;
    bitwise_and(temp, other);

    return temp;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType>& Tensor<DataType>::operator&= (const Tensor<OtherType>& other) noexcept
{
    bitwise_and(*this, other);
    return *this;
}

// Bitwise or
template <typename DataType>
template <typename OtherType>
inline Tensor<DataType> Tensor<DataType>::operator| (const OtherType& scalar) const
{
    Tensor<DataType> temp = *this;
    bitwise_or(temp, scalar);

    return temp;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType>& Tensor<DataType>::operator|= (const OtherType& scalar) noexcept
{
    bitwise_or(*this, scalar);
    return *this;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType> Tensor<DataType>::operator| (const Tensor<OtherType>& other) const
{
    Tensor<DataType> temp = *this;
    bitwise_or(temp, other);

    return temp;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType>& Tensor<DataType>::operator|= (const Tensor<OtherType>& other) noexcept
{
    bitwise_or(*this, other);
    return *this;
}

// Bitwise xor
template <typename DataType>
template <typename OtherType>
inline Tensor<DataType> Tensor<DataType>::operator^ (const OtherType& scalar) const
{
    Tensor<DataType> temp = *this;
    bitwise_xor(temp, scalar);

    return temp;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType>& Tensor<DataType>::operator^= (const OtherType& scalar) noexcept
{
    bitwise_xor(*this, scalar);
    return *this;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType> Tensor<DataType>::operator^ (const Tensor<OtherType>& other) const
{
    Tensor<DataType> temp = *this;
    bitwise_xor(temp, other);

    return temp;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType>& Tensor<DataType>::operator^= (const Tensor<OtherType>& other) noexcept
{
    bitwise_xor(*this, other);
    return *this;
}


// Compare equal
template <typename DataType>
template <typename OtherType>
inline Tensor<uint8_t> Tensor<DataType>::operator== (const OtherType& scalar) const
{
    return compare_equal(*this, scalar);
}

// Compare not equal
template <typename DataType>
template <typename OtherType>
inline Tensor<uint8_t> Tensor<DataType>::operator!= (const OtherType& scalar) const
{
    return compare_not_equal(*this, scalar);
}

// Compare less than
template <typename DataType>
template <typename OtherType>
inline Tensor<uint8_t> Tensor<DataType>::operator< (const OtherType& scalar) const
{
    return compare_less_than(*this, scalar);
}

// Compare greater than
template <typename DataType>
template <typename OtherType>
inline Tensor<uint8_t> Tensor<DataType>::operator> (const OtherType& scalar) const
{
    return compare_greater_than(*this, scalar);
}

// Compare less than or equal
template <typename DataType>
template <typename OtherType>
inline Tensor<uint8_t> Tensor<DataType>::operator<= (const OtherType& scalar) const
{
    return compare_less_or_equal(*this, scalar);
}

// Compare greater than or equal
template <typename DataType>
template <typename OtherType>
inline Tensor<uint8_t> Tensor<DataType>::operator>= (const OtherType& scalar) const
{
    return compare_greater_or_equal(*this, scalar);
}

// Add
template <typename DataType>
template <typename OtherType>
inline Tensor<DataType> Tensor<DataType>::operator+ (const OtherType& scalar) const
{
    Tensor<DataType> temp = *this;
    add(temp, scalar);

    return temp;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType>& Tensor<DataType>::operator+= (const OtherType& scalar) noexcept
{
    add(*this, scalar);
    return *this;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType> Tensor<DataType>::operator+ (const Tensor<OtherType>& other) const
{
    Tensor<DataType> temp = *this;
    add(temp, other);

    return temp;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType>& Tensor<DataType>::operator+= (const Tensor<OtherType>& other) noexcept
{
    add(*this, other);
    return *this;
}

// Subtract
template <typename DataType>
template <typename OtherType>
inline Tensor<DataType> Tensor<DataType>::operator- (const OtherType& scalar) const
{
    Tensor<DataType> temp = *this;
    subtract(temp, scalar);

    return temp;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType>& Tensor<DataType>::operator-= (const OtherType& scalar) noexcept
{
    subtract(*this, scalar);
    return *this;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType> Tensor<DataType>::operator- (const Tensor<OtherType>& other) const
{
    Tensor<DataType> temp = *this;
    subtract(temp, other);

    return temp;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType>& Tensor<DataType>::operator-= (const Tensor<OtherType>& other) noexcept
{
    subtract(*this, other);
    return *this;
}

// Multiply
template <typename DataType>
template <typename OtherType>
inline Tensor<DataType> Tensor<DataType>::operator* (const OtherType& scalar) const
{
    Tensor<DataType> temp = *this;
    multiply(temp, scalar);

    return temp;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType>& Tensor<DataType>::operator*= (const OtherType& scalar) noexcept
{
    multiply(*this, scalar);
    return *this;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType> Tensor<DataType>::operator* (const Tensor<OtherType>& other) const
{
    Tensor<DataType> temp = *this;
    multiply(temp, other);

    return temp;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType>& Tensor<DataType>::operator*= (const Tensor<OtherType>& other) noexcept
{
    multiply(*this, other);
    return *this;
}

// Divide
template <typename DataType>
template <typename OtherType>
inline Tensor<DataType> Tensor<DataType>::operator/ (const OtherType& scalar) const
{
    Tensor<DataType> temp = *this;
    divide(temp, scalar);

    return temp;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType>& Tensor<DataType>::operator/= (const OtherType& scalar)
{
    divide(*this, scalar);
    return *this;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType> Tensor<DataType>::operator/ (const Tensor<OtherType>& other) const
{
    Tensor<DataType> temp = *this;
    divide(temp, other);

    return temp;
}

template <typename DataType>
template <typename OtherType>
inline Tensor<DataType>& Tensor<DataType>::operator/= (const Tensor<OtherType>& other)
{
    divide(*this, other);
    return *this;
}

// Matrix multiplication
template <typename DataType>
template <typename OtherType>
inline Tensor<DataType> Tensor<DataType>::mul(const Tensor<OtherType>& other) const
{
    return matrix_multiply(*this, other);
}

// ----------------------------------------------------------------------------
// Functions

template <typename DataType> template <typename DstType>
inline Tensor<DstType> Tensor<DataType>::as() const
{
    Tensor<DstType> output(this->shape);
    for (int i = 0; i < this->shape.total(); ++i)
        output.data[i] = (DstType) this->data[i];

    return output;
}

template <typename DataType>
inline void Tensor<DataType>::reshape(const Shape &shape)
{
    TNT_ASSERT(shape.total() == this->shape.total(),
               InvalidParameterException("Tensor::reshape()",
                                         __FILE__,
                                         __LINE__,
                                         "The reshape operator cannot change the number of elements."));

    this->shape = shape;
}

template <typename DataType>
inline Tensor<DataType> Tensor<DataType>::transpose() const
{
    TNT_ASSERT(this->shape.num_axes() == 2,
               InvalidParameterException("Tensor::transpose()",
                                         __FILE__,
                                         __LINE__,
                                         "Transpose requires a 2D tensor"));

    const int rows = this->shape[0];
    const int cols = this->shape[1];

    SelfType transposed(Shape{cols, rows});

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            transposed.data[c * rows + r] = this->data[r * cols + c];

    return transposed;
}

TEST_CASE_TEMPLATE("Tensor::transpose()", T, test_data_types)
{
    { // Square matrix
        T rmajor[4] = {1, 2, 3, 4};
        T cmajor[4] = {1, 3, 2, 4};

        Tensor<T> tensor(Shape{2, 2}, AlignedPtr<T>(rmajor, 4));
        REQUIRE((tensor.transpose() == Tensor<T>(Shape{2, 2}, AlignedPtr<T>(cmajor, 4))));
    }

    { // Vertical rectangular matrix
        T rmajor[6] = {1, 2, 3, 4, 5, 6};
        T cmajor[6] = {1, 3, 5, 2, 4, 6};

        Tensor<T> tensor(Shape{3, 2}, AlignedPtr<T>(rmajor, 6));
        REQUIRE((tensor.transpose() == Tensor<T>(Shape{2, 3}, AlignedPtr<T>(cmajor, 6))));
    }

    { // Horizontal rectangular matrix
        T rmajor[6] = {1, 2, 3, 4, 5, 6};
        T cmajor[6] = {1, 4, 2, 5, 3, 6};

        Tensor<T> tensor(Shape{2, 3}, AlignedPtr<T>(rmajor, 6));
        REQUIRE((tensor.transpose() == Tensor<T>(Shape{3, 2}, AlignedPtr<T>(cmajor, 6))));
    }
}

// ----------------------------------------------------------------------------
// Tensor Utility Constructors

template <typename DataType>
inline Tensor<DataType> zeros(const Shape& shape)
{
    return Tensor<DataType>(shape, (DataType) 0);
}

template <typename DataType>
inline Tensor<DataType> zeros_like(const Tensor<DataType>& tensor)
{
    return Tensor<DataType>(tensor.shape, (DataType) 0);
}

template <typename DataType>
inline Tensor<DataType> ones(const Shape& shape)
{
    return Tensor<DataType>(shape, (DataType) 1);
}

template <typename DataType>
inline Tensor<DataType> ones_like(const Tensor<DataType>& tensor)
{
    return Tensor<DataType>(tensor.shape, (DataType) 1);
}

template <typename DataType>
inline Tensor<DataType> identity(const Shape& shape)
{
    TNT_ASSERT(shape.num_axes() == 2,
               InvalidParameterException("Tensor::identity()", __FILE__, __LINE__, "Can only make the identity of a 2 dimensional tensor."))

    int smaller_dim = std::min(shape[0], shape[1]);

    Tensor<DataType> tensor = zeros<DataType>(shape);
    for (int i = 0; i < smaller_dim; ++i)
        tensor(i, i) = 1;

    return tensor;
}

TEST_CASE_TEMPLATE("Tensor::identity()", T, test_data_types)
{
    { // 2x2 identity
        T data[4] = {1, 0,
                     0, 1};
        REQUIRE((identity<T>(Shape{2, 2}) == Tensor<T>(Shape{2, 2}, AlignedPtr<T>(data, 4))));
    }

    { // 3x3 identity
        T data[9] = {1, 0, 0,
                     0, 1, 0,
                     0, 0, 1};
        REQUIRE((identity<T>(Shape{3, 3}) == Tensor<T>(Shape{3, 3}, AlignedPtr<T>(data, 9))));
    }

    { // 3x5 identity
        T data[15] = {1, 0, 0, 0, 0,
                      0, 1, 0, 0, 0,
                      0, 0, 1, 0, 0};
        REQUIRE((identity<T>(Shape{3, 5}) == Tensor<T>(Shape{3, 5}, AlignedPtr<T>(data, 15))));
    }

    REQUIRE_THROWS((identity<T>(Shape{1})));
    REQUIRE_THROWS((identity<T>(Shape{2, 2, 2})));
}

template <typename DataType>
inline Tensor<DataType> arrange(const DataType& begin, const DataType& end, const DataType& step)
{
    TNT_ASSERT(end > begin,
               InvalidParameterException("Tensor::arrange()",
                                         __FILE__,
                                         __LINE__,
                                         "End must be greater than start. End: "
                                          + std::to_string(end)
                                          + " Begin: "
                                          + std::to_string(begin)))

    const int len = (int) floor((end - begin) / (float) step) + 1;
    Tensor<DataType> tensor(Shape{len});

    DataType value = begin;
    for (int i = 0; i < len; ++i, value += step)
        tensor.data[i] = value;

    return tensor;
}

template <typename DataType>
inline Tensor<DataType> uniform(const Shape& shape, const DataType& begin, const DataType& end)
{
    Tensor<DataType> tensor(shape);

    std::random_device device;
    std::mt19937 rng(device());
    auto dist = std::is_integral<DataType>::value ?
                    std::uniform_int_distribution<DataType>(begin, end) :
                    std::uniform_real_distribution<DataType>(begin, end);

    for (typename Tensor<DataType>::IteratorType it = tensor.begin(); it != tensor.end(); ++it)
        *it = dist(rng);

    return tensor;
}

// ----------------------------------------------------------------------------
// Tensor Stream Operator

template <typename DataType>
TNT_EXPORT inline std::ostream& operator<<(std::ostream& stream, const Tensor<DataType>& tensor)
{
    std::stringstream pretty_data;

    pretty_data << std::string(tensor.shape.num_axes(), '[');

    for (int i = 0; i < tensor.shape.total(); ++i) {
        pretty_data << (double) tensor.data[i];

        std::string newlines;
        for (int j = 0; j < tensor.shape.num_axes(); ++j) {
            if ((i + 1) % tensor.shape.total(j) == 0) {
                pretty_data << "]";
                newlines += "\n";
            }
        }

        if (i < tensor.shape.total() - 1)
            pretty_data << ", " << newlines << (newlines.empty() ? "" : std::string(11 + tensor.shape.num_axes(), ' ') + "[");
    }

    stream << "Tensor: {"                                                                            << std::endl;
    stream << "    "         << tensor.shape                                                         << std::endl;
    stream << "    Type: "   << std::string(1, TypeInfo<DataType>::type) << TypeInfo<DataType>::bits << std::endl;
    stream << "    Data: { " << pretty_data.str()                                                    << std::endl;
    stream << "    }"                                                                                << std::endl;
    stream << "}";

    return stream;
}

// ----------------------------------------------------------------------------

} // namespace tnt

// ----------------------------------------------------------------------------

#endif // TNT_TENSOR_IMPL_HPP
