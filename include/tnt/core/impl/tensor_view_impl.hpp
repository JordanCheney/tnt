#ifndef TNT_TENSOR_VIEW_IMPL_HPP
#define TNT_TENSOR_VIEW_IMPL_HPP

#include <tnt/core/tensor_view.hpp>
#include <tnt/utils/testing.hpp>

#include <iostream>

namespace tnt
{

// ----------------------------------------------------------------------------
// TensorViewIterator Constructors

template <typename DataType>
inline TensorViewIterator<DataType>::TensorViewIterator() noexcept
{
    this->data   = nullptr;
    this->offset = 0;
}

TEST_CASE_TEMPLATE("TensorViewIterator()", T, test_data_types)
{
    TensorViewIterator<T> it;

    REQUIRE(it.data   == nullptr);
    REQUIRE(it.offset == 0);
    REQUIRE(it.index  == Index());
    REQUIRE(it.stride == Stride());
}

template <typename DataType>
inline TensorViewIterator<DataType>::TensorViewIterator(DataType* data, int offset, const Shape& shape, const Stride& stride)
{
    this->data   = data;
    this->offset = offset;
    this->index  = Index(shape);
    this->stride = stride;
}

TEST_CASE_TEMPLATE("TensorViewIterator(DataType*, int, const Shape&, const Stride&)", T, test_data_types)
{
    T data[6] = {1, 2, 3, 4, 5, 6};

    TensorViewIterator<T> it(data, 0, Shape{2, 3}, Stride{3, 1});

    REQUIRE(*it.data   == 1);
    REQUIRE(it.offset  == 0);
    REQUIRE((it.index  == Index(Shape{2, 3})));
    REQUIRE((it.stride == Stride{3, 1}));
}

// ----------------------------------------------------------------------------
// TensorViewIterator Operators

template <typename DataType>
inline bool TensorViewIterator<DataType>::operator== (const TensorViewIterator<DataType>& other) const noexcept
{
    return this->data   == other.data
            && this->offset == other.offset
            && this->index  == other.index
            && this->stride == other.stride;
}

template <typename DataType>
inline bool TensorViewIterator<DataType>::operator !=(const TensorViewIterator<DataType>& other) const noexcept
{
    return this->data   != other.data
            || this->offset != other.offset
            || this->index  != other.index
            || this->stride != other.stride;
}

template <typename DataType>
inline DataType TensorViewIterator<DataType>::operator* () const
{
    return *(this->data + offset + this->index.distance(this->stride));
}

template <typename DataType>
inline DataType& TensorViewIterator<DataType>::operator* ()
{
    return *(this->data + offset + this->index.distance(this->stride));
}

template <typename DataType>
inline void TensorViewIterator<DataType>::operator++ (int)
{
    this->index++;
}

TEST_CASE_TEMPLATE("TensorViewIterator::operator++()", T, test_data_types)
{
    T data[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    { // 1-D
        Shape shape{8};
        TensorView<T> view(shape, Stride(shape), 0, data);

        int idx = 0;
        for (TensorViewIterator<T> it = view.begin(); it != view.end(); it++, idx++)
            REQUIRE(*it == data[idx]);
    }

    { // 2-D
        Shape shape{2, 4};
        TensorView<T> view(shape, Stride(shape), 0, data);

        int idx = 0;
        for (TensorViewIterator<T> it = view.begin(); it != view.end(); it++, idx++)
            REQUIRE(*it == data[idx]);
    }

    { // 3-D
        Shape shape{2, 2, 2};
        TensorView<T> view(shape, Stride(shape), 0, data);

        int idx = 0;
        for (TensorViewIterator<T> it = view.begin(); it != view.end(); it++, idx++)
            REQUIRE(*it == data[idx]);
    }

    { // 4-D
        Shape shape{2, 1, 2, 2};
        TensorView<T> view(shape, Stride(shape), 0, data);

        int idx = 0;
        for (TensorViewIterator<T> it = view.begin(); it != view.end(); it++, idx++)
            REQUIRE(*it == data[idx]);
    }
}

// ----------------------------------------------------------------------------
// TensorView Constructors

template <typename DataType>
inline TensorView<DataType>::TensorView() noexcept
{
    this->offset = 0;
    this->data = nullptr;
}

template <typename DataType>
inline TensorView<DataType>::TensorView(const Shape& shape, const Stride& stride, int offset, DataType* data)
{
    this->shape  = shape;
    this->stride = stride;
    this->offset = offset;
    this->data   = data;
}

template <typename DataType>
inline TensorView<DataType>::TensorView(const TensorView<DataType>& other)
{
    TNT_ASSERT(this->shape == other.shape,
               "TensorViews must be the same shape to copy data between themselves");

    auto it = this->begin();
    auto other_it = other.begin();

    for ( ; it != this->end(); it++, other_it++)
        *it = *other_it;
}

template <typename DataType>
inline TensorView<DataType>& TensorView<DataType>::operator= (const TensorView<DataType>& other)
{
    TNT_ASSERT(this->shape == other.shape,
               InvalidParameterException("TensorView::operator=(TensorView&&)",
                                         __FILE__,
                                         __LINE__,
                                         "TensorViews must be the same shape to copy data between themselves"));

    auto it = this->begin();
    auto other_it = other.begin();

    for ( ; it != this->end(); it++, other_it++)
        *it = *other_it;

    return *this;
}

template <typename DataType>
inline TensorView<DataType>::TensorView(TensorView<DataType>&& other) noexcept
{
    this->shape = other.shape;
    this->stride = other.stride;
    this->offset = other.offset;
    this->data = other.data;

    other.shape = Shape();
    other.stride = Stride();
    other.offset = 0;
    other.data = nullptr;
}

template <typename DataType>
inline TensorView<DataType>& TensorView<DataType>::operator= (TensorView<DataType>&& other) noexcept
{
    this->shape = other.shape;
    this->stride = other.stride;
    this->offset = other.offset;
    this->data = other.data;

    other.shape = Shape();
    other.stride = Stride();
    other.offset = 0;
    other.data = nullptr;

    return *this;
}

// ----------------------------------------------------------------------------
// TensorView Iterator

template <typename DataType>
inline typename TensorView<DataType>::IteratorType TensorView<DataType>::begin() const
{
    return TensorViewIterator<DataType>(this->data, this->offset, this->shape, this->stride);
}

template <typename DataType>
inline typename TensorView<DataType>::IteratorType TensorView<DataType>::end() const
{
    TensorViewIterator<DataType> it(this->data, this->offset, this->shape, this->stride);
    if (this->shape.num_axes() != 0) {
        it.index.loc[0] = this->shape[0];
        for (int i = 1; i < this->shape.num_axes(); ++i)
            it.index.loc[i] = 0;
    }

    return it;
}

template <typename DataType>
inline typename TensorView<DataType>::ConstIteratorType TensorView<DataType>::cbegin() const
{
    return TensorViewIterator<DataType>(this->data, this->offset, this->shape, this->stride);
}

template <typename DataType>
inline typename TensorView<DataType>::ConstIteratorType TensorView<DataType>::cend() const
{
    TensorViewIterator<DataType> it(this->data, this->offset, this->shape, this->stride);
    if (this->shape.num_axes() != 0) {
        it.index.loc[0] = this->shape[0];
        for (int i = 1; i < this->shape.num_axes(); ++i)
            it.index.loc[i] = 0;
    }

    return it;
}

// ----------------------------------------------------------------------------
// TensorView Operators

template <typename DataType>
inline bool TensorView<DataType>::operator== (const TensorView<DataType>& other) const noexcept
{
    return this->shape  == other.shape
            && this->stride == other.stride
            && this->offset == other.offset
            && this->data   == other.data;
}

template <typename DataType>
inline bool TensorView<DataType>::operator!= (const TensorView<DataType>& other) const noexcept
{
    return this->shape  != other.shape
            || this->stride != other.stride
            || this->offset != other.offset
            || this->data   != other.data;
}

template <typename DataType>
inline TensorView<DataType>::operator DataType()
{
    TNT_ASSERT(this->data != nullptr,
               OutOfBoundsAccessException("TensorView::operator DataType()",
                                          __FILE__,
                                          __LINE__,
                                          "Attempted type promotion on an empty tensor view."))

    return this->data[this->offset];
}

template <typename DataType> template <typename OtherType>
inline TensorView<DataType>& TensorView<DataType>::operator= (const OtherType& _scalar)
{
    static_assert(std::is_arithmetic<OtherType>::value,
                   "TensorView operator=() expects a Scalar, Tensor, or TensorView");

    const DataType scalar = static_cast<DataType>(_scalar);

    for (IteratorType it = this->begin(); it != this->end(); it++)
        *it = scalar;

    return *this;
}

template <typename DataType> template <typename ... IndexType>
inline TensorView<DataType> TensorView<DataType>::operator()(IndexType... indices) const
{
    std::vector<Range> ranges = make_range_list(indices...);

    TNT_ASSERT(ranges.size() <= (size_t) this->shape.num_axes(),
               InvalidParameterException("TensorView::operator()",
                                         __FILE__,
                                         __LINE__,
                                         "The slice operation cannot accept more ranges then it has dimensions."))

    int offset = this->offset;

    std::vector<int> new_shape;
    for (size_t i = 0; i < ranges.size(); ++i) {
        int start = ranges[i].begin, end = ranges[i].end;
        if (start < 0) start = this->shape[i] + start + 1;
        if (end   < 0) end   = this->shape[i] + end + 1;

        BOUNDS_CHECK("TensorView::operator()", start, 0, this->shape[i] + 1)
        BOUNDS_CHECK("TensorView::operator()", end,   0, this->shape[i] + 1)

        offset += start * this->stride[i];
        new_shape.push_back(end - start);
    }

    for (size_t i = ranges.size(); i < (size_t) this->shape.num_axes(); ++i)
        new_shape.push_back(this->shape[i]);

    return SelfType(new_shape, this->stride, offset, this->data);
}

TEST_CASE_TEMPLATE("TensorView::operator()", T, test_data_types)
{
    T data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    TensorView<T> tensor(Shape{3, 3}, Stride{3, 1}, 0, data);

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

    { // Multiple slices
        TensorView<T> view1 = tensor(Range(1, -1), Range(1, -1));
        TensorView<T> view2 = view1(1, Range());

        REQUIRE((view2.shape == Shape{1, 2}));
        REQUIRE(T(view2(0, 0)) == 8);
        REQUIRE(T(view2(0, 1)) == 9);
    }
}

template <typename DataType>
inline DataType TensorView<DataType>::max() const noexcept
{
    DataType max_val = -std::numeric_limits<DataType>::max();
    for (IteratorType it = this->begin(); it != this->end(); ++it)
        if (max_val < *it)
            max_val = *it;

    return max_val;
}

template <typename DataType>
inline DataType TensorView<DataType>::min() const noexcept
{
    DataType min_val = std::numeric_limits<DataType>::max();
    for (IteratorType it = this->begin(); it != this->end(); ++it)
        if (min_val > *it)
            min_val = *it;

    return min_val;
}

template <typename DataType>
inline DataType TensorView<DataType>::mean() const noexcept
{
    return this->sum() / this->shape.total();
}

template <typename DataType>
inline DataType TensorView<DataType>::sum() const noexcept
{
    DataType sum = 0;
    for (IteratorType it = this->begin(); it != this->end(); ++it)
        sum += *it;

    return sum;
}

} // namespace tnt

#endif // TNT_TENSOR_VIEW_IMPL_HPP
