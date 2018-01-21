#ifndef TNT_TENSOR_VIEW_HPP
#define TNT_TENSOR_VIEW_HPP

#include <tnt/core/aligned_ptr.hpp>
#include <tnt/core/shape.hpp>
#include <tnt/core/stride.hpp>
#include <tnt/core/range.hpp>
#include <tnt/core/index.hpp>

namespace tnt
{

/// \brief An iterator over a non-contiguous TensorView
///
/// \requires Type `Data` shall be arithmetic
template <typename Data>
class TNT_EXPORT TensorViewIterator
{
    static_assert(std::is_arithmetic<Data>::value,
                    "TensorViewIterator requires type `Data` is arithmetic");

public:
    using DataType = Data;
    using SelfType = TensorViewIterator<DataType>;

// ----------------------------------------------------------------------------
// Constructors

    TensorViewIterator() noexcept;
    TensorViewIterator(DataType* data, int offset, const Shape& shape, const Stride& stride);

// ----------------------------------------------------------------------------
// Operators

    bool operator== (const SelfType& other) const noexcept;
    bool operator!= (const SelfType& other) const noexcept;

    DataType operator* () const;
    DataType& operator* ();

    void operator++(int);

// ----------------------------------------------------------------------------
// Members

    DataType* data;
    int offset;
    Index index;
    Stride stride;
};

/// \brief A non-contiguous view of a [Tensor](*::Tensor)
///
/// \requires Type `Data` shall be arithmetic
template <typename Data>
class TNT_EXPORT TensorView
{
    static_assert(std::is_arithmetic<Data>::value,
                    "TensorView requires type `Data` is arithmetic");

public:
    using DataType          = Data;
    using SelfType          = TensorView<DataType>;
    using IteratorType      = TensorViewIterator<DataType>;
    using ConstIteratorType = const TensorViewIterator<DataType>;

// ----------------------------------------------------------------------------
// Constructors

    TensorView() noexcept;
    TensorView(const Shape& shape, const Stride& stride, int offset, DataType* data);

    TensorView(const SelfType&);
    SelfType& operator=(const SelfType&);

    TensorView(SelfType&&) noexcept;
    SelfType& operator=(SelfType&&) noexcept;

// ----------------------------------------------------------------------------
// Iterators

    IteratorType begin() const;
    IteratorType end() const;

    ConstIteratorType cbegin() const;
    ConstIteratorType cend() const;

// ----------------------------------------------------------------------------
// Operators

    bool operator== (const SelfType& other) const noexcept;
    bool operator!= (const SelfType& other) const noexcept;

    operator DataType();

    template <typename OtherType>
    SelfType& operator= (const OtherType& other);

    template <typename ... IndexType>
    SelfType operator() (IndexType... indices) const;

// ----------------------------------------------------------------------------
// Functions

    DataType max()    const noexcept;
    DataType min()    const noexcept;
    DataType mean()   const noexcept;
    DataType median() const noexcept;
    DataType sum()    const noexcept;

// ----------------------------------------------------------------------------
// Members

    Shape     shape;
    Stride    stride;
    int       offset;
    DataType* data;
};

} // namespace tnt

#endif // TNT_TENSOR_VIEW_HPP
