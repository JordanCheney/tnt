#ifndef TNT_TENSOR_HPP
#define TNT_TENSOR_HPP

#include <tnt/core/export.hpp>

#include <tnt/core/aligned_ptr.hpp>
#include <tnt/core/shape.hpp>
#include <tnt/core/stride.hpp>
#include <tnt/core/range.hpp>
#include <tnt/core/tensor_view.hpp>

#include <vector>
#include <random>
#include <type_traits>

namespace tnt
{

/// tnt::Tensor
/// An N-Dimensional tensor class
///
/// \requires Type `Data` shall be arithmetic
template <typename Data>
class TNT_EXPORT Tensor
{
    static_assert(std::is_arithmetic<Data>::value, "Type `Data` must be arithmetic");

public:
    using DataType          = Data;
    using SelfType          = Tensor<DataType>;
    using ViewType          = TensorView<DataType>;
    using MaskType          = Tensor<uint8_t>;
    using IteratorType      = DataType*;
    using ConstIteratorType = const DataType*;
    using PtrType           = AlignedPtr<DataType>;

// ----------------------------------------------------------------------------
// Constructors

    Tensor() noexcept;
    Tensor(const Shape& shape);
    Tensor(const Shape& shape, const PtrType& data);
    Tensor(const Shape& shape, const DataType& value);

    Tensor(const SelfType &other);
    SelfType& operator=(const SelfType& other);

    Tensor(SelfType&& other) noexcept;
    SelfType& operator=(SelfType&& other) noexcept;

// ----------------------------------------------------------------------------
// Iterators

    IteratorType begin() const noexcept;
    IteratorType end() const noexcept;

    ConstIteratorType cbegin() const noexcept;
    ConstIteratorType cend() const noexcept;

// ----------------------------------------------------------------------------
// Operators

    // Relationship
    bool operator== (const SelfType& other) const noexcept;
    bool operator!= (const SelfType& other) const noexcept;

    // Promotion
    operator DataType();
    operator TensorView<DataType>();

    // Set Value
    SelfType& operator= (const DataType& scalar) noexcept;

    /// Extract a non-contiguous view of a tensor.
    ///
    /// \requires Type `IndexType` is an integer or [tnt::Range]() object
    /// \notes If the number of provided indices is less than the dimensionality
    /// of the tensor, the missing indices shall be treated as full ranges over
    /// the appropriate dimensions.
    template <typename ... IndexType>
    ViewType operator()(IndexType... indices) const;

    // Masks
    template <typename T> MaskType operator== (const T& scalar) const;
    template <typename T> MaskType operator!= (const T& scalar) const;
    template <typename T> MaskType operator<  (const T& scalar) const;
    template <typename T> MaskType operator<= (const T& scalar) const;
    template <typename T> MaskType operator>  (const T& scalar) const;
    template <typename T> MaskType operator>= (const T& scalar) const;

    /// \brief Per element bitwise operations
    ///
    /// \requires Type `DataType` is an integer type
    SelfType operator~  () const;

    template <typename T> SelfType  operator&   (const T& scalar) const;
    template <typename T> SelfType& operator&=  (const T& scalar) noexcept;
    template <typename T> SelfType  operator|   (const T& scalar) const;
    template <typename T> SelfType& operator|=  (const T& scalar) noexcept;
    template <typename T> SelfType  operator^   (const T& scalar) const;
    template <typename T> SelfType& operator^=  (const T& scalar) noexcept;

    template <typename T> SelfType  operator&   (const Tensor<T>& other) const;
    template <typename T> SelfType& operator&=  (const Tensor<T>& other) noexcept;
    template <typename T> SelfType  operator|   (const Tensor<T>& other) const;
    template <typename T> SelfType& operator|=  (const Tensor<T>& other) noexcept;
    template <typename T> SelfType  operator^   (const Tensor<T>& other) const;
    template <typename T> SelfType& operator^=  (const Tensor<T>& other) noexcept;

    // Math
    template <typename T> SelfType  operator+  (const T& scalar) const;
    template <typename T> SelfType& operator+= (const T& scalar) noexcept;
    template <typename T> SelfType  operator-  (const T& scalar) const;
    template <typename T> SelfType& operator-= (const T& scalar) noexcept;
    template <typename T> SelfType  operator*  (const T& scalar) const;
    template <typename T> SelfType& operator*= (const T& scalar) noexcept;
    template <typename T> SelfType  operator/  (const T& scalar) const;
    template <typename T> SelfType& operator/= (const T& scalar);

    template <typename T> SelfType  operator+  (const Tensor<T>& other) const;
    template <typename T> SelfType& operator+= (const Tensor<T>& other) noexcept;
    template <typename T> SelfType  operator-  (const Tensor<T>& other) const;
    template <typename T> SelfType& operator-= (const Tensor<T>& other) noexcept;
    template <typename T> SelfType  operator*  (const Tensor<T>& other) const;
    template <typename T> SelfType& operator*= (const Tensor<T>& other) noexcept;
    template <typename T> SelfType  operator/  (const Tensor<T>& other) const;
    template <typename T> SelfType& operator/= (const Tensor<T>& other);

    // Matrix multiplication
    template <typename T> SelfType mul(const Tensor<T>& other) const;

// ----------------------------------------------------------------------------
// Statistics

    DataType max()    const noexcept;
    DataType min()    const noexcept;
    DataType mean()   const noexcept;
    DataType median() const noexcept;
    DataType sum()    const noexcept;

    SelfType max(int axis)    const;
    SelfType min(int axis)    const;
    SelfType mean(int axis)   const;
    SelfType median(int axis) const;
    SelfType sum(int axis)    const;

// ----------------------------------------------------------------------------
// Functions

    template <typename DstType>
    Tensor<DstType> as() const;

    void reshape(const Shape& shape);

    SelfType transpose() const;

// ----------------------------------------------------------------------------
// Members

    Shape   shape;
    PtrType data;
};

// ----------------------------------------------------------------------------
// Useful constructors

template <typename DataType>
Tensor<DataType> zeros(const Shape& shape);

template <typename DataType>
Tensor<DataType> zeros_like(const Tensor<DataType>& tensor);

template <typename DataType>
Tensor<DataType> ones(const Shape& shape);

template <typename DataType>
Tensor<DataType> ones_like(const Tensor<DataType>& tensor);

template <typename DataType>
Tensor<DataType> identity(const Shape& shape);

template <typename DataType>
Tensor<DataType> arrange(const DataType& begin, const DataType& end, const DataType& step = 1);

template <typename DataType>
Tensor<DataType> uniform(const Shape& shape, const DataType& begin = 0, const DataType& end = 1);

// ----------------------------------------------------------------------------

} // namespace tnt

#endif // TNT_TENSOR_HPP
