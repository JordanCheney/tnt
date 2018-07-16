#ifndef TNT_ALIGNED_PTR_IMPL_HPP
#define TNT_ALIGNED_PTR_IMPL_HPP

#include <tnt/core/aligned_ptr.hpp>
#include <tnt/utils/testing.hpp>
#include <tnt/utils/simd.hpp>

#include <memory>
#include <ostream>
#include <cstdlib>
#include <unistd.h>

namespace tnt
{

// ----------------------------------------------------------------------------
// Device specific memory management

namespace detail
{

template <typename DataType>
TNT_INL DataType* aligned_malloc(size_t size)
{
    if (size == 0)
        return nullptr;

    size_t aligned_size = AlignSIMDType<DataType>::aligned_buffer_size(size) * sizeof(DataType);

    void* buffer;
    if (posix_memalign(&buffer, 32, aligned_size)) {
        throw std::bad_alloc();
    }

    memset(buffer, 0, aligned_size);

    return static_cast<DataType*>(buffer);
}

} // namespace detail

// ----------------------------------------------------------------------------
// Constructors

template <typename DataType>
inline AlignedPtr<DataType>::AlignedPtr() noexcept
{
    this->data = nullptr;
    this->size = 0;
}

TEST_CASE_TEMPLATE("AlignedPtr()", T, test_data_types)
{
    AlignedPtr<T> ptr1;
    REQUIRE(ptr1.is_null() == true);
    REQUIRE(ptr1.size == 0);
}

template <typename DataType>
inline AlignedPtr<DataType>::AlignedPtr(size_t size) // Size is the number of elements not the number of bytes
{
    this->data = detail::aligned_malloc<DataType>(size);
    this->size = size;
}

TEST_CASE_TEMPLATE("AlignedPtr(size_t)", T, test_data_types)
{
    AlignedPtr<T> ptr1(10);
    REQUIRE(ptr1.is_null() == false);
    REQUIRE(ptr1.size == 10);

    AlignedPtr<T> ptr2(50);
    REQUIRE(ptr2.is_null() == false);
    REQUIRE(ptr2.size == 50);
}

template <typename DataType>
inline AlignedPtr<DataType>::AlignedPtr(DataType* const data, size_t size)
{
    this->data = detail::aligned_malloc<DataType>(size);
    this->size = size;

    memcpy(this->data, data, sizeof(DataType) * size);
}

TEST_CASE_TEMPLATE("AlignedPtr(DataType* const, size_t)", T, test_data_types)
{
    T data[5] = {1, 2, 3, 4, 5};

    AlignedPtr<T> ptr1(data, 5);
    REQUIRE(ptr1.is_null() == false);
    REQUIRE(ptr1.size == 5);
    REQUIRE(ptr1[1] == 2);
    REQUIRE(ptr1[3] == 4);

    REQUIRE_THROWS(ptr1[5]);

    AlignedPtr<T> ptr2(data, 3);
    REQUIRE(ptr2.is_null() == false);
    REQUIRE(ptr2.size == 3);
    REQUIRE(ptr2[1] == 2);

    REQUIRE_THROWS(ptr2[3]);
}

template <typename DataType>
inline AlignedPtr<DataType>::AlignedPtr(const SelfType& other)
{
    this->data = detail::aligned_malloc<DataType>(other.size);
    this->size = other.size;

    memcpy(this->data, other.data, sizeof(DataType) * this->size);
}

TEST_CASE_TEMPLATE("AlignedPtr(const AlignedPtr&)", T, test_data_types)
{
    T data[5] = {1, 2, 3, 4, 5};

    AlignedPtr<T> ptr1(data, 5);
    AlignedPtr<T> ptr2(ptr1);

    REQUIRE(ptr2.is_null() == false);
    REQUIRE(ptr2.size == 5);
    REQUIRE(ptr2[1] == 2);

    ptr2[2] = 10;
    REQUIRE(ptr2[2] == 10);
    REQUIRE(ptr1[2] == 3);
}

template <typename DataType>
inline AlignedPtr<DataType>& AlignedPtr<DataType>::operator =(const SelfType& other)
{
    this->data = detail::aligned_malloc<DataType>(other.size);
    this->size = other.size;

    memcpy(this->data, other.data, sizeof(DataType) * this->size);

    return *this;
}

TEST_CASE_TEMPLATE("AlignedPtr = const AlignedPtr", T, test_data_types)
{
    T data[5] = {1, 2, 3, 4, 5};

    AlignedPtr<T> ptr1(data, 5);
    AlignedPtr<T> ptr2 = ptr1;

    REQUIRE(ptr2.is_null() == false);
    REQUIRE(ptr2.size == 5);
    REQUIRE(ptr2[1] == 2);

    ptr2[2] = 10;
    REQUIRE(ptr2[2] == 10);
    REQUIRE(ptr1[2] == 3);
}

template <typename DataType>
inline AlignedPtr<DataType>::AlignedPtr(SelfType&& other) noexcept
{
    this->data = other.data;
    this->size = other.size;

    other.data = nullptr;
    other.size = 0;
}

TEST_CASE_TEMPLATE("AlignedPtr(AlignedPtr&&)", T, test_data_types)
{
    T data[5] = {1, 2, 3, 4, 5};

    AlignedPtr<T> ptr1(data, 5);
    AlignedPtr<T> ptr2(std::move(ptr1));

    REQUIRE(ptr2.is_null() == false);
    REQUIRE(ptr2.size == 5);
    REQUIRE(ptr2[1] == 2);

    REQUIRE(ptr1.is_null() == true);
    REQUIRE(ptr1.size == 0);
}

template <typename DataType>
inline AlignedPtr<DataType>& AlignedPtr<DataType>::operator =(SelfType&& other) noexcept
{
    this->data = other.data;
    this->size = other.size;

    other.data = nullptr;
    other.size = 0;

    return *this;
}

TEST_CASE_TEMPLATE("AlignedPtr = AlignedPtr&&", T, test_data_types)
{
    T data[5] = {1, 2, 3, 4, 5};

    AlignedPtr<T> ptr1(data, 5);
    AlignedPtr<T> ptr2 = std::move(ptr1);

    REQUIRE(ptr2.is_null() == false);
    REQUIRE(ptr2.size == 5);
    REQUIRE(ptr2[1] == 2);

    REQUIRE(ptr1.is_null() == true);
    REQUIRE(ptr1.size == 0);
}

// ----------------------------------------------------------------------------
// Destructor

template <typename DataType>
inline AlignedPtr<DataType>::~AlignedPtr() noexcept
{
    if (this->data) {
        free(this->data);
        this->data = nullptr;
    }
}

// ----------------------------------------------------------------------------
// Operators

template <typename DataType>
inline bool AlignedPtr<DataType>::operator==(const SelfType& other) const noexcept
{
    return this->size == other.size
            && memcmp(this->data, other.data, sizeof(DataType) * this->size) == 0;
}

template <typename DataType>
inline bool AlignedPtr<DataType>::operator!=(const SelfType& other) const noexcept
{
    return this->size != other.size
            || memcmp(this->data, other.data, sizeof(DataType) * this->size) != 0;
}

TEST_CASE_TEMPLATE("AlignedPtr::operator== / AlignedPtr::operator!=", T, test_data_types)
{
    T data1[5] = {1, 3, 5, 7, 9};
    T data2[5] = {2, 4, 6, 8, 10};

    AlignedPtr<T> ptr1(data1, 5);
    AlignedPtr<T> ptr2(data2, 5);
    AlignedPtr<T> ptr3(data1, 3);

    REQUIRE(ptr1 == ptr1);
    REQUIRE(ptr2 == ptr2);
    REQUIRE(ptr3 == ptr3);

    REQUIRE(ptr1 != ptr2);
    REQUIRE(ptr2 != ptr1);

    REQUIRE(ptr1 != ptr3);
    REQUIRE(ptr3 != ptr1);

    REQUIRE(ptr2 != ptr3);
    REQUIRE(ptr3 != ptr2);
}

template <typename DataType>
TNT_INL DataType AlignedPtr<DataType>::operator[](size_t idx) const
{
    BOUNDS_CHECK("AlignedPtr::operator[]", idx, 0, this->size)
    return this->data[idx];
}

template <typename DataType>
TNT_INL DataType& AlignedPtr<DataType>::operator[](size_t idx)
{
    BOUNDS_CHECK("AlignedPtr::operator[]", idx, 0, this->size)
    return this->data[idx];
}

TEST_CASE_TEMPLATE("AlignedPtr Index Operator", T, test_data_types)
{
    T data[5] = {1, 2, 3, 4, 5};

    AlignedPtr<T> ptr(data, 5);

    REQUIRE(ptr[0] == 1);
    REQUIRE(ptr[1] == 2);
    REQUIRE(ptr[4] == 5);

    REQUIRE_THROWS(ptr[5]);

    ptr[2] = 10;
    ptr[3] = 100;

    REQUIRE(ptr[0] == 1);
    REQUIRE(ptr[2] == 10);
    REQUIRE(ptr[3] == 100);

    REQUIRE_THROWS(ptr[5]);
}

// ----------------------------------------------------------------------------
// Functions

template <typename DataType>
TNT_INL bool AlignedPtr<DataType>::is_null() const noexcept
{
    return this->data == nullptr;
}

TEST_CASE_TEMPLATE("AlignedPtr::is_null()", T, test_data_types)
{
    AlignedPtr<T> ptr1;
    REQUIRE(ptr1.is_null());

    T data[5] = {1, 2, 3, 4, 5};
    ptr1 = AlignedPtr<T>(data, 5);
    REQUIRE(!ptr1.is_null());
}

// ----------------------------------------------------------------------------
// Stream operator

template <typename DataType>
TNT_EXPORT inline std::ostream& operator<<(std::ostream& stream, const AlignedPtr<DataType>& ptr)
{
    if (ptr.is_null())
        return stream << "[]";

    stream << "[" << (double) ptr[0];
    for (size_t i = 1; i < ptr.size; ++i)
        stream << ", " << (double) ptr[i];
    return stream << "]";
}

// ----------------------------------------------------------------------------

} // namespace tnt

// ----------------------------------------------------------------------------

#endif // TNT_ALIGNED_PTR_IMPL_HPP
