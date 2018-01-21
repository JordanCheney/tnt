#ifndef TNT_ALIGNED_PTR_HPP
#define TNT_ALIGNED_PTR_HPP

#include <tnt/utils/errors.hpp>

namespace tnt
{

/// \brief A managed pointer to aligned memory
///
/// An AlignedPtr provides an interface for basic operations on aligned memory
/// like
///
///     1. Allocation
///     2. Copying
///     3. Moving
///     4. Comparison
///     5. Indexing
///     6. Destruction
///
/// \requires Type `Data` is arithmetic
template <typename Data>
class TNT_EXPORT AlignedPtr
{
    static_assert(std::is_arithmetic<Data>::value,
                    "AlignedPtr requires that type `Data` be arithmetic");

public:
    using DataType = Data;
    using SelfType = AlignedPtr<DataType>;

    /// \brief Construct a default pointer
    ///
    /// [Size](tnt::AlignedPtr<Data>::size) is set to 0 and
    /// [Size](tnt::AlignedPtr<Data>::data) is set to `nullptr`.
    AlignedPtr() noexcept;

    /// \brief Construct a new buffer with at least the given number of elements
    ///
    /// [Size](tnt::AlignedPtr<Data>::size) is set to [size](*::size) and
    /// [data](tnt::AlignedPtr<Data>::data) will be allocated with at least
    /// `sizeof(DataType) * size` bytes.
    /// \param size The number of elements to allocate space for
    AlignedPtr(size_t size);

    /// \brief Construct a new buffer with at least size_t elements and copy
    /// the contents of [other](*::other) into the buffer.
    ///
    /// [Size](tnt::AlignedPtr<Data>::size) is set to `size` and [data](tnt::AlignedPtr<Data>::size)
    /// will be allocated with at least `sizeof(DataType) * size` bytes.
    /// Exactly `sizeof(DataType) * size` bytes will be copied from [other](*::other)
    /// into the beginning of the buffer.
    /// \param other A pointer to a buffer, allocated on the CPU, of at least
    /// `sizeof(DataType) * size` bytes
    /// \param size The number of elements to allocate space for
    /// \requires Parameter `other` must be at least `sizeof(DataType) * size`
    /// bytes.
    AlignedPtr(DataType* other, size_t size);

    // Copy
    AlignedPtr(const SelfType& other);
    SelfType& operator=(const SelfType& other);

    // Move
    AlignedPtr(SelfType&& other) noexcept;
    SelfType& operator=(SelfType&& other) noexcept;

    // Destructor
    ~AlignedPtr() noexcept;

    /// \brief Check if 2 AlignedPtrs are equivalent
    ///
    /// DevicePtrs are considered equivalent if all of the following are true:
    ///
    ///     1. They have the same type
    ///     2. They have the same length
    ///     3. The contents of their buffers are equivalent
    /// \returns True if the pointers are equivalent, false otherwise
    bool operator==(const SelfType& other) const noexcept;

    /// \brief Check if 2 AlignedPtrs are not equivalent
    ///
    /// DevicePtrs are considered not equivalent if any of the following are true:
    ///
    ///     1. They have different types
    ///     2. They have different lenghts
    ///     3. The contents of their buffers are not equivalent
    /// \returns True is the pointers are not equivalent, false otherwise
    bool operator!=(const SelfType& other) const noexcept;

    /// \brief Access the pointer buffer at a specific index
    ///
    /// \param index The index at which to access the buffer
    /// \returns A copy of the data at the given index
    /// \notes Bounds checking on the index is performed by default. It can be
    /// disabled by `#define DISABLE_CHECKS` before this function is called.
    DataType  operator[](size_t index) const;

    /// \brief Get a reference into the pointer buffer at a specific index
    ///
    /// \param index The index at which to access the buffer
    /// \returns A reference to the data at the given index
    /// \notes Bounds checking on the index is performed by default. It can be
    /// disabled by `#define DISABLE_CHECKS` before this function is called.
    DataType& operator[](size_t index);

    /// \brief Check if the AlignedPtr's buffer is null
    /// \returns True is the buffer is null, false otherwise
    bool is_null() const noexcept;

    // Members
    DataType* data; //< A buffer
    size_t size; //< The number of items in the buffer
};

// ----------------------------------------------------------------------------

} // namespace tnt

// ----------------------------------------------------------------------------

#endif // TNT_ALIGNED_PTR_HPP
