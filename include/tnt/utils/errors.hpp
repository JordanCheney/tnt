#ifndef TNT_ERRORS_HPP
#define TNT_ERRORS_HPP

#include <tnt/core/export.hpp>

#include <string>
#include <cstring>
#include <stdexcept>

// ----------------------------------------------------------------------------
// Condition Macro

#ifndef DISABLE_CHECKS

#define TNT_ASSERT(condition, error) \
{                                    \
    bool result = (condition);       \
    if (!result)                     \
        throw error;                 \
}

#else

#define TNT_ASSERT(condition, cleanup, error)

#endif

// ----------------------------------------------------------------------------
// Bound Check Macro

#ifndef DISABLE_CHECKS

#define BOUNDS_CHECK(function, index, start, end)                   \
{                                                                   \
    if (index < start || index >= end)                              \
        throw tnt::OutOfBoundsAccessException(                      \
                  function,                                         \
                  __FILE__,                                         \
                  __LINE__,                                         \
                  std::string("Out of bounds access. ")           + \
                  std::string("Index: ")  + std::to_string(index) + \
                  std::string("Range: [") + std::to_string(start) + \
                  std::string(" -> ")     + std::to_string(end)   + \
                  std::string(")"));                                \
}

#else

#define BOUNDS_CHECK(function, index, start, end)

#endif

namespace tnt
{

// ----------------------------------------------------------------------------
// Base Exception Class

class TNT_EXPORT BaseException
{
public:
    BaseException(const std::string& error,
                  const std::string& function,
                  const std::string& file,
                  const int line,
                  const std::string& msg)
{
    std::string error_str = error;
    error_str += "\n\tFunction: " + function;
    error_str += "\n\tLocation: " + file + ":" + std::to_string(line);
    error_str += "\n\tMessage: " + msg;

    error_msg = new char[error_str.size() + 1]; // 1 for null-terminator
    strcpy(error_msg, error_str.c_str());
}

    virtual ~BaseException()
    {
        delete[] error_msg;
    }

    virtual char* what() const
    {
        return error_msg;
    }

protected:
    char* error_msg;
};

// ----------------------------------------------------------------------------
// Invalid Parameter Exception

class TNT_EXPORT InvalidParameterException : public BaseException
{
public:
    InvalidParameterException(const std::string& function,
                              const std::string& file,
                              const int line,
                              const std::string& msg)
        : BaseException("InvalidParameterException", function, file, line, msg)
    {}
};

// ----------------------------------------------------------------------------
// End of Iterator Exception

class TNT_EXPORT EndOfIteratorException : public BaseException
{
public:
    EndOfIteratorException(const std::string& function,
                           const std::string& file,
                           const int line,
                           const std::string& msg)
        : BaseException("EndOfIteratorException", function, file, line, msg)
    {}
};

// ----------------------------------------------------------------------------
// Unit Test Check Failed Exception

class TNT_EXPORT UnitTestCheckFailedException : public BaseException
{
public:
    UnitTestCheckFailedException(const std::string& function,
                                 const std::string& file,
                                 const int line,
                                 const std::string& msg)
        : BaseException("UnitTestCheckFailedException", function, file, line, msg)
    {}
};

// ----------------------------------------------------------------------------
// Out Of Bounds Access Exception

class TNT_EXPORT OutOfBoundsAccessException : public BaseException
{
public:
    OutOfBoundsAccessException(const std::string& function,
                               const std::string& file,
                               const int line,
                               const std::string& msg)
        : BaseException("OutOfBoundsAccessException", function, file, line, msg)
    {}
};

// ----------------------------------------------------------------------------
// Out Of Bounds Access Exception

class TNT_EXPORT FeatureNotSupportedException : public BaseException
{
public:
    FeatureNotSupportedException(const std::string& function,
                                 const std::string& file,
                                 const int line,
                                 const std::string& msg)
        : BaseException("FeatureNotSupportedException", function, file, line, msg)
    {}
};

// ----------------------------------------------------------------------------

} // namespace tnt

// ----------------------------------------------------------------------------

#endif // TNT_ERRORS_HPP

