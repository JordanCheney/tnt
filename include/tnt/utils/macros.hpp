#ifndef TNT_MACROS_HPP
#define TNT_MACROS_HPP

#if defined TNT_LIBRARY
#  if defined _WIN32 || defined __CYGWIN__
#    define TNT_EXPORT __declspec(dllexport)
#  else
#    define TNT_EXPORT __attribute__((visibility("default")))
#  endif
#else
#  if defined _WIN32 || defined __CYGWIN__
#    define TNT_EXPORT __declspec(dllimport)
#  else
#    define TNT_EXPORT
#  endif
#endif

#if __GNUC__
#define TNT_INL __attribute__((__always_inline__)) inline
#elif _MSC_VER
#define SIMDPP_INL __forceinline
#else
#define SIMDPP_INL inline
#endif

#endif // TNT_MACROS_HPP
