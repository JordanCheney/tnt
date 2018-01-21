#ifndef TNT_EXPORT_HPP
#define TNT_EXPORT_HPP

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

#endif // TNT_EXPORT_HPP

