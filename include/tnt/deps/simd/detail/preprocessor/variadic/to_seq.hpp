# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Edward Diener 2011.                                    *
#  *     (C) Copyright Paul Mensonides 2011.                                  *
#  *     Distributed under the Boost Software License, Version 1.0. (See      *
#  *     accompanying file LICENSE_1_0.txt or copy at                         *
#  *     http://www.boost.org/LICENSE_1_0.txt)                                *
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef SIMDPP_PREPROCESSOR_VARIADIC_TO_SEQ_HPP
# define SIMDPP_PREPROCESSOR_VARIADIC_TO_SEQ_HPP
#
#include <tnt/deps/simd/detail/preprocessor/config/config.hpp>
#include <tnt/deps/simd/detail/preprocessor/tuple/to_seq.hpp>
#
# /* SIMDPP_PP_VARIADIC_TO_SEQ */
#
# if SIMDPP_PP_VARIADICS
#    define SIMDPP_PP_VARIADIC_TO_SEQ(...) SIMDPP_PP_TUPLE_TO_SEQ((__VA_ARGS__))
# endif
#
# endif
