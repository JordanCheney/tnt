/*  Copyright (C) 2016  Povilas Kanapickas <povilas@radix.lt>

    Distributed under the Boost Software License, Version 1.0.
        (See accompanying file LICENSE_1_0.txt or copy at
            http://www.boost.org/LICENSE_1_0.txt)
*/

#ifndef LIBSIMDPP_SIMD_TYPES_GENERIC_INL
#define LIBSIMDPP_SIMD_TYPES_GENERIC_INL

#include <tnt/deps/simd/setup_arch.h>
#include <tnt/deps/simd/expr.h>
#include <tnt/deps/simd/detail/expr/bit_and.h>
#include <tnt/deps/simd/detail/expr/bit_andnot.h>
#include <tnt/deps/simd/detail/expr/bit_not.h>
#include <tnt/deps/simd/detail/expr/bit_or.h>
#include <tnt/deps/simd/detail/expr/blend.h>
#include <tnt/deps/simd/detail/expr/f_abs.h>
#include <tnt/deps/simd/detail/expr/f_add.h>
#include <tnt/deps/simd/detail/expr/f_fmadd.h>
#include <tnt/deps/simd/detail/expr/f_fmsub.h>
#include <tnt/deps/simd/detail/expr/f_mul.h>
#include <tnt/deps/simd/detail/expr/f_neg.h>
#include <tnt/deps/simd/detail/expr/f_sub.h>
#include <tnt/deps/simd/detail/expr/i_abs.h>
#include <tnt/deps/simd/detail/expr/i_add.h>
#include <tnt/deps/simd/detail/expr/i_add_sat.h>
#include <tnt/deps/simd/detail/expr/i_mul.h>
#include <tnt/deps/simd/detail/expr/i_mull.h>
#include <tnt/deps/simd/detail/expr/i_neg.h>
#include <tnt/deps/simd/detail/expr/i_sub.h>
#include <tnt/deps/simd/detail/expr/i_sub_sat.h>
#include <tnt/deps/simd/detail/expr/scalar.h>
#include <tnt/deps/simd/detail/expr/splat_n.h>
#include <tnt/deps/simd/detail/expr/test_bits.h>
#include <tnt/deps/simd/detail/expr/vec.h>

namespace simdpp {
namespace SIMDPP_ARCH_NAMESPACE {
namespace detail {

template<class R, class E> struct expr_eval_wrapper {
static SIMDPP_INL R eval(const E& e) { return expr_eval<R, E>::eval(e); }
};

} // namespace detail
} // namespace SIMDPP_ARCH_NAMESPACE
} // namespace simdpp

#endif
