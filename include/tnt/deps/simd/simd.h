/*  Copyright (C) 2011-2013  Povilas Kanapickas <povilas@radix.lt>

    Distributed under the Boost Software License, Version 1.0.
        (See accompanying file LICENSE_1_0.txt or copy at
            http://www.boost.org/LICENSE_1_0.txt)
*/

#ifndef LIBSIMDPP_SIMD_H
#define LIBSIMDPP_SIMD_H

/* The following file sets up the preprocessor variables and includes the
   required system headers for the specific architecture
*/
#include <tnt/deps/simd/setup_arch.h>

#include <cstdlib>


#include <tnt/deps/simd/core/align.h>
#include <tnt/deps/simd/core/aligned_allocator.h>
#include <tnt/deps/simd/core/bit_and.h>
#include <tnt/deps/simd/core/bit_andnot.h>
#include <tnt/deps/simd/core/bit_not.h>
#include <tnt/deps/simd/core/bit_or.h>
#include <tnt/deps/simd/core/bit_xor.h>
#include <tnt/deps/simd/core/blend.h>
#include <tnt/deps/simd/core/cache.h>
#include <tnt/deps/simd/core/cast.h>
#include <tnt/deps/simd/core/cmp_eq.h>
#include <tnt/deps/simd/core/cmp_ge.h>
#include <tnt/deps/simd/core/cmp_gt.h>
#include <tnt/deps/simd/core/cmp_le.h>
#include <tnt/deps/simd/core/cmp_lt.h>
#include <tnt/deps/simd/core/cmp_neq.h>
#include <tnt/deps/simd/core/extract.h>
#include <tnt/deps/simd/core/extract_bits.h>
#include <tnt/deps/simd/core/f_abs.h>
#include <tnt/deps/simd/core/f_add.h>
#include <tnt/deps/simd/core/f_ceil.h>
#include <tnt/deps/simd/core/f_div.h>
#include <tnt/deps/simd/core/f_floor.h>
#include <tnt/deps/simd/core/f_fmadd.h>
#include <tnt/deps/simd/core/f_fmsub.h>
#include <tnt/deps/simd/core/f_isnan.h>
#include <tnt/deps/simd/core/f_isnan2.h>
#include <tnt/deps/simd/core/f_max.h>
#include <tnt/deps/simd/core/f_min.h>
#include <tnt/deps/simd/core/f_mul.h>
#include <tnt/deps/simd/core/f_neg.h>
#include <tnt/deps/simd/core/f_reduce_add.h>
#include <tnt/deps/simd/core/f_reduce_max.h>
#include <tnt/deps/simd/core/f_reduce_min.h>
#include <tnt/deps/simd/core/f_reduce_mul.h>
#include <tnt/deps/simd/core/f_rcp_e.h>
#include <tnt/deps/simd/core/f_rcp_rh.h>
#include <tnt/deps/simd/core/f_rsqrt_e.h>
#include <tnt/deps/simd/core/f_rsqrt_rh.h>
#include <tnt/deps/simd/core/f_sign.h>
#include <tnt/deps/simd/core/f_sqrt.h>
#include <tnt/deps/simd/core/f_sub.h>
#include <tnt/deps/simd/core/f_trunc.h>
#include <tnt/deps/simd/core/for_each.h>
#include <tnt/deps/simd/core/i_abs.h>
#include <tnt/deps/simd/core/i_add.h>
#include <tnt/deps/simd/core/i_add_sat.h>
#include <tnt/deps/simd/core/i_avg.h>
#include <tnt/deps/simd/core/i_avg_trunc.h>
#include <tnt/deps/simd/core/i_div_p.h>
#include <tnt/deps/simd/core/i_max.h>
#include <tnt/deps/simd/core/i_min.h>
#include <tnt/deps/simd/core/i_mul.h>
#include <tnt/deps/simd/core/i_mull.h>
#include <tnt/deps/simd/core/i_neg.h>
#include <tnt/deps/simd/core/i_popcnt.h>
#include <tnt/deps/simd/core/i_reduce_add.h>
#include <tnt/deps/simd/core/i_reduce_and.h>
#include <tnt/deps/simd/core/i_reduce_max.h>
#include <tnt/deps/simd/core/i_reduce_min.h>
#include <tnt/deps/simd/core/i_reduce_mul.h>
#include <tnt/deps/simd/core/i_reduce_or.h>
#include <tnt/deps/simd/core/i_reduce_popcnt.h>
#include <tnt/deps/simd/core/i_shift_l.h>
#include <tnt/deps/simd/core/i_shift_r.h>
#include <tnt/deps/simd/core/i_sub.h>
#include <tnt/deps/simd/core/i_sub_sat.h>
#include <tnt/deps/simd/core/insert.h>
#include <tnt/deps/simd/core/load.h>
#include <tnt/deps/simd/core/load_packed2.h>
#include <tnt/deps/simd/core/load_packed3.h>
#include <tnt/deps/simd/core/load_packed4.h>
#include <tnt/deps/simd/core/load_splat.h>
#include <tnt/deps/simd/core/load_u.h>
#include <tnt/deps/simd/core/make_float.h>
#include <tnt/deps/simd/core/make_int.h>
#include <tnt/deps/simd/core/make_uint.h>
#include <tnt/deps/simd/core/make_shuffle_bytes_mask.h>
#include <tnt/deps/simd/core/move_l.h>
#include <tnt/deps/simd/core/move_r.h>
#include <tnt/deps/simd/core/permute2.h>
#include <tnt/deps/simd/core/permute4.h>
#include <tnt/deps/simd/core/permute_bytes16.h>
#include <tnt/deps/simd/core/permute_zbytes16.h>
#include <tnt/deps/simd/core/set_splat.h>
#include <tnt/deps/simd/core/shuffle1.h>
#include <tnt/deps/simd/core/shuffle2.h>
#include <tnt/deps/simd/core/shuffle4x2.h>
#include <tnt/deps/simd/core/shuffle_bytes16.h>
#include <tnt/deps/simd/core/shuffle_zbytes16.h>
#include <tnt/deps/simd/core/splat.h>
#include <tnt/deps/simd/core/splat_n.h>
#include <tnt/deps/simd/core/store_first.h>
#include <tnt/deps/simd/core/store.h>
#include <tnt/deps/simd/core/store_last.h>
#include <tnt/deps/simd/core/store_masked.h>
#include <tnt/deps/simd/core/store_packed2.h>
#include <tnt/deps/simd/core/store_packed3.h>
#include <tnt/deps/simd/core/store_packed4.h>
#include <tnt/deps/simd/core/store_u.h>
#include <tnt/deps/simd/core/stream.h>
#include <tnt/deps/simd/core/test_bits.h>
#include <tnt/deps/simd/core/to_float32.h>
#include <tnt/deps/simd/core/to_float64.h>
#include <tnt/deps/simd/core/to_int16.h>
#include <tnt/deps/simd/core/to_int32.h>
#include <tnt/deps/simd/core/to_int64.h>
#include <tnt/deps/simd/core/to_int8.h>
#include <tnt/deps/simd/core/to_mask.h>
#include <tnt/deps/simd/core/transpose.h>
#include <tnt/deps/simd/core/unzip_hi.h>
#include <tnt/deps/simd/core/unzip_lo.h>
#include <tnt/deps/simd/core/zip_hi.h>
#include <tnt/deps/simd/core/zip_lo.h>
#include <tnt/deps/simd/detail/cast.h>
#include <tnt/deps/simd/detail/cast.inl>
#include <tnt/deps/simd/detail/insn/conv_to_mask.inl>

#include <tnt/deps/simd/detail/altivec/load1.h>

#include <tnt/deps/simd/detail/neon/math_int.h>
#include <tnt/deps/simd/detail/neon/memory_store.h>
#include <tnt/deps/simd/detail/neon/shuffle.h>

#include <tnt/deps/simd/detail/null/bitwise.h>
#include <tnt/deps/simd/detail/null/compare.h>
#include <tnt/deps/simd/detail/null/mask.h>
#include <tnt/deps/simd/detail/null/math.h>
#include <tnt/deps/simd/detail/null/memory.h>
#include <tnt/deps/simd/detail/null/set.h>
#include <tnt/deps/simd/detail/null/shuffle.h>
#include <tnt/deps/simd/detail/null/transpose.h>

#include <tnt/deps/simd/detail/extract128.h>

#include <tnt/deps/simd/types.h>
#include <tnt/deps/simd/types/generic.h>
#include <tnt/deps/simd/types/empty_expr.h>
#include <tnt/deps/simd/types/float32.h>
#include <tnt/deps/simd/types/float32x4.h>
#include <tnt/deps/simd/types/float32x8.h>
#include <tnt/deps/simd/types/float64.h>
#include <tnt/deps/simd/types/float64x2.h>
#include <tnt/deps/simd/types/float64x4.h>
#include <tnt/deps/simd/types/fwd.h>
#include <tnt/deps/simd/types/int16.h>
#include <tnt/deps/simd/types/int16x16.h>
#include <tnt/deps/simd/types/int16x8.h>
#include <tnt/deps/simd/types/int32.h>
#include <tnt/deps/simd/types/int32x4.h>
#include <tnt/deps/simd/types/int32x8.h>
#include <tnt/deps/simd/types/int64.h>
#include <tnt/deps/simd/types/int64x2.h>
#include <tnt/deps/simd/types/int64x4.h>
#include <tnt/deps/simd/types/int8.h>
#include <tnt/deps/simd/types/int8x16.h>
#include <tnt/deps/simd/types/int8x32.h>
#include <tnt/deps/simd/types/traits.h>
#include <tnt/deps/simd/expr.inl>

#include <tnt/deps/simd/operators/bit_and.h>
#include <tnt/deps/simd/operators/bit_or.h>
#include <tnt/deps/simd/operators/bit_xor.h>
#include <tnt/deps/simd/operators/bit_not.h>
#include <tnt/deps/simd/operators/cmp_eq.h>
#include <tnt/deps/simd/operators/cmp_neq.h>
#include <tnt/deps/simd/operators/cmp_ge.h>
#include <tnt/deps/simd/operators/cmp_gt.h>
#include <tnt/deps/simd/operators/cmp_le.h>
#include <tnt/deps/simd/operators/cmp_lt.h>
#include <tnt/deps/simd/operators/f_add.h>
#include <tnt/deps/simd/operators/f_div.h>
#include <tnt/deps/simd/operators/f_mul.h>
#include <tnt/deps/simd/operators/f_sub.h>
#include <tnt/deps/simd/operators/i_add.h>
#include <tnt/deps/simd/operators/i_mul.h>
#include <tnt/deps/simd/operators/i_shift_l.h>
#include <tnt/deps/simd/operators/i_shift_r.h>
#include <tnt/deps/simd/operators/i_sub.h>

/** @def SIMDPP_NO_DISPATCHER
    Disables internal dispatching functionality. If the internal dispathcher
    mechanism is not needed, the user can define the @c SIMDPP_NO_DISPATCHER.
*/
#ifndef SIMDPP_NO_DISPATCHER
#include <tnt/deps/simd/dispatch/dispatcher.h>
#include <tnt/deps/simd/dispatch/make_dispatcher.h>
#endif

#include <tnt/deps/simd/capabilities.h>

namespace simdpp {
using namespace SIMDPP_ARCH_NAMESPACE;
namespace detail {
using namespace ::simdpp::SIMDPP_ARCH_NAMESPACE::detail;
} // namespace detail
} // namespace simdpp

#endif
