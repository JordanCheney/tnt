// This file is generated by tools/gen_operators.pl. CHANGES WILL BE OVERWRITTEN
/*  Copyright (C) 2013-2014  Povilas Kanapickas <povilas@radix.lt>

    Distributed under the Boost Software License, Version 1.0.
        (See accompanying file LICENSE_1_0.txt or copy at
            http://www.boost.org/LICENSE_1_0.txt)
*/

#ifndef LIBSIMDPP_SIMDPP_CORE_I_SHIFT_L_OPERATOR_H
#define LIBSIMDPP_SIMDPP_CORE_I_SHIFT_L_OPERATOR_H

#ifndef LIBSIMDPP_SIMD_H
    #error "This file must be included through simd.h"
#endif

#include <tnt/deps/simd/types.h>
#include <tnt/deps/simd/capabilities.h>
#include <tnt/deps/simd/detail/insn/i_shift_l.h>
#include <tnt/deps/simd/detail/insn/i_shift_l_v.h>
#include <tnt/deps/simd/detail/not_implemented.h>

namespace simdpp {
namespace SIMDPP_ARCH_NAMESPACE {

// -----------------------------------------------------------------------------
// shift by scalar

/** Shifts 8-bit values left by @a count bits while shifting in zeros.

    @code
    r0 = a0 << count
    ...
    rN = aN << count
    @endcode
*/
template<unsigned N, class E> SIMDPP_INL
int8<N,expr_empty> operator<<(const int8<N,E>& a, unsigned count)
{
    uint8<N> qa = a.eval();
    return detail::insn::i_shift_l(qa, count);
}

template<unsigned N, class E> SIMDPP_INL
uint8<N,expr_empty> operator<<(const uint8<N,E>& a, unsigned count)
{
    return detail::insn::i_shift_l(a.eval(), count);
}

/** Shifts 16-bit values left by @a count bits while shifting in zeros.

    @code
    r0 = a0 << count
    ...
    rN = aN << count
    @endcode
*/
template<unsigned N, class E> SIMDPP_INL
int16<N,expr_empty> operator<<(const int16<N,E>& a, unsigned count)
{
    uint16<N> qa = a.eval();
    return detail::insn::i_shift_l(qa, count);
}

template<unsigned N, class E> SIMDPP_INL
uint16<N,expr_empty> operator<<(const uint16<N,E>& a, unsigned count)
{
    return detail::insn::i_shift_l(a.eval(), count);
}

/** Shifts 32-bit values left by @a count bits while shifting in zeros.

    @code
    r0 = a0 << count
    ...
    rN = aN << count
    @endcode
*/
template<unsigned N, class E> SIMDPP_INL
int32<N,expr_empty> operator<<(const int32<N,E>& a, unsigned count)
{
    uint32<N> qa = a.eval();
    return detail::insn::i_shift_l(qa, count);
}

template<unsigned N, class E> SIMDPP_INL
uint32<N,expr_empty> operator<<(const uint32<N,E>& a, unsigned count)
{
    return detail::insn::i_shift_l(a.eval(), count);
}

/** Shifts 64-bit values left by @a count bits while shifting in zeros.

    @code
    r0 = a0 << count
    ...
    rN = aN << count
    @endcode
*/
template<unsigned N, class E> SIMDPP_INL
int64<N,expr_empty> operator<<(const int64<N,E>& a, unsigned count)
{
    uint64<N> qa = a.eval();
    return detail::insn::i_shift_l(qa, count);
}

template<unsigned N, class E> SIMDPP_INL
uint64<N,expr_empty> operator<<(const uint64<N,E>& a, unsigned count)
{
    return detail::insn::i_shift_l(a.eval(), count);
}

// -----------------------------------------------------------------------------
// shift by vector

/** Shifts 8-bit values left by the number of bits in corresponding element
    in the given count vector. Zero bits are shifted in.

    @code
    r0 = a0 << count0
    ...
    rN = aN << countN
    @endcode
*/
template<unsigned N, class E> SIMDPP_INL
int8<N,expr_empty> operator<<(const int8<N,E>& a, const uint8<N,E>& count)
{
#if SIMDPP_HAS_INT8_SHIFT_R_BY_VECTOR
    uint8<N> qa = a.eval();
    return detail::insn::i_shift_l_v(qa, count.eval());
#else
    return SIMDPP_NOT_IMPLEMENTED_TEMPLATE2(E, a, count);
#endif
}

template<unsigned N, class E> SIMDPP_INL
uint8<N,expr_empty> operator<<(const uint8<N,E>& a, const uint8<N,E>& count)
{
#if SIMDPP_HAS_UINT8_SHIFT_R_BY_VECTOR
    return detail::insn::i_shift_l_v(a.eval(), count.eval());
#else
    return SIMDPP_NOT_IMPLEMENTED_TEMPLATE2(E, a, count);
#endif
}

/** Shifts 16-bit values left by the number of bits in corresponding element
    in the given count vector. Zero bits are shifted in.

    @code
    r0 = a0 << count0
    ...
    rN = aN << countN
    @endcode
*/
template<unsigned N, class E> SIMDPP_INL
int16<N,expr_empty> operator<<(const int16<N,E>& a, const uint16<N,E>& count)
{
#if SIMDPP_HAS_INT16_SHIFT_R_BY_VECTOR
    uint16<N> qa = a.eval();
    return detail::insn::i_shift_l_v(qa, count.eval());
#else
    return SIMDPP_NOT_IMPLEMENTED_TEMPLATE2(E, a, count);
#endif
}

template<unsigned N, class E> SIMDPP_INL
uint16<N,expr_empty> operator<<(const uint16<N,E>& a, const uint16<N,E>& count)
{
#if SIMDPP_HAS_UINT16_SHIFT_R_BY_VECTOR
    return detail::insn::i_shift_l_v(a.eval(), count.eval());
#else
    return SIMDPP_NOT_IMPLEMENTED_TEMPLATE2(E, a, count);
#endif
}

/** Shifts 32-bit values left by the number of bits in corresponding element
    in the given count vector. Zero bits are shifted in.

    @code
    r0 = a0 << count0
    ...
    rN = aN << countN
    @endcode
*/
template<unsigned N, class E> SIMDPP_INL
int32<N,expr_empty> operator<<(const int32<N,E>& a, const uint32<N,E>& count)
{
#if SIMDPP_HAS_INT32_SHIFT_R_BY_VECTOR
    uint32<N> qa = a.eval();
    return detail::insn::i_shift_l_v(qa, count.eval());
#else
    return SIMDPP_NOT_IMPLEMENTED_TEMPLATE2(E, a, count);
#endif
}

template<unsigned N, class E> SIMDPP_INL
uint32<N,expr_empty> operator<<(const uint32<N,E>& a, const uint32<N,E>& count)
{
#if SIMDPP_HAS_UINT32_SHIFT_R_BY_VECTOR
    return detail::insn::i_shift_l_v(a.eval(), count.eval());
#else
    return SIMDPP_NOT_IMPLEMENTED_TEMPLATE2(E, a, count);
#endif
}

// -----------------------------------------------------------------------------
// shift by compile-time constant

} // namespace SIMDPP_ARCH_NAMESPACE
} // namespace simdpp

#endif
