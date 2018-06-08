#ifndef TNT_LINEAR_MATRIX_MULTIPLY_IMPL_HPP
#define TNT_LINEAR_MATRIX_MULTIPLY_IMPL_HPP

#include <tnt/linear/matrix_multiply.hpp>
#include <tnt/utils/simd.hpp>
#include <tnt/utils/testing.hpp>

namespace tnt
{

namespace detail
{

/// This code is adapted from:
/// https://github.com/pytorch/glow/blob/master/lib/Backends/CPU/libjit/libjit_matmul.cpp
template <typename DataType>
struct OptimizedMatrixMultiply<DataType>
{
    using VecType = typename SIMDType<DataType>::VecType;

    static Tensor<DataType> eval(const Tensor<DataType>& left, const Tensor<DataType>& right)
    {
        constexpr int BLOCK_ROWS = 256;
        constexpr int BLOCK_COLS = 128;

        constexpr int REGS_A = 3;
        constexpr int REGS_B = 4;
        constexpr int REGS_B_SIZE = REGS_B * OptimalSIMDSize<DataType>::value;

        Tensor<DataType> result = zeros<DataType>(Shape{left.shape[0], right.shape[1]});

        const DataType* l_ptr = left.data.data;
        const DataType* r_ptr = right.data.data;
        DataType*       o_ptr = result.data.data;

        const int m = result.shape[0];
        const int n = result.shape[1];
        const int k = left.shape[1];

        /// Tile A into mc * kc blocks, where mc and kc are chosen to approximately fit
        /// the L2 cache on recent Intel processors (e.g., 256 KB for Skylake).  Stream
        /// kc * n panels of B through memory to compute each mc * n block of C.
        /// \p a is an \p m x \p k row-major matrix;
        /// \p b is a \p k x \p n row-major matrix;
        /// \p c is a \p m x \p n row-major matrix.
        /// \p lda, \p ldb, and \p ldc are the leading dimensions of A, B, and C,
        /// respectively.
        for (int c = 0; c < k; c += BLOCK_COLS) {
            int bounded_c = std::min(k - c, BLOCK_COLS);
            for (int r = 0; r < m; r += BLOCK_ROWS) {
                int bounded_r = std::min(m - r, BLOCK_ROWS);

                const DataType* l_block = l_ptr + r * k + c;
                const DataType* r_block = r_ptr + c * n;
                DataType*       o_block = o_ptr + r * n;

                /// Compute a portion of C one block at a time.  Handle ragged edges with calls
                /// to a slow but general helper.
                ///
                /// The tiling scheme naturally divides the input matrices into 2 parts each;
                /// one tiled section, and three "ragged" edges.
                ///
                /// --------------------    -------
                /// | A00*B00 | A00*B01|    | A00 |   -------------
                /// -------------------- += ------- * | B00 | B01 |
                /// | A10*B00 | A10*B01|    | A10 |   -------------
                /// --------------------    -------
                ///
                /// We can process this as 4 separate matrix multiplications.  A00*B00 is the
                /// perfectly-tiled portion, which we handly with a 4x16 dot-product kernel.
                /// The ragged edges are (ideally) less critical, so we handle them with a call
                /// to a general matrix-multiplication for odd sizes.
                for (int rb = 0; rb < n - REGS_B_SIZE + 1; rb += REGS_B_SIZE) {
                    for (int ra = 0; ra < bounded_r - REGS_A + 1; ra += REGS_A) {
                        const DataType* l_inner_block = l_block + bounded_c * ra;
                        const DataType* r_inner_block = r_block + rb;
                        DataType* o_inner_block       = o_block + n * ra + rb;
                        matmul_dot<REGS_A, REGS_B>(n, bounded_c, l_inner_block, r_inner_block, o_inner_block);
                    }
                }

                int rem_a = (bounded_r / REGS_A) * REGS_A;
                int rem_b = (n / REGS_B_SIZE) * REGS_B_SIZE;

                if (rem_a < bounded_r) {
                    const DataType* l_inner_block = l_block + bounded_c * rem_a;
                    const DataType* r_inner_block = r_block;
                    DataType* o_inner_block       = o_block + rem_a * n;

                    matmul_odd(bounded_r - rem_a, rem_b, bounded_c, l_inner_block, r_inner_block, o_inner_block);
                }

                if (rem_b < n) {
                    const DataType* l_inner_block = l_block;
                    const DataType* r_inner_block = r_block + rem_b;
                    DataType* o_inner_block       = o_block + rem_b;

                    matmul_odd(rem_a, n - rem_b, bounded_c, l_inner_block, r_inner_block, o_inner_block);
                }

                if (rem_a < bounded_r && rem_b < n) {
                    const DataType* l_inner_block = l_block + bounded_c * rem_a;
                    const DataType* r_inner_block = r_block + rem_b;
                    DataType* o_inner_block       = o_block + rem_a * n + rem_b;

                    matmul_odd(bounded_r - rem_a, n - rem_b, bounded_c, l_inner_block, r_inner_block, o_inner_block);
                }
            }
        }

        return result;
    }

private:
    /// Naive gemm helper to handle oddly-sized matrices.
    static void matmul_odd(int m, int n, int k, const DataType* a, const DataType* b, DataType* c)
    {
        for (int p = 0; p < k; p++) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                   c[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
    }

    /// Compute a RAxRB block of C using a vectorized dot product, where RA is the
    /// number of registers to load from matrix A, and RB is the number of registers
    /// to load from matrix B.
    template <int regsA, int regsB>
    static void matmul_dot(int l_cols, int r_cols, const DataType* l_ptr, const DataType* r_ptr, DataType* o_ptr)
    {
        const DataType zero_val = 0;

        VecType csum[regsA][regsB];
        for (int ai = 0; ai < regsA; ai++) {
            for (int bi = 0; bi < regsB; bi++) {
                csum[ai][bi] = simdpp::load_splat<VecType>(&zero_val);
            }
        }

        for (int p = 0; p < l_cols; p++) {
            // Perform the DOT product.
            for (int bi = 0; bi < regsB; bi++) {
                VecType bb = LoadSIMDType<DataType, DataType>::load(r_ptr + p * r_cols + bi * 8);
                for (int ai = 0; ai < regsA; ai++) {
                    VecType aa = simdpp::load_splat<VecType>(l_ptr + ai * l_cols + p);
                    csum[ai][bi] = simdpp::add(csum[ai][bi], MultiplySIMD<DataType>::run(aa, bb));
                }
            }
        }

        // Accumulate the results into C.
        for (int ai = 0; ai < regsA; ai++) {
            for (int bi = 0; bi < regsB; bi++) {
                DataType* ptr = o_ptr + ai * r_cols + bi * 8;

                VecType block = LoadSIMDType<DataType, DataType>::load(ptr);
                VecType result = simdpp::add(block, csum[ai][bi]);
                simdpp::store(ptr, result);
            }
        }
    }
};

} // namespace detail

// ----------------------------------------------------------------------------
// Unit tests

using multiply_data_types = doctest::Types<uint16_t, uint32_t,
                                           int16_t, int32_t,
                                           float, double>;

TEST_CASE_TEMPLATE("matrix_multiply(const Tensor<T>&, const Tensor<T>&)", T, multiply_data_types)
{
    using TensorType = Tensor<T>;

    { // 2x2
        T left_data[4]   = {1, 2, 3, 4};
        T right_data[4]  = {5, 6, 7, 8};
        T result_data[4] = {19, 22, 43, 50};

        TensorType left(Shape{2, 2}, AlignedPtr<T>(left_data, 4));
        TensorType right(Shape{2, 2}, AlignedPtr<T>(right_data, 4));
        TensorType result(Shape{2, 2}, AlignedPtr<T>(result_data, 4));

        REQUIRE(matrix_multiply(left, right) == result);
        REQUIRE(left.mul(right) ==  result);
    }

    { // 2x3 * 3x4
        T left_data[6] = {1, 2, 3, 1, 2, 3};
        T right_data[12] = {4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7};
        T result_data[8] = {24, 30, 36, 42, 24, 30, 36, 42};

        TensorType left(Shape{2, 3}, AlignedPtr<T>(left_data, 6));
        TensorType right(Shape{3, 4}, AlignedPtr<T>(right_data, 12));
        TensorType result(Shape{2, 4}, AlignedPtr<T>(result_data, 8));

        REQUIRE(matrix_multiply(left, right) == result);
        REQUIRE((left.mul(right) == result));
    }

    { // 4x4
        T left_data[16]   = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        T right_data[16]  = {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1};
        T result_data[16] = {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1};

        TensorType left(Shape{4, 4}, AlignedPtr<T>(left_data, 16));
        TensorType right(Shape{4, 4}, AlignedPtr<T>(right_data, 16));
        TensorType result(Shape{4, 4}, AlignedPtr<T>(result_data, 16));

        REQUIRE(matrix_multiply(left, right) == result);
        REQUIRE((left.mul(right) == result));
    }

    REQUIRE_THROWS(matrix_multiply(TensorType(Shape{2, 2}), TensorType(Shape{3, 3})));
    REQUIRE_THROWS(matrix_multiply(TensorType(Shape{3, 3, 2}), TensorType(Shape{2, 3, 3})));
}

} // namespace tnt

#endif // TNT_LINEAR_MATRIX_MULTIPLY_IMPL_HPP
