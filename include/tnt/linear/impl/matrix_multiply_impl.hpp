#ifndef TNT_LINEAR_MATRIX_MULTIPLY_IMPL_HPP
#define TNT_LINEAR_MATRIX_MULTIPLY_IMPL_HPP

#include <tnt/linear/matrix_multiply.hpp>
#include <tnt/deps/cblas.hpp>
#include <tnt/utils/testing.hpp>

namespace tnt
{

namespace detail
{

template <typename LeftType, typename RightType>
struct OptimizedMatrixMultiply<LeftType, RightType,
          typename std::enable_if<std::is_integral<LeftType>::value>::type>
{
    static Tensor<LeftType> eval(const Tensor<LeftType>& left, const Tensor<RightType>& right)
    {
        Tensor<LeftType> result(Shape{left.shape[0], right.shape[1]});

        const LeftType*  l_ptr = left.data.data;
        const RightType* r_ptr = right.data.data;
        LeftType*        o_ptr = result.data.data;

        const int l_step = left.shape[1];
        const int r_step = right.shape[1];

        for (int n = 0; n < left.shape[0]; ++n) {
            const LeftType* l_row = l_ptr + n * l_step;
            for (int k = 0; k < right.shape[1]; ++k) {
                const RightType* r_col = r_ptr + k;

                LeftType total = 0;
                for (int m = 0; m < left.shape[1]; ++m)
                    total += l_row[m] * r_col[m * r_step];

                o_ptr[n * r_step + k] = total;
            }
        }

        return result;
    }
};

template <typename RightType>
struct OptimizedMatrixMultiply<float, RightType, void>
{
    static Tensor<float> eval(const Tensor<float>& left, const Tensor<RightType>& _right)
    {
        Tensor<float> right = _right.template as<float>();

        const int M = left.shape[0];
        const int N = right.shape[1];
        const int K = left.shape[1];

        Tensor<float> result(Shape{M, N});

        /* See https://www.christophlassner.de/using-blas-from-c-with-row-major-data.html */
        /* For trick to use fortran order (col-major) */
        cblas_sgemm(CblasColMajor,    /* Memory order */
                    CblasNoTrans,     /* Left is transposed? */
                    CblasNoTrans,     /* Right is transposed? */
                    N,                /* Rows of left */
                    M,                /* Columns of right */
                    K,                /* Columns of left and rows of right */
                    1.,               /* Alpha (scale after mult) */
                    right.data.data,  /* Left data */
                    N,                /* lda, this is really for 2D mats */
                    left.data.data,   /* right data */
                    K,                /* ldb, this is really for 2D mats */
                    0.,               /* beta (scale on result before addition) */
                    result.data.data, /* result data */
                    N);               /* ldc, this is really for 2D mats */

        return result;
    }
};

template <typename RightType>
struct OptimizedMatrixMultiply<double, RightType, void>
{
    static Tensor<double> eval(const Tensor<double>& left, const Tensor<RightType>& _right)
    {
        Tensor<double> right = _right.template as<double>();

        const int M = left.shape[0];
        const int N = right.shape[1];
        const int K = left.shape[1];

        Tensor<double> result(Shape{M, N});

        /* See https://www.christophlassner.de/using-blas-from-c-with-row-major-data.html */
        /* For trick to use fortran order (col-major) */
        cblas_dgemm(CblasColMajor,    /* Memory order */
                    CblasNoTrans,     /* Left is transposed? */
                    CblasNoTrans,     /* Right is transposed? */
                    N,                /* Rows of left */
                    M,                /* Columns of right */
                    K,                /* Columns of left and rows of right */
                    1.,               /* Alpha (scale after mult) */
                    right.data.data,  /* Left data */
                    N,                /* lda, this is really for 2D mats */
                    left.data.data,   /* right data */
                    K,                /* ldb, this is really for 2D mats */
                    0.,               /* beta (scale on result before addition) */
                    result.data.data, /* result data */
                    N);               /* ldc, this is really for 2D mats */

        return result;
    }
};

} // namespace detail

// ----------------------------------------------------------------------------
// Unit tests

TEST_CASE_TEMPLATE("matrix_multiply(const Tensor<T>&, const Tensor<T>&)", T, test_data_types)
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
