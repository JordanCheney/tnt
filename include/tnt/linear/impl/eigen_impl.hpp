#ifndef TNT_LINEAR_EIGEN_IMPL_HPP
#define TNT_LINEAR_EIGEN_IMPL_HPP

#include <tnt/linear/eigen.hpp>
#include <tnt/utils/testing.hpp>

namespace tnt
{

namespace detail
{

template <typename DataType>
struct OptimizedEigenvalues<DataType>
{
    static Tensor<DataType> eval(const Tensor<DataType>& tensor, int max_iterations, float eps)
    {
        Tensor<DataType> A = tensor;
        int p, q;

        // Find largest off diagonal value
        for (int i = 0; i < max_iterations; ++i) {
            DataType max_value = -std::numeric_limits<DataType>::max();
            for (int r = 0; r < A.shape[0]; ++r) {
                for (int c = 0; c < A.shape[1]; ++c) {
                    if (r == c)
                        continue;

                    const DataType value = std::abs((DataType) A(r, c));
                    if (value > max_value) {
                        max_value = value;
                        p = r;
                        q = c;
                    }
                }
            }

            if (max_value < eps) // basically diagonal
                break;

            const DataType pp = A(p, p);
            const DataType qq = A(q, q);
            const DataType pq = A(p, q);

            const DataType tau = (qq - pp) / (2 * pq);
            const DataType phi = ((tau > 0) - (tau < 0)) / (std::abs(tau) + std::sqrt(1 + std::pow(tau, 2)));

            const DataType c = 1 / std::sqrt(1 + std::pow(phi, 2));
            const DataType s = phi * c;

            Tensor<DataType> givens = identity<DataType>(tensor.shape);
            givens(p, p) =  c;
            givens(q, q) =  c;
            givens(p, q) =  s;
            givens(q, p) = -s;

            A = givens.transpose().mul(A);
            A = A.mul(givens);
        }

        Tensor<DataType> evals(Shape{tensor.shape[0]});
        for (int r = 0; r < tensor.shape[0]; ++r)
            evals(r) = (DataType) A(r, r);

        return evals;
    }
};

} // namespace detail

TEST_CASE_TEMPLATE("eigenvalues()", T, test_float_data_types)
{
    { // 3x3 case
        T data[9] = {2, 0, 0, 0, 3, 4, 0, 4, 9};
        T expected[3] = {2, 1, 11};

        REQUIRE((approx_equal(eigenvalues(Tensor<T>(Shape{3, 3}, AlignedPtr<T>(data, 9))),
                                          Tensor<T>(Shape{3}, AlignedPtr<T>(expected, 3)))));
    }

    REQUIRE_THROWS((eigenvalues(Tensor<T>(Shape{3, 4}))));
    REQUIRE_THROWS((eigenvalues(Tensor<T>(Shape{2, 2, 2}))));
}

} // namespace tnt

#endif // TNT_LINEAR_EIGEN_IMPL_HPP
