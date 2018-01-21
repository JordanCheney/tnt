#ifndef TNT_EIGEN_HPP
#define TNT_EIGEN_HPP

#include <tnt/core/tensor.hpp>

namespace tnt
{

namespace detail
{

template <typename DataType, typename Enable = void>
struct OptimizedEigenvalues
{
    static Tensor<DataType> eval(const Tensor<DataType>&, int, float);
};

} // namespace detail

/// \brief Compute the eigenvalues of a square 2D tensor
///
/// \param tensor A 2D floating point tensor to compute eigenvalues from
/// \param max_iterations The maximum number of iterations to run before stopping
/// \param eps Early stopping criteria. If the largest off-diagonal value of the
/// intermediate matrix is less than eps, stop early.
/// \requires Type `DataType` shall be floating
/// \requires Parameter [tensor](*::tensor)shall be 2D
template <typename DataType>
inline Tensor<DataType> eigenvalues(const Tensor<DataType>& tensor, int max_iterations = 1000, float eps = 0.001)
{
    static_assert(std::is_floating_point<DataType>::value,
                    "eigenvalues() requires floating point data");

    TNT_ASSERT(tensor.shape.num_axes() == 2,
               InvalidParameterException("eigenvalues()",
                                         __FILE__,
                                         __LINE__,
                                         "Eigen decomposition requires a two dimensional matrix"))

    TNT_ASSERT(tensor.shape[0] == tensor.shape[1],
               InvalidParameterException("eigenvalues()",
                                         __FILE__,
                                         __LINE__,
                                         "Eigen decomposition requires a square matrix."))

    return detail::OptimizedEigenvalues<DataType>::eval(tensor, max_iterations, eps);
}

} // namespace tnt

#endif // TNT_EIGEN_HPP
